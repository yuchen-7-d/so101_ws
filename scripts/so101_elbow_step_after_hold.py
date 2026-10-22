"""ID3从本次冻结起点减少8计数；默认只读，--enable执行；不回程。

其余ID1–5保持，ID6不写。空夹爪，手在下方准备承接，不接触连杆。
异常立即尝试释放；正常到位后监测5秒，最后3秒提示承托并释放。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import tempfile
import time

ROOT = Path('/home/yu/so101_ws')
HOLD = ROOT / 'scripts/so101_hold_open_pose.py'
HOLD_SHA = 'a57c38ba1ab41d9a7e58dce2d27524d23667ced94cf0a1f62f099e3ceb4eea97'
GEOMETRY = Path('/tmp/so101_elbow_step_x47pzmdj.json')
GEOMETRY_SHA = '66ee2856b3fcee58f8bc26e2253a654139715eaecce72059f69af41286c67da3'
if hashlib.sha256(HOLD.read_bytes()).hexdigest() != HOLD_SHA:
    raise RuntimeError('原位保持模块已改变，需重新核对')
sys.path.insert(0, str(ROOT / 'scripts'))
import so101_hold_open_pose as hold
base, require = hold.base, hold.require


def load_reference():
    proof = hold.load_reference()
    raw = GEOMETRY.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == GEOMETRY_SHA, '单轴几何报告已改变')
    geometry = json.loads(raw)
    require(geometry['sampled_geometry_complete'] is True and geometry['error'] is None
            and geometry['axis'] == 3 and geometry['delta_counts'] == -8
            and geometry['execution_available'] is False, '单轴采样未完成或范围不匹配')
    for path, sha in {**geometry['input_files'], **geometry['model_files']}.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha,
                '几何报告的输入发生变化：' + path)
    variants = geometry['start_pose_variants']
    samples = geometry['checked_samples']
    require(len(variants) == 14 and len(samples) == 126
            and all(not s['penetrating_contacts'] for s in samples), '几何采样不完整或含穿透')
    for vi, variant in enumerate(variants):
        start = variant['counts']
        require(len(start) == 6 and all(type(v) is int and 0 <= v <= 4095 for v in start)
                and start[5] == 2143, '几何起点数据异常')
        for si in range(9):
            expected = list(start)
            expected[2] -= si
            item = samples[vi*9 + si]
            require(item['variant'] == vi and item['sample'] == si
                    and item['diagnostic_counts'] == expected, '单轴路径不匹配')
    proof['step_geometry'] = {'path': str(GEOMETRY), 'sha256': GEOMETRY_SHA,
                              'scope': geometry['source_scope'], 'variants': variants}
    proof['hold_module_sha256'] = HOLD_SHA
    return proof


def create_journal(directory, snapshot):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
    fd, path = tempfile.mkstemp(prefix='elbow_step_after_hold_'+stamp, suffix='.jsonl', dir=directory)
    journal = hold.DurableJournal(path, os.fdopen(fd, 'w', encoding='utf-8'))
    try:
        journal.append('before_any_write', snapshot)
        dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        journal.close()
        raise
    return journal


class StepBus(hold.HoldBus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_goals = {}
        self.step_armed = self.step_attempted = False

    def authorize(self, goals):
        require(not self.enabled, '本次只能授权一次起点')
        super().authorize(goals)
        self.original_goals = dict(goals)

    def arm_single_step(self):
        require(self.enabled and not self.step_armed and self.phase == 'elbow_step',
                '单步运动阶段不匹配')
        goal = self.original_goals[3] - 8
        require(0 <= goal <= 4095, '单步目标跨过计数边界')
        self.goals[3] = goal
        self.step_armed = True
        return goal

    def exchange(self, sid, instruction, params, size, tolerate_alarm=False):
        if instruction == 3 and sid == 3 and params and params[0] == base.GOAL and self.step_armed:
            require(self.phase == 'elbow_step' and not self.step_attempted
                    and len(params) == 3 and size == 0
                    and int.from_bytes(bytes(params[1:]), 'little') == self.original_goals[3]-8,
                    '仅允许一次ID3减8目标，禁止回程、重发或追加')
            self.step_attempted = True  # Mark before sending, including an ACK failure.
        return super().exchange(sid, instruction, params, size, tolerate_alarm)


class ElbowStep(hold.PoseHold):
    def __init__(self, bus, proof, *, journal_dir, emit=print, clock=time.monotonic,
                 sleep=time.sleep, journal_factory=create_journal):
        super().__init__(bus, proof, journal_dir=journal_dir, emit=emit, clock=clock,
                         sleep=sleep, journal_factory=journal_factory)
        self.frozen = {}
        self.baseline = None
        self.step_goal = None
        self.step_readback = self.response_complete = False
        self.step_frames = []

    def preflight(self):
        super().preflight()
        self.frozen = dict(self.targets)
        start = [self.frozen[i] for i in base.ARM_IDS] + [self.gripper_start]
        accepted = [v['counts'] for v in self.proof['step_geometry']['variants']]
        require(start in accepted,
                '本次起点不在已检查的14种保存姿态中；零写入停止，请保存新快照再核对，不要按计数推拉')
        self.step_goal = self.frozen[3] - 8
        self.emit(f'固定单程计划：ID3 {self.frozen[3]} → {self.step_goal}；不自动返回。')

    def monitor(self, expected_torque, tolerance, expected_output):
        started = self.clock()
        states = super().monitor(expected_torque, tolerance, expected_output)
        for sid in (1, 2, 4, 5):
            require(abs(states[sid].position-self.targets[sid]) <= 3,
                    f'ID{sid}偏离本次保持点超过3计数')
        g0 = self.frozen.get(3, self.targets[3])
        lo, hi = (g0-10, g0+2) if self.bus.step_armed else (g0-3, g0+3)
        require(lo <= states[3].position <= hi, 'ID3反馈超出本次局部范围，停止且不回程')
        e = hold.frame_fields(self.bus.read(3, 40, 26))
        base.check_health(3, base.Feedback(e['position'], e['load'], e['voltage'],
                                         e['temperature'], e['status']), *self.configs[3])
        require(lo <= e['position'] <= hi and e['torque'] == expected_torque
                and e['goal'] == self.targets[3] and e['output_limit'] == expected_output
                and e['speed_limit'] == 50 and e['acceleration'] == 1 and e['lock'] == 1,
                'ID3完整反馈或控制寄存器异常')
        require(self.clock()-started <= .5, '含ID3完整反馈的整轮读取超过0.5秒')
        self.step_frames.append({'elapsed_s': self.clock()-self.bus.started,
                                 'phase': self.bus.phase, **e})
        return e

    def collect_baseline(self):
        self.bus.phase = 'loaded_baseline'
        end, window = self.clock()+3, []
        while self.clock() < end:
            e = self.monitor(1, base.HOLD_TOLERANCE, 100)
            now = self.clock()
            require(abs(e['position']-self.frozen[3]) <= 2,
                    'ID3有载偏差超过2计数，停止；不调整目标补偿')
            if e['speed_raw_signed'] != 0:
                window = []
            else:
                window.append((now, e['position']))
                if max(p for _, p in window)-min(p for _, p in window) > 1:
                    window = [window[-1]]
                if len(window) >= 3 and window[-1][0]-window[0][0] >= .5:
                    self.baseline = int(statistics.median(p for _, p in window))
                    return
            self.sleep(.05)
        raise base.ControlError('ID3有载基线3秒内未稳定；未发送运动目标')

    def arrived(self, e):
        return (e['speed_raw_signed'] == 0
                and abs(e['position']-self.step_goal) <= 2
                and abs(e['position']-(self.baseline-8)) <= 2)

    def hold(self):
        self.bus.phase = 'ramp'
        previous = 0
        for level in (25, 50, 75, 100):
            self.monitor(1, base.HOLD_TOLERANCE, previous)
            started = self.clock()
            for sid in base.ARM_IDS:
                self.bus.set_checked(sid, base.OUTPUT, level)
                require(self.clock()-started <= .5, '输出分级写入超过0.5秒')
            self.monitor(1, base.HOLD_TOLERANCE, level)
            self.sleep(.1)
            previous = level
        self.emit('采集有载基线：手在下方准备承接，保持间隙，不接触或推拉连杆。稳定后自动执行一次。')
        self.collect_baseline()
        self.emit(f'固定起点G0={self.frozen[3]}；有载P0={self.baseline}；目标={self.step_goal}。')
        self.emit(f'同时核验：目标±2，及相对P0减少8±2；3秒内未达标即停止，不加输出。')
        began = self.clock()
        self.journal.append('before_single_step', self.report())
        require(self.clock()-began <= .5, '运动前记录保存超过0.5秒，停止')
        e = self.monitor(1, base.HOLD_TOLERANCE, 100)
        require(e['speed_raw_signed'] == 0 and abs(e['position']-self.baseline) <= 1,
                '发出运动前有载基线已改变')
        self.bus.phase = 'elbow_step'
        require(self.bus.arm_single_step() == self.step_goal, '单步目标不一致')
        self.targets[3] = self.step_goal
        began = self.clock()
        self.bus.set_checked(3, base.GOAL, self.step_goal)
        require(self.clock()-began <= .5, '单步写入及读回超过0.5秒')
        self.step_readback = True
        end, stable, next_print = began+3, [], self.clock()
        while self.clock() < end:
            e = self.monitor(1, base.HOLD_TOLERANCE, 100)
            now = self.clock()
            if now >= next_print:
                self.emit(f"ID3实际={e['position']}；目标误差={e['position']-self.step_goal:+d}；"
                          f"相对P0={e['position']-self.baseline:+d}；输出={e['load']}/1000")
                next_print = now+.25
            stable = stable+[now] if self.arrived(e) else []
            if now <= end and len(stable) >= 3 and stable[-1]-stable[0] >= .15:
                break
            self.sleep(.05)
        else:
            raise base.ControlError('ID3单步3秒内未同时满足目标与相对位移要求；不执行回程')
        self.emit('单步反馈达标；再监测5秒，最后3秒提示承托，然后释放，不返回起点。')
        self.bus.phase = 'after_step'
        end, prompted = self.clock()+5, False
        while self.clock() < end:
            if end-self.clock() <= 3 and not prompted:
                prompted = True
                self.bus.phase = 'support_requested'
                self.emit('还剩约3秒：现在从下方托住机械臂，保持支撑直到释放完成。')
            e = self.monitor(1, base.HOLD_TOLERANCE, 100)
            require(abs(e['position']-self.step_goal) <= 2
                    and abs(e['position']-(self.baseline-8)) <= 2, '到位后ID3偏差增大，停止且不回程')
            self.sleep(.05)
        self.response_complete = True

    def report(self):
        r = super().report()
        if r['status'] == 'current_pose_hold_complete':
            r['status'] = 'elbow_negative_eight_complete' if self.response_complete else 'incomplete'
        r.update(test='elbow_negative_eight_after_current_hold',
                 frozen_start_targets=self.frozen, loaded_baseline=self.baseline,
                 step_goal=self.step_goal, step_goal_write_attempted=self.bus.step_attempted,
                 step_goal_readback_confirmed=self.step_readback,
                 step_response_complete=self.response_complete, step_frames=self.step_frames,
                 motion_path_executed=self.response_complete, return_executed=False,
                 arm_tolerance_counts=3, active_axis_local_counts=[-10, 2],
                 settle_tolerance_counts=2, motion_timeout_seconds=3, seconds=5,
                 note='One fixed ID3 step; P0 validates displacement only, never shifts the goal. '
                      'Saved nominal geometry, not a dynamics or full feedback envelope proof.')
        return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--enable', action='store_true', help='使能ID1–5，执行ID3减8单程；默认只读')
    args = parser.parse_args(argv)
    controller = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        proof = load_reference()
        import serial
        print('空夹爪，关闭其他串口程序；手在下方准备承接，不接触连杆；异常或结束释放ID1–5。', flush=True)
        print('ID3单程减8，其余关节保持本次读数，ID6不写入；输出100、速度50、加速度1。', flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            controller = ElbowStep(StepBus(port), proof, journal_dir=ROOT/'data/real_control_runs',
                                   emit=lambda text: print(text, flush=True))
            r = controller.run(args.enable)
        print('状态：', r['status'], flush=True)
        return 0
    except KeyboardInterrupt:
        print('已中止；请查看逐关节释放读回。', flush=True)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print('本次未完成：', error, file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if controller is not None and controller.journal is not None:
            print('持久journal：', controller.journal.path, flush=True)
            print('释放读回：', controller.release_result, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
