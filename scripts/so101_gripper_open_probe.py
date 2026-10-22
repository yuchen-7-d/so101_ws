"""ID6 空夹爪独立单程张开检查：新起点 +16 计数，不返回。

默认只读预览；--enable 才执行。只适用于已记录的 2083 附近释放姿态。
沿用原控制器的输出100、速度50、加速度1、到位±2及3秒超时。
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT = Path('/home/yu/so101_ws')
SOURCE = ROOT/'data/pose_mapping_checks/current_start_20669zol.json'
SOURCE_SHA = 'f5d434a90c3f3b54c3fdd309a1d95d6dbf0b7fb41f52f9008f2b88a723e0600f'
STOP = ROOT/'data/real_control_runs/gripper_cycle_20261022T203435_2e5_r740.jsonl'
STOP_SHA = '7d9727014015e415dfdb7fa1c0240feb1eeb6384e36e75f2145f538ffc3a8a59'
DEPENDENCIES = {
    'so101_gripper_cycle.py': 'ad2df28ca9cbb65d322b020073dbec4de0759790d291b0b868ab2d61673ff8aa',
    'so101_hold_current.py': '6888f895dcf069791e8f9b2db19a172e73d8697ac6656c439f8b72dbea60f0a3',
    'so101_center_gripper.py': 'aed92daf1e1d27d07307bfdca3c6d2acf0c0255efd36615efc11503d9c817205',
    'so101_verify_gripper_center.py': '78dfb0ea0f5187df61ba3f554b02316f3745c3b13fa676c3762e8fdd1e860f75',
}
for _name, _sha in DEPENDENCIES.items():
    if hashlib.sha256((ROOT/'scripts'/_name).read_bytes()).hexdigest() != _sha:
        raise RuntimeError('控制依赖已改变，需重新核对：'+_name)
sys.path.insert(0, str(ROOT/'scripts'))
import so101_gripper_cycle as base

START_LOW, START_HIGH, DELTA = 2081, 2085, 16


def require(condition, message):
    if not condition:
        raise base.ControlError(message)


def load_reference():
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, '释放后起点快照改变')
    state = json.loads(raw)
    require(state['success'] is True and state['status'] == 'read_only_start_snapshot'
            and type(state['register_writes']) is int and state['register_writes'] == 0,
            '释放后快照无效')
    require(state['mapping_sha256'] == hashlib.sha256(
        (ROOT/'data/so101_joint_mapping.json').read_bytes()).hexdigest(), '软件映射已改变')
    require(len(state['feedback_rounds']) == 3, '快照缺少三帧反馈')
    for r in state['feedback_rounds']:
        require(len(r['states']) == 6 and {s['servo_id'] for s in r['states']} == set(base.IDS),
                '快照关节不完整')
        for s in r['states']:
            require(s['torque_enabled'] is False and s['configuration_locked'] is True
                    and s['speed_raw'] == s['output_raw'] == s['status'] == 0,
                    '快照不是静止释放状态')
            if s['servo_id'] == 6:
                require(s['position_counts'] == 2083 and s['output_limit'] == 0,
                        '快照ID6不是本次释放点2083')
    configs = {k: v['raw_hex'] for k, v in state['config_before'].items()}
    for sid in base.IDS:
        require(configs[str(sid)] == state['config_after'][str(sid)]['raw_hex'], '快照配置不稳定')
    raw = STOP.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == STOP_SHA, '原失败记录改变')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(rows[-1]['stage'] == 'final_cleanup', '原记录缺少最终释放结果')
    stopped = rows[-1]['snapshot']
    require(stopped['test'] == 'id6_observed_wide_empty_cycle'
            and stopped['status'] == 'failed_no_return' and stopped['current_goal'] == 2080
            and stopped['release'] == {'servo_id': 6, 'readback': {'40': 0, '48': 0},
                                       'errors': [], 'confirmed': True}
            and stopped['journal_error'] is None and stopped['initial_configs'] == configs,
            '原停止/释放/配置与新快照不匹配')
    return {
        'scope': 'independent empty gripper positive step only; original wide cycle remains incomplete',
        'released_snapshot': {'path': str(SOURCE), 'sha256': SOURCE_SHA},
        'stopped_record': {'path': str(STOP), 'sha256': STOP_SHA},
        'configuration_hex': configs,
        'supported_pose_counts': {str(s['servo_id']): s['position_counts'] for s in state['states']},
        'dependency_sha256': DEPENDENCIES,
    }


def fixed_probe_path(start):
    require(type(start) is int and START_LOW <= start <= START_HIGH,
            '本次起点须在2081–2085；停止而不自动恢复旧位置')
    return [start, start+DELTA]


class ProbeBus(base.GripperBus):
    def authorize_path(self, path):
        require(len(path) == 2 and all(type(p) is int for p in path)
                and path == fixed_probe_path(path[0]), '只允许新起点加16的冻结路径')
        self.allowed_goals = frozenset(path)
        self.enabled = True


class OpeningProbe(base.GripperCycle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_progress_at = 0.0

    def preflight(self):
        self.bus.phase = 'preflight'
        require(self.open_only is True, '本次只提供独立单程张开检查')
        for sid in base.IDS:
            raw = self.bus.read(sid, 0, 40)
            require(raw.hex() == self.proof['configuration_hex'][str(sid)],
                    f'ID{sid}完整配置与释放快照不一致')
            require(raw[2] == 0 and raw[5] == sid and raw[8] == 1 and raw[18] == 12
                    and raw[30] == 1 and raw[33] == 0
                    and base.u16(raw[9:11]) == 0 and base.u16(raw[11:13]) == 4095,
                    f'ID{sid}不符合已记录的单圈模式')
            self.configs[sid] = raw
        require(100 <= base.u16(self.configs[6][16:18]) <= 1000, 'ID6设备输出上限不适用')
        observations = []
        for index in range(3):
            if index:
                self.sleep(.1)
            states = {sid: self.read_frame(sid) for sid in base.IDS}
            for sid, s in states.items():
                require(s['torque'] == s['speed_raw_signed'] == s['output_raw_signed'] == 0
                        and s['lock'] == 1, f'ID{sid}需静止、释放且配置锁定')
                self.initial.setdefault(sid, s['position'])
                if sid != 6:
                    limit = 3 if sid == 4 else 12
                    require(abs(s['position']-self.proof['supported_pose_counts'][str(sid)]) <= limit,
                            f'ID{sid}已离开本次支撑姿态')
            s = states[6]
            require(START_LOW <= s['position'] <= START_HIGH and s['output_limit'] == 0,
                    'ID6需在2081–2085，输出上限为0')
            observations.append((self.clock(), s['position']))
        require(observations[-1][0]-observations[0][0] >= .2
                and max(p for _, p in observations)-min(p for _, p in observations) <= 2,
                'ID6三帧未稳定')
        self.start = observations[-1][1]
        self.path = fixed_probe_path(self.start)
        self.current_goal = self.start
        self.feedback_bounds = (self.start-2, self.path[-1]+2)
        require(base.u16(self.configs[6][9:11]) <= self.feedback_bounds[0]
                < self.feedback_bounds[1] <= base.u16(self.configs[6][11:13]), '反馈走廊超设备限位')
        self.emit(f'独立单程张开：{self.start} → {self.path[-1]}；+16计数；不返回。')
        self.emit('输出100/1000、速度50、加速度1；到位±2计数、每步3秒；PID/offset不变。')

    def recheck_after_journal(self):
        super().recheck_after_journal()
        for sid in range(1, 6):
            require(self.bus.read(sid, 0, 40) == self.configs[sid], f'ID{sid}保存日志后配置改变')
        require(self.path == fixed_probe_path(self.start), '冻结路径改变')

    def monitor(self, goal, torque, output, corridor):
        state = super().monitor(goal, torque, output, corridor)
        if (self.bus.phase in ('opening_only', 'opened_observation') and goal == self.path[-1]
                and self.clock() >= self.next_progress_at):
            self.next_progress_at = self.clock()+.25
            self.emit(f"ID6反馈：目标={goal}，实际={state['position']}，"
                      f"误差={state['position']-goal:+d}，速度={state['speed_raw_signed']}，"
                      f"输出幅值={state['load']}/1000，电压={state['voltage']:.1f}V")
        return state

    def cycle(self):
        super().cycle()
        self.open_response_complete = False
        self.bus.phase = 'opened_observation'
        target = self.path[-1]
        self.emit('单程到位；继续监测2秒后释放ID6，不返回起点。')
        end = self.clock()+2
        while self.clock() < end:
            self.monitor(target, 1, 100, (target-2, target+2))
            self.sleep(.05)
        self.check_config()
        self.open_response_complete = True

    def report(self):
        r = super().report()
        r.update(test='id6_positive_sixteen_probe', mode='opening_probe',
                 start_window=[START_LOW, START_HIGH], delta_counts=DELTA,
                 opening_probe_response_complete=self.open_response_complete,
                 original_wide_cycle_completed=False, motion_authorized_for_arm_ids=False,
                 note='Independent positive step with fixed fresh start; no return, zero/PID adjustment or grasp.')
        if r['status'] == 'empty_gripper_open_only_complete':
            r['status'] = 'empty_gripper_opening_probe_complete'
        if self.path:
            samples = [s for s in self.samples if s['servo_id'] == 6
                       and s['phase'] in ('opening_only', 'opened_observation')
                       and s['goal'] == self.path[-1]]
            if samples:
                last = samples[-1]
                r['last_probe_feedback'] = {k: last[k] for k in
                    ('elapsed_s', 'position', 'goal', 'speed_raw_signed', 'output_raw_signed', 'voltage')}
                r['last_probe_feedback']['error_counts'] = last['position']-last['goal']
                r['last_probe_feedback']['from_frozen_start_counts'] = last['position']-self.start
        return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--enable', action='store_true', help='实际执行；不提供则只读预览')
    args = parser.parse_args(argv)
    controller = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        proof = load_reference()
        import serial
        print('空夹爪；稳定支撑机械臂；夹指运动区空置；关闭其他串口程序。', flush=True)
        print('执行ID6单程张开；ID1–5不使能，结束或异常释放ID6。' if args.enable
              else '只读预览；寄存器写入=0，不执行运动。', flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15, write_timeout=.15, exclusive=True) as port:
            controller = OpeningProbe(ProbeBus(port), proof, journal_dir=ROOT/'data/real_control_runs',
                                      open_only=True)
            result = controller.run(args.enable)
        print(f"状态：{result['status']}；起点={result['start']}；写指令={result['writes_sent']}", flush=True)
        return 0
    except KeyboardInterrupt:
        print('已中止；不执行回程。', flush=True)
        return 130
    except (base.ControlError, RuntimeError, OSError, ValueError, KeyError) as error:
        print(f'本次未完成：{error}', file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if controller is not None:
            last = controller.report().get('last_probe_feedback')
            if last:
                print('最后一次运动反馈：', last, flush=True)
            if controller.journal is not None:
                print('持久journal：', controller.journal.path, flush=True)
                print('ID6释放读回：', controller.release_result, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
