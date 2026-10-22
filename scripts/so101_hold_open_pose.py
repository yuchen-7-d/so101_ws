"""固定当前姿态保持10秒；默认只读，--enable才使能ID1–5。

目标只取新读数，输出固定分级至100/1000，ID6不写入。
手在下方准备承接，最后3秒托住；退出或异常释放本次控制的关节。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT = Path('/home/yu/so101_ws')
SOURCE = ROOT/'data/real_control_runs/gripper_cycle_20261022T205227_pifksuzk.jsonl'
SOURCE_SHA = '9a58da30f85c91c666532c5bf3705dc97cda859d27beb6ec5cc6a66c1b26407a'
MAPPING_SHA = 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8'
DEPENDENCIES = {
    'so101_hold_current.py': '6888f895dcf069791e8f9b2db19a172e73d8697ac6656c439f8b72dbea60f0a3',
    'so101_gripper_cycle.py': 'ad2df28ca9cbb65d322b020073dbec4de0759790d291b0b868ab2d61673ff8aa',
    'so101_center_gripper.py': 'aed92daf1e1d27d07307bfdca3c6d2acf0c0255efd36615efc11503d9c817205',
    'so101_verify_gripper_center.py': '78dfb0ea0f5187df61ba3f554b02316f3745c3b13fa676c3762e8fdd1e860f75',
}
for _name, _sha in DEPENDENCIES.items():
    if hashlib.sha256((ROOT/'scripts'/_name).read_bytes()).hexdigest() != _sha:
        raise RuntimeError('依赖已改变，需重新核对：'+_name)
sys.path.insert(0, str(ROOT/'scripts'))
import so101_hold_current as base
from so101_gripper_cycle import frame_fields, DurableJournal


def require(condition, message):
    if not condition:
        raise base.ControlError(message)


def load_reference():
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, '已完成张开记录改变')
    require(hashlib.sha256((ROOT/'data/so101_joint_mapping.json').read_bytes()).hexdigest()
            == MAPPING_SHA, '软件映射改变')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(rows[-1]['stage'] == 'final_cleanup', '缺少张开后释放记录')
    s = rows[-1]['snapshot']
    require(s['test'] == 'id6_open_to_measured_2144'
            and s['status'] == 'empty_gripper_open_2144_complete'
            and s['open_2144_response_complete'] is True
            and s['fault'] is None and s['journal_error'] is None
            and s['release'] == {'servo_id': 6, 'readback': {'40': 0, '48': 0},
                                'errors': [], 'confirmed': True}, '张开/释放未完整通过')
    require(s['last_motion_feedback']['position'] == 2143
            and s['last_motion_feedback']['goal'] == 2144, '张开最终反馈不匹配')
    pose = dict(s['initial_positions'])
    pose['6'] = 2143
    return {'source': {'path': str(SOURCE), 'sha256': SOURCE_SHA},
            'configuration_hex': s['initial_configs'], 'reference_pose_counts': pose,
            'mapping_sha256': MAPPING_SHA, 'dependency_sha256': DEPENDENCIES}


def create_journal(directory, snapshot):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
    fd, path = tempfile.mkstemp(prefix='arm_current_hold_'+stamp, suffix='.jsonl', dir=directory)
    journal = DurableJournal(path, os.fdopen(fd, 'w', encoding='utf-8'))
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


class HoldBus(base.Bus):
    def __init__(self, port, clock=time.monotonic):
        super().__init__(port)
        self.clock, self.started = clock, clock()
        self.phase, self.events = 'preflight', []
        self.goals, self.enabled, self.release_values = {}, False, {}

    def authorize(self, goals):
        require(set(goals) == set(base.ARM_IDS)
                and all(type(v) is int and 0 <= v <= 4095 for v in goals.values()), '保持目标不完整')
        self.goals, self.enabled = dict(goals), True

    def exchange(self, sid, instruction, params, size, tolerate_alarm=False):
        if instruction == 2:
            require(len(params) == 2 and params[1] == size and 1 <= size <= 40
                    and 0 <= params[0] <= 255-size, '读取范围错误')
        elif instruction == 3:
            require(self.enabled and sid in base.ARM_IDS and params
                    and params[0] in base.WRITE_WIDTHS, '只允许本次保持写ID1–5的控制寄存器')
            address = params[0]
            value = int.from_bytes(bytes(params[1:]), 'little')
            allowed = {40: (0, 1), 41: (1,), 42: (self.goals[sid],), 46: (50,),
                       48: (0, 25, 50, 75, 100)}
            require(size == 0 and len(params) == base.WRITE_WIDTHS[address]+1
                    and value in allowed[address], '禁止改变冻结目标或超出固定输出分级')
        else:
            raise base.ControlError('禁止非读写白名单指令')
        entry = {'elapsed_s': self.clock()-self.started, 'phase': self.phase, 'servo_id': sid,
                 'instruction': instruction, 'parameters_hex': bytes(params).hex(), 'acknowledged': False}
        self.events.append(entry)
        try:
            result = super().exchange(sid, instruction, params, size, tolerate_alarm)
            entry.update(response_payload_hex=result.hex(), acknowledged=True)
            if self.phase == 'release' and instruction == 2 and params[0] in (40, 48):
                self.release_values.setdefault(str(sid), {})[str(params[0])] = base.u16(result)
            return result
        except BaseException as error:
            entry['error'] = f'{type(error).__name__}: {error}'
            raise


class PoseHold(base.HoldController):
    def __init__(self, bus, proof, *, journal_dir, emit=print, clock=time.monotonic,
                 sleep=time.sleep, journal_factory=create_journal):
        super().__init__(bus, limit=100, seconds=10, emit=emit, clock=clock, sleep=sleep)
        self.proof, self.journal_dir, self.journal_factory = proof, journal_dir, journal_factory
        self.journal, self.journal_error = None, None
        self.status, self.fault, self.release_result = 'not_started', None, None
        self.raw_configs, self.samples, self.gripper_start = {}, [], None

    def check_configs(self):
        for sid in base.ALL_IDS:
            raw = self.bus.read(sid, 0, 40)
            require(raw.hex() == self.proof['configuration_hex'][str(sid)],
                    f'ID{sid}配置与张开记录不同')
            self.raw_configs[sid] = raw

    def check_released_start(self):
        for sid in base.ALL_IDS:
            s = frame_fields(self.bus.read(sid, 40, 26))
            base.check_health(sid, base.Feedback(s['position'], s['load'], s['voltage'],
                                               s['temperature'], s['status']), *self.configs[sid])
            require(s['torque'] == s['speed_raw_signed'] == s['output_raw_signed'] == 0
                    and s['lock'] == 1, f'ID{sid}需静止、释放且配置锁定')
            require(abs(s['position']-self.proof['reference_pose_counts'][str(sid)]) <= 3,
                    f'ID{sid}已离开本次参考姿态，停止而不自动恢复')
            if sid in base.ARM_IDS:
                require(abs(s['position']-self.targets[sid]) <= 3, f'ID{sid}偏离冻结起点')
            else:
                require(s['output_limit'] == 0, 'ID6输出上限须为0')
                if self.gripper_start is None:
                    self.gripper_start = s['position']
                require(abs(s['position']-self.gripper_start) <= 3, 'ID6张开位置发生变化')

    def preflight(self):
        self.check_configs()
        super().preflight()
        self.check_released_start()
        self.emit(f'目标仅取本次反馈：{self.targets}；ID6保持释放。')

    def prepare(self):
        self.journal = self.journal_factory(self.journal_dir, self.report())
        self.bus.phase = 'post_journal_recheck'
        self.check_configs()
        self.check_released_start()
        self.bus.authorize(self.targets)
        self.bus.phase = 'prepare'
        super().prepare()

    def monitor(self, expected_torque, tolerance, expected_output):
        began = self.clock()
        states = super().monitor(expected_torque, tolerance, expected_output)
        require(abs(states[4].position-self.targets[4]) <= min(tolerance, 3), 'ID4偏离保持点超过3计数')
        g = frame_fields(self.bus.read(6, 40, 26))
        base.check_health(6, base.Feedback(g['position'], g['load'], g['voltage'],
                                           g['temperature'], g['status']), *self.configs[6])
        require(g['torque'] == g['output_limit'] == g['speed_raw_signed'] == g['output_raw_signed'] == 0
                and g['lock'] == 1 and abs(g['position']-self.gripper_start) <= 3,
                'ID6释放状态或张开位置改变')
        require(self.clock()-began <= .5, '整轮反馈超过0.5秒')
        self.samples.append({'elapsed_s': self.clock()-self.bus.started, 'phase': self.bus.phase,
            'output_limit': expected_output, 'states': {str(i): {
                **vars(s), 'error_counts': s.position-self.targets[i]} for i,s in states.items()},
            'gripper': g})
        return states

    def hold(self):
        self.bus.phase = 'ramp'
        previous = 0
        for level in (25, 50, 75, 100):
            self.monitor(1, base.HOLD_TOLERANCE, previous)
            began = self.clock()
            for sid in base.ARM_IDS:
                self.bus.set_checked(sid, base.OUTPUT, level)
                require(self.clock()-began <= .5, '输出上限写入超过0.5秒')
            self.monitor(1, base.HOLD_TOLERANCE, level)
            self.sleep(.1)
            previous = level
        self.bus.phase = 'holding'
        end, next_print = self.clock()+10, self.clock()
        catch_prompted = False
        self.emit('原位保持10秒；手在下方准备承接。最后3秒会提示托住，结束即释放。')
        while self.clock() < end:
            if end-self.clock() <= 3 and not catch_prompted:
                catch_prompted = True
                self.bus.phase = 'support_requested'
                self.emit('还剩约3秒：现在从下方托住机械臂，保持支撑直到释放完成。')
            states = self.monitor(1, base.HOLD_TOLERANCE, 100)
            now = self.clock()
            if now >= next_print:
                detail = '  '.join(f'ID{i}:{s.position}(差{s.position-self.targets[i]:+d},输出{s.load}/1000)'
                                   for i,s in states.items())
                self.emit(f'剩余 {max(0,end-now):.1f}s  {detail}')
                next_print = now+.5
            self.sleep(.05)

    def release(self):
        self.bus.phase = 'release'
        unconfirmed = super().release()
        if self.touched:
            self.release_result = {'servo_ids': sorted(self.touched), 'unconfirmed_ids': unconfirmed,
                'readback': self.bus.release_values, 'confirmed': not unconfirmed}
        return unconfirmed

    def run(self, enable=False):
        primary = None
        try:
            super().run(enable)
            self.status = 'current_pose_hold_complete' if enable else 'read_only_preview'
        except BaseException as error:
            primary = error
            self.status = 'failed_after_write' if self.touched else 'failed_before_write'
            if self.touched and (not self.release_result or not self.release_result['confirmed']):
                self.status = 'release_unconfirmed'
            self.fault = f'{type(error).__name__}: {error}'
            raise
        finally:
            if self.journal is not None:
                try:
                    self.journal.append('final_cleanup', self.report())
                except BaseException as error:
                    self.journal_error = f'{type(error).__name__}: {error}'
                try:
                    self.journal.close()
                except BaseException as error:
                    self.journal_error = self.journal_error or str(error)
                if self.journal_error and primary is None:
                    self.status = 'journal_save_failed'
                    raise base.ControlError('释放已尝试，但最终记录保存失败：'+self.journal_error)
        return self.report()

    def report(self):
        return {'schema_version': 1, 'test': 'arm_current_pose_hold_after_open2144', 'status': self.status,
            'proof': self.proof, 'targets': self.targets, 'gripper_start': self.gripper_start,
            'initial_configs': {str(k):v.hex() for k,v in self.raw_configs.items()},
            'samples': self.samples, 'events': self.bus.events,
            'writes_sent': sum(e['instruction']==3 for e in self.bus.events),
            'release': self.release_result, 'fault': self.fault, 'journal_error': self.journal_error,
            'seconds': 10, 'output_limit': 100, 'arm_tolerance_counts': 12, 'wrist_tolerance_counts': 3,
            'gripper_tolerance_counts': 3, 'gripper_written': False, 'motion_path_executed': False,
            'independent_support_verified': False, 'model_mapping_verified': False, 'grasp_verified': False,
            'note': 'Fixed fresh pose only; physical support conditions require user observation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--enable', action='store_true', help='使能ID1–5保持10秒；默认只读')
    args = parser.parse_args(argv)
    controller = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        proof = load_reference()
        import serial
        print('关闭其他串口程序。手在下方准备承接，异常或结束会释放ID1–5；ID6不写入。', flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15, write_timeout=.15, exclusive=True) as port:
            controller = PoseHold(HoldBus(port), proof, journal_dir=ROOT/'data/real_control_runs',
                                  emit=lambda text: print(text, flush=True))
            r = controller.run(args.enable)
        print('状态：', r['status'], flush=True)
        return 0
    except KeyboardInterrupt:
        print('已中止；请查看逐关节释放读回。', flush=True)
        return 130
    except (base.ControlError, RuntimeError, OSError, ValueError, KeyError) as error:
        print('本次未完成：', error, file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if controller is not None and controller.journal is not None:
            print('持久journal：', controller.journal.path, flush=True)
            print('释放读回：', controller.release_result, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
