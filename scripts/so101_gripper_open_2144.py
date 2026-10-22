"""ID6 空夹爪从2083附近分步张开至2144，不返回。

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
SOURCE = ROOT/'data/real_control_runs/gripper_cycle_20261022T204835_u8445o8c.jsonl'
SOURCE_SHA = 'c51e4ce3112b46f002a7f917996706ba600c7677e475f4e7b6650f3f36329655'
MANUAL = ROOT/'data/pose_mapping_checks/current_start_hc2lrx8k.json'
MANUAL_SHA = 'ba36a70964a8004cf63a2f3404d5a20e88763d36ce747ad0129cc141550dfcca'
MAPPING_SHA = 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8'
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

START_LOW, START_HIGH, OPEN_TARGET = 2081, 2085, 2144


def require(condition, message):
    if not condition:
        raise base.ControlError(message)


def load_reference():
    require(hashlib.sha256((ROOT/'data/so101_joint_mapping.json').read_bytes()).hexdigest()
            == MAPPING_SHA, '软件映射已改变')
    raw = MANUAL.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANUAL_SHA, '手动张开2144的快照改变')
    manual = json.loads(raw)
    require(manual['success'] is True and manual['status'] == 'read_only_start_snapshot'
            and type(manual['register_writes']) is int and manual['register_writes'] == 0
            and manual['mapping_sha256'] == MAPPING_SHA, '手动2144快照无效')
    configs = {k: v['raw_hex'] for k, v in manual['config_before'].items()}
    require(set(configs) == {str(i) for i in base.IDS}, '手动快照配置不完整')
    for sid in base.IDS:
        require(configs[str(sid)] == manual['config_after'][str(sid)]['raw_hex'], '手动快照配置不稳定')
    for r in manual['feedback_rounds']:
        hand6 = next(s for s in r['states'] if s['servo_id'] == 6)
        require(hand6['position_counts'] == 2144 and hand6['torque_enabled'] is False
                and hand6['speed_raw'] == hand6['output_raw'] == hand6['output_limit'] == 0,
                '手动张开参考不符')
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, '最近合拢成功记录改变')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require([row['stage'] for row in rows] == ['before_any_write', 'final_cleanup'],
            '合拢记录缺少起点备份或最终释放结果')
    closed = rows[-1]['snapshot']
    require(closed['test'] == 'id6_negative_sixteen_probe'
            and closed['status'] == 'empty_gripper_closing_probe_complete'
            and closed['closing_probe_response_complete'] is True
            and closed['path'] == [2098, 2082] and closed['current_goal'] == 2082
            and closed['fault'] is None and closed['journal_error'] is None
            and closed['initial_configs'] == configs
            and closed['release'] == {'servo_id': 6, 'readback': {'40': 0, '48': 0},
                                     'errors': [], 'confirmed': True},
            '合拢记录未完整通过并释放，或配置与手动参考不符')
    samples = [s for s in closed['samples']
               if s['servo_id'] == 6 and s['phase'] == 'closed_observation']
    require(len(samples) >= 3 and samples[-1]['elapsed_s']-samples[0]['elapsed_s'] >= 1.9
            and all(s['position'] == 2083 and s['goal'] == 2082
                    and s['speed_raw_signed'] == s['output_raw_signed'] == s['status'] == 0
                    for s in samples), '缺少合拢后2083稳定反馈')
    pose = dict(closed['initial_positions'])
    pose['6'] = samples[-1]['position']
    return {
        'scope': 'empty opening only; manual2144 reference; original wide cycle remains incomplete',
        'closed_probe_record': {'path': str(SOURCE), 'sha256': SOURCE_SHA},
        'manual_2144_snapshot': {'path': str(MANUAL), 'sha256': MANUAL_SHA},
        'configuration_hex': configs, 'supported_pose_counts': pose,
        'mapping_sha256': MAPPING_SHA, 'dependency_sha256': DEPENDENCIES,
    }


def fixed_open_path(start):
    require(type(start) is int and START_LOW <= start <= START_HIGH,
            '本次起点须在2081–2085；停止而不自动恢复旧位置')
    path = [start]
    while path[-1] < OPEN_TARGET:
        path.append(min(OPEN_TARGET, path[-1]+16))
    return path


class Open2144Bus(base.GripperBus):
    def authorize_path(self, path):
        require(bool(path) and all(type(p) is int for p in path)
                and path == fixed_open_path(path[0]), '只允许从冻结起点分步张开至2144的完整路径')
        self.allowed_goals = frozenset(path)
        self.enabled = True


class Open2144(base.GripperCycle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_progress_at = 0.0

    def preflight(self):
        self.bus.phase = 'preflight'
        require(self.open_only is True, '本次只提供分步张开至2144')
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
        self.path = fixed_open_path(self.start)
        self.current_goal = self.start
        self.feedback_bounds = (self.start-2, self.path[-1]+2)
        require(base.u16(self.configs[6][9:11]) <= self.feedback_bounds[0]
                < self.feedback_bounds[1] <= base.u16(self.configs[6][11:13]), '反馈走廊超设备限位')
        self.emit(f'分步张开固定路径：{self.path}；每步≤16；不返回。')
        self.emit('输出100/1000、速度50、加速度1；到位±2计数、每步3秒；PID/offset不变。')

    def recheck_after_journal(self):
        super().recheck_after_journal()
        for sid in range(1, 6):
            require(self.bus.read(sid, 0, 40) == self.configs[sid], f'ID{sid}保存日志后配置改变')
        require(self.path == fixed_open_path(self.start), '冻结路径改变')

    def monitor(self, goal, torque, output, corridor):
        state = super().monitor(goal, torque, output, corridor)
        if (self.bus.phase in ('opening_only', 'opened_observation') and goal != self.start
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
        self.emit('已到2144目标；继续监测2秒后释放ID6，不返回起点。')
        end = self.clock()+2
        while self.clock() < end:
            self.monitor(target, 1, 100, (target-2, target+2))
            self.sleep(.05)
        self.check_config()
        self.open_response_complete = True

    def report(self):
        r = super().report()
        r.update(test='id6_open_to_measured_2144', mode='open_2144_only',
                 start_window=[START_LOW, START_HIGH], open_target=OPEN_TARGET,
                 open_2144_response_complete=self.open_response_complete,
                 original_wide_cycle_completed=False, motion_authorized_for_arm_ids=False,
                 note='Bounded opening to manually observed 2144 with fixed fresh start; no return, zero/PID adjustment or grasp.')
        if r['status'] == 'empty_gripper_open_only_complete':
            r['status'] = 'empty_gripper_open_2144_complete'
        if self.path:
            samples = [s for s in self.samples if s['servo_id'] == 6
                       and s['phase'] in ('opening_only', 'opened_observation')
                       and s['goal'] != self.start]
            if samples:
                last = samples[-1]
                r['last_motion_feedback'] = {k: last[k] for k in
                    ('elapsed_s', 'position', 'goal', 'speed_raw_signed', 'output_raw_signed', 'voltage')}
                r['last_motion_feedback']['error_counts'] = last['position']-last['goal']
                r['last_motion_feedback']['from_frozen_start_counts'] = last['position']-self.start
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
        print('执行ID6分步张开至2144；ID1–5不使能，结束或异常释放ID6。' if args.enable
              else '只读预览；寄存器写入=0，不执行运动。', flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15, write_timeout=.15, exclusive=True) as port:
            controller = Open2144(Open2144Bus(port), proof, journal_dir=ROOT/'data/real_control_runs',
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
            last = controller.report().get('last_motion_feedback')
            if last:
                print('最后一次运动反馈：', last, flush=True)
            if controller.journal is not None:
                print('持久journal：', controller.journal.path, flush=True)
                print('ID6释放读回：', controller.release_result, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
