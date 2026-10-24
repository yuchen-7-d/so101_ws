"""ID5从冻结起点增加8计数；默认只读，--enable执行；不自动返回。

空夹爪。ID1/2/3/4保持本次反馈，ID6不写入。输出100、速度50、加速度1。
手在下方留出间隙：模型预测腕旋转约0.70度，夹爪中心位移约0.15毫米。
到位后监测5秒，最后3秒提示承托；结束或异常释放本次使能的ID1–5。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import importlib.util
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
GEOMETRY = Path('/tmp/so101_wrist_roll_step_evec9_h2.json')
GEOMETRY_SHA = '44b95196bbf42a57118aa6a1a732faa49c1bfcf9cfb7c38d8c7d474feba9bb79'
START = ROOT / 'data/real_control_runs/wrist_positive_eight_20261022T222026_4fzx0ru7.jsonl'
START_SHA = '6a4c9af03132953f7ba15d3862b9cdd33dfa63e9c94091ec77eff2f4ebb12f9c'
VALIDATOR = ROOT / 'scripts/so101_check_wrist_roll_step.py'
VALIDATOR_SHA = '3d09829b100bc8ae3e0d6eb6aa4ad2d55a2ee0e21c219734d5e8a58d892dfcdd'
EXPECTED_STARTS = [[8, 964, 4073, 1950, count, 2143] for count in (3282, 3283)]
EXPECTED_START = EXPECTED_STARTS[0]
ALLOWED_ACTIVE_STARTS = (3282, 3283)
AXIS, DELTA = 5, 8
OTHER_ARM_IDS = (1, 2, 3, 4)
MOTION_PHASE = 'wrist_roll_step'
REFERENCE_HOLD_MARGIN = 3

if hashlib.sha256(HOLD.read_bytes()).hexdigest() != HOLD_SHA:
    raise RuntimeError('原位保持模块已改变，需重新核对')
sys.path.insert(0, str(ROOT / 'scripts'))
import so101_hold_open_pose as hold
base, require = hold.base, hold.require


def checked_bytes(path, sha):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, '输入文件已改变：' + str(path))
    return raw


def entry_bounds(sid):
    """Fixed holding bands; ID5 can start only at 3282 or 3283, ID6 at 2143."""
    require(sid in base.ALL_IDS, '关节ID无效')
    if sid == AXIS:
        return ALLOWED_ACTIVE_STARTS
    margin = REFERENCE_HOLD_MARGIN if sid in OTHER_ARM_IDS else 0
    reference = EXPECTED_START[sid - 1]
    return reference - margin, reference + margin


def load_reference():
    proof = hold.load_reference()
    geometry = json.loads(checked_bytes(GEOMETRY, GEOMETRY_SHA))
    require(geometry['kind'] == 'so101_saved_wrist_roll_step_geometry'
            and geometry['sampled_geometry_complete'] is True and geometry['error'] is None
            and geometry['axis'] == AXIS and geometry['delta_counts'] == DELTA
            and geometry['execution_available'] is False and geometry['status'] == 'not_for_motion'
            and geometry['start_variants_counts'] == EXPECTED_STARTS
            and geometry['sample_counts'] == {'nominal': 18, 'holding_error_grid': 2268},
            '腕旋转几何报告未完成或起点组不匹配')
    for path, sha in {**geometry['input_files'], **geometry['model_and_reference_files']}.items():
        checked_bytes(path, sha)
    require(geometry['input_files'][str(START)] == START_SHA, '报告来源腕俯仰日志不匹配')
    require(len(geometry['variants']) == 2, '缺少两个分别检查的起点')
    proof_variants = []
    for start, variant in zip(EXPECTED_STARTS, geometry['variants']):
        require(variant['start_counts'] == start and variant['sampled_geometry_complete'] is True
                and variant['sample_counts'] == {'nominal': 9, 'holding_error_grid': 1134},
                '单起点几何采样未完成或不匹配')
        samples = variant['checked_samples']
        require(len(samples) == 1143 and all(not s['penetrating_contacts'] for s in samples),
                '采样不完整或存在穿透')
        nominal = [s for s in samples if s['group'] == 'nominal']
        expected_nominal = []
        for delta in range(DELTA + 1):
            p = start.copy()
            p[AXIS - 1] += delta
            expected_nominal.append(p)
        require([s['diagnostic_counts'] for s in nominal] == expected_nominal, '名义单轴路径不匹配')
        expected_grid = set()
        for deviations in itertools.product((-3, 0, 3), repeat=4):
            for delta in range(-3, DELTA + 3):
                p = start.copy()
                p[AXIS - 1] += delta
                for sid, deviation in zip(OTHER_ARM_IDS, deviations):
                    p[sid - 1] += deviation
                expected_grid.add(tuple(p))
        grid = [s for s in samples if s['group'] == 'holding_error_grid']
        require(len(grid) == 1134 and {tuple(s['diagnostic_counts']) for s in grid} == expected_grid,
                '保持误差采样范围不匹配')
        proof_variants.append({'label': variant['label'], 'counts': start.copy(),
                               'nominal_samples': nominal,
                               'sample_counts': {'nominal': 9, 'holding_error_grid': 1134}})
    # Reuse only the pinned, pure-data validator; its MuJoCo imports are lazy
    # inside analyze(), which is never called by this controller.
    checked_bytes(VALIDATOR, VALIDATOR_SHA)
    spec = importlib.util.spec_from_file_location('roll_saved_journal_validator', VALIDATOR)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    rows = [json.loads(line) for line in checked_bytes(START, START_SHA).splitlines() if line.strip()]
    require(validator.validate_journal(rows) == EXPECTED_STARTS, '腕俯仰日志的两个释放前姿态不匹配')
    state = rows[-1]['snapshot']
    for sid in range(1, 7):
        require(state['initial_configs'][str(sid)] == proof['configuration_hex'][str(sid)],
                f'ID{sid}配置与已有记录不符')
    proof['reference_pose_counts'] = {str(i): c for i, c in enumerate(EXPECTED_START, 1)}
    proof['step_geometry'] = {
        'path': str(GEOMETRY), 'sha256': GEOMETRY_SHA,
        'variants': proof_variants, 'sample_counts': {'nominal': 18, 'holding_error_grid': 2268},
        'scope': 'discrete saved model geometry; not continuous, dynamic or live scene proof',
        'gripper_fixed_counts': 2143}
    proof['source_wrist_journal'] = {'path': str(START), 'sha256': START_SHA,
                                        'scope': 'stable pre-release pose, not current device state'}
    proof['journal_validator'] = {'path': str(VALIDATOR), 'sha256': VALIDATOR_SHA}
    proof['hold_module_sha256'] = HOLD_SHA
    return proof


def create_journal(directory, snapshot):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
    fd, path = tempfile.mkstemp(prefix='wrist_roll_positive_eight_' + stamp,
                               suffix='.jsonl', dir=directory)
    stream = None
    try:
        stream = os.fdopen(fd, 'w', encoding='utf-8')
        journal = hold.DurableJournal(path, stream)
        journal.append('before_any_write', snapshot)
        dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return journal
    except BaseException:
        if stream is None:
            os.close(fd)
        else:
            stream.close()
        raise


class StepBus(hold.HoldBus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_goals = {}
        self.step_armed = self.step_attempted = False

    def authorize(self, goals):
        require(not self.enabled, '本次只能授权一次起点')
        require(goals.get(AXIS) in ALLOWED_ACTIVE_STARTS
                and all(entry_bounds(sid)[0] <= goals.get(sid, -1) <= entry_bounds(sid)[1]
                        for sid in base.ARM_IDS), '起点超出本次两个分支或保持范围')
        super().authorize(goals)
        self.original_goals = dict(goals)

    def arm_single_step(self):
        require(self.enabled and not self.step_armed and self.phase == MOTION_PHASE,
                '单步运动阶段不匹配')
        goal = self.original_goals[AXIS] + DELTA
        require(0 <= goal <= 4095, '单步目标跨过计数边界')
        self.goals[AXIS] = goal
        self.step_armed = True
        return goal

    def exchange(self, sid, instruction, params, size, tolerate_alarm=False):
        if instruction == 3 and sid == AXIS and params and params[0] == base.GOAL and self.step_armed:
            require(self.phase == MOTION_PHASE and not self.step_attempted
                    and len(params) == 3 and size == 0
                    and int.from_bytes(bytes(params[1:]), 'little') == self.original_goals[AXIS] + DELTA,
                    '仅允许一次ID5加8目标，禁止回程、重发或追加')
            self.step_attempted = True  # Set before I/O, even if the ACK is lost.
        return super().exchange(sid, instruction, params, size, tolerate_alarm)


class WristRollStep(hold.PoseHold):
    def __init__(self, bus, proof, *, journal_dir, emit=print, clock=time.monotonic,
                 sleep=time.sleep, journal_factory=create_journal):
        super().__init__(bus, proof, journal_dir=journal_dir, emit=emit, clock=clock,
                         sleep=sleep, journal_factory=journal_factory)
        self.frozen = {}
        self.baseline = None
        self.step_goal = None
        self.step_readback = self.response_complete = False
        self.step_frames = []
        self.selected_geometry_start = None

    def check_released_start(self):
        # Also called again after journal creation, before authorizing any writes.
        super().check_released_start()
        for sid in base.ALL_IDS:
            s = hold.frame_fields(self.bus.read(sid, 40, 26))
            base.check_health(sid, base.Feedback(s['position'], s['load'], s['voltage'],
                                               s['temperature'], s['status']), *self.configs[sid])
            require(s['torque'] == s['output_limit'] == 0
                    and s['speed_raw_signed'] == s['output_raw_signed'] == 0 and s['lock'] == 1,
                    f'ID{sid}未静止释放或配置未锁定；零写入停止')
            lo, hi = entry_bounds(sid)
            require(lo <= s['position'] <= hi,
                    f"ID{sid}启动位置={s['position']}，超出参考范围{lo}–{hi}；零写入停止")

    def preflight(self):
        super().preflight()
        self.frozen = dict(self.targets)
        start = [self.frozen[i] for i in base.ARM_IDS] + [self.gripper_start]
        require(all(entry_bounds(sid)[0] <= count <= entry_bounds(sid)[1]
                    for sid, count in enumerate(start, 1)),
                '冻结起点超出已检查参考范围；零写入停止')
        changes = {sid: {'reference': EXPECTED_START[sid - 1], 'fresh_hold_goal': self.frozen[sid]}
                   for sid in OTHER_ARM_IDS if self.frozen[sid] != EXPECTED_START[sid - 1]}
        if changes:
            self.emit(f'参考保持范围内的起点差异：{changes}；保持目标采用本次读数，绝对边界不平移。')
        self.selected_geometry_start = next(
            pose.copy() for pose in EXPECTED_STARTS if pose[AXIS - 1] == self.frozen[AXIS])
        self.step_goal = self.frozen[AXIS] + DELTA
        self.emit(f'固定单程计划：ID5 {self.frozen[AXIS]} → {self.step_goal}；不自动返回。')

    def monitor(self, expected_torque, tolerance, expected_output):
        started = self.clock()
        # Keep base health/control/saturation checks. ID5 has its own motion
        # corridor; IDs 1-4 (including wrist flex) stay within both the fresh
        # goal's three-count band and the unchanged reference absolute band.
        states = base.HoldController.monitor(self, expected_torque, tolerance, expected_output)
        g = hold.frame_fields(self.bus.read(6, 40, 26))
        base.check_health(6, base.Feedback(g['position'], g['load'], g['voltage'],
                                          g['temperature'], g['status']), *self.configs[6])
        require(g['torque'] == g['output_limit'] == g['speed_raw_signed'] == g['output_raw_signed'] == 0
                and g['lock'] == 1 and g['position'] == EXPECTED_START[5],
                'ID6释放状态或本次2143张开位置改变，停止且不写入夹爪')
        self.samples.append({'elapsed_s': self.clock() - self.bus.started, 'phase': self.bus.phase,
                             'output_limit': expected_output,
                             'states': {str(i): {**vars(s), 'error_counts': s.position-self.targets[i]}
                                        for i, s in states.items()}, 'gripper': g})
        for sid in OTHER_ARM_IDS:
            require(abs(states[sid].position - self.targets[sid]) <= 3,
                    f'ID{sid}偏离本次保持点超过3计数')
            lo, hi = entry_bounds(sid)
            require(lo <= states[sid].position <= hi,
                    f'ID{sid}超出原参考绝对范围{lo}–{hi}，停止；不随新起点扩大范围')
        require(self.samples[-1]['gripper']['position'] == EXPECTED_START[5],
                'ID6偏离本次几何检查的2143位置，停止且不写入夹爪')
        g0 = self.frozen.get(AXIS, self.targets[AXIS])
        lo, hi = (g0 - 2, g0 + DELTA + 2) if self.bus.step_armed else (g0 - 3, g0 + 3)
        require(lo <= states[AXIS].position <= hi, 'ID5反馈超出本次局部范围，停止且不回程')
        e = hold.frame_fields(self.bus.read(AXIS, 40, 26))
        base.check_health(AXIS, base.Feedback(e['position'], e['load'], e['voltage'],
                                            e['temperature'], e['status']), *self.configs[AXIS])
        require(lo <= e['position'] <= hi and e['torque'] == expected_torque
                and e['goal'] == self.targets[AXIS] and e['output_limit'] == expected_output
                and e['speed_limit'] == 50 and e['acceleration'] == 1 and e['lock'] == 1,
                'ID5完整反馈或控制寄存器异常')
        require(self.clock() - started <= .5, '含ID5完整反馈的整轮读取超过0.5秒')
        self.step_frames.append({'elapsed_s': self.clock() - self.bus.started,
                                 'phase': self.bus.phase, **e})
        return e

    def collect_baseline(self):
        self.bus.phase = 'loaded_baseline'
        end, window = self.clock() + 3, []
        while self.clock() < end:
            e = self.monitor(1, base.HOLD_TOLERANCE, 100)
            now = self.clock()
            require(abs(e['position'] - self.frozen[AXIS]) <= 2,
                    'ID5有载偏差超过2计数，停止；不调整目标补偿')
            if e['speed_raw_signed'] != 0:
                window = []
            else:
                window.append((now, e['position']))
                if max(p for _, p in window) - min(p for _, p in window) > 1:
                    window = [window[-1]]
                if now <= end and len(window) >= 3 and window[-1][0] - window[0][0] >= .5:
                    self.baseline = int(statistics.median(p for _, p in window))
                    return
            self.sleep(.05)
        raise base.ControlError('ID5有载基线3秒内未稳定；未发送运动目标')

    def arrived(self, e):
        return (e['speed_raw_signed'] == 0 and abs(e['position'] - self.step_goal) <= 2
                and abs(e['position'] - (self.baseline + DELTA)) <= 2)

    def hold(self):
        self.bus.phase = 'ramp'
        previous = 0
        for level in (25, 50, 75, 100):
            self.monitor(1, base.HOLD_TOLERANCE, previous)
            started = self.clock()
            for sid in base.ARM_IDS:
                self.bus.set_checked(sid, base.OUTPUT, level)
                require(self.clock() - started <= .5, '输出分级写入超过0.5秒')
            self.monitor(1, base.HOLD_TOLERANCE, level)
            self.sleep(.1)
            previous = level
        self.emit('采集有载基线：手在下方留出间隙，不接触或推拉连杆；夹指与线缆周围留出旋转空间。')
        self.collect_baseline()
        self.emit(f'固定起点G0={self.frozen[AXIS]}；有载P0={self.baseline}；目标={self.step_goal}。')
        self.emit('同时核验：目标±2，及相对P0增加8±2；3秒内未达标即停止，不加输出。')
        began = self.clock()
        self.journal.append('before_single_step', self.report())
        require(self.clock() - began <= .5, '运动前记录保存超过0.5秒，停止')
        e = self.monitor(1, base.HOLD_TOLERANCE, 100)
        require(e['speed_raw_signed'] == 0 and abs(e['position'] - self.baseline) <= 1,
                '发出运动前有载基线已改变')
        self.bus.phase = MOTION_PHASE
        require(self.bus.arm_single_step() == self.step_goal, '单步目标不一致')
        self.targets[AXIS] = self.step_goal
        began = self.clock()
        self.bus.set_checked(AXIS, base.GOAL, self.step_goal)
        require(self.clock() - began <= .5, '单步写入及读回超过0.5秒')
        self.step_readback = True
        end, stable, next_print = began + 3, [], self.clock()
        while self.clock() < end:
            e = self.monitor(1, base.HOLD_TOLERANCE, 100)
            now = self.clock()
            if now >= next_print:
                self.emit(f"ID5实际={e['position']}；目标误差={e['position'] - self.step_goal:+d}；"
                          f"相对P0={e['position'] - self.baseline:+d}；输出={e['load']}/1000")
                next_print = now + .25
            stable = stable + [now] if self.arrived(e) else []
            if now <= end and len(stable) >= 3 and stable[-1] - stable[0] >= .15:
                break
            self.sleep(.05)
        else:
            raise base.ControlError('ID5单步3秒内未同时满足目标与相对位移要求；不执行回程')
        self.emit('单步反馈达标；再监测5秒，最后3秒提示承托，然后释放，不返回起点。')
        self.bus.phase = 'after_step'
        end, prompted = self.clock() + 5, False
        while self.clock() < end:
            if end - self.clock() <= 3 and not prompted:
                prompted = True
                self.bus.phase = 'support_requested'
                self.emit('还剩约3秒：现在从下方托住机械臂，保持支撑直到释放完成。')
            e = self.monitor(1, base.HOLD_TOLERANCE, 100)
            require(abs(e['position'] - self.step_goal) <= 2
                    and abs(e['position'] - (self.baseline + DELTA)) <= 2,
                    '到位后ID5偏差增大，停止且不回程')
            self.sleep(.05)
        self.response_complete = True

    def report(self):
        r = super().report()
        if r['status'] == 'current_pose_hold_complete':
            r['status'] = 'wrist_roll_positive_eight_complete' if self.response_complete else 'incomplete'
        r.update(test='wrist_roll_positive_eight_independent_probe', axis=AXIS,
                 frozen_start_targets=self.frozen, loaded_baseline=self.baseline,
                 step_goal=self.step_goal, delta_counts=DELTA, independent_new_diagnostic=True,
                 step_goal_write_attempted=self.bus.step_attempted,
                 step_goal_readback_confirmed=self.step_readback,
                 step_response_complete=self.response_complete, step_frames=self.step_frames,
                 motion_path_executed=self.response_complete, return_executed=False,
                 arm_tolerance_counts=3, gripper_tolerance_counts=0,
                 wrist_tolerance_counts=3, wrist_tolerance_scope='ID4 holding',
                 active_axis_settle_tolerance_counts=2,
                 selected_geometry_start_counts=self.selected_geometry_start,
                 allowed_active_start_counts=list(ALLOWED_ACTIVE_STARTS),
                 geometry_variant_frozen_for_run=True,
                 entry_bounds_counts={str(sid): list(entry_bounds(sid)) for sid in base.ALL_IDS},
                 absolute_other_arm_bounds_counts={str(sid): list(entry_bounds(sid))
                                                   for sid in OTHER_ARM_IDS},
                 reference_hold_margin_counts=REFERENCE_HOLD_MARGIN,
                 hold_goals_use_fresh_feedback=True,
                 active_axis_local_counts=[-2, DELTA + 2],
                 settle_tolerance_counts=2, motion_timeout_seconds=3, seconds=5,
                 note='One positive eight-count ID5 step; P0 checks displacement, never changes the goal. '
                      'Discrete saved geometry is not continuous collision or dynamics validation.')
        return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--enable', action='store_true', help='使能ID1–5，执行ID5加8单程；默认只读')
    args = parser.parse_args(argv)
    controller = None

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        proof = load_reference()
        import serial
        print('空夹爪，关闭其他串口程序；手在下方准备承接，不接触连杆；结束或异常释放ID1–5。', flush=True)
        print('ID5单程加8；其余臂关节保持本次读数，ID6不写入；输出100、速度50、加速度1。', flush=True)
        print('模型预计腕旋转约0.70度、夹爪中心位移约0.15毫米；最后3秒提示后再托住。', flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            controller = WristRollStep(StepBus(port), proof, journal_dir=ROOT / 'data/real_control_runs',
                                      emit=lambda text: print(text, flush=True))
            r = controller.run(args.enable)
        print('状态：', r['status'], flush=True)
        return 0
    except KeyboardInterrupt:
        print('已中止；请查看逐关节释放读回。', flush=True)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError, LookupError) as error:
        print('本次未完成：', error, file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if controller is not None and controller.journal is not None:
            print('持久journal：', controller.journal.path, flush=True)
            print('释放读回：', controller.release_result, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
