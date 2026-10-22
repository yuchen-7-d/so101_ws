"""Empty ID6 cycle near the manually observed 52 mm opening.

Default: read-only preview. --enable: current ~2144 -> 2048 -> frozen start.
Only ID6 volatile control registers can be written; no arm movement or grasp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/home/yu/so101_ws')
SOURCE = ROOT/'data/pose_mapping_checks/current_start_hc2lrx8k.json'
SOURCE_SHA = 'ba36a70964a8004cf63a2f3404d5a20e88763d36ce747ad0129cc141550dfcca'
OPEN_RECORD = ROOT/'data/real_control_runs/gripper_cycle_20261021T213754_wnvtc3su.jsonl'
OPEN_SHA = 'bc969904a66469ed64055509d293b898c1a43af50340333f4ba68ad3920a5daa'
DEPENDENCIES = {
    'so101_gripper_cycle.py': 'ad2df28ca9cbb65d322b020073dbec4de0759790d291b0b868ab2d61673ff8aa',
    'so101_hold_current.py': '6888f895dcf069791e8f9b2db19a172e73d8697ac6656c439f8b72dbea60f0a3',
    'so101_center_gripper.py': 'aed92daf1e1d27d07307bfdca3c6d2acf0c0255efd36615efc11503d9c817205',
    'so101_verify_gripper_center.py': '78dfb0ea0f5187df61ba3f554b02316f3745c3b13fa676c3762e8fdd1e860f75',
}
for _name, _sha in DEPENDENCIES.items():
    if hashlib.sha256((ROOT/'scripts'/_name).read_bytes()).hexdigest() != _sha:
        raise RuntimeError('控制依赖已改变，请重新核对：'+_name)
sys.path.insert(0, str(ROOT/'scripts'))
import so101_gripper_cycle as base

START_LOW, START_HIGH, INNER_TARGET = 2142, 2146, 2048


def require(condition, message):
    if not condition:
        raise base.ControlError(message)


def load_reference():
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, '手动张开快照改变')
    state = json.loads(raw)
    require(state['success'] is True and state['status'] == 'read_only_start_snapshot'
            and type(state['register_writes']) is int and state['register_writes'] == 0,
            '手动张开快照无效')
    require(state['mapping_sha256'] == hashlib.sha256((ROOT/'data/so101_joint_mapping.json').read_bytes()).hexdigest(),
            '关节计数映射已改变')
    for r in state['feedback_rounds']:
        require(len(r['states']) == 6 and all(s['torque_enabled'] is False
                and s['speed_raw'] == s['output_raw'] == s['status'] == 0 for s in r['states']),
                '原快照不是静止释放状态')
        s = next(s for s in r['states'] if s['servo_id'] == 6)
        require(s['position_counts'] == 2144 and s['output_limit'] == 0, '原张开点不匹配')
    raw = OPEN_RECORD.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OPEN_SHA, '已验证张开记录改变')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(rows[-1]['stage'] == 'final_cleanup', '缺少旧张开释放结果')
    opened = rows[-1]['snapshot']
    require(opened['status'] == 'empty_gripper_open_only_complete'
            and opened['open_response_complete'] is True
            and opened['current_goal'] == 2048 and opened['path'][-1] == 2048
            and opened['release']['confirmed'] is True
            and opened['release']['readback'] == {'40': 0, '48': 0}
            and opened['journal_error'] is None, '旧2048张开动作未完整通过')
    configs = {k: v['raw_hex'] for k, v in state['config_before'].items()}
    require(configs['6'] == opened['initial_configs']['6'], '两端点ID6配置不一致')
    for sid in range(1, 7):
        require(configs[str(sid)] == state['config_after'][str(sid)]['raw_hex'], '快照配置不稳定')
    return {
        'scope': 'empty gripper only; 2144 manually observed; powered wide cycle still unverified',
        'manual_snapshot': {'path': str(SOURCE), 'sha256': SOURCE_SHA},
        'powered_2048_reference': {'path': str(OPEN_RECORD), 'sha256': OPEN_SHA},
        'configuration_hex': configs,
        'supported_pose_counts': {str(s['servo_id']): s['position_counts'] for s in state['states']},
        'dependency_sha256': DEPENDENCIES,
    }


def fixed_wide_path(start):
    require(type(start) is int and START_LOW <= start <= START_HIGH,
            '本次起点须在2142–2146；不要修改旧程序的启动范围')
    down = [start]
    while down[-1] > INNER_TARGET:
        down.append(max(INNER_TARGET, down[-1]-16))
    return down + down[-2::-1]


class WideBus(base.GripperBus):
    """Separate capability for this exact path; old GripperBus stays unchanged."""
    def authorize_path(self, path):
        require(bool(path) and path == fixed_wide_path(path[0]), '仅允许冻结的受限往返路径')
        require(all(type(p) is int and INNER_TARGET <= p <= START_HIGH for p in path), '目标超范围')
        self.allowed_goals = frozenset(path)
        self.enabled = True


class WideCycle(base.GripperCycle):
    def preflight(self):
        self.bus.phase = 'preflight'
        require(self.open_only is False, '本次仅提供固定空夹爪往返')
        expected = self.proof['configuration_hex']
        for sid in base.IDS:
            raw = self.bus.read(sid, 0, 40)
            require(raw.hex() == expected[str(sid)], f'ID{sid}完整配置与测量快照不一致')
            require(raw[2] == 0 and raw[5] == sid and raw[8] == 1 and raw[18] == 12
                    and raw[30] == 1 and raw[33] == 0
                    and base.u16(raw[9:11]) == 0 and base.u16(raw[11:13]) == 4095,
                    f'ID{sid}不满足已验证单圈模式')
            self.configs[sid] = raw
        require(100 <= base.u16(self.configs[6][16:18]) <= 1000, 'ID6设备输出上限不适用')
        observations = []
        for index in range(3):
            if index:
                self.sleep(.1)
            states = {sid: self.read_frame(sid) for sid in base.IDS}
            for sid, s in states.items():
                require(s['torque'] == 0 and s['speed_raw_signed'] == 0 and s['output_raw_signed'] == 0
                        and s['lock'] == 1, f'ID{sid}需静止、释放且配置锁定')
                self.initial.setdefault(sid, s['position'])
                if sid != 6:
                    limit = 3 if sid == 4 else 12
                    require(abs(s['position']-self.proof['supported_pose_counts'][str(sid)]) <= limit,
                            f'ID{sid}已离开本次支撑姿态，停止而不自动恢复')
            s = states[6]
            require(START_LOW <= s['position'] <= START_HIGH and s['output_limit'] == 0,
                    'ID6需在2142–2146，输出上限为0')
            observations.append((self.clock(), s['position']))
        require(observations[-1][0]-observations[0][0] >= .2
                and max(p for _, p in observations)-min(p for _, p in observations) <= 2,
                'ID6三帧未稳定')
        self.start = observations[-1][1]
        self.path = fixed_wide_path(self.start)
        self.current_goal = self.start
        self.feedback_bounds = (INNER_TARGET-2, self.start+2)
        require(base.u16(self.configs[6][9:11]) <= self.feedback_bounds[0]
                < self.feedback_bounds[1] <= base.u16(self.configs[6][11:13]), '反馈走廊超出设备限位')
        self.emit(f'空夹爪固定路径：{self.path}；每步≤16，输出100/1000。')

    def recheck_after_journal(self):
        super().recheck_after_journal()
        for sid in range(1, 6):
            require(self.bus.read(sid, 0, 40) == self.configs[sid], f'ID{sid}保存日志后配置改变')
        require(self.path == fixed_wide_path(self.start), '冻结路径改变')

    def cycle(self):
        super().cycle()
        # Successful return only. A failure in any preceding step skips this code.
        self.bus.phase = 'returned_observation'
        self.emit('已返回本次起点；保持监测2秒后释放ID6。')
        end = self.clock()+2
        while self.clock() < end:
            self.monitor(self.start, 1, 100, (self.start-2, self.start+2))
            self.sleep(.05)

    def report(self):
        r = super().report()
        r.update(test='id6_observed_wide_empty_cycle', mode='wide_empty_cycle',
                 inner_target=INNER_TARGET, manual_open_reference=2144,
                 start_window=[START_LOW, START_HIGH], step_counts_max=16,
                 powered_wide_cycle_response_complete=self.cycle_response_complete,
                 motion_authorized_for_arm_ids=False,
                 note='Empty cycle only; returns to frozen fresh start on success; no fault return or grasp.')
        if r['status'] == 'empty_gripper_cycle_complete':
            r['status'] = 'empty_gripper_wide_cycle_complete'
        return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--enable', action='store_true', help='执行空夹爪受限往返；未提供则只读预览')
    args = parser.parse_args(argv)
    controller = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        proof = load_reference()
        import serial
        print('空夹爪；稳定支撑机械臂，夹指运动区域保持空置；关闭其他串口程序。', flush=True)
        print('执行ID6受限往返；ID1–5不使能，结束或异常释放ID6。' if args.enable
              else '只读预览；寄存器写入=0。', flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15, write_timeout=.15, exclusive=True) as port:
            controller = WideCycle(WideBus(port), proof, journal_dir=ROOT/'data/real_control_runs')
            result = controller.run(args.enable)
        print(f"状态：{result['status']}；起点={result['start']}；写指令={result['writes_sent']}", flush=True)
        return 0
    except KeyboardInterrupt:
        print('已中止；不执行故障回程。', flush=True)
        return 130
    except (base.ControlError, RuntimeError, OSError, ValueError, KeyError) as error:
        print(f'本次未完成：{error}', file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if controller is not None and controller.journal is not None:
            print('持久journal：', controller.journal.path, flush=True)
            print('ID6释放读回：', controller.release_result, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
