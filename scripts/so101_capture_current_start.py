"""只读保存当前 SO101 起点、配置和新映射摘要；不执行任何动作。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path('/home/yu/so101_ws')
MAPPING_SHA256 = 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8'
EXPECTED_OFFSETS = {1: 0, 2: 0x0C9D, 3: 0, 4: 1822, 5: 0, 6: 3901}
IDS = range(1, 7)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def new_report():
    return {
        'schema_version': 1, 'kind': 'so101_current_start_with_configuration',
        'backend': 'serial', 'operation': 'read_state', 'success': False,
        'status': 'not_completed', 'execution_available': False,
        'model_mapping_verified': False, 'register_writes': 0, 'error': '',
        'started_at_ns': time.time_ns(),
        'states': [], 'events': [], 'capture_events': [], 'feedback_rounds': [],
        'config_before': {}, 'config_after': {},
        'events_scope': 'events contains the selected final six feedback reads; capture_events contains every request',
        'scope': 'static_read_only_snapshot_not_motion_or_power_cycle_verification',
    }


def collect(bus, mapping, report, decode_config, frame_fields, sleep=time.sleep):
    """Bus 必须由 main 的 ReadOnlyBus 提供；单串口会话内先后检查配置。"""
    report['capture_events'] = bus.events

    def read_configuration(phase, target):
        bus.phase = phase
        for sid in IDS:
            raw = bus.read(sid, 0, 40)
            require(len(raw) == 40, f'ID{sid} 配置长度错误')
            config = decode_config(raw)
            config['sampled_at_ns'] = time.time_ns()
            target[str(sid)] = config
            require(config['id'] == sid and config['endianness'] == 0
                    and config['mode'] == 0 and config['phase'] == 12
                    and config['resolution'] == 1
                    and config['minimum'] == 0 and config['maximum'] == 4095,
                    f'ID{sid} 模式、相位、分辨率或设备限位与已检查配置不符')
            require(config['offset_raw'] == EXPECTED_OFFSETS[sid],
                    f"ID{sid} offset=0x{config['offset_raw']:04X}，"
                    f"本次映射要求0x{EXPECTED_OFFSETS[sid]:04X}；停止，不自动修复")

    read_configuration('start_config_before', report['config_before'])
    for index in range(3):
        if index:
            sleep(0.2)
        bus.phase = f'start_feedback_{index + 1}'
        first_event = len(bus.events)
        states = []
        report['states'] = states
        report['feedback_rounds'].append({'index': index + 1, 'states': states})
        for sid in IDS:
            raw = bus.read(sid, 40, 26)
            require(len(raw) == 26, f'ID{sid} 反馈长度错误')
            f = frame_fields(raw)
            state = {
                'servo_id': sid, 'position_counts': f['position'], 'goal_counts': f['goal'],
                'speed_raw': f['speed_raw_signed'], 'output_raw': f['output_raw_signed'],
                'torque_enabled': bool(f['torque']), 'output_limit': f['output_limit'],
                'voltage': f['voltage'], 'temperature_c': f['temperature'],
                'status': f['status'], 'configuration_locked': f['lock'] == 1,
                'sampled_at_ns': time.time_ns(), 'raw_hex': raw.hex(),
            }
            states.append(state)
            report['events'] = bus.events[first_event:]
            require(f['torque'] == 0 and f['speed_raw_signed'] == 0
                    and f['output_raw_signed'] == 0 and f['status'] == 0
                    and 0 <= f['position'] <= 4095,
                    f'ID{sid} 不是扭矩关闭、静止且无报警状态：{state}')
            config = bytes.fromhex(report['config_before'][str(sid)]['raw_hex'])
            require(max(6.0, config[15] / 10) <= f['voltage'] <= min(7.4, config[14] / 10)
                    and f['temperature'] < min(55, config[13]),
                    f'ID{sid} 电压或温度未通过现有检查范围：{state}')
            require(f['lock'] == 1 and 0 <= f['output_limit'] <= 1000,
                    f'ID{sid} 配置未锁定或输出上限数值异常')
    read_configuration('start_config_after', report['config_after'])
    for sid in IDS:
        key = str(sid)
        require(report['config_before'][key]['raw_hex'] == report['config_after'][key]['raw_hex'],
                f'ID{sid} 读取期间配置发生变化')
        values = [r['states'][sid - 1]['position_counts'] for r in report['feedback_rounds']]
        # 只比较短时位置变化；不用于生成或折返任何运动目标。
        offsets = [(v - values[0] + 2048) % 4096 - 2048 for v in values]
        require(max(offsets) - min(offsets) <= 4,
                f'ID{sid} 三轮读数变化超过4计数：{values}；本次快照不通过')
    require(len(report['events']) == 6 and len(bus.events) == 30
            and all(e['instruction'] == 2 and e.get('acknowledged') is True for e in bus.events),
            '完整读取证据不符合预期')
    report['mapping_reference_counts'] = {
        key: ref['position_count'] for key, ref in mapping['joint_references'].items()}
    for s in report['states'][:5]:
        key = str(s['servo_id'])
        ref = mapping['joint_references'][key]
        sign = mapping['joint_directions'][key]['direction']
        difference = (s['position_counts'] - ref['position_count'] + 2048) % 4096 - 2048
        s['model_angle_nearest_branch_deg'] = ref['model_angle_deg'] + sign * difference * 360 / 4096
    report['configuration_matches_mapping'] = True
    report['position_spread_limit_counts'] = 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    args = parser.parse_args(argv)
    report = new_report()
    mapping_path = ROOT / 'data/so101_joint_mapping.json'
    print('只读采集当前起点；保持稳定支撑，关闭其他串口程序。不会使能扭矩或清理写入。', flush=True)
    try:
        mapping_bytes = mapping_path.read_bytes()
        require(sha(mapping_bytes) == MAPPING_SHA256,
                '当前映射摘要与本次 ID2 迁移结果不同；停止，不访问串口')
        mapping = json.loads(mapping_bytes)
        report['mapping_path'] = str(mapping_path)
        report['mapping_sha256'] = sha(mapping_bytes)
        report['mapping_snapshot'] = mapping
        report['expected_offsets_raw'] = EXPECTED_OFFSETS
        sys.path.insert(0, str(ROOT / 'scripts'))
        import serial
        from so101_verify_gripper_center import ReadOnlyBus
        from so101_center_gripper import decode_config
        from so101_gripper_cycle import frame_fields

        with serial.Serial(args.port, baudrate=1_000_000, timeout=0.15,
                           write_timeout=0.15, exclusive=True) as port:
            bus = ReadOnlyBus(port)
            collect(bus, mapping, report, decode_config, frame_fields)
        require(mapping_path.read_bytes() == mapping_bytes, '读取期间软件映射发生变化')
        report['success'] = True
        report['status'] = 'read_only_start_snapshot'
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['finished_at_ns'] = time.time_ns()
    report['register_writes'] = sum(e['instruction'] != 2 for e in report['capture_events'])
    # 自动生成新文件名；不会覆盖旧起点、旧计划或配置。
    directory = ROOT / 'data/pose_mapping_checks'
    directory.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='current_start_', suffix='.json', dir=directory)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('读取通过：', report['success'])
    print('寄存器写入次数：', report['register_writes'])
    for s in report['states']:
        config = report['config_before'][str(s['servo_id'])]
        angle = s.get('model_angle_nearest_branch_deg')
        suffix = f'，模型诊断角={angle:+.3f}°' if angle is not None else ''
        print(f"ID{s['servo_id']}：位置={s['position_counts']}，offset=0x{config['offset_raw']:04X}，"
              f"扭矩={int(s['torque_enabled'])}，电压={s['voltage']:.1f}V{suffix}")
    print('状态：', report['status'])
    if report['error']:
        print('原因：', report['error'])
    print('完整记录：', filename)
    print('这是离线检查用的起点快照；未验证模型映射或抓取。')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
