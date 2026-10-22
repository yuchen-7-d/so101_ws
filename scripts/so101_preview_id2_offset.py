"""Offline ID2 count-origin proposal; no serial, ROS, MuJoCo or apply mode.

Only reads saved reports/model/mapping and writes a new report under /tmp.
The proposed offset is not a joint-angle calibration or a motion command.
"""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET


K = 4096 / 360
SOURCES = [
    'https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation',
    'https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/motors/feetech/feetech.py',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), '数值不是有限实数')
    return float(value)


def analyze(args, report):
    approach_path = args.approach.resolve()
    raw = approach_path.read_bytes()
    approach = json.loads(raw)
    require(approach['kind'] == 'so101_saved_approach_geometry_check'
            and approach['status'] == 'not_for_motion'
            and approach['execution_available'] is False
            and type(approach['writes_sent']) is int and approach['writes_sent'] == 0,
            '输入不是零写入的离线接近检查')
    require(approach['sampled_geometry_complete'] is True
            and approach['count_boundary_ids'] == [2], '本工具只处理已检查路径的 ID2 跨界')
    for group in ('input_files', 'current_model_files'):
        require(bool(approach[group]), '缺少输入文件摘要')
        for path, expected in approach[group].items():
            require(digest(path) == expected, f'文件已变化或摘要不匹配：{path}')

    root = args.workspace.resolve()
    mapping_path = root / 'data/so101_joint_mapping.json'
    model_path = root / 'simulation/so101/so101_new_calib.xml'
    for path in (mapping_path, model_path):
        require(approach['current_model_files'].get(str(path)) == digest(path),
                f'输入报告没有核对当前文件：{path}')
    mapping = json.loads(mapping_path.read_bytes())
    direction = mapping['joint_directions']['2']
    require(direction['joint_name'] == 'shoulder_lift'
            and type(direction['direction']) is int and direction['direction'] == 1,
            '本工具限定 ID2 shoulder_lift 正方向映射')
    reference = mapping['joint_references']['2']
    old_ref = reference['position_count']
    require(type(old_ref) is int and 0 <= old_ref <= 4095, '参考计数无效')
    ref_angle = number(reference['model_angle_deg'])

    states = [s for s in approach['state_snapshot'] if s['servo_id'] == 2]
    require(len(states) == 1 and states[0]['position_counts'] == args.observed_count,
            '用户读数与离线起点不同；需要重新采集并检查起点')
    require(args.observed_offset_raw == 0, '本次方案只适用于 ID2 当前 offset_raw=0')
    require(states[0]['torque_enabled'] is False, '保存的起点扭矩不是关闭状态')
    q0 = number(approach['start_model_angles_deg'][1])
    require(abs(q0 - (ref_angle + ((args.observed_count-old_ref+2048) % 4096-2048)/K)) < 1e-8,
            '保存的起点角与当前映射不一致')

    # Put model q=0 near count 2048. Do not center the current bent pose.
    shift = round(2048 - (old_ref - ref_angle*K))
    new_ref = old_ref + shift
    new_offset = -shift  # Feetech: feedback = encoder - offset (single-turn branch).
    require(abs(new_offset) <= 2047 and 0 <= new_ref <= 4095,
            '本工具限定 11 位幅值偏移和单圈参考计数')
    offset_raw = abs(new_offset) | (0x800 if new_offset < 0 else 0)

    def new_count(angle):
        return new_ref + (number(angle)-ref_angle)*K

    new_start = new_count(q0)
    require(abs(new_start-round(new_start)) < 1e-8, '新起点不是整数计数')
    branch_translation = round(new_start) - args.observed_count

    xml = ET.parse(model_path).getroot()
    compiler = xml.find('compiler')
    require(compiler is not None and compiler.get('angle') == 'radian', '模型角度单位不符合约定')
    joints = xml.findall(".//joint[@name='shoulder_lift']")
    actuators = xml.findall("./actuator/position[@joint='shoulder_lift']")
    require(len(joints) == len(actuators) == 1, '无法唯一找到 ID2 关节和位置执行器')
    joint_range = [math.degrees(number(float(x))) for x in joints[0].attrib['range'].split()]
    control_range = [math.degrees(number(float(x))) for x in actuators[0].attrib['ctrlrange'].split()]
    require(len(joint_range) == len(control_range) == 2, '模型限位格式无效')
    qlow, qhigh = max(joint_range[0], control_range[0]), min(joint_range[1], control_range[1])
    require(qlow < qhigh, '模型限位没有有效交集')
    full_range = [new_count(qlow), new_count(qhigh)]
    require(0 <= full_range[0] <= full_range[1] <= 4095, '平移后模型全范围仍有计数越界')

    plan_paths = [Path(p) for p in approach['input_files'] if Path(p).name == 'report.json']
    require(len(plan_paths) == 1, '无法唯一找到原候选规划报告')
    plan = json.loads(plan_paths[0].read_bytes())
    require(len(plan['samples']) == 41, '原候选样本数量不是 41')
    checked = approach['checked_samples']
    require(bool(checked) and all(s['penetrating_contacts'] == [] for s in checked),
            '接近报告不完整或含穿透记录')
    angles = [number(s['model_angles_deg'][1]) for s in checked]
    angles += [math.degrees(number(s['model_q_rad'][1])) for s in plan['samples']]
    require(all(qlow-1e-8 <= q <= qhigh+1e-8 for q in angles), '样本角超出原模型限位')
    translated = [new_count(q) for q in angles]
    require(all(0 <= c <= 4095 for c in translated), '样本平移后仍越界')
    translation_error = max(abs(new_count(q) - (
        args.observed_count + (q-q0)*K + branch_translation)) for q in angles)
    require(translation_error < 1e-8, '各点未使用同一连续分支平移')
    angle_error = max(abs(ref_angle + (new_count(q)-new_ref)/K - q) for q in angles)
    require(angle_error < 1e-8, '平移改变了模型角度')
    ranges = [r for r in approach['continuous_count_ranges_diagnostic_only'] if r['servo_id'] == 2]
    require(len(ranges) == 1, '缺少 ID2 原计数范围')
    prior = ranges[0]
    for key, value in [('minimum', min(translated)), ('maximum', max(translated))]:
        require(abs(value-(number(prior[key])+branch_translation)) < 1e-8,
                '重新计算的路径范围与原报告不一致')

    report.update(
        status='offline_offset_preview',
        input_files={str(approach_path): hashlib.sha256(raw).hexdigest(),
                     **approach['input_files'], **approach['current_model_files']},
        user_reported_config={'source': 'CLI transcription; not read by this program',
                              'servo_id': 2, 'position_count': args.observed_count,
                              'offset_raw': args.observed_offset_raw},
        proposal={'servo_id': 2, 'offset_signed': new_offset, 'offset_raw': offset_raw,
                  'offset_raw_hex': f'0x{offset_raw:04X}', 'count_phase_shift': shift,
                  'continuous_branch_translation': branch_translation,
                  'old_reference_count': old_ref, 'new_reference_count': new_ref,
                  'reference_angle_deg_unchanged': ref_angle,
                  'start_count_predicted': round(new_start),
                  'pregrasp_count_predicted': new_count(math.degrees(plan['samples'][0]['model_q_rad'][1])),
                  'path_count_range_predicted': [min(translated), max(translated)],
                  'model_limit_deg_unchanged': [qlow, qhigh],
                  'model_limit_count_range_predicted': full_range},
        checked_angle_samples=len(angles), max_angle_change_deg=angle_error,
        max_translation_error_counts=translation_error,
        model_mapping_verified=mapping.get('model_mapping_verified') is True,
        requires_device_readback=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approach', required=True, type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('/home/yu/so101_ws'))
    parser.add_argument('--observed-count', required=True, type=int)
    parser.add_argument('--observed-offset-raw', required=True, type=int)
    args = parser.parse_args()
    report = {'schema_version': 1, 'kind': 'so101_id2_offset_proposal', 'status': 'invalid',
              'execution_available': False, 'register_writes': 0, 'project_files_modified': [],
              'hardware_offset_applied': False, 'software_mapping_applied': False,
              'references': SOURCES,
              'limitations': ['Saved data only; no live device or target freshness check.',
                              'Offset encoding and single-turn response require hardware readback.',
                              'Joint/base/gripper accuracy remains unverified.',
                              'No new collision, dynamics or mechanical-limit validation.',
                              'Do not use predicted counts with the current device offset.',
                              'Do not apply the software reference before verified device change.']}
    rc = 0
    try:
        analyze(args, report)
        p = report['proposal']
        print('离线预览；未访问串口，未修改项目配置。')
        print(f"ID2 候选中位修正：{p['offset_signed']}；编码原始值：{p['offset_raw_hex']}")
        print(f"软件参考计数候选：{p['old_reference_count']} → {p['new_reference_count']}；参考角和方向保持不变")
        print(f"平移后的起点预测：{p['start_count_predicted']}")
        print(f"平移后的预抓取计数预测：{p['pregrasp_count_predicted']:.2f}")
        lo, hi = p['path_count_range_predicted']
        print(f'整条已保存路径的计数范围预测：{lo:.2f}–{hi:.2f}')
        lo, hi = p['model_limit_count_range_predicted']
        print(f'ID2 模型全范围的计数范围预测：{lo:.2f}–{hi:.2f}')
        print(f"核对角度样本：{report['checked_angle_samples']}；最大角度变化：{report['max_angle_change_deg']:.3g}°")
    except Exception as error:
        report.update(status='invalid', error=f'{type(error).__name__}: {error}')
        print('预览未完成：', report['error'])
        rc = 1
    with tempfile.NamedTemporaryFile('w', dir='/tmp', prefix='so101_id2_offset_', suffix='.json',
                                     encoding='utf-8', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        saved_path = stream.name
    print('状态：', report['status'], '；execution_available=False；寄存器写入=0')
    print('完整记录：', saved_path)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
