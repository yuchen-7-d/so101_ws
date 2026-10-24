#!/usr/bin/env python3
"""Compare saved manual endpoints with existing mappings and explicit XML ranges.

Standard library only. No servo, ROS, MuJoCo, camera, or execution dependency.
This diagnostic does not calibrate a reference or generate motor commands.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path('/home/yu/so101_ws')
REGISTRY = Path('/home/yu/ros2_ws/data/so101_joint_range_observations/joint_ranges_05kzoo_b.json')
REGISTRY_SHA = 'f07e10d656704903deca38a2af5033d4d4145dfce09d4b95b85341a861ddfe97'
PINS = {
    'data/so101_joint_mapping.json': 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8',
    'data/gripper_gap_third_point_20261023.json': 'd4412593585c5e288d951ae758b1e09d39c302fe6023a2e4c7fdd382550233d0',
    'simulation/so101/so101_new_calib.xml': 'a65f2fe17fadfb66710a1e29cf2168c0a726e803d43ae56a6f7c17d0063f6b02',
    'simulation/so101/scene_grasp.xml': '657b0387a443b26ba225d826e845a380e86773e5342cd7617187ad0ac67fde73',
    'scripts/so101_plan_grasp.py': 'bb6abd9e08a3738fa0d85f78fcdec8da1c3adf84a766338941447ff7223dda58',
}
NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
REASONS = {
    'user_reported_intrinsic_mechanical_stop': '自身挡块（用户观察）',
    'user_reported_environment_contact_table': '桌面接触（用户观察）',
    'user_reported_cable_restriction': '线缆限制（用户观察）',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), '数值无效')
    return float(value)


def count(value):
    require(type(value) is int, '计数必须为整数')
    return value


def pinned_read(path, expected, inputs):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == expected, f'文件摘要已变化，停止使用本批诊断：{path}')
    inputs[str(path)] = digest
    return data


def explicit_xml_ranges(xml_bytes):
    """Read explicit attributes, not an MJCF compiler or collision checker."""
    root = ET.fromstring(xml_bytes)
    compiler = root.find('compiler')
    require(compiler is not None and compiler.get('angle') == 'radian', '本程序只处理当前弧度制模型')
    require(compiler.get('autolimits') == 'true', '模型 autolimits 发生变化')
    result = {}
    for name in NAMES:
        joints = [j for j in root.findall('./worldbody//joint') if j.get('name') == name]
        acts = [a for a in root.findall('./actuator/position') if a.get('name') == name]
        require(len(joints) == len(acts) == 1, f'{name} 缺少唯一关节或位置执行器')
        joint, act = joints[0], acts[0]
        require(joint.get('type') == 'hinge' and act.get('joint') == name, f'{name} 结构变化')
        require(joint.get('limited') != 'false' and act.get('ctrllimited') != 'false', f'{name} 限位配置变化')
        jr = [number(float(x)) for x in joint.attrib['range'].split()]
        ar = [number(float(x)) for x in act.attrib['ctrlrange'].split()]
        require(len(jr) == len(ar) == 2 and jr[0] < jr[1] and ar[0] < ar[1], f'{name} 范围无效')
        effective = [max(jr[0], ar[0]), min(jr[1], ar[1])]
        require(effective[0] < effective[1], f'{name} 范围交集为空')
        result[name] = {
            'joint_range_rad': jr, 'actuator_ctrlrange_rad': ar,
            'explicit_range_intersection_deg': [math.degrees(x) for x in effective],
        }
    return result


def continuous_angles(first_raw, first_continuous, endpoint_counts,
                      reference_counts, reference_angle_deg, direction):
    """Choose one branch at the first sample, then retain the recorded unwrap."""
    for value in (first_raw, first_continuous, reference_counts, *endpoint_counts):
        count(value)
    require(0 <= first_raw <= 4095 and 0 <= reference_counts <= 4095, '原始或参考计数越界')
    require(first_continuous % 4096 == first_raw, '首帧连续计数与原始计数不匹配')
    require(type(direction) is int and direction in (-1, 1), '方向无效')
    principal = (first_raw - reference_counts + 2048) % 4096 - 2048
    require(principal != -2048, '首帧与参考恰好相差半圈，分支不明确')
    lifted_reference = first_continuous - principal
    angle = number(reference_angle_deg)
    values = [angle + direction * (c - lifted_reference) * 360.0 / 4096 for c in endpoint_counts]
    return lifted_reference, values


def compare_interval(angles, limits):
    angles = [number(x) for x in angles]
    limits = [number(x) for x in limits]
    require(len(angles) == len(limits) == 2 and limits[0] < limits[1], '对比区间无效')
    observed = sorted(angles)
    return {
        'diagnostic_angle_interval_deg': observed,
        'numeric_endpoints_inside_xml': [limits[0] <= x <= limits[1] for x in angles],
        'lower_excess_deg': max(0.0, limits[0] - observed[0]),
        'upper_excess_deg': max(0.0, observed[1] - limits[1]),
        'observed_encoder_span_deg': observed[1] - observed[0],
        'xml_interval_span_deg': limits[1] - limits[0],
        'interval_midpoint_difference_deg': sum(observed) / 2 - sum(limits) / 2,
        'midpoint_difference_is_calibration_correction': False,
    }


def coverage_flags(raw_counts, measured_interval):
    require(len(measured_interval) == 2, '测量覆盖范围无效')
    low, high = map(count, measured_interval)
    require(low < high, '测量覆盖范围无效')
    return [not low <= count(c) <= high for c in raw_counts]


def verify_saved_observation(obs, registry, inputs):
    sid = count(obs['joint_id'])
    proof = obs['proof']
    doc = json.loads(pinned_read(proof['report_path'], proof['report_sha256'], inputs))
    require(doc['joint_id'] == sid and doc['status'] == 'manual_endpoints_recorded_not_limits_verified', '源报告不匹配')
    require(type(doc['register_writes']) is int and doc['register_writes'] == 0, '源报告不是零写入记录')
    require(doc['error'] is None and doc['configuration_unchanged'] is True, '源报告未正常完成')
    require(doc['config_before'] == doc['config_after'], '源记录配置不一致')
    require(doc['mapping_sha256'] == PINS['data/so101_joint_mapping.json'], '源记录映射不一致')
    for key, conf in doc['config_before'].items():
        require(conf['offset_raw'] == registry['baseline_snapshot']['offset_raw_by_id'][key], 'offset 不一致')
    observation = doc['observation']
    require(observation['continuous_tracking_valid'] is True, '连续计数失效')
    require(observation['two_distinct_endpoints_recorded'] is True, '缺少两个端点')
    saved_range = registry['joint_ranges'][str(sid)]
    classified = {saved_range[k]['label']: saved_range[k] for k in ('lower_endpoint', 'upper_endpoint')}
    require(set(classified) == {'a', 'b'}, '分类记录缺少端点标签')
    for label in ('a', 'b'):
        ep = obs['endpoints'][label]
        source = observation['markers'][label]
        for key in ('raw_counts', 'continuous_counts', 'stable_intervals_counts', 'sample_index'):
            require(ep[key] == source[key], '汇总端点与原始报告不一致')
            require(classified[label][key] == ep[key], '停止原因分类记录与端点不一致')
        require(classified[label]['boundary_type'] in REASONS, '停止原因分类尚未支持')
    values = sorted(obs['endpoints'][k]['continuous_counts'][sid - 1] for k in ('a', 'b'))
    require(values == observation['endpoint_interval_counts'] == obs['observed_continuous_interval_counts'], '端点区间不一致')
    journal = pinned_read(proof['journal_path'], proof['journal_sha256'], inputs)
    first = None
    final = None
    for line in journal.splitlines():
        item = json.loads(line)
        if first is None and item['kind'] == 'sample':
            first = item['data']
        if item['kind'] == 'final':
            final = item['data']
    require(first is not None and first['index'] == 0 and final == doc, '过程记录与报告不一致')
    return doc, first


def build_report(workspace=ROOT, registry_path=REGISTRY):
    inputs = {}
    registry = json.loads(pinned_read(registry_path, REGISTRY_SHA, inputs))
    require(registry['status'] == 'all_six_joint_observations_saved_limits_not_verified', '观察记录状态不匹配')
    blobs = {name: pinned_read(Path(workspace) / name, digest, inputs) for name, digest in PINS.items()}
    mapping = json.loads(blobs['data/so101_joint_mapping.json'])
    gripper = json.loads(blobs['data/gripper_gap_third_point_20261023.json'])
    require(mapping['model_mapping_verified'] is False and gripper['model_mapping_verified'] is False, '标定状态已变化，应重新核对')
    ranges = explicit_xml_ranges(blobs['simulation/so101/so101_new_calib.xml'])
    observations = sorted(registry['observations'], key=lambda o: o['joint_id'])
    require([o['joint_id'] for o in observations] == list(range(1, 7)), '必须恰好包含六轴记录')
    rows = []
    for obs in observations:
        sid = obs['joint_id']
        name = NAMES[sid - 1]
        doc, first = verify_saved_observation(obs, registry, inputs)
        saved_range = registry['joint_ranges'][str(sid)]
        classified = {saved_range[k]['label']: saved_range[k] for k in ('lower_endpoint', 'upper_endpoint')}
        endpoints = [classified[label] for label in ('a', 'b')]
        continuous = [ep['continuous_counts'][sid - 1] for ep in endpoints]
        raw = [ep['raw_counts'][sid - 1] for ep in endpoints]
        if sid <= 5:
            reference = mapping['joint_references'][str(sid)]
            config = mapping['joint_directions'][str(sid)]
            require(config['joint_name'] == name, '映射关节名称不一致')
            ref_count, ref_angle = reference['position_count'], reference['model_angle_deg']
            direction = config['direction']
            extrapolated = None
            coverage = None
            basis = 'existing_arm_mapping_not_verified'
        else:
            candidate = gripper['mapping_candidate']
            require(candidate['offset_raw'] == doc['config_before']['6']['offset_raw'], '夹爪候选 offset 不一致')
            require(math.isclose(candidate['radians_per_count'], 2 * math.pi / 4096, rel_tol=0, abs_tol=1e-15), '夹爪比例已变化')
            ref_count = candidate['reference_counts']
            ref_angle = math.degrees(candidate['reference_model_rad'])
            direction = candidate['direction']
            coverage = candidate['measured_count_interval']
            extrapolated = coverage_flags(raw, coverage)
            basis = 'gripper_candidate_extrapolation_not_verified'
        lifted, angles = continuous_angles(first['raw_counts'][sid - 1], first['continuous_counts'][sid - 1],
                                           continuous, ref_count, ref_angle, direction)
        limits = ranges[name]['explicit_range_intersection_deg']
        rows.append({
            'joint_id': sid, 'joint_name': name, 'basis': basis,
            'raw_endpoint_counts_A_B': raw, 'continuous_endpoint_counts_A_B': continuous,
            'reference_count_in_recorded_branch': lifted,
            'first_sample_raw_count': first['raw_counts'][sid - 1],
            'first_sample_continuous_count': first['continuous_counts'][sid - 1],
            'reference_angle_deg': ref_angle, 'direction': direction,
            'diagnostic_endpoint_angles_deg_A_B': angles,
            'xml_ranges': ranges[name], 'comparison': compare_interval(angles, limits),
            'gripper_measured_count_interval': coverage,
            'gripper_endpoint_extrapolated_A_B': extrapolated,
            'stopping_reasons_A_B': [ep['boundary_type'] for ep in endpoints],
            'saved_full_endpoint_poses': obs['endpoints'],
            'measurement_condition': obs.get('measurement_condition'),
            'source_report': obs['proof']['report_path'],
            'source_report_sha256': obs['proof']['report_sha256'],
            'model_mapping_verified': False, 'software_limits_verified': False,
        })
    return {
        'schema_version': 1, 'status': 'not_for_motion',
        'scope': 'saved_active_joint_numeric_comparison_only',
        'execution_available': False, 'register_writes': 0,
        'model_mapping_verified': False, 'software_limits_changed': False,
        'device_limits_changed': False, 'model_limits_changed': False,
        'collision_checked': False, 'mujoco_model_loaded': False,
        'source_files': inputs, 'joints': rows,
        'numeric_outside_xml_ids': [r['joint_id'] for r in rows if not all(r['comparison']['numeric_endpoints_inside_xml'])],
        'user_reported_external_boundary_ids': [2, 5],
        'limitations': [
            'Uses saved endpoint records, not live feedback or a current motion start.',
            'Selects the reference branch once at the first recorded sample; does not wrap each endpoint independently.',
            'Reads explicit XML joint/actuator ranges only; does not compile MJCF or check meshes, contacts, FK or IK.',
            'Numeric agreement does not verify reference angles, directions or physical clearance.',
            'ID2 table contact and ID5 cable restrictions do not establish intrinsic joint stops.',
            'ID6 values outside the prior gap-observation interval are unverified candidate extrapolations, not gap predictions.',
            'Interval midpoint differences are not zero-offset corrections; no mapping or limits are fitted or changed.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线对照手动端点、现有映射与 XML 范围；不连接串口，不运行 MuJoCo。')
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.workspace, args.registry)
        # Abort if inputs changed while the diagnostic was running.
        for path, digest in report['source_files'].items():
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'运行期间文件变化：{path}')
        folder = Path(tempfile.mkdtemp(prefix='so101_range_model_comparison_'))
        saved = folder / 'report.json'
        with saved.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        print('离线数值对照；未访问串口，未加载 MuJoCo，未修改映射或限位。')
        print('同一连续分支换算 A/B；模型范围为 XML 关节范围与执行器范围的交集。')
        for row in report['joints']:
            sid = row['joint_id']
            a, b = row['diagnostic_endpoint_angles_deg_A_B']
            lo, hi = row['xml_ranges']['explicit_range_intersection_deg']
            check = row['comparison']
            note = '数值在范围内' if all(check['numeric_endpoints_inside_xml']) else '存在数值超界'
            print(f'ID{sid}: A={a:+.3f}°，B={b:+.3f}°；XML=[{lo:+.3f}, {hi:+.3f}]°；{note}')
            reasons = [REASONS.get(r, r) for r in row['stopping_reasons_A_B']]
            print(f'  A原因={reasons[0]}；B原因={reasons[1]}')
            print(f"  编码器跨度={check['observed_encoder_span_deg']:.3f}°，XML跨度={check['xml_interval_span_deg']:.3f}°；"
                  f"下侧超界={check['lower_excess_deg']:.3f}°，上侧超界={check['upper_excess_deg']:.3f}°")
            if sid == 6:
                print(f"  原夹爪开口测量覆盖={row['gripper_measured_count_interval']}；"
                      f"A/B是否外推={row['gripper_endpoint_extrapolated_A_B']}；不能据此外推值确认夹爪标定。")
        print('数值超出 XML 范围的关节：', report['numeric_outside_xml_ids'])
        print('ID2桌面边界、ID5线缆边界保留原分类；本次没有生成软件限位。')
        print('状态：not_for_motion；未检查原来的穿模是否消除。')
        print('完整记录：', saved)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        print(f'离线对照停止：{type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
