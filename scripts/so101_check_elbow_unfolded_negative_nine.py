"""从新起点快照检查ID3减少9计数；没有串口或ROS入口。"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path('/home/yu/so101_ws')
SOURCE = ROOT / 'data/pose_mapping_checks/current_start_6qsmm3aa.json'
MAPPING = ROOT / 'data/so101_joint_mapping.json'
PLAN = Path('/tmp/so101_ros_plan_cq24l2e1/report.json')
GEOMETRY = Path('/tmp/so101_third_point_gripper_check_urybagdw.json')
GRIPPER = ROOT / 'data/gripper_gap_third_point_20261023.json'
PINNED = {
    SOURCE: '6590f2e2654f6c224c1412228ec966ffd0273453e7d26861fff8a742d142c770',
    MAPPING: 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8',
    PLAN: 'fddc54511232cb5ae1e1a830bf0e6400a53ddb7d1a1685ee7e1f3b30582ec428',
    GEOMETRY: '24b430819799d9e9242698c1e7932298d461448fbe21bb39a39bd4a8eb54ed60',
    GRIPPER: 'd4412593585c5e288d951ae758b1e09d39c302fe6023a2e4c7fdd382550233d0',
}
EXPECTED_START = [7, 952, 3975, 1946, 3286, 2143]
AXIS = 3
DELTA = -9
OFFSETS = [0, 0x0C9D, 0, 0x071E, 0, 0x0F3D]
SCALE = 2 * math.pi / 4096
OTHER_INDICES = (0, 1, 3, 4)
EXPECTED_SAMPLES = {'nominal': 10, 'holding_error_grid': 1215}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f'输入文件已变化：{path}')
    return raw


def validate_snapshot(state, mapping):
    require(state['kind'] == 'so101_current_start_with_configuration'
            and state['backend'] == 'serial' and state['operation'] == 'read_state'
            and state['success'] is True and state['status'] == 'read_only_start_snapshot'
            and state['execution_available'] is False and not state['error']
            and type(state['register_writes']) is int and state['register_writes'] == 0
            and state['configuration_matches_mapping'] is True
            and state['mapping_sha256'] == PINNED[MAPPING]
            and state['mapping_snapshot'] == mapping, '起点快照或软件映射不匹配')
    require(len(state['capture_events']) == 30
            and all(e['instruction'] == 2 and e['acknowledged'] is True
                    for e in state['capture_events']), '快照不是完整的只读采集')
    require(len(state['feedback_rounds']) == 3
            and state['states'] == state['feedback_rounds'][-1]['states'], '快照缺少三轮反馈或最终字段不一致')
    for sid, offset in enumerate(OFFSETS, 1):
        before, after = state['config_before'][str(sid)], state['config_after'][str(sid)]
        raw = bytes.fromhex(before['raw_hex'])
        require(before['raw_hex'] == after['raw_hex'] and len(raw) == 40
                and before['id'] == sid and before['offset_raw'] == offset
                and int.from_bytes(raw[31:33], 'little') == offset
                and before['mode'] == 0 and before['phase'] == 12
                and before['resolution'] == 1 and before['minimum'] == 0
                and before['maximum'] == 4095
                and raw[2] == 0 and raw[5] == sid and raw[8] == 1
                and raw[18] == 12 and raw[30] == 1 and raw[33] == 0
                and int.from_bytes(raw[9:11], 'little') == 0
                and int.from_bytes(raw[11:13], 'little') == 4095, f'ID{sid}配置发生变化')
    for row in state['feedback_rounds']:
        frames = row['states']
        require([s['servo_id'] for s in frames] == list(range(1, 7))
                and [s['position_counts'] for s in frames] == EXPECTED_START,
                '三轮反馈未稳定在本次保存起点')
        for s in frames:
            raw = bytes.fromhex(s['raw_hex'])
            require(len(raw) == 26 and raw[0] == raw[25] == 0 and raw[15] == 1
                    and int.from_bytes(raw[16:18], 'little') == s['position_counts']
                    and int.from_bytes(raw[2:4], 'little') == s['goal_counts']
                    and raw[8:10] == raw[18:20] == raw[20:22] == bytes(2)
                    and abs(raw[22] / 10 - s['voltage']) < 1e-9
                    and raw[23] == s['temperature_c'], '反馈原始字节与快照字段不一致')
            require(s['torque_enabled'] is False and s['configuration_locked'] is True
                    and s['speed_raw'] == 0 and s['output_raw'] == 0
                    and s['output_limit'] == 0 and s['status'] == 0
                    and 6.0 <= s['voltage'] <= 7.4 and s['temperature_c'] < 55,
                    f"ID{s['servo_id']}静止或健康条件不满足")
    return EXPECTED_START.copy()


def load_inputs(report):
    blobs = {p: checked_bytes(p, sha) for p, sha in PINNED.items()}
    report['input_files'] = {str(p): sha for p, sha in PINNED.items()}
    mapping, candidate = json.loads(blobs[MAPPING]), json.loads(blobs[GRIPPER])
    plan, geometry = json.loads(blobs[PLAN]), json.loads(blobs[GEOMETRY])
    manifest = {**plan['files'], **geometry['current_model_files'], **candidate['input_files']}
    for path, sha in manifest.items():
        checked_bytes(path, sha)
    report['model_and_reference_files'] = manifest
    start = validate_snapshot(json.loads(blobs[SOURCE]), mapping)
    report['start_counts'] = start
    report['start_snapshot'] = str(SOURCE)
    report['source_scope'] = 'three stable read-only rounds in saved snapshot; not live state at planning or execution'
    report['fresh_start_verified'] = False
    report['saved_target'] = {k: plan['input_target'][k] for k in
                            ('session_id', 'frame_id', 'captured_at_unix_s', 'center_m')}
    report['saved_target']['freshness_verified'] = False
    return start, mapping, candidate, plan


def initial_angles(start, mapping):
    angles = []
    for sid, count in enumerate(start[:5], 1):
        reference = mapping['joint_references'][str(sid)]
        sign = mapping['joint_directions'][str(sid)]['direction']
        require(type(sign) is int and sign in (-1, 1), '关节方向无效')
        delta = (count - reference['position_count'] + 2048) % 4096 - 2048
        angles.append(math.radians(reference['model_angle_deg']) + sign * delta * SCALE)
    return angles


def sample_counts(start):
    require(len(start) == 6 and all(type(c) is int and 0 <= c <= 4095 for c in start),
            '需要六个单圈整数计数')
    # Nominal path: only elbow decreases, one count per sample, no return.
    for distance in range(abs(DELTA) + 1):
        point = start.copy()
        point[AXIS - 1] -= distance
        yield 'nominal', point
    # These are diagnostic poses, NOT a path or servo commands. Sample the
    # observed holding-error scale at -3, 0, +3 on the other four arm axes.
    # Include elbow motion down to G0-11 and initial drift up to G0+3.
    # The active goal remains G0-9; these samples never become motor commands.
    for deviations in itertools.product((-3, 0, 3), repeat=4):
        for delta in range(DELTA - 2, 4):
            point = start.copy()
            point[AXIS - 1] += delta
            for index, deviation in zip(OTHER_INDICES, deviations):
                point[index] += deviation
            yield 'holding_error_grid', point


def penetrating_contacts(named_contacts):
    found = []
    for names, distance in named_contacts:
        require(math.isfinite(distance), '接触距离不是有限数')
        if distance >= -1e-5:
            continue
        # Keep exactly the existing table/box support-contact exception.
        if set(names) == {'floor', 'a_box_geom'} and distance >= -0.0005:
            continue
        found.append({'geoms': names, 'penetration_mm': -distance * 1000})
    return found


def evaluate_samples(report, start, mapping, evaluate_pose):
    q0 = initial_angles(start, mapping)
    signs = [mapping['joint_directions'][str(i)]['direction'] for i in range(1, 6)]
    report['start_model_angles_deg'] = [math.degrees(q) for q in q0]
    counts_by_group = {'nominal': 0, 'holding_error_grid': 0}
    first_tcp = last_tcp = None
    for group, counts in sample_counts(start):
        require(all(type(c) is int and 0 <= c <= 4095 for c in counts),
                '诊断样本超出单圈计数；不做取模修正')
        # Only initial q0 chooses a branch; every subsequent sample is continuous.
        angles = [q + (c - b) * SCALE * sign
                  for q, c, b, sign in zip(q0, counts[:5], start[:5], signs)]
        sample = {'group': group, 'index': counts_by_group[group],
                  'diagnostic_counts': counts, 'model_angles_deg': [math.degrees(q) for q in angles]}
        sample.update(evaluate_pose(angles))
        require(len(sample['tcp_m']) == 3 and all(math.isfinite(x) for x in sample['tcp_m']),
                'TCP不是有限三维坐标')
        report['checked_samples'].append(sample)
        counts_by_group[group] += 1
        report['sample_counts'] = dict(counts_by_group)
        if sample['penetrating_contacts']:
            report['first_penetrating_sample'] = sample
            raise ValueError(f'{group}样本{sample["index"]}出现穿透，停止检查')
        if group == 'nominal':
            first_tcp = sample['tcp_m'] if first_tcp is None else first_tcp
            last_tcp = sample['tcp_m']
    require(counts_by_group == EXPECTED_SAMPLES, '采样不完整')
    report['nominal_tcp_displacement_mm'] = [(b - a) * 1000 for a, b in zip(first_tcp, last_tcp)]
    report['sampled_geometry_complete'] = True


def analyze(report):
    start, mapping, candidate, plan = load_inputs(report)
    # No model is loaded when this file is imported for offline code checks.
    import mujoco
    import numpy as np
    sys.path.insert(0, str(ROOT / 'scripts'))
    import so101_plan_grasp as planning
    planner = planning.CandidatePlanner(ROOT / 'simulation/so101/scene_grasp.xml', mapping)
    model, data = planner.model, planner.data
    reference = json.loads((ROOT / 'data/base_reference_initial.json').read_bytes())
    target = plan['input_target']
    center, rotation = planning.candidate_transform(np.asarray(target['center_m']),
                                                   target['square_yaw_deg'], reference)
    size = np.asarray(target['size_m'])
    box = planner.lookup(mujoco.mjtObj.mjOBJ_GEOM, 'a_box_geom')
    require(np.allclose(model.geom_size[box], size / 2, atol=1e-9, rtol=0), '盒子尺寸发生变化')
    data.qpos[planner.box_qadr:planner.box_qadr + 3] = center - [0, 0, size[2] / 2]
    yaw = math.atan2(rotation[1, 2], rotation[0, 2])
    data.qpos[planner.box_qadr + 3:planner.box_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    grip = candidate['mapping_candidate']
    require(grip['offset_raw'] == OFFSETS[5] and grip['reference_counts'] == 1913
            and grip['direction'] == 1 and abs(grip['radians_per_count'] - SCALE) < 1e-15
            and grip['measured_count_interval'][0] <= start[5] <= grip['measured_count_interval'][1],
            '夹爪候选映射不匹配或需要外推')
    g = grip['reference_model_rad'] + (start[5] - 1913) * SCALE
    gj = planner.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    ga = planner.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper')
    require(max(model.jnt_range[gj, 0], model.actuator_ctrlrange[ga, 0]) <= g <=
            min(model.jnt_range[gj, 1], model.actuator_ctrlrange[ga, 1]), '夹爪角超限')
    report['gripper_angle_deg'] = math.degrees(g)

    def evaluate_pose(angles):
        q = np.asarray(angles)
        require(np.all(q >= planner.limits[:, 0]) and np.all(q <= planner.limits[:, 1]),
                '样本超出原模型关节限位，停止检查')
        data.qpos[planner.qadr], data.qpos[planner.gripper_qadr] = q, g
        mujoco.mj_forward(model, data)
        named = []
        for contact in data.contact[:data.ncon]:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                     or f'geom_{geom}' for geom in (contact.geom1, contact.geom2)]
            named.append((names, float(contact.dist)))
        return {'tcp_m': data.site_xpos[planner.tcp].copy().tolist(),
                'joint_limit_margin_deg': np.rad2deg(np.minimum(q - planner.limits[:, 0],
                                                               planner.limits[:, 1] - q)).tolist(),
                'penetrating_contacts': penetrating_contacts(named)}

    evaluate_samples(report, start, mapping, evaluate_pose)
    for path, sha in {**report['input_files'], **report['model_and_reference_files']}.items():
        checked_bytes(path, sha)


def new_report():
    return {'schema_version': 1, 'kind': 'so101_unfolded_snapshot_elbow_negative_nine_geometry',
            'status': 'not_for_motion', 'execution_available': False, 'register_writes': 0,
            'model_mapping_verified': False, 'sampled_geometry_complete': False,
            'axis': AXIS, 'delta_counts': DELTA, 'nominal_sample_step_counts': 1,
            'expected_sample_counts': dict(EXPECTED_SAMPLES),
            'holding_error_grid': {'elbow_delta_counts': [DELTA - 2, 3],
                                   'other_arm_axes_delta_samples': [-3, 0, 3],
                                   'gripper_fixed_counts': 2143,
                                   'is_motion_path': False},
            'checked_samples': [], 'error': None,
            'limitations': ['Saved read-only snapshot and saved box position, not a live environment.',
                            'Provisional arm mapping, base transform and gripper gap fit.',
                            'Discrete geometry samples only; no continuous, dynamic or force proof.',
                            'Gripper fixed at 2143; its drift, hands, cables and external supports are not covered.']}


def main():
    report = new_report()
    print('离线检查手动展开后新快照的ID3减少9计数；不读取串口，不执行运动。')
    try:
        analyze(report)
    except (Exception, KeyboardInterrupt) as exc:
        report['sampled_geometry_complete'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='so101_elbow_unfolded_negative9_step_',
                                     suffix='.json', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        saved = stream.name
    if 'start_counts' in report:
        start = report['start_counts']
        print('只读快照中的稳定姿态：', start)
        print(f'名义单轴路径：ID3 {start[2]} → {start[2]+DELTA}；其他轴固定，ID6={start[5]}')
    print('已检查样本：', report.get('sample_counts', {}), '；预期：', EXPECTED_SAMPLES)
    print('单轴及保持误差几何采样完成：', report['sampled_geometry_complete'])
    if 'nominal_tcp_displacement_mm' in report:
        print('名义路径预测TCP位移(mm)：', report['nominal_tcp_displacement_mm'])
    if report.get('first_penetrating_sample'):
        print('首个穿透样本：', json.dumps(report['first_penetrating_sample'], ensure_ascii=False))
    if report['error']:
        print('原因：', report['error'])
    print('状态：not_for_motion；几何采样结果不代表实机运动已经通过。')
    print('完整记录：', saved)
    return 0 if report['sampled_geometry_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
