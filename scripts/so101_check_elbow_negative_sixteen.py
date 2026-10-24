"""从腕旋转测试日志的保持末态检查ID3减少16计数；没有串口或ROS入口。"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path('/home/yu/so101_ws')
SOURCE = ROOT / 'data/real_control_runs/wrist_roll_positive_eight_20261022T223024_mg_5ykva.jsonl'
MAPPING = ROOT / 'data/so101_joint_mapping.json'
PLAN = Path('/tmp/so101_ros_plan_cq24l2e1/report.json')
GEOMETRY = Path('/tmp/so101_third_point_gripper_check_urybagdw.json')
GRIPPER = ROOT / 'data/gripper_gap_third_point_20261023.json'
PINNED = {
    SOURCE: '41bf1c90856cb256c826b91f925fb088c6959c9a346901fe82b3ab82c4e9c7ab',
    MAPPING: 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8',
    PLAN: 'fddc54511232cb5ae1e1a830bf0e6400a53ddb7d1a1685ee7e1f3b30582ec428',
    GEOMETRY: '24b430819799d9e9242698c1e7932298d461448fbe21bb39a39bd4a8eb54ed60',
    GRIPPER: 'd4412593585c5e288d951ae758b1e09d39c302fe6023a2e4c7fdd382550233d0',
}
EXPECTED_START = [8, 964, 4073, 1950, 3289, 2143]
AXIS = 3
DELTA = -16
OFFSETS = [0, 0x0C9D, 0, 0x071E, 0, 0x0F3D]
SCALE = 2 * math.pi / 4096
OTHER_INDICES = (0, 1, 3, 4)
EXPECTED_SAMPLES = {'nominal': 17, 'holding_error_grid': 1782}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f'输入文件已变化：{path}')
    return raw


def validate_journal(rows):
    require(rows and rows[-1]['stage'] == 'final_cleanup', '腕旋转记录缺少最终收尾')
    state = rows[-1]['snapshot']
    require(state['status'] == 'wrist_roll_positive_eight_complete'
            and state['test'] == 'wrist_roll_positive_eight_independent_probe'
            and state['step_response_complete'] is True
            and state['step_goal_readback_confirmed'] is True
            and state['return_executed'] is False
            and state['fault'] is None and state['journal_error'] is None
            and state['proof']['mapping_sha256'] == PINNED[MAPPING], '腕旋转测试未完整通过或映射改变')
    require(state['frozen_start_targets'] == {'1': 8, '2': 964, '3': 4073, '4': 1950, '5': 3282}
            and state['loaded_baseline'] == 3282 and state['step_goal'] == 3290
            and state['axis'] == 5 and state['delta_counts'] == 8,
            '腕旋转测试目标或有载基线不匹配')
    release = state['release']
    require(release['confirmed'] is True and release['unconfirmed_ids'] == []
            and release['servo_ids'] == [1, 2, 3, 4, 5], '腕旋转测试释放未确认')
    for sid in range(1, 6):
        r = release['readback'][str(sid)]
        require(all(type(r[k]) is int and r[k] == 0 for k in ('40', '48')),
                f'ID{sid}释放读回不满足')
    events = state['events']
    require(events and all(e['acknowledged'] is True for e in events), '腕旋转测试存在未确认通讯')
    goals = []
    for e in events:
        require(e['instruction'] in (2, 3), '腕旋转日志包含其他指令')
        if e['instruction'] != 3:
            continue
        raw = bytes.fromhex(e['parameters_hex'])
        require(e['servo_id'] in (1, 2, 3, 4, 5) and raw[0] in (40, 41, 42, 46, 48),
                '腕旋转日志包含未预期写入')
        if raw[0] == 42:
            goals.append((e['servo_id'], int.from_bytes(raw[1:], 'little')))
    require(goals == [(1, 8), (2, 964), (3, 4073), (4, 1950), (5, 3282), (5, 3290)],
            '腕旋转日志包含额外运动目标或回程')
    for sid, offset in enumerate(OFFSETS, 1):
        raw = bytes.fromhex(state['initial_configs'][str(sid)])
        require(len(raw) == 40 and raw[5] == sid and raw[33] == 0
                and raw[18] == 12 and raw[30] == 1
                and int.from_bytes(raw[31:33], 'little') == offset
                and int.from_bytes(raw[9:11], 'little') == 0
                and int.from_bytes(raw[11:13], 'little') == 4095,
                f'ID{sid}日志配置不匹配')
    phases = ('after_step', 'support_requested')
    samples = [s for s in state['samples'] if s['phase'] in phases]
    require({s['phase'] for s in samples} == set(phases) and len(samples) >= 20
            and samples[-1]['elapsed_s'] - samples[0]['elapsed_s'] >= 4.8,
            '腕旋转测试缺少到位后的持续反馈')
    for sample in samples:
        frames, gripper = sample['states'], sample['gripper']
        require([frames[str(i)]['position'] for i in range(1, 6)] + [gripper['position']]
                == EXPECTED_START, '到位后姿态并非本次单一参考姿态')
        require(gripper['torque'] == gripper['output_limit'] == gripper['speed_raw_signed']
                == gripper['output_raw_signed'] == 0 and gripper['lock'] == 1,
                '夹爪释放状态不匹配')
        for frame in [*frames.values(), gripper]:
            require(frame['status'] == 0 and 6.0 <= frame['voltage'] <= 7.4
                    and frame['temperature'] < 55, '日志中的健康状态异常')
    feedback = [s for s in state['step_frames'] if s['phase'] in phases]
    require(len(feedback) >= 20 and all(s['position'] == 3289 and s['speed_raw_signed'] == 0
                                      for s in feedback), '腕旋转到位后的完整反馈不稳定')
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
    rows = [json.loads(line) for line in blobs[SOURCE].splitlines() if line.strip()]
    start = validate_journal(rows)
    report['start_counts'] = start
    report['source_journal'] = str(SOURCE)
    report['source_scope'] = 'stable feedback before release; not fresh post-release state'
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
    # Include elbow motion down to G0-18 and initial drift up to G0+3.
    # The active goal remains G0-16; these samples never become motor commands.
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
    return {'schema_version': 1, 'kind': 'so101_saved_elbow_negative_sixteen_geometry',
            'status': 'not_for_motion', 'execution_available': False, 'register_writes': 0,
            'model_mapping_verified': False, 'sampled_geometry_complete': False,
            'axis': AXIS, 'delta_counts': DELTA, 'nominal_sample_step_counts': 1,
            'expected_sample_counts': dict(EXPECTED_SAMPLES),
            'holding_error_grid': {'elbow_delta_counts': [DELTA - 2, 3],
                                   'other_arm_axes_delta_samples': [-3, 0, 3],
                                   'gripper_fixed_counts': 2143,
                                   'is_motion_path': False},
            'checked_samples': [], 'error': None,
            'limitations': ['Pre-release stable feedback and saved box position, not a live environment.',
                            'Provisional arm mapping, base transform and gripper gap fit.',
                            'Discrete geometry samples only; no continuous, dynamic or force proof.',
                            'Gripper fixed at 2143; its drift, hands, cables and external supports are not covered.']}


def main():
    report = new_report()
    print('离线检查腕旋转测试日志中保存姿态的ID3减少16计数；不读取串口，不执行运动。')
    try:
        analyze(report)
    except (Exception, KeyboardInterrupt) as exc:
        report['sampled_geometry_complete'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='so101_elbow_negative16_step_',
                                     suffix='.json', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        saved = stream.name
    if 'start_counts' in report:
        start = report['start_counts']
        print('释放前保存的稳定姿态：', start)
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
