"""Saved-pose ID3 -8-count geometry preview. No hardware or ROS access."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path('/home/yu/so101_ws')
JOURNAL = ROOT / 'data/real_control_runs/arm_current_hold_20261022T210254_4lgy7w4w.jsonl'
GEOMETRY = Path('/tmp/so101_third_point_gripper_check_urybagdw.json')
PLAN = Path('/tmp/so101_ros_plan_cq24l2e1/report.json')
GRIPPER = ROOT / 'data/gripper_gap_third_point_20261023.json'
MAPPING = ROOT / 'data/so101_joint_mapping.json'
PINNED = {
    JOURNAL: 'd7d75172e3b6a94578b664dbb36291da167acc220bc46df7e1d4c50f5a59b325',
    GEOMETRY: '24b430819799d9e9242698c1e7932298d461448fbe21bb39a39bd4a8eb54ed60',
    PLAN: 'fddc54511232cb5ae1e1a830bf0e6400a53ddb7d1a1685ee7e1f3b30582ec428',
    GRIPPER: 'd4412593585c5e288d951ae758b1e09d39c302fe6023a2e4c7fdd382550233d0',
    MAPPING: 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8',
}
SCALE = 2 * math.pi / 4096


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_bytes(path, expected):
    path = Path(path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f'输入文件已变化：{path}')
    return raw


def validate_counts(counts):
    require(len(counts) == 6 and all(type(c) is int and 0 <= c <= 4095
                                   for c in counts), '需要六个0–4095整数计数')


def step_counts(start):
    """Only ID3 changes; include start and endpoint, no modulo wrapping."""
    validate_counts(start)
    require(start[2] >= 8, 'ID3减8会跨过单圈边界')
    for delta in range(0, -9, -1):
        point = list(start)
        point[2] += delta
        yield point


def initial_angles(counts, mapping):
    validate_counts(counts)
    result = []
    for sid, count in enumerate(counts[:5], 1):
        ref = mapping['joint_references'][str(sid)]
        sign = mapping['joint_directions'][str(sid)]['direction']
        require(type(sign) is int and sign in (-1, 1), '关节方向无效')
        delta = (count - ref['position_count'] + 2048) % 4096 - 2048
        result.append(math.radians(ref['model_angle_deg']) + sign * delta * SCALE)
    return result


def load_inputs(report):
    blobs = {p: checked_bytes(p, sha) for p, sha in PINNED.items()}
    report['input_files'] = {str(p): sha for p, sha in PINNED.items()}
    geometry = json.loads(blobs[GEOMETRY])
    candidate = json.loads(blobs[GRIPPER])
    model_files = dict(geometry['current_model_files'])
    model_files.update(candidate['input_files'])
    for path, sha in model_files.items():
        checked_bytes(path, sha)
    report['model_files'] = model_files
    plan, mapping = json.loads(blobs[PLAN]), json.loads(blobs[MAPPING])
    for path, sha in plan['files'].items():
        checked_bytes(path, sha)
    rows = [json.loads(line) for line in blobs[JOURNAL].splitlines() if line.strip()]
    require(rows[-1]['stage'] == 'final_cleanup', '保持日志没有最终收尾记录')
    held = rows[-1]['snapshot']
    require(held['status'] == 'current_pose_hold_complete' and held['fault'] is None
            and held['journal_error'] is None and held['release']['confirmed'] is True,
            '保持日志未完成或释放未确认')
    require(held['proof']['mapping_sha256'] == PINNED[MAPPING], '保持日志映射不匹配')
    offsets = [0, 0x0C9D, 0, 0x071E, 0, 0x0F3D]
    for sid, offset in enumerate(offsets, 1):
        config = bytes.fromhex(held['initial_configs'][str(sid)])
        require(len(config) == 40 and int.from_bytes(config[31:33], 'little') == offset,
                f'ID{sid}日志中的offset不匹配')
    frozen = [held['targets'][str(i)] for i in range(1, 6)] + [held['gripper_start']]
    validate_counts(frozen)
    variants = [{'label': 'frozen_hold_targets', 'counts': frozen}]
    seen = {tuple(frozen)}
    for sample in held['samples']:
        if sample['phase'] not in ('holding', 'support_requested'):
            continue
        counts = [sample['states'][str(i)]['position'] for i in range(1, 6)]
        counts.append(sample['gripper']['position'])
        validate_counts(counts)
        require(counts[5] == 2143, '日志中的夹爪位置发生变化')
        require(max(abs(c - b) for c, b in zip(counts[:5], frozen[:5])) <= 3,
                '日志中存在超出本次已核对范围的姿态')
        if tuple(counts) not in seen:
            seen.add(tuple(counts))
            variants.append({'label': f"{sample['phase']}@{sample['elapsed_s']:.6f}s",
                             'counts': counts})
    require(len(held['samples']) > 0 and len(variants) > 1, '日志缺少姿态采样')
    report['start_pose_variants'] = variants
    report['source_scope'] = 'saved hold targets and sampled feedback; not fresh post-release state'
    report['saved_target'] = {k: plan['input_target'][k] for k in
                              ('session_id', 'frame_id', 'captured_at_unix_s', 'center_m')}
    report['saved_target']['freshness_verified'] = False
    return frozen, variants, mapping, candidate, plan


def analyze(report):
    frozen, variants, mapping, candidate, plan = load_inputs(report)
    # These imports and MuJoCo model loading occur only when the user runs main.
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
    require(np.allclose(model.geom_size[box], size / 2, atol=1e-9, rtol=0), '盒子模型尺寸变化')
    data.qpos[planner.box_qadr:planner.box_qadr+3] = center - [0, 0, size[2]/2]
    yaw = math.atan2(rotation[1, 2], rotation[0, 2])
    data.qpos[planner.box_qadr+3:planner.box_qadr+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    grip = candidate['mapping_candidate']
    require(grip['reference_counts'] == 1913 and grip['direction'] == 1
            and abs(grip['radians_per_count'] - SCALE) < 1e-15,
            '夹爪候选映射不匹配')
    g = grip['reference_model_rad'] + (frozen[5] - 1913) * SCALE
    gj = planner.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    ga = planner.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper')
    require(max(model.jnt_range[gj, 0], model.actuator_ctrlrange[ga, 0]) <= g <=
            min(model.jnt_range[gj, 1], model.actuator_ctrlrange[ga, 1]), '夹爪角超限')
    q0 = np.asarray(initial_angles(frozen, mapping))
    signs = np.array([mapping['joint_directions'][str(i)]['direction'] for i in range(1, 6)])
    report['gripper_angle_deg'] = math.degrees(g)
    report['frozen_model_angles_deg'] = np.rad2deg(q0).tolist()
    report['predicted_tcp_displacements_mm'] = []
    for variant_index, variant in enumerate(variants):
        first_tcp = None
        for sample_index, counts in enumerate(step_counts(variant['counts'])):
            # Keep the original continuous angle branch throughout every sweep.
            q = q0 + (np.array(counts[:5]) - np.array(frozen[:5])) * SCALE * signs
            require(np.all(q >= planner.limits[:, 0]) and np.all(q <= planner.limits[:, 1]),
                    f'姿态{variant_index}样本{sample_index}超出模型限位')
            data.qpos[planner.qadr], data.qpos[planner.gripper_qadr] = q, g
            mujoco.mj_forward(model, data)
            contacts = []
            for contact in data.contact[:data.ncon]:
                if contact.dist >= -1e-5:
                    continue
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                         or f'geom_{geom}' for geom in (contact.geom1, contact.geom2)]
                if set(names) == {'floor', 'a_box_geom'} and contact.dist >= -0.0005:
                    continue
                contacts.append({'geoms': names, 'penetration_mm': float(-contact.dist * 1000)})
            tcp = data.site_xpos[planner.tcp].copy()
            sample = {'variant': variant_index, 'sample': sample_index,
                      'diagnostic_counts': counts, 'model_angles_deg': np.rad2deg(q).tolist(),
                      'tcp_m': tcp.tolist(), 'penetrating_contacts': contacts}
            report['checked_samples'].append(sample)
            if contacts:
                report['first_penetrating_sample'] = sample
                raise ValueError(f'姿态{variant_index}样本{sample_index}出现穿透，停止检查')
            if first_tcp is None:
                first_tcp = tcp
        report['predicted_tcp_displacements_mm'].append(((tcp - first_tcp)*1000).tolist())
    require(len(report['checked_samples']) == 9 * len(variants), '单轴采样不完整')
    report['sampled_geometry_complete'] = True


def main():
    report = {'schema_version': 1, 'kind': 'so101_saved_elbow_step_geometry',
              'status': 'not_for_motion', 'execution_available': False, 'register_writes': 0,
              'model_mapping_verified': False, 'sampled_geometry_complete': False,
              'axis': 3, 'delta_counts': -8, 'max_sample_step_counts': 1,
              'checked_samples': [], 'error': None,
              'limitations': ['Saved poses and saved box position, not live environment.',
                              'Provisional model mapping and gripper gap fit.',
                              'Sampled geometry only; no dynamics, cables, hand or force validation.']}
    print('离线检查保存姿态的ID3减少8计数；不读取串口，不执行运动。')
    try:
        analyze(report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='so101_elbow_step_',
                                     suffix='.json', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        saved = stream.name
    variants = report.get('start_pose_variants', [])
    if variants:
        c = variants[0]['counts']
        print(f'保持目标的诊断路径：ID3 {c[2]} → {c[2]-8}；其他轴固定，ID6={c[5]}')
    print('已保存姿态种类：', len(variants), '；已检查几何样本：', len(report['checked_samples']))
    print('单轴采样完成：', report['sampled_geometry_complete'])
    if report.get('predicted_tcp_displacements_mm'):
        print('首种姿态预测TCP位移(mm)：', report['predicted_tcp_displacements_mm'][0])
    if report.get('first_penetrating_sample'):
        print('首个穿透样本：', json.dumps(report['first_penetrating_sample'], ensure_ascii=False))
    if report['error']:
        print('原因：', report['error'])
    print('状态：not_for_motion；仅保存姿态的单轴几何检查，不能直接用于实机运动。')
    print('完整记录：', saved)
    return 0 if report['sampled_geometry_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
