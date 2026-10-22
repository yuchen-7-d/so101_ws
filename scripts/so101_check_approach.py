"""Replay saved SO101 snapshots in MuJoCo; no camera, ROS or serial access.

This checks one hypothetical gripper sweep and synchronized joint interpolation.
It is not a motion executor, a live target check, or continuous collision proof.
"""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np

import so101_plan_grasp as planning


ROOT = Path('/home/yu/so101_ws')
STEP_RAD = math.radians(0.5)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(state_path, plan_path, report):
    state, state_hash = read_json(state_path)
    plan, plan_hash = read_json(plan_path)
    report['input_files'] = {
        str(state_path): state_hash, str(plan_path): plan_hash,
    }
    require(state.get('success') is True and state.get('backend') == 'serial',
            '起点不是成功的串口读取报告')
    require(type(state.get('register_writes')) is int
            and state['register_writes'] == 0, '起点报告不是零寄存器写入')
    require(len(state.get('events', [])) == 6
            and all(e.get('instruction') == 2 and e.get('acknowledged') is True
                    for e in state['events']), '起点报告缺少六次成功读取')
    states = state['states']
    require(len(states) == 6 and sorted(s['servo_id'] for s in states) == list(range(1, 7)),
            '起点须包含唯一的 ID1–6')
    by_id = {s['servo_id']: s for s in states}
    for s in states:
        require(type(s['position_counts']) is int and 0 <= s['position_counts'] <= 4095,
                f"ID{s['servo_id']} 起点计数无效")
        require(s['torque_enabled'] is False and s['speed_raw'] == 0
                and s['output_raw'] == 0 and s['status'] == 0,
                f"ID{s['servo_id']} 快照不是扭矩关闭、静止且无报警状态")
    report['state_snapshot'] = states

    require(plan.get('status') == 'not_for_motion'
            and plan.get('execution_available') is False
            and type(plan.get('writes_sent')) is int and plan['writes_sent'] == 0,
            '候选报告类型不符合离线检查约定')
    samples = plan['samples']
    expected = [('pregrasp', 0)] + [('descend', i) for i in range(20)] + [
        ('lift_geometry_only', i) for i in range(20)]
    require(len(samples) == 41 and all(
        (s['phase'], s['index']) == phase and s['ik_ok'] is True
        and 0 <= s['position_error_mm'] <= 1 and 0 <= s['orientation_error_deg'] <= 2
        and s['penetrating_contacts'] == []
        for s, phase in zip(samples, expected)), '输入候选未通过原有 41 点检查')

    reference_path = ROOT / 'data/base_reference_initial.json'
    mapping_path = ROOT / 'data/so101_joint_mapping.json'
    calibration_path = ROOT / 'data/calibration/table_plane_pose.npz'
    for path in (reference_path, mapping_path):
        require(plan['files'].get(str(path)) == digest(path), f'候选生成后文件已改变：{path}')
    target = plan['input_target']
    require(target['valid'] is True and target['coordinate_system'] == 'table', '目标无效')
    require(target['calibration']['sha256'] == digest(calibration_path), '候选标定摘要不匹配')
    report['saved_target'] = {k: target[k] for k in (
        'session_id', 'frame_id', 'captured_at_unix_s', 'center_m', 'square_yaw_deg', 'calibration')}
    mapping, _ = read_json(mapping_path)
    reference, _ = read_json(reference_path)
    center, rotation = planning.candidate_transform(
        np.asarray(target['center_m'], dtype=float), target['square_yaw_deg'], reference)
    require(np.allclose(center, plan['coordinate_conversion']['assumed_model_top_center_m'],
                        atol=1e-9, rtol=0), '重新计算的坐标转换与候选不一致')
    report['coordinate_conversion'] = plan['coordinate_conversion']
    scene = ROOT / 'simulation/so101/scene_grasp.xml'
    planner = planning.CandidatePlanner(scene, mapping)
    model, data = planner.model, planner.data
    report['current_model_files'] = {str(path): digest(path) for path in (
        scene, scene.parent / 'so101_new_calib.xml',
        scene.parent / 'moving_jaw_collision_assets.xml', Path(planning.__file__),
        mapping_path, reference_path, calibration_path)}
    report['model_mapping_verified'] = mapping.get('model_mapping_verified') is True

    counts = np.array([by_id[i]['position_counts'] for i in range(1, 6)], dtype=float)
    q0, signs = [], []
    for i, name in enumerate(planning.ARM_NAMES, 1):
        ref = mapping['joint_references'][str(i)]
        direction = mapping['joint_directions'][str(i)]
        require(direction['joint_name'] == name and direction['direction'] in (-1, 1),
                f'ID{i} 映射名称或方向无效')
        sign = direction['direction']
        # Only choose the diagnostic angle representation; never wrap motion commands.
        delta = (counts[i-1] - ref['position_count'] + 2048) % 4096 - 2048
        q0.append(math.radians(ref['model_angle_deg'] + sign * delta * 360 / 4096))
        signs.append(sign)
    q0, signs = np.array(q0), np.array(signs)
    q1 = planning.finite_vector(samples[0]['model_q_rad'], 5, '预抓取关节角')
    report['start_model_angles_deg'] = np.rad2deg(q0).tolist()
    report['start_angle_scope'] = 'nearest branch diagnostic using unverified joint mapping'

    def continuous_counts(q):
        return counts + (q-q0) * 4096 / (2*math.pi) / signs

    all_q = np.vstack([q0] + [planning.finite_vector(s['model_q_rad'], 5, '候选关节角')
                               for s in samples])
    require(np.isfinite(all_q).all() and np.all(all_q >= planner.limits[:, 0]-1e-9)
            and np.all(all_q <= planner.limits[:, 1]+1e-9), '起点或候选关节角超出模型限位')
    all_counts = np.array([continuous_counts(q) for q in all_q])
    report['continuous_count_ranges_diagnostic_only'] = [
        {'servo_id': i+1, 'start': int(counts[i]),
         'pregrasp': float(continuous_counts(q1)[i]),
         'minimum': float(all_counts[:, i].min()), 'maximum': float(all_counts[:, i].max())}
        for i in range(5)]
    report['count_boundary_ids'] = [i+1 for i in range(5)
        if all_counts[:, i].min() < 0 or all_counts[:, i].max() > 4095]

    size = planning.finite_vector(target['size_m'], 3, '纸盒尺寸')
    box_geom = planner.lookup(mujoco.mjtObj.mjOBJ_GEOM, 'a_box_geom')
    require(np.allclose(model.geom_size[box_geom], size/2, atol=1e-9, rtol=0), '模型盒子尺寸不匹配')
    data.qpos[planner.box_qadr:planner.box_qadr+3] = center - [0, 0, size[2]/2]
    yaw = math.atan2(rotation[1, 2], rotation[0, 2])
    data.qpos[planner.box_qadr+3:planner.box_qadr+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    g0, g1 = 0.0, planning.finite_scalar(plan['model_gripper_angle_rad'], '模型夹爪角')
    gj = planner.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    ga = planner.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper')
    glow = max(model.jnt_range[gj, 0], model.actuator_ctrlrange[ga, 0])
    ghigh = min(model.jnt_range[gj, 1], model.actuator_ctrlrange[ga, 1])
    require(glow <= min(g0, g1) <= max(g0, g1) <= ghigh, '假设夹爪角超出模型限位')
    report['gripper_assumption'] = {
        'start_rad': g0, 'candidate_rad': g1, 'measured_count': by_id[6]['position_counts'],
        'count_to_angle_calibrated': False,
        'note': 'Model angles are assumptions; no ID6 target counts are generated.'}

    # Replay endpoint FK first so a changed model cannot silently reuse old IK results.
    data.qpos[planner.qadr], data.qpos[planner.gripper_qadr] = q1, g1
    mujoco.mj_forward(model, data)
    pos_error = float(np.linalg.norm(data.site_xpos[planner.tcp]-samples[0]['target_tcp_m'])*1000)
    rot_error = float(np.rad2deg(np.linalg.norm(planning.rotation_error(
        data.site_xmat[planner.tcp].reshape(3, 3), rotation))))
    report['pregrasp_replay_errors'] = {'position_mm': pos_error, 'orientation_deg': rot_error}
    require(pos_error <= 1 and rot_error <= 2, '当前模型重放预抓取误差超过原阈值')

    report['checked_samples'] = []

    def check(phase, index, fraction, q, gripper):
        data.qpos[planner.qadr], data.qpos[planner.gripper_qadr] = q, gripper
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact[:data.ncon]:
            if contact.dist >= -1e-5:
                continue
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                     or f'geom{int(g)}' for g in (contact.geom1, contact.geom2)]
            # Exactly preserve the candidate planner's standing box/floor allowance.
            if set(names) == {'floor', 'a_box_geom'} and contact.dist >= -.0005:
                continue
            contacts.append({'geoms': names, 'penetration_mm': float(-contact.dist*1000)})
        sample = {'phase': phase, 'index': index, 'fraction': float(fraction),
                  'model_angles_deg': np.rad2deg(q).tolist(), 'gripper_angle_rad': float(gripper),
                  'penetrating_contacts': contacts}
        report['checked_samples'].append(sample)
        if contacts:
            report['first_failure'] = sample
            return False
        return True

    report['sampled_geometry_complete'] = False
    if not check('start', 0, 0, q0, g0):
        return
    ng = max(1, math.ceil(abs(g1-g0)/STEP_RAD))
    for i in range(1, ng+1):
        if not check('assumed_gripper_sweep', i, i/ng, q0, g0+(g1-g0)*i/ng):
            return
    n = max(1, math.ceil(float(np.max(np.abs(q1-q0)))/STEP_RAD))
    for i in range(1, n+1):
        if not check('synchronized_approach', i, i/n, q0+(q1-q0)*i/n, g1):
            return
    report['sampled_geometry_complete'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema_version': 1, 'kind': 'so101_saved_approach_geometry_check',
              'status': 'not_for_motion', 'execution_available': False, 'writes_sent': 0,
              'scope': 'saved snapshots; current model; synchronized joint interpolation',
              'max_sample_step_deg': 0.5,
              'limitations': ['No live target or live joint freshness claim.',
                              'Base transform and joint/gripper mapping remain unverified.',
                              'No continuous collision or dynamics verification.',
                              'Cables, external supports and other real obstacles are not modeled.',
                              'No carried-box simulation or hardware commands.']}
    code = 0
    try:
        analyze(args.state.resolve(), args.plan.resolve(), report)
        print('起点模型角度(度)：', np.round(report['start_model_angles_deg'], 3))
        for item in report['continuous_count_ranges_diagnostic_only']:
            print(f"ID{item['servo_id']} 连续分支诊断：起点={item['start']}，"
                  f"预抓取={item['pregrasp']:.2f}，范围={item['minimum']:.2f}–{item['maximum']:.2f}")
        print('超出0–4095的关节：', report['count_boundary_ids'])
        print('已检查几何样本数：', len(report['checked_samples']))
        print('假设夹爪扫掠及同步接近采样完成：', report['sampled_geometry_complete'])
        if 'first_failure' in report:
            print('首个穿透样本：', json.dumps(report['first_failure'], ensure_ascii=False))
    except Exception as error:
        report.update(status='invalid', error=f'{type(error).__name__}: {error}')
        print('检查未完成：', report['error'])
        code = 1
    with tempfile.NamedTemporaryFile('w', prefix='so101_approach_check_', suffix='.json',
                                     encoding='utf-8', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        path = stream.name
    print('状态：', report['status'], '；只检查离线模型，不代表可执行。')
    print('完整记录：', path)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
