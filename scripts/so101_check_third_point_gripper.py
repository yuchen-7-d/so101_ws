"""Replay saved SO101 snapshots in MuJoCo; no camera, ROS or serial access.

This checks a provisional measured opening, synchronized approach and saved path.
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

import sys
sys.path.insert(0, '/home/yu/so101_ws/scripts')
import so101_plan_grasp as planning


ROOT = Path('/home/yu/so101_ws')
STEP_RAD = math.radians(0.5)
EXPECTED_GRIPPER_CANDIDATE_SHA = 'd4412593585c5e288d951ae758b1e09d39c302fe6023a2e4c7fdd382550233d0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def load_gripper_candidate(path, state, state_hash, mapping_hash, report):
    candidate, candidate_hash = read_json(path)
    require(candidate_hash == EXPECTED_GRIPPER_CANDIDATE_SHA,
            '夹爪候选文件已改变；需要重新核对测量记录')
    require(candidate['kind'] == 'so101_gripper_two_point_fit_with_third_observation'
            and candidate['status'] == 'offline_candidate_not_verified'
            and candidate['execution_available'] is False
            and type(candidate['register_writes']) is int
            and candidate['register_writes'] == 0, '夹爪候选类型不符合约定')
    require(state.get('status') == 'read_only_start_snapshot'
            and state.get('configuration_matches_mapping') is True
            and state.get('mapping_sha256') == mapping_hash,
            '起点不是与当前映射配套的配置快照')
    for source, sha in candidate['input_files'].items():
        require(digest(Path(source)) == sha, f'夹爪拟合后模型或网格已改变：{source}')
    observations = candidate['observations']
    require(len(observations) == 3
            and [o['position_counts'] for o in observations] == [1913, 2047, 2144]
            and [o['role'] for o in observations] == ['fit', 'fit', 'holdout'], '需要两份拟合记录及第三份独立观测')
    parent, parent_hash = read_json(Path(candidate['parent_candidate']['path']))
    require(parent_hash == candidate['parent_candidate']['sha256'], '原拟合候选已改变')
    for key in ('reference_counts', 'reference_model_rad', 'direction', 'radians_per_count', 'offset_raw'):
        require(candidate['mapping_candidate'][key] == parent['mapping_candidate'][key], '第三点不能反向改写原拟合参数')
    require(candidate['third_point_check']['not_used_for_fitting'] is True
            and candidate['third_point_check']['validation_passed'] is None, '第三点尚未作精度验收')
    bound_to_observation = False
    for o in observations:
        measured, measured_hash = read_json(Path(o['snapshot_path']))
        require(measured_hash == o['snapshot_sha256'], '测量快照已改变')
        require(measured['success'] is True and type(measured['register_writes']) is int
                and measured['register_writes'] == 0, '测量快照无效')
        require(measured['config_before']['6']['offset_raw'] == 3901
                and measured['config_after']['6']['offset_raw'] == 3901,
                '测量快照的ID6计数编号已改变')
        for r in measured['feedback_rounds']:
            s = next(s for s in r['states'] if s['servo_id'] == 6)
            require(s['position_counts'] == o['position_counts']
                    and s['torque_enabled'] is False and s['speed_raw'] == 0
                    and s['output_raw'] == 0 and s['output_limit'] == 0,
                    '测量快照夹爪未稳定释放')
        bound_to_observation |= measured_hash == state_hash
    require(bound_to_observation, '本次只接受与三次间距测量配套的已保存起点')
    offsets = {1: 0, 2: 0x0C9D, 3: 0, 4: 0x071E, 5: 0, 6: 0x0F3D}
    for sid, offset in offsets.items():
        before, after = state['config_before'][str(sid)], state['config_after'][str(sid)]
        require(before['offset_raw'] == after['offset_raw'] == offset,
                f'ID{sid} 起点计数编号不匹配')
        for config in (before, after):
            require(config['mode'] == 0 and config['phase'] == 12
                    and config['resolution'] == 1
                    and config['minimum'] == 0 and config['maximum'] == 4095,
                    f'ID{sid} 配置与测量条件不一致')
    mapping = candidate['mapping_candidate']
    require(mapping['offset_raw'] == 3901 and mapping['direction'] == 1
            and mapping['reference_counts'] == 1913
            and mapping['measured_count_interval'] == [1913, 2144], '夹爪映射条件不匹配')
    scale = planning.finite_scalar(mapping['radians_per_count'], '夹爪计数比例')
    require(abs(scale - 2*math.pi/4096) < 1e-15, '不允许改变编码器比例')
    qref = planning.finite_scalar(mapping['reference_model_rad'], '夹爪候选参考角')
    current_count = next(s['position_counts'] for s in state['states'] if s['servo_id'] == 6)
    open_count = candidate['model_preview_open_count']
    require(type(open_count) is int and open_count == 2144, '只复现已实测张开点')
    require(1913 <= current_count <= 2144, '不能外推到测量区间之外')
    g0, g1 = (qref+(c-1913)*scale for c in (current_count, open_count))
    report['gripper_candidate'] = {
        'file': str(path), 'sha256': candidate_hash, 'mapping': mapping,
        'observations': observations, 'fit': candidate['fit'],
        'third_point_check': candidate['third_point_check'],
        'motor_validation': candidate['motor_validation'],
        'start_rad': g0, 'preview_open_rad': g1,
        'preview_open_observed_count': open_count,
        'model_mapping_verified': False, 'hardware_commands_generated': False}
    return g0, g1


def analyze(state_path, plan_path, gripper_path, report):
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
    mapping, mapping_hash = read_json(mapping_path)
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
    g0, g1 = load_gripper_candidate(gripper_path, state, state_hash, mapping_hash, report)
    report['original_plan_gripper_angle_rad'] = plan['model_gripper_angle_rad']
    gj = planner.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    ga = planner.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper')
    glow = max(model.jnt_range[gj, 0], model.actuator_ctrlrange[ga, 0])
    ghigh = min(model.jnt_range[gj, 1], model.actuator_ctrlrange[ga, 1])
    require(glow <= min(g0, g1) <= max(g0, g1) <= ghigh, '假设夹爪角超出模型限位')
    report['gripper_assumption'] = {
        'start_rad': g0, 'candidate_rad': g1, 'measured_count': by_id[6]['position_counts'],
        'count_to_angle_calibrated': False,
        'note': 'Two-point reference fit; provisional, not independently validated; no motion commands.'}

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

    report['full_saved_path_sampled_geometry_complete'] = False
    report['sampled_geometry_complete'] = False
    if not check('start', 0, 0, q0, g0):
        return
    ng = max(1, math.ceil(abs(g1-g0)/STEP_RAD))
    for i in range(1, ng+1):
        if not check('candidate_measured_gripper_sweep', i, i/ng, q0, g0+(g1-g0)*i/ng):
            return
    n = max(1, math.ceil(float(np.max(np.abs(q1-q0)))/STEP_RAD))
    for i in range(1, n+1):
        if not check('synchronized_approach', i, i/n, q0+(q1-q0)*i/n, g1):
            return
    report['sampled_geometry_complete'] = True
    for sample in samples:
        q = planning.finite_vector(sample['model_q_rad'], 5, '保存路径关节角')
        if not check('saved_path_' + sample['phase'], sample['index'], 0.0, q, g1):
            return
        pos_error = float(np.linalg.norm(data.site_xpos[planner.tcp]-sample['target_tcp_m'])*1000)
        rot_error = float(np.rad2deg(np.linalg.norm(planning.rotation_error(
            data.site_xmat[planner.tcp].reshape(3, 3), rotation))))
        report['checked_samples'][-1].update(position_error_mm=pos_error, orientation_error_deg=rot_error)
        require(pos_error <= 1 and rot_error <= 2, '保存路径重放误差超过原阈值')
    report['full_saved_path_sampled_geometry_complete'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gripper-map', type=Path, default=ROOT/'data/gripper_gap_third_point_20261023.json')
    args = parser.parse_args()
    report = {'schema_version': 1, 'kind': 'so101_third_point_gripper_path_geometry_check',
              'status': 'not_for_motion', 'execution_available': False, 'writes_sent': 0,
              'scope': 'saved snapshots; candidate measured opening; synchronized approach and all 41 saved path samples',
              'max_sample_step_deg': 0.5,
              'limitations': ['No live target or live joint freshness claim.',
                              'Base transform and joint/gripper mapping remain unverified.',
                              'No continuous collision or dynamics verification.',
                              'Cables, external supports and other real obstacles are not modeled.',
                              'No carried-box simulation or hardware commands.',
                              'Two measurement points used for fitting, not independent validation.',
                              'Physical box clearance and TCP contact geometry remain unverified.']}
    code = 0
    try:
        analyze(args.state.resolve(), args.plan.resolve(), args.gripper_map.resolve(), report)
        gap_check = report['gripper_candidate']['third_point_check']
        print(f"第三点：模型预测={gap_check['predicted_gap_mm']:.2f}mm，实测={gap_check['measured_gap_mm']:.2f}mm，"
              f"差值={gap_check['measured_minus_predicted_mm']:+.2f}mm；未作标定精度验收")
        print('本次仅重放手动张开到2144的快照；未生成实机运动目标。')
        print('起点模型角度(度)：', np.round(report['start_model_angles_deg'], 3))
        for item in report['continuous_count_ranges_diagnostic_only']:
            print(f"ID{item['servo_id']} 连续分支诊断：起点={item['start']}，"
                  f"预抓取={item['pregrasp']:.2f}，范围={item['minimum']:.2f}–{item['maximum']:.2f}")
        print('超出0–4095的关节：', report['count_boundary_ids'])
        print('已检查几何样本数：', len(report['checked_samples']))
        print('夹爪候选角度(度)：', np.round(np.rad2deg([report['gripper_candidate']['start_rad'],
            report['gripper_candidate']['preview_open_rad']]), 3))
        print('测量候选开口下的接近采样完成：', report['sampled_geometry_complete'])
        print('接近及保存的41点路径全部采样完成：', report['full_saved_path_sampled_geometry_complete'])
        if 'first_failure' in report:
            print('首个穿透样本：', json.dumps(report['first_failure'], ensure_ascii=False))
    except Exception as error:
        report.update(status='invalid', error=f'{type(error).__name__}: {error}')
        print('检查未完成：', report['error'])
        code = 1
    with tempfile.NamedTemporaryFile('w', prefix='so101_third_point_gripper_check_', suffix='.json',
                                     encoding='utf-8', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        path = stream.name
    print('状态：', report['status'], '；只检查离线模型，不代表可执行。')
    print('完整记录：', path)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
