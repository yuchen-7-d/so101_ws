"""纸盒目标 -> MuJoCo 候选路径；只计算，不连接串口，不发送运动。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import mujoco
import numpy as np

ARM_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
# 与 read_box_measurement.py 的旧假设一致；尚未实测，绝不能成为实机标定。
ASSUMED_MODEL_OFFSET_M = np.array([0.064635292, 0.0, 0.0])


def atomic_write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            name = stream.name
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def finite_vector(value, length, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f'{label} 必须是 {length} 个有限数值')
    return array


def finite_scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} 必须是有限数值')
    return float(value)


def validate_target(target, now, max_age, allow_offline, calibration_path):
    if target.get('schema_version') != 1 or target.get('kind') != 'so101_box_grasp_target':
        raise ValueError('目标格式不匹配；请使用 so101_grasp_target.py 生成的新目标')
    if target.get('valid') is not True:
        raise ValueError(f"当前没有有效目标：{target.get('reason', 'unknown')}")
    if target.get('coordinate_system') != 'table':
        raise ValueError('目标坐标系不是 table')
    mode = target.get('source_mode')
    if mode not in ('live', 'offline'):
        raise ValueError('未知目标来源')
    if (not isinstance(target.get('session_id'), str) or not target['session_id']
            or type(target.get('frame_id')) is not int or target['frame_id'] < 1):
        raise ValueError('目标缺少有效会话编号或帧编号')
    captured = finite_scalar(target.get('captured_at_unix_s'), '采集时间')
    published = finite_scalar(target.get('published_at_unix_s'), '发布时间')
    if captured > now + 0.1 or published > now + 0.1 or published < captured:
        raise ValueError('目标时间戳不一致或来自未来')
    if mode == 'live' and now - captured > max_age:
        raise ValueError(f'目标已过期：{now - captured:.2f} 秒 > {max_age:.2f} 秒')
    if mode == 'offline' and not allow_offline:
        raise ValueError('静态图片仅用于离线预览；需明确指定 --allow-offline')
    center = finite_vector(target.get('center_m'), 3, '顶面中心')
    size = finite_vector(target.get('size_m'), 3, '盒尺寸')
    if not np.allclose(size, [0.04, 0.04, 0.074], atol=1e-6, rtol=0):
        raise ValueError('当前模型只适配竖放的 40×40×74 mm 纸盒')
    if abs(center[2] + size[2]) > 1e-5:
        raise ValueError('顶面高度与已知盒高不一致')
    score = finite_scalar(target.get('max_edge_error_mm'), '尺寸误差')
    if not 0 <= score <= 3:
        raise ValueError('顶面尺寸检查未通过')
    yaw = finite_scalar(target.get('square_yaw_deg'), '顶面方向')
    if not -45 <= yaw < 45:
        raise ValueError('正方形方向应归一化到 [-45, 45) 度')
    expected_hash = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
    if target.get('calibration', {}).get('sha256') != expected_hash:
        raise ValueError('目标使用的标定文件与当前文件不一致')
    return center, size, yaw


def candidate_transform(center, yaw_deg, reference):
    axes = np.column_stack([finite_vector(reference[key], 3, key) for key in
                            ('x_axis_table', 'y_axis_table', 'z_axis_table')])
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5) or abs(np.linalg.det(axes) - 1) > 1e-5:
        raise ValueError('底座参考的三个轴不是右手正交坐标系')
    if not np.allclose(axes.T @ np.array([0., 0., 1.]), [0., 0., -1.], atol=1e-5):
        raise ValueError('桌面法向不匹配；当前场景仅支持水平桌面且 table 正Z朝下')
    origin = finite_vector(reference['origin_table_m'], 3, '底座参考原点')
    center_model = axes.T @ (center - origin) + ASSUMED_MODEL_OFFSET_M
    yaw = np.deg2rad(yaw_deg)
    closing = axes.T @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    if abs(closing @ down) > 1e-5:
        raise ValueError('当前顶抓求解只支持水平桌面')
    # 方形顶面的一条边没有箭头；与现有仿真一致，选[-90,90)度的夹持方向。
    closing_yaw = (math.atan2(closing[1], closing[0]) + math.pi / 2) % math.pi - math.pi / 2
    closing = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.0])
    rotation = np.column_stack([down, np.cross(closing, down), closing])
    return center_model, rotation


def load_historical_pose_mapping(workspace, pose_bytes):
    """原始姿态只用与它绑定的原映射解释；不得回退到当前映射。"""
    path = Path(workspace) / 'data/so101_initial_pose_mapping.json'
    source_bytes = path.read_bytes()
    record = json.loads(source_bytes)
    if (record.get('schema_version') != 1
            or record.get('kind') != 'so101_historical_pose_mapping'
            or record.get('scope') != 'historical_diagnostics_only'
            or record.get('execution_available') is not False):
        raise ValueError('历史姿态映射记录格式或用途不匹配')
    pose_hash = hashlib.sha256(pose_bytes).hexdigest()
    if record.get('pose_sha256') != pose_hash:
        raise ValueError('历史姿态文件已变化，不能沿用旧映射绑定')
    mapping = record['mapping']
    canonical = json.dumps(mapping, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    mapping_hash = hashlib.sha256(canonical).hexdigest()
    if record.get('mapping_canonical_sha256') != mapping_hash:
        raise ValueError('历史姿态映射内容与摘要不匹配')
    return mapping, {
        'source': str(path),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'pose_sha256': pose_hash,
        'mapping_canonical_sha256': mapping_hash,
        'scope': 'historical_diagnostics_only',
        'is_live_start': False,
    }


def mapping_diagnostics(mapping, pose, limits):
    diagnostics = []
    for i, name in enumerate(ARM_NAMES, 1):
        key = str(i)
        ref = mapping['joint_references'][key]
        config = mapping['joint_directions'][key]
        if config['joint_name'] != name or config['direction'] not in (-1, 1):
            raise ValueError(f'ID{i} 映射名称或方向错误')
        count = finite_scalar(pose['position_counts'][key], f'ID{i} 历史计数')
        anchor = finite_scalar(ref['position_count'], f'ID{i} 参考计数')
        if not 0 <= count <= 4095 or not 0 <= anchor <= 4095:
            raise ValueError(f'ID{i} 计数不在单圈范围')
        angle = finite_scalar(ref['model_angle_deg'], f'ID{i} 参考角度')
        delta = count - anchor
        principal = (delta + 2048) % 4096 - 2048
        model_deg = angle + config['direction'] * principal * 360 / 4096
        bound = np.rad2deg(limits[i - 1])
        diagnostics.append({'servo_id': i, 'joint_name': name, 'historical_count': count,
                            'model_angle_nearest_branch_deg': model_deg,
                            'model_limits_deg': bound.tolist(),
                            'nearest_branch_in_model_range': bool(bound[0] <= model_deg <= bound[1]),
                            'note': '最短环绕分支仅用于诊断，不用于选择真实运动路径'})
    return diagnostics


def check_live_snapshot(original, latest, now, max_age, calibration_path):
    """发布前检查旧快照仍未超时、生产者仍有效；不把新帧冒充同一快照。"""
    validate_target(original, now, max_age, False, calibration_path)
    validate_target(latest, now, max_age, False, calibration_path)
    if (latest.get('source_mode') != 'live'
            or latest['session_id'] != original['session_id']
            or latest['frame_id'] < original['frame_id']):
        raise ValueError('规划期间视觉来源、会话或帧序列发生变化')
    if latest['frame_id'] == original['frame_id'] and latest != original:
        raise ValueError('同一目标帧的内容发生变化')
    return 'same_frame' if latest['frame_id'] == original['frame_id'] else 'snapshot_not_latest_frame'


def rotation_error(current, target):
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, (target @ current.T).ravel())
    error = np.zeros(3)
    mujoco.mju_quat2Vel(error, quaternion, 1.0)
    return error


class CandidatePlanner:
    def __init__(self, model_path, mapping):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.mapping = mapping
        joints = [self.lookup(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_NAMES]
        self.qadr = self.model.jnt_qposadr[joints]
        self.dadr = self.model.jnt_dofadr[joints]
        acts = [self.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_NAMES]
        self.limits = np.column_stack([
            np.maximum(self.model.jnt_range[joints, 0], self.model.actuator_ctrlrange[acts, 0]),
            np.minimum(self.model.jnt_range[joints, 1], self.model.actuator_ctrlrange[acts, 1]),
        ])
        self.tcp = self.lookup(mujoco.mjtObj.mjOBJ_SITE, 'grasp_center_40mm')
        self.gripper_qadr = self.model.jnt_qposadr[self.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')]
        self.box_qadr = self.model.jnt_qposadr[self.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'a_box_free')]
        self.jac_pos = np.zeros((3, self.model.nv))
        self.jac_rot = np.zeros((3, self.model.nv))

    def lookup(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result == -1:
            raise ValueError(f'模型缺少 {name}')
        return result

    def solve(self, target, rotation, position_only=False):
        for _ in range(300):
            mujoco.mj_forward(self.model, self.data)
            pos = target - self.data.site_xpos[self.tcp]
            rot = rotation_error(self.data.site_xmat[self.tcp].reshape(3, 3), rotation)
            if np.linalg.norm(pos) <= .001 and (position_only or np.linalg.norm(rot) <= np.deg2rad(2)):
                break
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.tcp)
            if position_only:
                jac, error = self.jac_pos[:, self.dadr], pos
            else:
                jac = np.vstack([self.jac_pos[:, self.dadr], .05 * self.jac_rot[:, self.dadr]])
                error = np.concatenate([pos, .05 * rot])
            error *= min(1, .005 / max(np.linalg.norm(error), 1e-12))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .01**2 * np.eye(len(error)), error)
            delta *= min(1, .05 / max(np.max(np.abs(delta)), 1e-12))
            self.data.qpos[self.qadr] = np.clip(self.data.qpos[self.qadr] + delta,
                                               self.limits[:, 0], self.limits[:, 1])
        mujoco.mj_forward(self.model, self.data)
        pos_mm = float(np.linalg.norm(target - self.data.site_xpos[self.tcp]) * 1000)
        rot_deg = float(np.rad2deg(np.linalg.norm(rotation_error(
            self.data.site_xmat[self.tcp].reshape(3, 3), rotation))))
        q = self.data.qpos[self.qadr].copy()
        counts = []
        for i, angle in enumerate(np.rad2deg(q), 1):
            ref = self.mapping['joint_references'][str(i)]
            direction = self.mapping['joint_directions'][str(i)]['direction']
            counts.append(float(ref['position_count'] + (angle - ref['model_angle_deg']) * 4096 / 360 / direction))
        contacts = []
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist >= -1e-5:
                continue
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f'geom{int(g)}'
                     for g in (contact.geom1, contact.geom2)]
            if set(names) == {'floor', 'a_box_geom'} and contact.dist >= -.0005:
                continue
            contacts.append({'geoms': names, 'penetration_mm': float(-contact.dist * 1000)})
        return {'ik_ok': pos_mm <= 1 and rot_deg <= 2, 'target_tcp_m': target.tolist(),
                'position_error_mm': pos_mm, 'orientation_error_deg': rot_deg,
                'model_q_rad': q.tolist(), 'candidate_counts_unwrapped': counts,
                'candidate_counts_in_single_turn': all(0 <= c <= 4095 for c in counts),
                'penetrating_contacts': contacts}

    def plan(self, center, rotation, size):
        # 数学种子，非实机起点；不生成从当前机械臂到此种子的运动。
        self.data.qpos[self.qadr] = 0
        self.data.qpos[self.gripper_qadr] = .6
        box_center = center - np.array([0, 0, size[2] / 2])
        self.data.qpos[self.box_qadr:self.box_qadr + 3] = box_center
        yaw = math.atan2(rotation[1, 2], rotation[0, 2])
        self.data.qpos[self.box_qadr + 3:self.box_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        # 沿用已有仿真的顶面上方10mm、向下进入10mm、沿夹持方向-1mm偏置。
        lateral = -.001 * rotation[:, 2]
        pre = center + np.array([0, 0, .01]) + lateral
        grasp = center + np.array([0, 0, -.01]) + lateral
        # 与已有仿真一致：先检查抬回预抓取高度（20mm），不自行增加行程。
        lift = pre.copy()
        self.solve(pre, rotation, position_only=True)
        samples = []
        for phase, points in [('pregrasp', [pre]),
                              ('descend', np.linspace(pre, grasp, 21)[1:]),
                              ('lift_geometry_only', np.linspace(grasp, lift, 21)[1:])]:
            for index, target in enumerate(points):
                result = self.solve(np.asarray(target), rotation)
                result.update(phase=phase, index=index)
                samples.append(result)
                if not result['ik_ok']:
                    return samples
        return samples


def build_report(workspace, target, allow_offline=False, max_age=2.0, now=None):
    workspace = Path(workspace)
    now = time.time() if now is None else now
    center, size, yaw = validate_target(target, now, max_age, allow_offline,
                                        workspace / 'data/calibration/table_plane_pose.npz')
    reference_path = workspace / 'data/base_reference_initial.json'
    mapping_path = workspace / 'data/so101_joint_mapping.json'
    pose_path = workspace / 'data/so101_initial_pose.json'
    reference = json.loads(reference_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    pose_bytes = pose_path.read_bytes()
    pose = json.loads(pose_bytes)
    historical_mapping, historical_reference = load_historical_pose_mapping(workspace, pose_bytes)
    center_model, rotation = candidate_transform(center, yaw, reference)
    planner = CandidatePlanner(workspace / 'simulation/so101/scene_grasp.xml', mapping)
    diagnostics = mapping_diagnostics(historical_mapping, pose, planner.limits)
    samples = planner.plan(center_model, rotation, size)
    return {'schema_version': 1, 'kind': 'so101_candidate_grasp_plan', 'status': 'not_for_motion',
            'writes_sent': 0, 'execution_available': False, 'generated_at_unix_s': time.time(),
            'input_target': target,
            'coordinate_conversion': {'table_top_center_m': center.tolist(),
                                      'assumed_model_top_center_m': center_model.tolist(),
                                      'assumed_model_offset_m': ASSUMED_MODEL_OFFSET_M.tolist(),
                                      'base_reference_status': reference.get('status'),
                                      'verified': False},
            'historical_pose_diagnostics': diagnostics,
            'historical_pose_mapping_reference': historical_reference,
            'historical_pose_source': str(pose_path), 'live_start_pose_available': False,
            'ik_seed': 'model_zero_not_live_pose', 'model_gripper_angle_rad': .6,
            'model_ik_complete': len(samples) == 41 and all(s['ik_ok'] for s in samples),
            'sampled_open_gripper_contact_count': sum(len(s['penetrating_contacts']) for s in samples),
            'path_validation_scope': 'IK samples only; no live-start approach, continuous collision check, closed-gripper or carried-box simulation',
            'samples': samples,
            'missing_for_execution': [
                '桌面到模型底座的实测变换；当前仍包含64.635292mm假设偏移',
                '一致的舵机计数与模型角映射；历史参考姿态换算存在模型越界',
                '当前反馈起点及本次任务的实机局部限位',
                '40mm纸盒对应夹爪开口、接触和抬升保夹条件',
                '从当前起点出发的路径和实际环境碰撞验证',
                '统一六轴的闭环实机执行器'],
            'files': {**{str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (reference_path, mapping_path, pose_path)},
                      historical_reference['source']: historical_reference['source_sha256']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--target', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-offline', action='store_true', help='仅允许旧图片参与候选模型计算')
    parser.add_argument('--max-age', type=float, default=2.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_age) or not 0 < args.max_age <= 5:
        parser.error('--max-age 必须大于0且不超过5秒')
    target_path = args.target or args.workspace / 'data/box_grasp_target.json'
    output_path = args.output or args.workspace / 'data/box_grasp_plan.json'
    protected_sources = [target_path] + [args.workspace / name for name in (
        'data/base_reference_initial.json', 'data/so101_joint_mapping.json',
        'data/so101_initial_pose.json', 'data/so101_initial_pose_mapping.json',
        'data/calibration/table_plane_pose.npz',
        'simulation/so101/scene_grasp.xml', 'simulation/so101/so101_new_calib.xml')]
    for source_path in protected_sources:
        if (output_path.resolve() == source_path.resolve()
                or (output_path.exists() and source_path.exists() and os.path.samefile(output_path, source_path))):
            parser.error('输出文件不能覆盖输入目标、标定、基准姿态或模型（包含文件别名）')
    record = {'schema_version': 1, 'kind': 'so101_candidate_grasp_plan', 'status': 'invalid',
              'writes_sent': 0, 'execution_available': False, 'generated_at_unix_s': time.time()}
    # 模型加载失败或计算被中断，也不能让调用者继续读上一次的路径。
    atomic_write_json(output_path, dict(record, reason='planning_started'))
    try:
        target = json.loads(target_path.read_text())
        record = build_report(args.workspace, target, args.allow_offline, args.max_age)
        if target['source_mode'] == 'live':
            latest = json.loads(target_path.read_text())
            record['target_state_at_publish'] = check_live_snapshot(
                target, latest, time.time(), args.max_age,
                args.workspace / 'data/calibration/table_plane_pose.npz')
        else:
            record['target_state_at_publish'] = 'offline_snapshot'
    except Exception as exc:
        # 即使IK已算完，发布前失检也丢弃该路径。
        record = {'schema_version': 1, 'kind': 'so101_candidate_grasp_plan', 'status': 'invalid',
                  'writes_sent': 0, 'execution_available': False,
                  'generated_at_unix_s': time.time(), 'reason': str(exc)}
        atomic_write_json(output_path, record)
        print(f'无法规划：{exc}\n记录：{output_path}')
        return 2
    atomic_write_json(output_path, record)
    conversion = record['coordinate_conversion']
    print('候选抓取计算；未连接串口，不能作为实机运动命令。')
    print('桌面顶面中心(m)：', np.round(conversion['table_top_center_m'], 5))
    print('假设模型顶面中心(m)：', np.round(conversion['assumed_model_top_center_m'], 5))
    for phase in ('pregrasp', 'descend', 'lift_geometry_only'):
        points = [s for s in record['samples'] if s['phase'] == phase]
        if points:
            last = points[-1]
            print(f"{phase}: TCP={np.round(last['target_tcp_m'], 5)}, "
                  f"误差={last['position_error_mm']:.3f}mm/{last['orientation_error_deg']:.2f}度, "
                  f"IK={last['ik_ok']}")
    print(f"全部模型IK完成：{record['model_ik_complete']}；采样穿透接触：{record['sampled_open_gripper_contact_count']}")
    print('历史姿态诊断采用与旧姿态绑定的原映射；它不是当前起点。')
    for joint in record['historical_pose_diagnostics']:
        if not joint['nearest_branch_in_model_range']:
            print(f"历史姿态映射冲突 ID{joint['servo_id']}: "
                  f"{joint['model_angle_nearest_branch_deg']:.2f}度，模型范围{np.round(joint['model_limits_deg'], 2)}")
    print(f'状态：not_for_motion（仅候选规划，实机抓取尚未执行）\n记录：{output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
