"""Inspect existing model jaw geometry offline. No servo commands or project edits."""
from pathlib import Path
import hashlib
import json
import math
import os

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/yu/so101_ws')
MODEL = ROOT / 'simulation/so101/so101_new_calib.xml'
TEST_RESULT = ROOT / 'data/so101_gripper_test_result.json'


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    grip_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    grip_adr = model.jnt_qposadr[grip_joint]
    grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_center_40mm')
    fixed_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'fixed_jaw_tip_collision')
    moving_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'moving_jaw_collision_0')
    moving_visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'moving_jaw_visual')
    fixed_visual = None
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == grip_body and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom]))
            if mesh_name == 'wrist_roll_follower_so101_v1':
                fixed_visual = geom
    if fixed_visual is None:
        raise RuntimeError('Fixed jaw visual mesh not found')
    cases = []
    meshes = []
    for label, q in [('previous_open_display', .6), ('zero_angle_still_open', 0.), ('joint_lower_bound_nearly_closed', float(model.jnt_range[grip_joint, 0]))]:
        data.qpos[grip_adr] = q
        mujoco.mj_forward(model, data)
        fromto = np.zeros(6)
        distance = mujoco.mj_geomDistance(model, data, fixed_tip, moving_tip, 1., fromto)
        cases.append({'label': label, 'model_q_rad': q, 'model_q_deg': math.degrees(q),
                      'tip_collision_geoms_nearest_distance_mm': distance * 1000,
                      'distance_kind': 'minimum separation of existing fixed-tip box and moving-tip convex collision geom; not a uniform real jaw width',
                      'tcp_world_m': data.site_xpos[tcp].tolist(),
                      'tcp_rotation_world_from_tcp': data.site_xmat[tcp].reshape(3, 3).tolist(),
                      'does_not_assign_a_servo_count': True})
        triangles = []
        for geom in (fixed_visual, moving_visual):
            mesh = model.geom_dataid[geom]
            verts = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
            world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            local = (world - data.xpos[grip_body]) @ data.xmat[grip_body].reshape(3, 3)
            planar = np.column_stack([local[:, 0], -local[:, 2]]) * 1000
            triangles.append(planar[faces])
        meshes.append(triangles)
    tcp_positions = np.array([case['tcp_world_m'] for case in cases])
    tcp_rotations = np.array([case['tcp_rotation_world_from_tcp'] for case in cases])
    report = {
        'kind': 'offline_gripper_display_geometry_review', 'not_for_motion': True,
        'hardware_writes_sent': 0, 'active_mapping_modified': False,
        'source_model': str(MODEL), 'source_model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        'reported_ID6_count_3951_is_not_converted_to_angle': True,
        'old_small_range_test': json.loads(TEST_RESULT.read_text()),
        'gripper_joint_limits_rad': model.jnt_range[grip_joint].tolist(),
        'tcp_body': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[tcp])),
        'tcp_local_position_m': model.site_pos[tcp].tolist(),
        'max_tcp_position_change_across_cases_m': float(np.max(np.linalg.norm(tcp_positions - tcp_positions[0], axis=1))),
        'max_tcp_rotation_matrix_change_across_cases': float(np.max(np.abs(tcp_rotations - tcp_rotations[0]))),
        'cases': cases,
        'display_recommendation': 'Use existing model lower joint bound for nearly-closed display only; do not identify 3951 with this angle or change the model limits.',
    }
    output = HERE / 'gripper_display_review.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False, 'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.8), sharex=True, sharey=True, layout='constrained')
    titles = ['旧显示：q = 0.6 rad', '模型零角：q = 0 rad', '模型最小角：q ≈ −0.17453 rad']
    for i, (axis, triangles, case) in enumerate(zip(axes, meshes, cases)):
        axis.add_collection(PolyCollection(triangles[0], facecolors='#4584b0', edgecolors='none', rasterized=True))
        axis.add_collection(PolyCollection(triangles[1], facecolors='#dc8536', edgecolors='none', rasterized=True))
        axis.scatter(model.site_pos[tcp, 0] * 1000, -model.site_pos[tcp, 2] * 1000,
                     marker='*', s=100, color='#167543', edgecolors='white', linewidths=.6, zorder=5)
        axis.set_title(titles[i] + '\n' + f'模型爪尖最短间隙 ≈ {case["tip_collision_geoms_nearest_distance_mm"]:.2f} mm', fontsize=12)
        axis.set_aspect('equal')
        axis.set_xlim(-32, 85)
        axis.set_ylim(-17, 116)
        axis.set_xlabel('夹爪局部 X / mm')
        axis.grid(alpha=.15)
    axes[0].set_ylabel('沿夹爪向前 −Z / mm')
    fig.suptitle('现有模型的夹爪几何：零角不等于完全闭合', fontsize=18)
    fig.supxlabel('蓝：固定爪网格；橙：活动爪网格；绿星：固定的 40 mm 盒抓取参照点\n仅离线展示；3951 未被标定为上述任何模型角，间隙来自模型碰撞几何。', fontsize=12)
    image_path = HERE / 'gripper_open_zero_nearly_closed.png'
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print('Report:', output)
    print('Figure:', image_path)
    print(json.dumps({k: report[k] for k in ('gripper_joint_limits_rad','tcp_body','tcp_local_position_m','max_tcp_position_change_across_cases_m','max_tcp_rotation_matrix_change_across_cases','cases')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
