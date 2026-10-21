"""Historical log -> conditional model FK and a side-view figure; no device I/O."""
from pathlib import Path
import hashlib
import json
import math
import os

import mujoco
import numpy as np

ROOT = Path('/home/yu/so101_ws')
HERE = Path(__file__).resolve().parent
LOG = ROOT / 'data/real_control_runs/shoulder_relative_probe_20261020T233307_ijyxou1a.json'
MAPPING = ROOT / 'data/pose_mapping_checks/wrist_model_e_ebbd4505/wrist_candidate_mapping.json'
MODEL = ROOT / 'simulation/so101/so101_new_calib.xml'
ARM_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
GRIPPER_Q = .6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_feedback(log):
    cycles, current = [], []
    for row in log['reads']:
        if row.get('phase') != 'step_+8' or row.get('address') != 56 or 'position' not in row:
            continue
        if row['servo_id'] == 1:
            current = []
        current.append(row)
        if [r['servo_id'] for r in current] == [1, 2, 3, 4, 5, 6]:
            cycles.append(current)
            current = []
    if not cycles:
        raise ValueError('No complete six-servo feedback sweep in step_+8')
    last = cycles[-1]
    positions = [r['position'] for r in last]
    stable = []
    for cycle in reversed(cycles):
        if [r['position'] for r in cycle] != positions:
            break
        stable.append(cycle)
    stable.reverse()
    return last, stable


def counts_to_q(counts, mapping):
    result = []
    for servo_id, count in enumerate(counts, 1):
        key = str(servo_id)
        reference = mapping['joint_references'][key]
        direction = mapping['joint_directions'][key]['direction']
        difference = (count - reference['position_count'] + 2048) % 4096 - 2048
        result.append(math.radians(reference['model_angle_deg'] + direction * difference * 360 / 4096))
    return np.array(result)


def compute():
    log = json.loads(LOG.read_text())
    mapping = json.loads(MAPPING.read_text())
    last, stable = select_feedback(log)
    feedback_counts = [row['position'] for row in last[:5]]
    goal_counts = [log['last_command_targets'][str(i)] for i in range(1, 6)]
    selected_goals = [row['command_target'] for row in last[:5]]
    if selected_goals != goal_counts:
        raise ValueError('Selected sweep target metadata differs from final command vector')
    shoulder_only = feedback_counts.copy()
    shoulder_only[1] = goal_counts[1]

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_NAMES]
    qadr = model.jnt_qposadr[joint_ids]
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_center_40mm')
    poses = {}
    for key, counts in [('feedback', feedback_counts), ('shoulder_only_target', shoulder_only), ('all_command_targets', goal_counts)]:
        data = mujoco.MjData(model)
        q = counts_to_q(counts, mapping)
        data.qpos[qadr] = q
        data.qpos[model.jnt_qposadr[grip]] = GRIPPER_Q
        mujoco.mj_forward(model, data)
        poses[key] = {
            'counts_1_to_5': counts,
            'model_q_rad': q.tolist(),
            'model_q_deg': np.rad2deg(q).tolist(),
            'tcp_position_model_m': data.site_xpos[tcp_id].tolist(),
            'tcp_rotation_model_from_tcp': data.site_xmat[tcp_id].reshape(3, 3).tolist(),
            'joint_anchors_model_m': {name: data.xanchor[jid].tolist() for name, jid in zip(ARM_NAMES, joint_ids)},
        }
    actual = poses['feedback']
    for key in ('shoulder_only_target', 'all_command_targets'):
        p = poses[key]
        delta = np.array(p['tcp_position_model_m']) - actual['tcp_position_model_m']
        rotation = np.array(p['tcp_rotation_model_from_tcp']) @ np.array(actual['tcp_rotation_model_from_tcp']).T
        quaternion, rotation_vector = np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        mujoco.mju_quat2Vel(rotation_vector, quaternion, 1.)
        p['difference_from_feedback'] = {
            'joint_counts': (np.array(p['counts_1_to_5']) - actual['counts_1_to_5']).tolist(),
            'tcp_delta_model_mm': (delta * 1000).tolist(),
            'tcp_distance_mm': float(np.linalg.norm(delta) * 1000),
            'tcp_orientation_geodesic_deg': float(np.rad2deg(np.linalg.norm(rotation_vector))),
            'tcp_rotation_vector_model_deg': np.rad2deg(rotation_vector).tolist(),
        }
    return {
        'schema_version': 1,
        'kind': 'historical_shoulder_probe_conditional_model_fk_comparison',
        'status': 'offline_not_for_motion',
        'hardware_writes_sent': 0,
        'execution_available': False,
        'source_files': {'log': str(LOG), 'candidate_mapping': str(MAPPING), 'model': str(MODEL)},
        'source_sha256': {str(path): digest(path) for path in (LOG, MAPPING, MODEL)},
        'log_status': log['status'],
        'log_fault': log['fault'],
        'selected_feedback': {
            'phase': 'step_+8',
            'selection': 'last complete six-servo feedback sweep before release',
            'positions_1_to_6': [row['position'] for row in last],
            'per_servo_elapsed_s': [row['elapsed_s'] for row in last],
            'sweep_span_ms': (last[-1]['elapsed_s'] - last[0]['elapsed_s']) * 1000,
            'simultaneous_sample': False,
            'identical_trailing_sweeps': len(stable),
            'stable_interval_elapsed_s': [stable[0][0]['elapsed_s'], stable[-1][-1]['elapsed_s']],
            'goal_counts_1_to_5': selected_goals,
            'latest_hardware_state_claimed': False,
            'raw_feedback_rows': last,
        },
        'assumptions': {
            'mapping_verified': False,
            'wrist_zero_basis': 'qualitative pose E wrist-axis gamma=90-degree hypothesis',
            'gripper_model_q_rad': GRIPPER_Q,
            'ID6_count_to_angle_calibrated': False,
            'ID6_reported_count_used_for_model_q': False,
            'side_plot': 'kinematic anchor schematic, not shell clearances or camera reconstruction',
            'coordinates': 'model coordinates; no measured table/base height correction applied',
            'no_new_zero_inferred': True,
            'scope': 'conditional FK of historical observations and comparison vectors, not commands',
        },
        'poses': poses,
    }


def make_figure(report):
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import MultipleLocator

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False, 'font.size': 12})
    fig = plt.figure(figsize=(13.2, 8.0), layout='constrained', facecolor='#f8fafc')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.04, 1.35], height_ratios=[1, .47])
    full = fig.add_subplot(grid[:, 0])
    zoom = fig.add_subplot(grid[0, 1])
    info = fig.add_subplot(grid[1, 1])
    info.axis('off')
    actual = report['poses']['feedback']
    pan = actual['model_q_rad'][0]
    forward = np.array([math.cos(pan), -math.sin(pan), 0.])
    origin = np.array(actual['joint_anchors_model_m']['shoulder_lift'])
    actual_tcp = np.array(actual['tcp_position_model_m'])
    colors = {'feedback': '#1463a3', 'shoulder_only_target': '#d86b16', 'all_command_targets': '#8d4cb2'}
    labels = {'feedback': '日志反馈姿态', 'shoulder_only_target': '仅 ID2 换为目标 123', 'all_command_targets': '全部关节换为日志目标'}
    styles = {'feedback': '-', 'shoulder_only_target': '--', 'all_command_targets': ':'}
    for key, pose in report['poses'].items():
        anchors = pose['joint_anchors_model_m']
        nodes = np.array([anchors[name] for name in ('shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')] + [pose['tcp_position_model_m']])
        side = np.column_stack([(nodes - origin) @ forward, nodes[:, 2]]) * 1000
        full.plot(side[:, 0], side[:, 1], styles[key], color=colors[key], linewidth=2.3 if key == 'feedback' else 1.6, marker='o', markersize=5, alpha=.92)
        if key == 'feedback':
            annotations = ['S 肩 / ID2', 'E 肘 / ID3', 'W 腕俯仰 / ID4', 'R 腕旋转 / ID5', 'TCP']
            shifts = [(-13, -20), (6, 9), (10, -12), (10, 5), (-26, 12)]
            for label, xy, shift in zip(annotations, side, shifts):
                full.annotate(label, xy, xytext=shift, textcoords='offset points', fontsize=10, color='#1c3346')
        tcp_delta = np.array(pose['tcp_position_model_m']) - actual_tcp
        point = np.array([tcp_delta @ forward, tcp_delta[2]]) * 1000
        zoom.scatter(*point, color=colors[key], s=85, label=labels[key], zorder=4)
        if key != 'feedback':
            zoom.annotate('', point, (0, 0), arrowprops={'arrowstyle': '->', 'color': colors[key], 'lw': 1.8})
    full.set_title('轴心侧视图\n姿态差很小，三条线几乎重合', fontsize=14)
    full.set_xlabel('相对肩轴的前向距离 / mm')
    full.set_ylabel('模型高度 Z / mm')
    full.set_xlim(-120, 105)
    full.set_ylim(90, 345)
    full.set_aspect('equal', adjustable='box')
    full.grid(alpha=.22)
    zoom.set_title('TCP 局部放大：以反馈模型 TCP 为原点', fontsize=14)
    zoom.set_xlabel('前向位移 / mm')
    zoom.set_ylabel('高度位移 / mm')
    zoom.set_xlim(-.22, 1.62)
    zoom.set_ylim(-.63, .18)
    zoom.set_aspect('equal', adjustable='box')
    zoom.xaxis.set_major_locator(MultipleLocator(.25))
    zoom.yaxis.set_major_locator(MultipleLocator(.1))
    zoom.axhline(0, color='#8393a1', lw=.7)
    zoom.axvline(0, color='#8393a1', lw=.7)
    zoom.grid(alpha=.22)
    zoom.legend(loc='upper right', fontsize=9, framealpha=.9)
    one = report['poses']['shoulder_only_target']['difference_from_feedback']
    all_goals = report['poses']['all_command_targets']['difference_from_feedback']
    text = (
        '日志末尾反馈： [67, 119, 3966, 968, 3300]\n'
        '仅改肩目标：     [67, 123, 3966, 968, 3300]\n'
        '全部日志目标： [67, 123, 3967, 968, 3300]\n\n'
        f'仅改 ID2：TCP 差 {one["tcp_distance_mm"]:.3f} mm；姿态差 {one["tcp_orientation_geodesic_deg"]:.3f}°\n'
        f'全部目标：TCP 差 {all_goals["tcp_distance_mm"]:.3f} mm；姿态差 {all_goals["tcp_orientation_geodesic_deg"]:.3f}°\n\n'
        '候选映射未校准；夹爪 q=0.6 是模型假设。\n'
        '历史反馈不是当前位置；此图不能用于运动或反推零位。'
    )
    info.text(0, 1, text, va='top', fontsize=11, linespacing=1.55, color='#253c4d')
    fig.suptitle('肩部小步日志的离线模型对照', fontsize=20, fontweight='bold')
    output = HERE / 'shoulder_feedback_comparison.png'
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


if __name__ == '__main__':
    report = compute()
    report_path = HERE / 'shoulder_feedback_fk.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    figure_path = make_figure(report)
    print('Report:', report_path)
    print('Figure:', figure_path)
    print(json.dumps({'selected_feedback': {k: v for k, v in report['selected_feedback'].items() if k != 'raw_feedback_rows'},
                      'comparisons': {k: v['difference_from_feedback'] for k, v in report['poses'].items() if k != 'feedback'}}, ensure_ascii=False, indent=2))
