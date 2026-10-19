from pathlib import Path
import json
import time
import mujoco
import mujoco.viewer
import numpy as np


def orientation_error_world(current_rotation, target_rotation):
    delta_rotation = target_rotation @ current_rotation.T

    delta_quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        delta_quat, delta_rotation.ravel()
    )

    error_world = np.zeros(3)
    mujoco.mju_quat2Vel(
        error_world, delta_quat, 1.0
    )

    return error_world

def sample_joint_path(path, progress):
    progress = np.clip(progress, 0.0, 1.0)

    path_position = progress * (len(path) - 1)

    left_index = min(int(path_position), len(path) - 2,)
    fraction = path_position - left_index

    q_left = path[left_index]
    q_right = path[left_index + 1]

    return(
        (1 - fraction) * q_left
        + fraction * q_right
    )

def main():
    workspase = Path(__file__).resolve().parents[1]

    placement = 'flat'

    measurement_sources = {
        'upright': ('box_top_measurement.json', 'square_yaw_deg'),
        'flat': ('box_flat_measurement.json', 'long_axis_yaw_deg'),
    }

    measurement_file, yaw_field = measurement_sources[placement]

    input_path = workspase / 'data' / measurement_file

    json_text = input_path.read_text(encoding='utf-8')
    measurement = json.loads(json_text)

    if measurement is None:
        print('这份记录没有通过尺寸筛选的检测结果')
        return

    center_m = measurement['center_m']
    yaw_deg = measurement[yaw_field]

    print(f'顶面中心:{center_m}')
    print(f'顶面方向:{yaw_deg:.2f}')

    x_table, y_table, z_table = center_m

    center_sim_demo = [x_table,-y_table,-z_table]
    yaw_sim_demo_deg = -yaw_deg

    print(f'仿真演示中心:{center_sim_demo}')
    print(f'仿真演示方向:{yaw_sim_demo_deg:.2f}')

    model_path = Path(
        workspase / 'simulation' / 'so101' / 'scene_grasp.xml'
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    box_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        'a_box_free',
    )
    if box_joint_id == -1:
        raise RuntimeError('找不到纸盒关节')

    fixed_jaw_geom_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_GEOM,'fixed_jaw_tip_collision'
    )
    moving_jaw_geom_ids = {
        mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_GEOM,f'moving_jaw_collision_{i}',
        )
        for i in range(4)
    }

    if fixed_jaw_geom_id < 0 or -1 in moving_jaw_geom_ids:
        raise RuntimeError('找不到夹爪几何碰撞')

    box_qpos_adr = model.jnt_qposadr[box_joint_id]
    box_qvel_adr = model.jnt_dofadr[box_joint_id]

    box_half_size = np.array([0.02,0.02,0.037])

    angle_rad = np.deg2rad(yaw_sim_demo_deg)
    c, s =np.cos(angle_rad), np.sin(angle_rad)

    box_rotation = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1],
    ])

    if placement == 'flat':
        lay_rotation = np.array([
            [0,0,1],
            [0,1,0],
            [-1,0,0]
        ], dtype=float)

        box_rotation = box_rotation @ lay_rotation

    vertical_half_extent = (
        np.abs(box_rotation[2, :]) @ box_half_size
    )

    box_center = np.asarray(center_sim_demo, dtype=float).copy()
    box_center[2] -= vertical_half_extent

    box_quat = np.zeros(4)
    mujoco.mju_mat2Quat(box_quat,box_rotation.ravel())

    data.qpos[box_qpos_adr:box_qpos_adr + 3] = box_center
    data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = box_quat

    #data.qpos[box_qpos_adr + 2] += 0.10 临时抬高测试
    print(f'初始中心高度:{data.qpos[box_qpos_adr + 2]:.4f}')

    mujoco.mj_forward(model,data)

    tcp_site_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center_40mm'
    )
    if tcp_site_id == -1:
        raise RuntimeError('找不到末端参考点')

    tcp_position = data.site_xpos[tcp_site_id].copy()

    pregrasp_target = np.asarray(center_sim_demo,dtype=float).copy()
    pregrasp_clearance = 0.01
    pregrasp_target[2] += pregrasp_clearance

    grasp_depth = 0.010

    grasp_target = np.asarray(
        center_sim_demo, dtype=float
    ).copy()

    grasp_target[2] -= grasp_depth

    if placement == 'upright':
        closing_direction = box_rotation[:, 0].copy()
    else:
        closing_direction = box_rotation[:, 1].copy()

    entry_clearance = 0.001
    entry_offset = -entry_clearance * closing_direction

    pregrasp_target += entry_offset
    grasp_target += entry_offset

    position_error = pregrasp_target - tcp_position

    print(f'末端参考点:{tcp_position}')
    print(f'预抓取目标:{pregrasp_target}')
    print(f'距目标:{np.linalg.norm(position_error) * 1000:.2f}')

    arm_joint_names = [
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
    ]

    arm_joint_ids = []

    for name in arm_joint_names:
        joint_id = mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_JOINT,name
        )
        if joint_id == -1:
            raise RuntimeError(f'找不到关节:{name}')

        arm_joint_ids.append(joint_id)

    arm_dof_adrs = model.jnt_dofadr[arm_joint_ids]

    jac_position = np.zeros((3,model.nv))

    mujoco.mj_jacSite(
        model,data,jac_position,None,tcp_site_id
    )

    arm_jacobian = jac_position[:,arm_dof_adrs]

    print(f'完整位置雅可比形状:{jac_position.shape}')
    print(f'机械臂位置雅可比形状:{arm_jacobian.shape}')

    max_cart_step = 0.005
    error_norm = np.linalg.norm(position_error)

    if error_norm < 1e-6:
        print('当前末端已到达目标位置')
        return

    step_error = position_error * min(1.0, max_cart_step / error_norm)

    J = arm_jacobian
    damping = 0.01

    system_matrix = (
        J @ J.T + damping**2 * np.eye(3)
    )

    delta_q = J.T @ np.linalg.solve(
        system_matrix, step_error
    )

    predicted_step = J @ delta_q

    print(f'关节微调(rad):{np.round(delta_q, 5)}')
    print(f'预计末端位移(mm):{np.round(predicted_step * 1000, 3)}')

    arm_qpos_adrs = model.jnt_qposadr[arm_joint_ids]

    trial_data = mujoco.MjData(model)
    trial_data.qpos[:] = data.qpos

    trial_data.qpos[arm_qpos_adrs] += delta_q

    mujoco.mj_forward(model,trial_data)

    trial_tcp = trial_data.site_xpos[tcp_site_id].copy()
    actual_step = trial_tcp - tcp_position

    remaining_error = np.linalg.norm(
        pregrasp_target - trial_tcp
    )

    print(f'重新计算的末端位移:{np.round(actual_step * 1000, 3)}')
    print(f'微调后距目标:{remaining_error * 1000:.2f}')

    max_iterations = 150
    position_tolerance = 0.001
    joint_limits = model.jnt_range[arm_joint_ids]

    for iteration in range(max_iterations):
        mujoco.mj_forward(model, trial_data)

        error = pregrasp_target - trial_data.site_xpos[tcp_site_id]
        distance = np.linalg.norm(error)

        if distance <= position_tolerance:
            break

        mujoco.mj_jacSite(
            model,trial_data,jac_position,None,tcp_site_id
        )

        J = jac_position[:,arm_dof_adrs]

        step_error = error * min(1.0, max_cart_step / distance)

        system_matrix = (
            J @ J.T + damping**2 * np.eye(3)
        )

        delta_q = J.T @ np.linalg.solve(
            system_matrix, step_error
        )

        largest_change = np.max(np.abs(delta_q))
        if largest_change > 0.05:
            delta_q *= 0.05 / largest_change

        trial_data.qpos[arm_qpos_adrs] = np.clip(
            trial_data.qpos[arm_qpos_adrs] + delta_q,
            joint_limits[:,0],
            joint_limits[:,1],
        )

    mujoco.mj_forward(model, trial_data)

    final_error = np.linalg.norm(
        pregrasp_target - trial_data.site_xpos[tcp_site_id]
    )

    print(f'位置求解通过:{final_error <= position_tolerance}')
    print(f'最终位置误差:{final_error * 1000:.3f} mm')

    tcp_rotation = trial_data.site_xmat[tcp_site_id].reshape(3, 3)
    approach_direction = tcp_rotation[:,0]

    print(f'末端接近方向:{np.round(approach_direction, 3)}')

    target_direction = np.array([0.0, 0.0, -1.0])

    target_closing_direction = closing_direction.copy()

    target_side_direction = np.cross(
        target_closing_direction,
        target_direction,
    )

    target_rotation = np.column_stack((
        target_direction,
        target_side_direction,
        target_closing_direction,
    ))

    print(f'目标夹持方向:{np.round(target_closing_direction, 3)}')
    print(f'目标旋转矩阵:\n{np.round(target_rotation, 3)}')

    cos_angle = np.dot(approach_direction, target_direction)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_deg = np.rad2deg(np.arccos(cos_angle))

    print(f'偏移竖直向下:{angle_deg:.2f}')

    jac_rotation = np.zeros((3, model.nv))
    direction_weight = 0.05

    max_combined_iterations = 200
    direction_tolerance_deg = 2.0

    def solve_pose_ik(
            ik_data,
            target_position,
            target_rotation,
            position_tol,
            rotation_tol_deg,
    ):

        for iteration in range(max_combined_iterations):
            mujoco.mj_forward(model,ik_data)

            position_error_now = (
                target_position - ik_data.site_xpos[tcp_site_id]
            )

            current_rotation = (
                ik_data.site_xmat[tcp_site_id].reshape(3,3)
            )

            rotation_error = orientation_error_world(
                current_rotation, target_rotation
            )

            angle_deg = np.rad2deg(np.linalg.norm(rotation_error))

            if(
                np.linalg.norm(position_error_now) <= position_tol
                and angle_deg <= rotation_tol_deg
            ):
                break

            mujoco.mj_jacSite(
                model,ik_data,jac_position,jac_rotation,tcp_site_id
            )

            J_position = jac_position[:, arm_dof_adrs]
            J_angular = jac_rotation[:, arm_dof_adrs]

            J_combined = np.vstack((
                J_position,
                direction_weight * J_angular,
            ))

            error_combined = np.concatenate((
                position_error_now,
                direction_weight * rotation_error
            ))

            combined_norm = np.linalg.norm(error_combined)

            step_combined = error_combined * min(
                1.0, max_cart_step / max(combined_norm, 1e-12)
            )

            combined_matrix = (
                J_combined @ J_combined.T + damping**2 * np.eye(6)
            )

            delta_q_comdined = J_combined.T @ np.linalg.solve(
                combined_matrix, step_combined
            )

            largest_change = np.max(np.abs(delta_q_comdined))
            if largest_change > 0.05:
                delta_q_comdined *= 0.05 / largest_change

            ik_data.qpos[arm_qpos_adrs] = np.clip(
                ik_data.qpos[arm_qpos_adrs] + delta_q_comdined,
                joint_limits[:,0],
                joint_limits[:,1],
            )

        mujoco.mj_forward(model, ik_data)

        final_position_error = np.linalg.norm(
            target_position - ik_data.site_xpos[tcp_site_id]
        )

        final_rotation = (
            ik_data.site_xmat[tcp_site_id].reshape(3,3)
        )

        final_rotation_error = orientation_error_world(
            final_rotation, target_rotation
        )

        final_rotation_error_deg = np.rad2deg(
            np.linalg.norm(final_rotation_error)
        )

        ok = (
            final_position_error <= position_tol
            and final_rotation_error_deg <= rotation_tol_deg
        )

        return ok, final_position_error, final_rotation_error_deg

    combined_ok, new_position_error, new_angle_deg = solve_pose_ik(
        trial_data,
        pregrasp_target,
        target_rotation,
        position_tolerance,
        direction_tolerance_deg,
    )

    print(f'联合更新后位置误差:{new_position_error * 1000:.3f} mm')
    print(f'联合更新后完整姿态误差:{new_angle_deg:.2f}')
    print(f'联合求解通过:{combined_ok}')

    current_rotation = (
        trial_data.site_xmat[tcp_site_id].reshape(3,3)
    )

    full_orientation_error = orientation_error_world(
        current_rotation,target_rotation
    )

    print(
        f'完整姿态误差:'
        f'{np.rad2deg(np.linalg.norm(full_orientation_error)):.2f}'
    )

    start_q = data.qpos[arm_qpos_adrs].copy()
    goal_q = trial_data.qpos[arm_qpos_adrs].copy()

    arm_actuator_ids = []

    for name in arm_joint_names:
        actuator_id = mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_ACTUATOR,name
        )

        if actuator_id == -1:
            raise RuntimeError(f'找不到执行器:{name}')

        arm_actuator_ids.append(actuator_id)

    ctrl_limits = model.actuator_ctrlrange[arm_actuator_ids]

    goal_in_ctrl_range = np.all(
        (goal_q >= ctrl_limits[:, 0])
        & (goal_q <= ctrl_limits[:, 1])
    )

    print(f'机械臂执行器编号:{arm_actuator_ids}')
    print(f'目标角度在控制范围内:{goal_in_ctrl_range}')

    path_data = mujoco.MjData(model)
    path_data.qpos[:] = trial_data.qpos

    num_path_samples = 101
    contact_sample_ids = []

    for index, alpha in enumerate(
        np.linspace(0.0, 1.0, num_path_samples)
    ):
        path_data.qpos[arm_qpos_adrs] = (
            (1.0-alpha) * start_q + alpha * goal_q
        )

        mujoco.mj_forward(model,path_data)

        for contact_index in range(path_data.ncon):
            contact = path_data.contact[contact_index]


            names = {
                mujoco.mj_id2name(
                    model,mujoco.mjtObj.mjOBJ_GEOM, int(g)
                )
                for g in contact.geom
            }

            if names == {'floor', 'a_box_geom'}:
                continue

            contact_sample_ids.append(index)
            break

    print(f'路径采样数:{num_path_samples}')
    print(f'有接触记录的采样数:{len(contact_sample_ids)}')

    num_descent_samples = 21

    descent_targets = np.linspace(
        pregrasp_target,
        grasp_target,
        num_descent_samples,
    )

    print(f'下降路径形状:{descent_targets.shape}')
    print(f'下降起点:{descent_targets[0]}')
    print(f'下降终点:{descent_targets[-1]}')

    descent_data = mujoco.MjData(model)
    descent_data.qpos[:] = trial_data.qpos

    gripper_joint_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_JOINT,'gripper'
    )

    if gripper_joint_id == -1:
        raise RuntimeError('找不到夹爪关节')

    gripper_qpos_adr = model.jnt_qposadr[gripper_joint_id]
    descent_data.qpos[gripper_qpos_adr] = 0.6

    descent_joint_path = []
    descent_errors = []

    for index, target_position in enumerate(descent_targets):
        ok, pos_error, rot_error = solve_pose_ik(
            descent_data,
            target_position,
            target_rotation,
            0.0002,
            0.5,
        )

        if not ok:
            raise RuntimeError(
                f'下降采样{index}求解失败'
                f'位置误差{pos_error * 1000:.3f}mm'
                f'姿态误差{rot_error:.3f}度'
            )

        descent_joint_path.append(descent_data.qpos[arm_qpos_adrs].copy())
        descent_errors.append([pos_error, rot_error])

    descent_joint_path = np.asarray(descent_joint_path)
    descent_errors = np.asarray(descent_errors)

    lift_joint_path = descent_joint_path[::-1].copy()
    lift_target = pregrasp_target.copy()

    print(f'抬起关节路径形状:{lift_joint_path.shape}')

    print(
        f'下降关节路径形状:{descent_joint_path.shape}'
        f'下降最大位置误差:{descent_errors[:, 0].max() * 1000:.3f}mm'
        f'下降最大姿态误差:{descent_errors[:, 1].max():.3f}'
    )

    descent_in_ctrl_range = np.all(
        (descent_joint_path >= ctrl_limits[:, 0])
        & (descent_joint_path <= ctrl_limits[:, 1])
    )

    joint_steps = np.diff(
        descent_joint_path, axis=0
    )

    max_joint_step_deg = np.rad2deg(
        np.max(np.abs(joint_steps))
    )

    print(f'下降角度在控制范围内:{descent_in_ctrl_range}')
    print(f'下降相邻最大角度变化:{max_joint_step_deg:.3f}度')

    check_data = mujoco.MjData(model)
    check_data.qpos[:] = descent_data.qpos
    descent_contacts = []

    for index,joint_angles in enumerate(descent_joint_path):
        check_data.qpos[arm_qpos_adrs] = joint_angles
        mujoco.mj_forward(model,check_data)

        for contact_index in range(check_data.ncon):
            contact = check_data.contact[contact_index]

            names = [
                mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g))
                or f'geom_{int(g)}'
                for g in contact.geom
            ]

            if set(names) == {'a_box_geom', 'floor'}:
                continue

            descent_contacts.append((
                index,
                names[0],
                names[1],
                float(contact.dist) * 1000,
            ))

    print(f'下降额外接触记录:{len(descent_contacts)}')

    if descent_contacts:
        worst = min(descent_contacts, key=lambda item: item[3])
        print(f'下降最小接触记录:{worst}')

    path_data.qpos[arm_qpos_adrs] = start_q
    mujoco.mj_forward(model,path_data)

    if (
        not combined_ok 
        or not goal_in_ctrl_range 
        or contact_sample_ids
        or not descent_in_ctrl_range
        or descent_contacts
    ):
        raise RuntimeError('求解、控制范围或路径检查未通过')

    sim_data = mujoco.MjData(model)
    sim_data.qpos[:] = data.qpos
    sim_data.qvel[:] = 0.0

    sim_data.ctrl[arm_actuator_ids] = start_q

    mujoco.mj_forward(model, sim_data)

    print(f'动力学初始时间:{sim_data.time:.3f}s')
    print(f'物理步长:{model.opt.timestep:.3f}s')

    box_geom_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_GEOM,'a_box_geom'
    )
    floor_geom_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_GEOM,'floor'
    )
    if min(box_geom_id,floor_geom_id) <0:
        raise RuntimeError('找不到几何体')

    allowed_pair = {box_geom_id, floor_geom_id}
    unexpected_pairs = set()

    moving_grasp_pairs = {
        tuple(sorted((geom_id, box_geom_id)))
        for geom_id in moving_jaw_geom_ids
    }

    grasp_pairs = moving_grasp_pairs | {
        tuple(sorted((fixed_jaw_geom_id, box_geom_id)))
    }

    gripper_actuator_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_ACTUATOR,'gripper'
    )
    if gripper_actuator_id == -1:
        raise RuntimeError('找不到夹爪执行器')

    gripper_open_angle = 0.6

    low, high = model.actuator_ctrlrange[gripper_actuator_id]
    if not low <= gripper_open_angle <= high:
        raise RuntimeError('超出夹爪张开范围')

    probe_data = mujoco.MjData(model)
    probe_data.qpos[:] = descent_data.qpos

    first_touch_angle = None

    for angule in np.linspace(gripper_open_angle, 0.0, 61):
        probe_data.qpos[gripper_qpos_adr] = angule
        mujoco.mj_forward(model, probe_data)

        for contact_index in range(probe_data.ncon):
            contact = probe_data.contact[contact_index]

            pair = tuple(sorted(int(g)for g in contact.geom))
            if pair in moving_grasp_pairs and contact.dist <= 0.0:
                first_touch_angle = float(angule)
                break

        if first_touch_angle is not None:
            break

    if first_touch_angle is None:
        print('采集区间内没有检测到活动爪与纸盒接触')
    else:
        print(f'首次活动爪接触角度:{first_touch_angle:.3f}rad')

    with mujoco.viewer.launch_passive(model,sim_data) as viewer:
        with viewer.lock():

            #viewer.opt.sitegroup[3] = 1
            #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.opt.geomgroup[2] = 0
            viewer.opt.geomgroup[3] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005,0,0],
                pos=np.asarray(pregrasp_target,dtype=float),
                mat=np.eye(3).ravel(),
                rgba=[1,1,0,1],
            )
            viewer.user_scn.ngeom = 1

        duration = 3
        steps_per_frame = 10
        open_start_time = duration + 1
        open_duration = 1
        descent_start_time = (
            open_start_time + open_duration + 1
        )
        descent_duration = 3

        close_start_time = (
            descent_start_time + descent_duration + 2.0
        )
        close_duration = 2.0

        lift_start_time = (
            close_start_time + close_duration + 2.0
        )
        lift_duration = 3.0

        box_height_before_lift = None

        gripper_close_angle = 0.0
        close_torque_limit = 0.15

        if not low <= gripper_close_angle <= high:
            raise RuntimeError('闭合目标超出夹爪控制范围')

        while viewer.is_running():
            frame_start = time.monotonic()

            with viewer.lock():
                viewer.opt.sitegroup[4] = 1
                for _ in range(steps_per_frame):
                    alpha = min(sim_data.time / duration, 1.0)
                    target_q = (
                        (1.0 - alpha) * start_q + alpha *goal_q
                    )

                    if sim_data.time >= descent_start_time:
                        descent_progress = (
                            (sim_data.time - descent_start_time)
                            / descent_duration
                        )

                        target_q = sample_joint_path(
                            descent_joint_path, descent_progress
                        )

                    if sim_data.time >= lift_start_time:
                        if box_height_before_lift is None:
                            box_height_before_lift = float(
                                sim_data.qpos[box_qpos_adr + 2 ]
                            )
                            box_height_peak = box_height_before_lift

                            snapshot_path = (
                                workspase / 'data' / 'grasp_before_lift.npz'
                            )

                            np.savez(
                                snapshot_path,
                                qpos=sim_data.qpos.copy(),
                                qvel=sim_data.qvel.copy(),
                                ctrl=sim_data.ctrl.copy(),
                                actuator_force=sim_data.actuator_force.copy(),
                                actuator_forcerange=model.actuator_forcerange.copy(),
                                sim_time=sim_data.time,
                            )

                            print(f'状态已保存至:{snapshot_path}')

                        lift_progress = (
                            (sim_data.time - lift_start_time)
                            / lift_duration
                        )

                        target_q = sample_joint_path(
                            lift_joint_path, lift_progress
                        )

                    sim_data.ctrl[arm_actuator_ids] = target_q

                    open_alpha = np.clip(
                        (sim_data.time - open_start_time) /
                        open_duration,0.0,1.0
                    )

                    gripper_target_angle = (
                        open_alpha * gripper_open_angle
                    )

                    if sim_data.time >= close_start_time:
                        close_alpha = np.clip(
                            (sim_data.time - close_start_time)
                            / close_duration,
                            0.0,
                            1.0,
                        )

                        gripper_target_angle = (
                            (1.0 - close_alpha) * gripper_open_angle
                            + close_alpha * gripper_close_angle
                        )

                        model.actuator_forcerange[
                            gripper_actuator_id
                        ] = [-close_torque_limit, close_torque_limit]

                    sim_data.ctrl[gripper_actuator_id] = (
                        gripper_target_angle
                    )

                    mujoco.mj_step(model,sim_data)

                    if box_height_before_lift is not None:
                        box_height_peak = max(
                            box_height_peak,
                            float(sim_data.qpos[box_qpos_adr + 2])
                        )

                    for contact_index in range(sim_data.ncon):
                        contact = sim_data.contact[contact_index]

                        if set(contact.geom) == allowed_pair:
                            continue

                        pair = tuple(sorted(int(g) for g in contact.geom))

                        if(
                            sim_data.time >= close_start_time
                            and pair in grasp_pairs
                        ):
                            continue

                        unexpected_pairs.add(pair)

            viewer.sync()
            wait_time = (
                steps_per_frame * model.opt.timestep -
                (time.monotonic() - frame_start)
            )
            if wait_time > 0:
                time.sleep(wait_time)

        print(f'动力学结束时间:{sim_data.time:.3f}s')

    print(f'额外接触对数量:{len(unexpected_pairs)}')
    print(f'额外接触数对:{sorted(unexpected_pairs)}')

    if sim_data.time < lift_start_time + lift_duration + 2:
        print('保持时间不足，请运行更久后关闭窗口')
        return

    mujoco.mj_forward(model,sim_data)

    actual_q = sim_data.qpos[arm_qpos_adrs].copy()
    joint_error_deg = np.rad2deg(lift_joint_path[-1] - actual_q)
    max_joint_error_deg = np.max(np.abs(joint_error_deg))

    actual_tcp = sim_data.site_xpos[tcp_site_id].copy()
    tcp_error_mm = np.linalg.norm(
        lift_target - actual_tcp
    ) * 1000

    print(f'最大关节角度误差:{max_joint_error_deg:.3f}')
    print(f'动力学末端位置误差:{tcp_error_mm:.3f}')

    actual_rotation = (
        sim_data.site_xmat[tcp_site_id].reshape(3, 3)
    )

    actual_rotation_error = orientation_error_world(
        actual_rotation, target_rotation
    )

    actual_rotation_error_deg = np.rad2deg(
        np.linalg.norm(actual_rotation_error)
    )

    actual_joint_speed = sim_data.qvel[arm_dof_adrs]
    max_joint_speed_deg_s = np.max(
        np.abs(np.rad2deg(actual_joint_speed))
    )

    print(f'动力学完整姿态误差:{actual_rotation_error_deg:.3f}')
    print(f'最大关节速度:{max_joint_speed_deg_s:.3f}')

    gripper_joint_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_JOINT,'gripper'
    )

    if gripper_joint_id == -1:
        raise RuntimeError('找不到关节')

    gripper_qpos_adr = model.jnt_qposadr[gripper_joint_id]
    actual_gripper_angle = sim_data.qpos[gripper_qpos_adr]

    gripper_error_deg = abs(np.rad2deg(
        gripper_close_angle - actual_gripper_angle
    ))

    print(f'实际夹爪角度:{actual_gripper_angle:.4f} rad')
    print(f'夹爪角度误差:{gripper_error_deg:.3f}')

    fixed_count = 0
    moving_count = 0
    fixed_normal_force = 0
    moving_normal_force = 0

    for contact_index in range(sim_data.ncon):
        contact = sim_data.contact[contact_index]
        pair = tuple(sorted(int(g) for g in contact.geom))

        if pair not in grasp_pairs:
            continue

        force = np.zeros(6)
        mujoco.mj_contactForce(
            model,sim_data,contact_index,force
        )
        normal_force = float(force[0])

        if fixed_jaw_geom_id in pair:
            fixed_count += 1
            fixed_normal_force += normal_force
        else:
            moving_count += 1
            moving_normal_force += normal_force

    print(
        f'固定夹爪接触点:{fixed_count},'
        f'法力向合计:{fixed_normal_force:.3f}N'
    )

    print(
        f'活动爪接触点:{moving_count},'
        f'法向力合计:{moving_normal_force:.3f}N'
    )

    final_box_height = float(
        sim_data.qpos[box_qpos_adr + 2]
    )

    peak_lift_mm = (
        box_height_peak - box_height_before_lift
    ) * 1000

    final_lift_mm =(
        final_box_height - box_height_before_lift
    ) * 1000

    print(f'纸盒最大升高:{peak_lift_mm}')
    print(f'最终升高:{final_lift_mm}')

    return

    print(f'结束中心高度:{data.qpos[box_qpos_adr + 2]:.4f}')

    mujoco.mj_forward(model,data)
    box_floor_contacts = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        if set(contact.geom) == {box_geom_id, floor_geom_id}:
            box_floor_contacts += 1

    print(f'纸盒和地面接触点数:{box_floor_contacts}')

    linear_velocity = data.qvel[box_qvel_adr:box_qvel_adr + 3]
    angular_velocity = data.qvel[box_qvel_adr + 3:box_qvel_adr + 6]

    linear_speed = np.linalg.norm(linear_velocity)
    angular_speed = np.linalg.norm(angular_velocity)

    print(f'纸盒线速度:{linear_speed:.6f}')
    print(f'纸盒角速度:{angular_speed:.6f}')


if __name__ == '__main__':
    main()
