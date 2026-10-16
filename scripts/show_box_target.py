from pathlib import Path
import json
import time
import mujoco
import mujoco.viewer
import numpy as np


def main():
    workspase = Path(__file__).resolve().parents[1]
    input_path = workspase / 'data' / 'box_top_measurement.json'

    json_text = input_path.read_text(encoding='utf-8')
    measurement = json.loads(json_text)

    if measurement is None:
        print('这份记录没有通过尺寸筛选的检测结果')
        return

    center_m = measurement['center_m']
    yaw_deg = measurement['square_yaw_deg']

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

    box_qpos_adr = model.jnt_qposadr[box_joint_id]
    box_qvel_adr = model.jnt_dofadr[box_joint_id]

    box_half_size = np.array([0.02,0.02,0.037])

    box_center = np.asarray(center_sim_demo,dtype=float).copy()
    box_center[2] -= box_half_size[2]

    angle_rad = np.deg2rad(yaw_sim_demo_deg)
    c, s =np.cos(angle_rad), np.sin(angle_rad)

    box_rotation = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1],
    ])

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

    target_closing_direction = box_rotation[:,0].copy()

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

    for iteration in range(max_combined_iterations):
        mujoco.mj_forward(model,trial_data)

        position_error_now = (
            pregrasp_target - trial_data.site_xpos[tcp_site_id]
        )

        approach_direction = (
            trial_data.site_xmat[tcp_site_id].reshape(3,3)[:,0].copy()
        )
        direction_error = target_direction - approach_direction

        cos_angle = np.clip(
            np.dot(approach_direction,target_direction),-1.0,1.0
        )

        angle_deg = np.rad2deg(np.arccos(cos_angle))

        if(
            np.linalg.norm(position_error_now) <= position_tolerance
            and angle_deg <= direction_tolerance_deg
        ):
            break

        mujoco.mj_jacSite(
            model,trial_data,jac_position,jac_rotation,tcp_site_id
        )

        J_angular = jac_rotation[:, arm_dof_adrs]

        J_direction = np.cross(J_angular.T, approach_direction).T

        direction_error = target_direction - approach_direction

        position_error_now = (
            pregrasp_target - trial_data.site_xpos[tcp_site_id]
        )
        J_position = jac_position[:, arm_dof_adrs]

        J_combined = np.vstack((
            J_position,
            direction_weight * J_direction,
        ))

        error_combined = np.concatenate((
            position_error_now,
            direction_weight * direction_error,
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

        trial_data.qpos[arm_qpos_adrs] = np.clip(
            trial_data.qpos[arm_qpos_adrs] + delta_q_comdined,
            joint_limits[:,0],
            joint_limits[:,1],
        )

    mujoco.mj_forward(model, trial_data)

    new_position_error = np.linalg.norm(
        pregrasp_target - trial_data.site_xpos[tcp_site_id]
    )

    new_direction = (
        trial_data.site_xmat[tcp_site_id].reshape(3,3)[:,0]
    )

    new_cos = np.clip(
        np.dot(new_direction, target_direction), -1.0, 1.0
    )
    new_angle_deg = np.rad2deg(np.arccos(new_cos))

    combined_ok = (
        new_position_error <= position_tolerance
        and new_angle_deg <= direction_tolerance_deg
    )

    print(f'联合更新后位置误差:{new_position_error * 1000:.3f} mm')
    print(f'联合更新后的方向夹角:{new_angle_deg:.2f}')
    print(f'联合求解通过:{combined_ok}')

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

        if path_data.ncon > 0:
            contact_sample_ids.append(index)

    print(f'路径采样数:{num_path_samples}')
    print(f'有接触记录的采样数:{len(contact_sample_ids)}')

    path_data.qpos[arm_qpos_adrs] = start_q
    mujoco.mj_forward(model,path_data)

    if not combined_ok or not goal_in_ctrl_range or contact_sample_ids:
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

    gripper_actuator_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_ACTUATOR,'gripper'
    )
    if gripper_actuator_id == -1:
        raise RuntimeError('找不到夹爪执行器')

    gripper_open_angle = 0.6

    low, high = model.actuator_ctrlrange[gripper_actuator_id]
    if not low <= gripper_open_angle <= high:
        raise RuntimeError('超出夹爪张开范围')

    with mujoco.viewer.launch_passive(model,sim_data) as viewer:
        with viewer.lock():

            #viewer.opt.sitegroup[3] = 1
            #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

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

        while viewer.is_running():
            frame_start = time.monotonic()

            with viewer.lock():
                viewer.opt.sitegroup[4] = 1
                for _ in range(steps_per_frame):
                    alpha = min(sim_data.time / duration, 1.0)
                    target_q = (
                        (1.0 - alpha) * start_q + alpha *goal_q
                    )
                    sim_data.ctrl[arm_actuator_ids] = target_q

                    open_alpha = np.clip(
                        (sim_data.time - open_start_time) /
                        open_duration,0.0,1.0
                    )

                    sim_data.ctrl[gripper_actuator_id] = (
                        open_alpha * gripper_open_angle
                    )

                    mujoco.mj_step(model,sim_data)

                    for contact_index in range(sim_data.ncon):
                        contact = sim_data.contact[contact_index]

                        if set(contact.geom) == allowed_pair:
                            continue

                        pair = tuple(sorted(int(g) for g in contact.geom))
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

    if sim_data.time < open_duration + open_start_time + 1:
        print('保持时间不足，请运行更久后关闭窗口')
        return

    mujoco.mj_forward(model,sim_data)

    actual_q = sim_data.qpos[arm_qpos_adrs].copy()
    joint_error_deg = np.rad2deg(goal_q - actual_q)
    max_joint_error_deg = np.max(np.abs(joint_error_deg))

    actual_tcp = sim_data.site_xpos[tcp_site_id].copy()
    tcp_error_mm = np.linalg.norm(
        pregrasp_target - actual_tcp
    ) * 1000

    print(f'最大关节角度误差:{max_joint_error_deg:.3f}')
    print(f'动力学末端位置误差:{tcp_error_mm:.3f}')

    actual_direction =(
        sim_data.site_xmat[tcp_site_id].reshape(3,3)[:,0]
    )

    direction_cos = np.clip(
        np.dot(actual_direction,target_direction),-1.0,1.0
    )

    direction_error_deg = np.rad2deg(
        np.arccos(direction_cos)
    )

    actual_joint_speed = sim_data.qvel[arm_dof_adrs]
    max_joint_speed_deg_s = np.max(
        np.abs(np.rad2deg(actual_joint_speed))
    )

    print(f'动力学方向夹角:{direction_error_deg:.3f}')
    print(f'最大关节速度:{max_joint_speed_deg_s:.3f}')

    gripper_joint_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_JOINT,'gripper'
    )

    if gripper_joint_id == -1:
        raise RuntimeError('找不到关节')

    gripper_qpos_adr = model.jnt_qposadr[gripper_joint_id]
    actual_gripper_angle = sim_data.qpos[gripper_qpos_adr]

    gripper_error_deg = abs(np.rad2deg(
        gripper_open_angle - actual_gripper_angle
    ))

    print(f'实际夹爪角度:{actual_gripper_angle:.4f} rad')
    print(f'夹爪角度误差:{gripper_error_deg:.3f}')

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
