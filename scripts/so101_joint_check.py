import time
import mujoco
import mujoco.viewer
import math
from pathlib import Path


def main():
    model_path = Path(
        '/home/yu/so101_reference/Simulation/SO101/scene.xml'
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    shoulder_pan_actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'shoulder_pan'
    )

    shoulder_pan_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        'shoulder_pan'
    )

    ids = {
        'shoulder_pan_actuator':shoulder_pan_actuator_id,
        'shoulder_pan_joint':shoulder_pan_joint_id,
    }

    for id_name, id_id in ids.items():
        if id_id == -1:
            raise RuntimeError(f'找不到:{id_name}')

    print(f'仿真开始:{data.time}')
    with mujoco.viewer.launch_passive(model, data) as windows:

        qpos_index = model.jnt_qposadr[shoulder_pan_joint_id]
        targets = [0.2, -0.2, 0.0]

        duration = 1.0
        steps = round(duration / model.opt.timestep)

        tolerance = 0.01

        lower = model.actuator_ctrlrange[shoulder_pan_actuator_id][0]
        upper = model.actuator_ctrlrange[shoulder_pan_actuator_id][1]

        for target in targets:

            inside_ctrl = lower <= target <= upper
            if not inside_ctrl:
                raise ValueError('shoulder_pan 超出范围')

            data.ctrl[shoulder_pan_actuator_id] = target

            for _ in range(steps):

                mujoco.mj_step(model,data)
                windows.sync()
                time.sleep(model.opt.timestep)

            actual = data.qpos[qpos_index]

            error = target - actual

            print(f'目标角度:{target:.6f}')
            print(f'实际角度:{actual:.6f}')
            print(f'误差:{error:.6f}')

            if abs(error) <= tolerance:
                print('到位')
            else:
                print('未到位')

    print(f'仿真结束时间:{data.time:.6f}')


if __name__ == '__main__':
    main()
