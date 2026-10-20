import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    model_path = (
        Path(__file__).resolve().parents[1]
        / 'simulation'
        / 'so101'
        / 'so101_new_calib.xml'
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    reference_deg = {
        'shoulder_pan': 0.0,
        'shoulder_lift': -13.97,
        'elbow_flex': 16.18,
        'wrist_flex': -2.21,
        'wrist_roll': 2.79,
        'gripper': 0.0,
    }

    for name, angle_deg in reference_deg.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id == -1:
            raise RuntimeError(f'找不到关节:{name}')

        angle = math.radians(angle_deg)
        lower, upper = model.jnt_range[joint_id]
        if not lower <= angle <= upper:
            raise RuntimeError(f'{name}:参考角度超出模型范围')

        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = angle

    reference_qpos = data.qpos.copy()
    mujoco.mj_forward(model, data)

    print('模型参考角度(度):', reference_deg)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.16, 0.0, 0.14]
            viewer.cam.distance = 0.65
            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

            # 纸盒尺寸为 40 × 40 × 74 mm，size 使用半尺寸
            box_half_size = np.array([0.020, 0.020, 0.037])

            # 检测结果给出的是顶面中心
            box_top_center = np.array([
                0.25140174, 0.00404917, 0.074,
            ])

            # 显示纸盒需要使用整个盒子的中心
            box_center = box_top_center.copy()
            box_center[2] -= box_half_size[2]

            angle_rad = math.radians(-30.67)
            c, s = math.cos(angle_rad), math.sin(angle_rad)

            box_rotation = np.array([
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ])

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=box_half_size,
                pos=box_center,
                mat=box_rotation.ravel(),
                rgba=[0.1, 0.4, 1.0, 0.6],
            )
            viewer.user_scn.ngeom = 1

        while viewer.is_running():
            with viewer.lock():
                data.qpos[:] = reference_qpos
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(0.03)


if __name__ == '__main__':
    main()
