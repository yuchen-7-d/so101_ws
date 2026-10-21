"""Historical arm feedback with visually near-closed model jaw. No hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from build_report import ARM_NAMES, HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify archived inputs and FK without opening a window')
    args = parser.parse_args()
    report = json.loads((HERE / 'shoulder_feedback_fk.json').read_text())
    for name, expected in report['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'离线对照的来源文件已改变，请先重新生成报告：{name}')
    model = mujoco.MjModel.from_xml_path(report['source_files']['model'])
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES]
    qadr = model.jnt_qposadr[joints]
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    grip_adr = model.jnt_qposadr[grip]
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_center_40mm')
    pose = report['poses']['feedback']

    def apply_pose():
        data.qpos[qadr] = pose['model_q_rad']
        # Display-only adjustment from the user observation, never an ID6 calibration.
        data.qpos[grip_adr] = float(model.jnt_range[grip, 0])
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        expected = pose['tcp_position_model_m']
        if not np.allclose(data.site_xpos[tcp], expected, atol=1e-12, rtol=0):
            raise RuntimeError('离线 FK 与保存的对照不一致')

    apply_pose()
    print('仅离线查看历史日志；未连接串口，不产生机械臂运动。')
    print('用户确认实物闭合、肩肘腕基本一致；夹爪按模型下限作近闭合显示。')
    print(f'仅显示用夹爪角={float(model.jnt_range[grip, 0]):.6f} rad；这不是 ID6=3951 的零位标定。')
    print('候选映射仍未验证；40mm纸盒抓取参照点固定在固定爪侧，不随ID6开合变化。')
    print('固定显示日志最后稳定反馈：', pose['counts_1_to_5'])
    print('鼠标可旋转、缩放模型；不播放运动，不计算动力学。关闭窗口退出。')
    if args.check_only:
        print('输入哈希和历史反馈 FK 校验通过；未打开图形窗口。')
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [.07, 0, .205]
            viewer.cam.distance = .68
            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
        while viewer.is_running():
            with viewer.lock():
                apply_pose()
            viewer.sync()
            time.sleep(.03)


if __name__ == '__main__':
    main()
