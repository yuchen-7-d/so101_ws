import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path("/home/yu/so101_ws")

# 上一次提供的读数快照，不是实时读取。
COUNTS = {
    1: 9,
    2: 3871,
    3: 4074,
    4: 1941,
    5: 3282,
    6: 1913,
}


def main():
    model_path = ROOT / "simulation/so101/so101_new_calib.xml"
    mapping_path = ROOT / "data/so101_joint_mapping.json"

    model_bytes = model_path.read_bytes()
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)

    # 在内存中构建场景，不修改原模型文件。
    xml = ET.fromstring(model_bytes)

    # 修正网格资源路径。
    xml.find("compiler").set(
        "meshdir", str(model_path.parent / "assets")
    )

    # 修正被 include 引用的 XML 路径。
    include_hashes = {}
    for include in xml.iter("include"):
        include_path = Path(include.attrib["file"])
        if not include_path.is_absolute():
            include_path = model_path.parent / include_path
        include_path = include_path.resolve()

        if not include_path.is_file():
            raise FileNotFoundError(
                f"找不到模型引用文件：{include_path}"
            )

        include.set("file", str(include_path))
        include_hashes[str(include_path)] = hashlib.sha256(
            include_path.read_bytes()
        ).hexdigest()

    # 添加假设位于 z=0 的模型桌面。
    ET.SubElement(
        xml.find("worldbody"),
        "geom",
        name="diagnostic_floor",
        type="plane",
        pos="0 0 0",
        size="1 1 0.05",
    )

    scene_xml = ET.tostring(xml, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)

    print("使用已提供的读数快照；不读取串口。")
    joint_checks = []

    for servo_id in range(1, 6):
        key = str(servo_id)
        info = mapping["joint_directions"][key]
        ref = mapping["joint_references"][key]
        name = info["joint_name"]

        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if jid < 0 or aid < 0:
            raise RuntimeError(f"模型缺少关节或执行器：{name}")

        delta = (
            COUNTS[servo_id] - ref["position_count"] + 2048
        ) % 4096 - 2048
        angle_deg = (
            ref["model_angle_deg"]
            + info["direction"] * delta * 360.0 / 4096
        )
        angle = math.radians(angle_deg)

        lower = max(
            model.jnt_range[jid, 0],
            model.actuator_ctrlrange[aid, 0],
        )
        upper = min(
            model.jnt_range[jid, 1],
            model.actuator_ctrlrange[aid, 1],
        )
        if not lower <= angle <= upper:
            raise RuntimeError(
                f"ID{servo_id} 超出模型范围，停止检查"
            )

        data.qpos[model.jnt_qposadr[jid]] = angle
        margin_deg = math.degrees(
            min(angle - lower, upper - angle)
        )
        joint_checks.append({
            "servo_id": servo_id,
            "joint_name": name,
            "angle_deg": angle_deg,
            "nearest_limit_margin_deg": margin_deg,
        })
        print(
            f"ID{servo_id}: {angle_deg:+.2f}°，"
            f"最近限位余量={margin_deg:.2f}°"
        )

    # 与上一轮静态窗口一致：夹爪角度仅为示意。
    gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
    )
    if gripper_id < 0:
        raise RuntimeError("模型缺少 gripper 关节")
    data.qpos[model.jnt_qposadr[gripper_id]] = 0.0

    # 固定姿态进行计算，不求 IK、不推进动力学。
    mujoco.mj_forward(model, data)

    def geom_label(geom_id):
        geom_id = int(geom_id)
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        body_id = int(model.geom_bodyid[geom_id])
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return f"{body}/{name or ('geom_' + str(geom_id))}"

    contacts = []
    for contact in data.contact[:data.ncon]:
        contacts.append({
            "geoms": [
                geom_label(contact.geom1),
                geom_label(contact.geom2),
            ],
            "distance_mm": float(contact.dist * 1000),
            # 沿用现有规划器的 0.01 mm 穿透判定阈值。
            "penetrating": bool(contact.dist < -1e-5),
        })

    conflicts = [
        item for item in contacts if item["penetrating"]
    ]

    report = {
        "schema_version": 1,
        "kind": "so101_start_model_contact_check",
        "status": "not_for_motion",
        "execution_available": False,
        "register_writes": 0,
        "source": "user_reported_counts_snapshot",
        "device_sample_time": None,
        "position_counts": COUNTS,
        "joint_checks": joint_checks,
        "model_mapping_verified": mapping.get(
            "model_mapping_verified", False
        ),
        "gripper_model_angle_rad": 0.0,
        "gripper_count_calibrated": False,
        "assumed_floor_z_m": 0.0,
        "scope": "single_pose_robot_and_assumed_floor",
        "excluded_from_scene": [
            "paper_box",
            "cables",
            "external_supports",
            "other_obstacles",
        ],
        "contacts": contacts,
        "penetrating_contact_count": len(conflicts),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "include_sha256": include_hashes,
        "scene_sha256": hashlib.sha256(
            scene_xml.encode()
        ).hexdigest(),
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir="/tmp",
        prefix="so101_start_contact_",
        suffix=".json",
        delete=False,
    ) as stream:
        json.dump(
            report, stream,
            ensure_ascii=False, indent=2, allow_nan=False,
        )
        saved_path = stream.name

    print(f"\n起点模型接触记录数：{len(contacts)}")
    print(f"超过判定阈值的穿透记录数：{len(conflicts)}")

    for item in conflicts:
        print(
            f"  {item['geoms'][0]} ↔ {item['geoms'][1]}："
            f"穿透 {-item['distance_mm']:.3f} mm"
        )

    print("状态：not_for_motion；仅检查一个静态模型姿态。")
    print("完整记录：", saved_path)


if __name__ == "__main__":
    main()
