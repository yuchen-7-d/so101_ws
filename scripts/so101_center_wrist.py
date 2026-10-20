import json
from pathlib import Path

from so101_read_position import read_servo_position, read_config_u16
from so101_set_id import read_servo_byte, write_servo_byte
from so101_gripper_test import write_servo_u16


def read_wrist_state(port, servo_id):
    return {
        "position": read_servo_position(port, servo_id),
        "mode": read_servo_byte(port, servo_id, 0x21),
        "torque": read_servo_byte(port, servo_id, 0x28),
        "reply_level": read_servo_byte(port, servo_id, 0x08),
        "offset_raw": read_config_u16(port, servo_id, 0x1F),
        "output_limit": read_config_u16(port, servo_id, 0x30),
        "minimum": read_config_u16(port, servo_id, 0x09),
        "maximum": read_config_u16(port, servo_id, 0x0B),
        "phase": read_servo_byte(port, servo_id, 0x12),
        "resolution": read_servo_byte(port, servo_id, 0x1E),
    }


def center_wrist(port):
    servo_id = 4

    before = read_wrist_state(port, servo_id)
    expected = {
        "mode": 0, "torque": 0, "reply_level": 1,
        "offset_raw": 0, "minimum": 0, "maximum": 4095,
        "phase": 0x0C, "resolution": 1,
    }

    if any(before[key] != value for key, value in expected.items()):
        raise RuntimeError(f"ID 4配置不符合本次校准条件:{before}")

    if not 3850 <= before["position"] <= 3890:
        raise RuntimeError("ID 4已离开本次参考位置，停止校准")

    record_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "id4_center_record.json"
    )
    record = {"before": before, "calibration_attempted": False}

    # x模式：已有同名记录就停止，避免重复校准。
    with record_path.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)

    try:
        write_servo_u16(port, servo_id, 0x30, 0)

        ready = read_wrist_state(port, servo_id)
        if ready["output_limit"] != 0:
            raise RuntimeError("输出限制未归零")

        if any(ready[key] != value for key, value in expected.items()):
            raise RuntimeError("校准前状态发生变化")

        if abs(ready["position"] - before["position"]) > 2:
            raise RuntimeError("腕部位置不稳定，停止校准")

        record["position_before"] = ready["position"]
        record["calibration_attempted"] = True
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        write_servo_byte(port, servo_id, 0x28, 128)

    finally:
        write_servo_byte(port, servo_id, 0x28, 0)
        torque = read_servo_byte(port, servo_id, 0x28)
        print(f"ID 4:退出扭矩={torque}")

        if torque != 0:
            raise RuntimeError("未确认扭矩关闭")

    after = read_wrist_state(port, servo_id)
    record["after"] = after
    record["shift"] = after["position"] - record["position_before"]

    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for key, value in expected.items():
        if key != "offset_raw" and after[key] != value:
            raise RuntimeError(f"校准后{key}异常:{after[key]}")

    if (
        not 2045 <= after["position"] <= 2050
        or after["offset_raw"] == before["offset_raw"]
        or after["output_limit"] != 0
    ):
        raise RuntimeError(f"中位校准读回未通过:{after}")

    print("校准前计数:", record["position_before"])
    print("校准后计数:", after["position"])
    print("计数平移量:", record["shift"])
    print("位置修正原始值:", after["offset_raw"])
    print("输出限制:", after["output_limit"])
    print("记录已保存至:", record_path)

def main():
    import serial

    with serial.Serial(
        "/dev/ttyUSB0",
        baudrate=1_000_000,
        timeout=0.5,
        write_timeout=0.5,
    ) as port:
        center_wrist(port)


if __name__ == "__main__":
    main()