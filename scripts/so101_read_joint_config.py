"""Read the six servos' current configuration without changing registers."""

import argparse


def read_joint_config(port, servo_id):
    from so101_read_position import read_config_u16, read_servo_position
    from so101_set_id import read_servo_byte

    position_before = read_servo_position(port, servo_id)
    state = {
        "servo_id": servo_id,
        "position_before": position_before,
        "mode": read_servo_byte(port, servo_id, 0x21),
        "torque": read_servo_byte(port, servo_id, 0x28),
        "minimum": read_config_u16(port, servo_id, 0x09),
        "maximum": read_config_u16(port, servo_id, 0x0B),
        "phase": read_servo_byte(port, servo_id, 0x12),
        "resolution": read_servo_byte(port, servo_id, 0x1E),
        # Keep the encoded value; it is not a conventional signed int16.
        "offset_raw": read_config_u16(port, servo_id, 0x1F),
        "voltage_raw": read_servo_byte(port, servo_id, 0x3E),
    }
    state["position_after"] = read_servo_position(port, servo_id)
    return state


def print_joint_config(state):
    before = state["position_before"]
    after = state["position_after"]
    print(
        f"ID {state['servo_id']}:位置={before}→{after},"
        f"模式={state['mode']},扭矩开关={state['torque']},"
        f"电压={state['voltage_raw'] * 0.1:.1f}V",
        flush=True,
    )
    print(
        f"  设备最小限位={state['minimum']},"
        f"设备最大限位={state['maximum']},"
        f"中位修正原始值={state['offset_raw']}"
        f"(0x{state['offset_raw']:04X}),"
        f"相位=0x{state['phase']:02X},"
        f"分辨率倍率={state['resolution']}",
        flush=True,
    )
    if state["mode"] != 0 or not (0 <= before < 4096 and 0 <= after < 4096):
        print("  当前反馈不满足本项目单圈角度换算条件。", flush=True)
    else:
        delta = (after - before + 2048) % 4096 - 2048
        if abs(delta) > 4:
            print(
                f"  读取期间位置变化 {delta:+d} 个计数，"
                "这组数据不能作为静止姿态记录。",
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="只读检查 ID 1–6 的位置及配置，不改变扭矩、目标或限位。"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if args.baudrate <= 0:
        parser.error("--baudrate 必须大于 0")

    try:
        import serial
    except ImportError:
        parser.exit(1, "缺少 pyserial，请使用之前能运行位置读取脚本的 Python 环境。\n")

    print("只发送读取请求；保持当前姿态，并关闭其他占用该串口的程序。", flush=True)
    try:
        with serial.Serial(
            args.port,
            baudrate=args.baudrate,
            timeout=0.5,
            write_timeout=0.5,
        ) as port:
            for servo_id in range(1, 7):
                try:
                    state = read_joint_config(port, servo_id)
                except (RuntimeError, OSError) as error:
                    raise RuntimeError(f"ID {servo_id} 读取中止：{error}") from error
                print_joint_config(state)
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"检查未完成：{error}\n")

    print(
        "读取完成。设备限位是当前配置值，不代表已验证的机械安全范围；"
        "本次读取不代表关节映射或抓取已经验收。",
        flush=True,
    )


if __name__ == "__main__":
    main()
