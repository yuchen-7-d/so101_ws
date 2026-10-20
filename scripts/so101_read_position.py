import serial
from so101_set_id import read_servo_byte
import json
from pathlib import Path


def read_servo_position(port, servo_id):
    fields = [servo_id, 4, 0x02, 0x38, 2]
    checksum = (~sum(fields)) & 0xff
    packet = bytes([0xff, 0xff] + fields + [checksum])

    port.reset_input_buffer()

    if port.write(packet) != len(packet):
        raise RuntimeError(f'ID{servo_id}:读取请求未完整发送')

    reply = port.read(8)

    if len(reply) != 8:
        raise RuntimeError(
            f'ID{servo_id}：应答错误'
            f'接收:{reply.hex(" ").upper()}'
        )

    if (
        reply[:2] != bytes([0xff,0xff])
        or reply[2] != servo_id
        or reply[3] != 4
    ):
        raise RuntimeError(f'ID{servo_id}应答格式错误')

    expected_checksum = (~sum(reply[2:-1])) & 0xff
    if reply[-1] != expected_checksum:
        raise RuntimeError(f'ID {servo_id}:校验和错误')

    if reply[4] != 0:
        raise RuntimeError(f'ID {servo_id}:状态码{reply[4]}')

    raw = reply[5] + (reply[6] << 8)
    position = raw & 0x7FFF

    if raw & 0x8000:
        position = -position

    return position

def read_config_u16(port, servo_id, address):
    low = read_servo_byte(port, servo_id, address)
    high = read_servo_byte(port, servo_id, address + 1)
    return low + (high << 8)

def main():
    reference_path = (
        Path(__file__).resolve().parents[1]
        / 'data'
        / 'so101_initial_pose.json'
    )
    record = json.loads(
        reference_path.read_text(encoding='utf-8')
    )
    reference_counts = record['position_counts']
    with serial.Serial(
        '/dev/ttyUSB0',
        baudrate=1_000_000,
        timeout=0.5,
        write_timeout=0.5,
    ) as port:
        for servo_id in range(1, 7):
            position = read_servo_position(port, servo_id)

            mode = read_servo_byte(port, servo_id, 0x21)
            torque_state = read_servo_byte(port, servo_id, 0x28)
            voltage_raw = read_servo_byte(port, servo_id, 0x3e)

            if mode != 0 or not 0 <= position < 4096:
                raise RuntimeError(
                    f'ID {servo_id}:当前反馈不适用于单圈角度换算'
                )

            reference_position = reference_counts[str(servo_id)]

            delta_counts = (
                position - reference_position + 2048
            ) % 4096 - 2048

            relative_deg = delta_counts * 360.0 / 4096

            print(
                f'ID {servo_id}:位置计数={position},'
                f'模式={mode},扭矩开关={torque_state},'
                f'电压={voltage_raw * 0.1:.1f}V,'
                f'编码器相对角度={relative_deg:+.2f}°'
            )
            if servo_id == 6:
                minimum = read_config_u16(port, servo_id, 0x09)
                maximum = read_config_u16(port, servo_id, 0x0B)
                phase = read_servo_byte(port, servo_id, 0x12)
                resolution = read_servo_byte(port, servo_id, 0x1E)

                acceleration = read_servo_byte(
                    port, servo_id, 0x29
                )
                speed_raw = read_config_u16(
                    port, servo_id, 0x2E
                )
                torque_limit = read_config_u16(
                    port, servo_id, 0x30
                )

                print(
                    f'ID 6运行参数:加速度={acceleration},'
                    f'速度原始值={speed_raw},'
                    f'扭矩限制={torque_limit}/1000'
                )

                print(
                    f'ID 6配置:最小限位={minimum},'
                    f'最大限位={maximum},'
                    f'相位=0x{phase:02X},'
                    f'分辨率倍率={resolution}'
                )


if __name__ == '__main__':
    main()