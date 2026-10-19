import serial
from so101_ping_packet import parse_ping_reply as parse_status_reply

def read_servo_byte(port, servo_id, address):
    fields = [servo_id, 4, 0x02, address, 1]
    checksum = (~sum(fields)) & 0xFF
    packet = bytes([0xFF, 0xFF] + fields + [checksum])

    port.reset_input_buffer()

    sent = port.write(packet)
    if sent != len(packet):
        raise RuntimeError('读取请求未完整发送')

    reply = port.read(7)

    if len(reply) != 7:
        raise RuntimeError(
            f'读取应答长度错误:{reply.hex(" ").upper()}'
        )

    if (
        reply[:2] != bytes([0xFF, 0xFF])
        or reply[2] != servo_id
        or reply[3] != 3
    ):
        raise RuntimeError('读取应答的包头、ID或长度字段错误')

    expected_checksum = (~sum(reply[2:-1])) & 0xFF
    if reply[-1] != expected_checksum:
        raise RuntimeError('读取应答校验和错误')

    if reply[4] != 0:
        raise RuntimeError(f'舵机状态码:{reply[4]}')

    return reply[5]

def build_write_byte_packet(servo_id, address, value):
    if not 0 <= servo_id <= 253:
        raise ValueError('舵机ID超出范围')

    if not 0 <= address <= 255 or not 0 <= value <= 255:
        raise ValueError('地址或数据超出单字节范围')

    length = 4
    instruction = 0x03

    fields = [servo_id, length, instruction, address, value]
    checksum = (~sum(fields)) & 0xFF

    return bytes([0xFF, 0xFF] + fields + [checksum])

def write_servo_byte(port, servo_id, address, value):
    packet = build_write_byte_packet(
        servo_id, address, value
    )

    port.reset_input_buffer()

    sent = port.write(packet)
    if sent != len(packet):
        raise RuntimeError('写入请求未完整发送')

    reply = port.read(6)

    print('发送:', packet.hex(' ').upper())
    print('接收:', reply.hex(' ').upper())

    reply_id, error = parse_status_reply(reply)

    allowed_ids = {servo_id}
    if address == 0x05:
        allowed_ids.add(value)

    if reply_id not in allowed_ids:
        raise RuntimeError(f'写入应答ID异常:{reply_id}')

    if error != 0:
        raise RuntimeError(f'写入状态码:{error}')

def main():
    old_id = 1
    new_id = 6

    with serial.Serial(
        '/dev/ttyUSB0',
        baudrate=1_000_000,
        timeout=0.5,
        write_timeout=0.5,
    ) as port:
        if read_servo_byte(port, old_id, 0x05) != old_id:
            raise RuntimeError('当前ID检查未通过')

        if read_servo_byte(port, old_id, 0x08) != 1:
            raise RuntimeError('应答等级不是1，停止写入')

        commands = [
            ('解锁', old_id, 0x37, 0, old_id),
            ('修改ID', old_id, 0x05, new_id, new_id),
            ('锁回', new_id, 0x37, 1, new_id),
        ]

        for name, send_id, address, value, check_id in commands:
            print(f'开始{name}')

            write_servo_byte(
                port, send_id, address, value
            )

            actual = read_servo_byte(
                port, check_id, address
            )

            if actual != value:
                raise RuntimeError(
                    f'{name}读回不一致:期望{value},实际{actual}'
                )

            print(f'{name}通过:{actual}')


if __name__ == '__main__':
    main()