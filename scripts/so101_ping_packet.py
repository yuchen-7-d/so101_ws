import serial


def build_ping_packet(servo_id):
    if not 0 <= servo_id <= 253:
        raise ValueError('舵机ID必须在0～253')

    length = 2
    instruction = 0x01

    checksum = (~(instruction + length + servo_id)) & 0xFF
    packet = bytes(
        [0xFF,0xFF,servo_id,length,instruction,checksum]
    )
    return packet

def parse_ping_reply(reply):
    if len(reply) != 6:
        raise ValueError('PING应答为6字节')

    if reply[:2] != bytes([0xFF,0xFF]) or reply[3] != 2:
        raise ValueError('包头或长度字段错误')

    expected_checksum = (~(sum(reply[2:-1]))) & 0xFF

    if reply[-1] != expected_checksum:
        raise ValueError('校验和不匹配')

    return reply[2], reply[4]

def main():
    with serial.Serial(
        '/dev/ttyUSB0',
        baudrate=1_000_000,
        timeout=0.5,
        write_timeout=0.5,
    ) as port:
        for target_id in range(1, 7):
            packet = build_ping_packet(target_id)

            port.reset_input_buffer()

            sent = port.write(packet)
            if sent != len(packet):
                raise RuntimeError(f'ID {target_id}:请求未完整发送')

            reply = port.read(6)

            try:
                reply_id, error = parse_ping_reply(reply)
            except ValueError as exc:
                raise RuntimeError(
                    f'ID {target_id}:应答无效，'
                    f'接收:{reply.hex(" ").upper()}'
                ) from exc

            if reply_id != target_id or error != 0:
                raise RuntimeError(
                    f'ID {target_id}:检查失败，'
                    f'应答ID:{reply_id},状态码:{error}'
                )

            print(f'ID {target_id}:通信通过，状态码:{error}')

    print('六只舵机通信检查通过')
if __name__ == '__main__':
    main()
