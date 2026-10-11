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

packet = build_ping_packet(1)
print(packet.hex(" ").upper())

def parse_ping_reply(reply):
    if len(reply) != 6:
        raise ValueError('PING应答为6字节')

    if reply[:2] != bytes([0xFF,0xFF]) or reply[3] != 2:
        raise ValueError('包头或长度字段错误')

    expected_checksum = (~(sum(reply[2:-1]))) & 0xFF

    if reply[-1] != expected_checksum:
        raise ValueError('校验和不匹配')

    return reply[2], reply[4]

reply = bytes.fromhex('FF FF 01 02 00 FC')
servo_id, error = parse_ping_reply(reply)
print(f'应答ID:{servo_id},状态码:{error}')
