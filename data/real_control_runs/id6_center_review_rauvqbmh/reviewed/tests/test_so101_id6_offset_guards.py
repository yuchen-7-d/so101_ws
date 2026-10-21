"""Exercise ID6 offset guards using protocol replies from a fake serial port."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import so101_gripper_test as gripper
import so101_read_position as reader


REFERENCE = {1: 64, 2: 3692, 3: 4059, 4: 1982, 5: 3250, 6: 3949}


class ReadOnlyPort:
    """Return real protocol READ replies and reject every other instruction."""

    def __init__(self, offset, position6=3961, torque6=0, fail_address=None):
        self.packets = []
        self.register_writes = []
        self.reply = b''
        self.fail_address = fail_address
        self.registers = {}
        for servo_id, position in REFERENCE.items():
            registers = {
                0x08: 1, 0x12: 0x0C, 0x1E: 1, 0x21: 0,
                0x28: 0, 0x29: 1, 0x3E: 67,
            }
            for address, value in (
                (0x09, 0), (0x0B, 4095), (0x1F, 0),
                (0x2E, 50), (0x30, 0), (0x38, position),
            ):
                registers[address] = value & 0xFF
                registers[address + 1] = value >> 8
            self.registers[servo_id] = registers
        self.registers[6].update({
            0x1F: offset & 0xFF, 0x20: offset >> 8,
            0x38: position6 & 0xFF, 0x39: position6 >> 8,
            0x28: torque6,
        })

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def reset_input_buffer(self):
        self.reply = b''

    def write(self, packet):
        self.packets.append(bytes(packet))
        if packet[4] != 0x02:
            self.register_writes.append(bytes(packet))
            raise AssertionError('A register write reached the fake serial port')
        if (
            packet[:2] != b'\xff\xff'
            or len(packet) != 8
            or packet[3] != 4
            or packet[-1] != (~sum(packet[2:-1]) & 0xFF)
        ):
            raise AssertionError('Invalid READ request')
        servo_id, address, count = packet[2], packet[5], packet[6]
        values = [self.registers[servo_id][address + i] for i in range(count)]
        fields = [servo_id, count + 2, 0, *values]
        self.reply = bytes([0xFF, 0xFF, *fields, ~sum(fields) & 0xFF])
        if servo_id == 6 and address == self.fail_address:
            self.reply = self.reply[:-1]
        return len(packet)

    def read(self, count):
        result, self.reply = self.reply[:count], self.reply[count:]
        return result


class InitialWriteReached(Exception):
    """Stop the legacy offset-zero path before any mutation takes place."""


class OffsetGuardTests(unittest.TestCase):
    def run_reader(self, port):
        output = io.StringIO()
        reference_text = json.dumps({
            'position_counts': {str(k): v for k, v in REFERENCE.items()},
        })
        with (
            patch.object(reader.serial, 'Serial', return_value=port),
            patch.object(reader.Path, 'read_text', return_value=reference_text),
            redirect_stdout(output),
        ):
            reader.main()
        self.assertEqual(port.register_writes, [])
        lines = output.getvalue().splitlines()
        for servo_id in range(1, 6):
            self.assertEqual(lines[servo_id - 1], (
                f'ID {servo_id}:位置计数={REFERENCE[servo_id]},'
                '模式=0,扭矩开关=0,电压=6.7V,编码器相对角度=+0.00°'
            ))
        return lines

    def test_zero_offset_preserves_relative_angle_and_counts(self):
        lines = self.run_reader(ReadOnlyPort(offset=0))
        self.assertEqual(lines[5], (
            'ID 6:位置计数=3961,模式=0,扭矩开关=0,电压=6.7V,'
            '编码器相对角度=+1.05°'
        ))
        self.assertIn('中位修正原始值=0(0x0000)', lines[-1])

    def test_nonzero_offsets_hide_stale_relative_angle_without_writes(self):
        for offset in (1854, 0x1001):
            with self.subTest(offset=offset):
                lines = self.run_reader(ReadOnlyPort(offset, position6=2048))
                self.assertEqual(lines[5], (
                    'ID 6:位置计数=2048,模式=0,扭矩开关=0,电压=6.7V,'
                    '编码器相对角度=未换算（ID6计数基准已改变）'
                ))
                self.assertIn(
                    f'中位修正原始值={offset}(0x{offset:04X})', lines[-1],
                )

    def test_legacy_actions_reject_nonzero_offset_before_any_register_write(self):
        for action in ('open', 'precontact'):
            for offset in (1854, 0x1001):
                for torque in (0, 1):
                    with self.subTest(action=action, offset=offset, torque=torque):
                        port = ReadOnlyPort(offset, torque6=torque)
                        with patch.object(gripper.serial, 'Serial', return_value=port):
                            with self.assertRaisesRegex(RuntimeError, '旧ID6绝对目标不可用'):
                                gripper.main(action)
                        self.assertEqual(port.register_writes, [])
                        self.assertEqual(
                            [(p[2], p[4], p[5]) for p in port.packets],
                            [(6, 0x02, 0x1F), (6, 0x02, 0x20)],
                        )

    def test_offset_read_failure_stops_legacy_actions_before_any_write(self):
        for action in ('open', 'precontact'):
            with self.subTest(action=action):
                port = ReadOnlyPort(0, fail_address=0x20)
                with patch.object(gripper.serial, 'Serial', return_value=port):
                    with self.assertRaisesRegex(RuntimeError, '读取应答长度错误'):
                        gripper.main(action)
                self.assertEqual(port.register_writes, [])

    def test_zero_offset_still_reaches_legacy_initial_torque_disable(self):
        for action in ('open', 'precontact'):
            with self.subTest(action=action):
                port = ReadOnlyPort(0)
                with (
                    patch.object(gripper.serial, 'Serial', return_value=port),
                    patch.object(
                        gripper, 'write_servo_byte', side_effect=InitialWriteReached,
                    ) as first_write,
                ):
                    with self.assertRaises(InitialWriteReached):
                        gripper.main(action)
                first_write.assert_called_once_with(port, 6, 0x28, 0)
                self.assertEqual(port.register_writes, [])


if __name__ == '__main__':
    unittest.main()
