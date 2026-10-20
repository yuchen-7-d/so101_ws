"""Protocol-level fault injection. Never opens a serial device."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import (
    ALL_IDS, ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError, HoldController,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeSerial:
    def __init__(self, clock, hook=None):
        self.clock, self.hook = clock, hook
        self.regs = {sid: bytearray(100) for sid in ALL_IDS}
        self.writes, self.events = [], []
        self.pending = b""
        self.auto_enable_on_goal = True
        self.allow_live_goal_writes = False
        for sid, position in zip(ALL_IDS, (29, 841, 3030, 1982, 3254, 4013)):
            r = self.regs[sid]
            r[5], r[8], r[13], r[14], r[15] = sid, 1, 70, 74, 40
            r[18], r[30], r[33] = 0x0C, 1, 0
            r[62], r[63] = 67, 27
            self.put(sid, 11, 4095)
            self.put(sid, 16, 1000)
            self.put(sid, OUTPUT, 1000)
            self.put(sid, GOAL, 2000)  # Deliberately stale goal.
            self.put(sid, 56, position)

    def put(self, sid, address, value):
        self.regs[sid][address:address+2] = value.to_bytes(2, "little")

    def get(self, sid, address):
        return int.from_bytes(self.regs[sid][address:address+2], "little")

    def reset_input_buffer(self):
        self.pending = b""

    def write(self, packet):
        assert packet[:2] == b"\xff\xff"
        assert packet[-1] == (~sum(packet[2:-1])) & 255
        sid, length, instruction, address = packet[2:6]
        assert len(packet) == length + 4
        params = packet[6:-1]
        self.events.append((instruction, sid, address, bytes(params)))
        self.clock.now += .0001
        if instruction == 2:
            payload = bytes(self.regs[sid][address:address+params[0]])
        else:
            assert instruction == 3
            if address == GOAL:
                # Guarantee that old targets never get powered during setup.
                assert self.get(sid, OUTPUT) == 0 or self.allow_live_goal_writes
            if address == OUTPUT and int.from_bytes(params, "little"):
                assert self.get(sid, GOAL) == self.get(sid, 56)
                assert self.regs[sid][TORQUE] == 1
            self.writes.append((sid, address, int.from_bytes(params, "little")))
            self.regs[sid][address:address+len(params)] = params
            if address == GOAL and self.auto_enable_on_goal:
                self.regs[sid][TORQUE] = 1
            payload = b""
        fields = bytes([sid, len(payload)+2, 0]) + payload
        self.pending = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])
        if self.hook:
            self.hook(self, instruction, sid, address, params)
        return len(packet)

    def read(self, size):
        result = self.pending[:size]
        self.pending = self.pending[size:]
        return result


class HoldTests(unittest.TestCase):
    def setup_control(self, hook=None, limit=100):
        clock = Clock()
        port = FakeSerial(clock, hook)
        messages = []
        controller = HoldController(Bus(port), limit=limit, seconds=.25,
                                    emit=messages.append, clock=clock.time, sleep=clock.sleep)
        return port, controller, messages

    def assert_released(self, port, ids=ARM_IDS):
        for sid in ids:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def test_default_does_not_write(self):
        port, control, _ = self.setup_control()
        control.run()
        self.assertEqual(port.writes, [])

    def test_current_goals_only_and_cleanup(self):
        port, control, _ = self.setup_control()
        original = {sid: bytes(r) for sid, r in port.regs.items()}
        control.run(enable=True)
        self.assert_released(port)
        self.assertEqual(bytes(port.regs[6]), original[6])
        for sid in ARM_IDS:
            self.assertEqual(bytes(port.regs[sid][:40]), original[sid][:40])
            self.assertEqual(port.get(sid, GOAL), port.get(sid, 56))
        self.assertTrue(all(sid in ARM_IDS and address in (40, 41, 42, 46, 48)
                            for sid, address, value in port.writes))
        self.assertEqual([value for sid, addr, value in port.writes
                          if sid == 2 and addr == OUTPUT and value], [25, 50, 75, 100])

    def test_existing_torque_is_never_disabled_by_preflight(self):
        port, control, _ = self.setup_control()
        port.regs[3][TORQUE] = 1
        with self.assertRaisesRegex(ControlError, "扭矩已开启"):
            control.run(True)
        self.assertEqual(port.writes, [])
        self.assertEqual(port.regs[3][TORQUE], 1)

    def test_unstable_start_has_no_writes(self):
        port, control, _ = self.setup_control()
        old_sleep = control.sleep
        def drift(duration):
            old_sleep(duration)
            port.put(3, 56, 3010)
        control.sleep = drift
        with self.assertRaisesRegex(ControlError, "位置漂移"):
            control.run(True)
        self.assertEqual(port.writes, [])

    def test_crc_failure_aborts_without_writing(self):
        def corrupt(p, ins, sid, addr, params):
            p.pending = p.pending[:-1] + bytes([p.pending[-1] ^ 1])
        port, control, _ = self.setup_control(corrupt)
        with self.assertRaisesRegex(ControlError, "校验和"):
            control.run(True)
        self.assertEqual(port.writes, [])

    def test_ack_loss_after_applied_write_still_releases(self):
        fired = False
        def lose(p, ins, sid, addr, params):
            nonlocal fired
            if not fired and ins == 3 and sid == 3 and addr == GOAL:
                fired = True
                p.pending = b""
        port, control, _ = self.setup_control(lose)
        with self.assertRaisesRegex(ControlError, "超时"):
            control.run(True)
        self.assertTrue(fired)
        self.assert_released(port)
        self.assertFalse(any(addr == OUTPUT and value for _, addr, value in port.writes))

    def test_runtime_drift_does_not_retarget_or_increase_limit(self):
        fired = False
        def drift(p, ins, sid, addr, params):
            nonlocal fired
            if not fired and ins == 3 and sid == 5 and addr == OUTPUT and params == b"\x64\x00":
                fired = True
                p.put(2, 56, 870)
        port, control, _ = self.setup_control(drift)
        with self.assertRaisesRegex(ControlError, "偏离保持点"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(port.get(2, GOAL), 841)
        self.assertLessEqual(max(value for _, addr, value in port.writes if addr == OUTPUT), 100)

    def test_raw_wrap_is_rejected_not_retargeted(self):
        def wrap(p, ins, sid, addr, params):
            if ins == 3 and sid == 5 and addr == OUTPUT and params == b"\x19\x00":
                p.put(1, 56, 4095)
        port, control, _ = self.setup_control(wrap)
        with self.assertRaisesRegex(ControlError, "偏离保持点"):
            control.run(True)
        self.assertEqual(port.get(1, GOAL), 29)
        self.assert_released(port)

    def test_release_continues_when_one_id_disconnected(self):
        enabled = False
        def disconnect(p, ins, sid, addr, params):
            nonlocal enabled
            if ins == 3 and sid == 5 and addr == OUTPUT and params == b"\x19\x00":
                enabled = True
            if enabled and sid == 2:
                p.pending = b""
        port, control, messages = self.setup_control(disconnect)
        with self.assertRaisesRegex(ControlError, r"未确认释放 ID=\[2\]"):
            control.run(True)
        self.assert_released(port, ids=(1, 3, 4, 5))
        self.assertTrue(any("ID 5: 已读回" in msg for msg in messages))

    def test_interrupt_after_enable_releases(self):
        fired = False
        def interrupt(p, ins, sid, addr, params):
            nonlocal fired
            if not fired and ins == 3 and sid == 3 and addr == TORQUE and params == b"\x01":
                fired = True
                raise KeyboardInterrupt
        port, control, _ = self.setup_control(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            control.run(True)
        self.assert_released(port)

    def test_whitelist_prevents_eeprom_gripper_and_calibration_writes(self):
        port, control, _ = self.setup_control()
        for sid, address, value in ((1, 31, 0), (1, 55, 0), (6, TORQUE, 1),
                                    (1, TORQUE, 128), (1, OUTPUT, 1000), (1, GOAL, 4096)):
            with self.assertRaises(ControlError):
                control.bus.write(sid, address, value)
        self.assertEqual(port.writes, [])

    def test_broken_output_stream_cannot_skip_cleanup(self):
        port, control, _ = self.setup_control()
        control.preflight()
        control.prepare()
        def broken(message):
            raise BrokenPipeError("closed stdout")
        control.emit = broken
        self.assertEqual(control.release(), [])
        self.assert_released(port)

    def test_feedback_time_budget_checked_between_reads(self):
        port, control, _ = self.setup_control()
        control.preflight()
        control.prepare()
        def slow(p, ins, sid, addr, params):
            if ins == 2:
                p.clock.now += .14
        port.hook = slow
        started = port.clock.time()
        with self.assertRaisesRegex(ControlError, "反馈累计"):
            control.monitor(1, 12, 0)
        self.assertLess(port.clock.time() - started, .7)
        port.hook = None
        self.assertEqual(control.release(), [])

    def test_corrupt_goal_readback_never_enables_output(self):
        fired = False
        def wrong(p, ins, sid, addr, params):
            nonlocal fired
            if not fired and ins == 3 and sid == 2 and addr == GOAL:
                fired = True
                p.put(sid, GOAL, 2000)
        port, control, _ = self.setup_control(wrong)
        with self.assertRaisesRegex(ControlError, "写后读回不一致"):
            control.run(True)
        self.assertFalse(any(addr == OUTPUT and value for _, addr, value in port.writes))
        self.assert_released(port)

    def test_runtime_overtemperature_releases(self):
        def hot(p, ins, sid, addr, params):
            if ins == 3 and sid == 5 and addr == OUTPUT and params == b"\x19\x00":
                p.regs[3][63] = 60
        port, control, _ = self.setup_control(hot)
        with self.assertRaisesRegex(ControlError, "温度"):
            control.run(True)
        self.assert_released(port)

    def test_persistent_saturation_releases_without_increasing_limit(self):
        def loaded(p, ins, sid, addr, params):
            if ins == 3 and sid == 5 and addr == OUTPUT and params == b"\x64\x00":
                p.put(2, 60, 100)
        port, control, _ = self.setup_control(loaded)
        control.seconds = 2
        with self.assertRaisesRegex(ControlError, "持续接近本次上限"):
            control.run(True)
        self.assert_released(port)
        self.assertLessEqual(max(value for _, addr, value in port.writes if addr == OUTPUT), 100)

    def test_alarm_packets_do_not_prevent_release_readback(self):
        alarm = False
        def alarming(p, ins, sid, addr, params):
            nonlocal alarm
            if ins == 3 and sid == 5 and addr == OUTPUT and params == b"\x19\x00":
                alarm = True
            if alarm:
                reply = bytearray(p.pending)
                reply[4] = 32
                reply[-1] = (~sum(reply[2:-1])) & 255
                p.pending = bytes(reply)
        port, control, _ = self.setup_control(alarming)
        with self.assertRaisesRegex(ControlError, "舵机报警"):
            control.run(True)
        self.assert_released(port)


if __name__ == "__main__":
    unittest.main()
