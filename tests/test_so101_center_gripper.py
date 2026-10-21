"""Packet-level mock tests. No pyserial import and no device access."""
from contextlib import redirect_stdout, redirect_stderr
import builtins
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import so101_center_gripper as center


class Clock:
    def __init__(self):
        self.now = 0.
    def time(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


class FakeSerial:
    """Synchronous exact ST packets; 128 models offset-only renumbering."""
    def __init__(self, clock, hook=None):
        self.clock, self.hook = clock, hook
        self.regs = {sid: bytearray(100) for sid in center.ALL_IDS}
        self.events, self.writes = [], []
        self.pending = b""
        self.center_position = 2048
        self.center_offset_extra = 0
        self.control = None
        for sid in center.ALL_IDS:
            r = self.regs[sid]
            r[0:5] = bytes([3, 10, 0, 9, 3])
            r[5], r[8], r[13], r[14], r[15] = sid, 1, 70, 74, 40
            r[18], r[30], r[33], r[62], r[63] = 12, 1, 0, 67, 27
            self.put(sid, 11, 4095)
            self.put(sid, 48, 1000)
            self.put(sid, 42, 999)  # Stale target must only be recorded, never used.
            r[55] = 1
            self.put(sid, 56, 194 if sid == 6 else sid*100)
    def put(self, sid, address, value):
        self.regs[sid][address:address+2] = value.to_bytes(2, 'little')
    def get(self, sid, address):
        return int.from_bytes(self.regs[sid][address:address+2], 'little')
    def reset_input_buffer(self):
        self.pending = b""
    def write(self, packet):
        assert packet[:2] == b'\xff\xff'
        assert packet[-1] == (~sum(packet[2:-1])) & 255
        sid, length, ins, address = packet[2:6]
        assert len(packet) == length+4
        params = bytes(packet[6:-1])
        self.events.append((ins, sid, address, params))
        self.clock.now += .001
        if ins == 2:
            payload = bytes(self.regs[sid][address:address+params[0]])
        else:
            assert ins == 3
            self.writes.append((sid, address, int.from_bytes(params, 'little')))
            self.regs[sid][address:address+len(params)] = params
            if (sid, address, params) == (6, 40, b'\x80'):
                offset = self.get(6, 56)-self.center_position+self.center_offset_extra
                assert -2047 <= offset <= 2047
                self.put(6, 31, abs(offset) | (0x800 if offset < 0 else 0))
                self.put(6, 56, self.center_position)
            payload = b''
        fields = bytes([sid, len(payload)+2, 0])+payload
        self.pending = b'\xff\xff'+fields+bytes([(~sum(fields)) & 255])
        if self.hook:
            self.hook(self, ins, sid, address, params)
        return len(packet)
    def read(self, size):
        result, self.pending = self.pending[:size], self.pending[size:]
        return result
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class CenterTests(unittest.TestCase):
    def make(self, hook=None, factory=None):
        clock = Clock()
        port = FakeSerial(clock, hook)
        messages = []
        directory = tempfile.TemporaryDirectory(prefix='center_mock_')
        self.addCleanup(directory.cleanup)
        args = {} if factory is None else {'journal_factory': factory}
        control = center.GripperCenterController(
            center.CenterBus(port, clock=clock.time), journal_dir=directory.name,
            emit=messages.append, clock=clock.time, sleep=clock.sleep, **args)
        port.control = control
        return port, control, messages
    def assert_safe_writes(self, port, count128=None):
        self.assertTrue(all(sid == 6 and (address, value) in ((40, 0), (40, 128), (48, 0))
                            for sid, address, value in port.writes))
        self.assertLessEqual(port.writes.count((6, 40, 128)), 1)
        if count128 is not None:
            self.assertEqual(port.writes.count((6, 40, 128)), count128)
    def assert_released(self, port, control):
        self.assertEqual(port.regs[6][40], 0)
        self.assertEqual(port.get(6, 48), 0)
        self.assertTrue(control.release_result['confirmed'])
    def journal_lines(self, control):
        return [json.loads(line) for line in control.journal.path.read_text().splitlines()]

    def test_preview_is_zero_writes_and_preserves_all_registers(self):
        port, control, _ = self.make()
        original = {sid: bytes(r) for sid, r in port.regs.items()}
        result = control.run()
        self.assertEqual(result['status'], 'read_only_preview')
        self.assertEqual(result['before']['position'], 194)
        self.assertEqual(result['before']['config']['offset_raw'], 0)
        self.assertEqual(len(bytes.fromhex(result['before']['control']['raw_hex'])), 16)
        self.assertEqual(result['before']['control']['goal_position_raw'], 999)
        self.assertEqual(result['before']['control']['lock'], 1)
        self.assertEqual(result['writes_sent'], 0)
        self.assertIsNone(control.journal)
        self.assertEqual(original, {sid: bytes(r) for sid, r in port.regs.items()})

    def test_apply_success_only_translates_offset_and_historical_endpoints(self):
        port, control, _ = self.make()
        original = {sid: bytes(r) for sid, r in port.regs.items()}
        result = control.run(True)
        self.assertEqual(result['status'], 'centering_readback_verified')
        self.assert_safe_writes(port, 1)
        self.assert_released(port, control)
        self.assertEqual(result['after']['config']['offset_raw'], 0x0F3E)
        self.assertEqual(result['after']['config']['offset_signed'], -1854)
        self.assertEqual(result['after']['position'], 2048)
        self.assertEqual(result['after']['predicted_observed_closed']['new_count'], 1711)
        self.assertEqual(result['after']['predicted_observed_open']['new_count'], 2048)
        self.assertFalse(result['power_cycle_persistence_verified'])
        self.assertFalse(result['mechanical_limits_verified'])
        self.assertFalse(result['grasp_verified'])
        for sid in range(1, 6):
            self.assertEqual(bytes(port.regs[sid]), original[sid])
        for address in range(40):
            if address not in (31, 32):
                self.assertEqual(port.regs[6][address], original[6][address])
        lines = self.journal_lines(control)
        self.assertEqual(lines[0]['stage'], 'before_any_write')
        self.assertEqual(lines[0]['snapshot']['before']['config']['offset_raw'], 0)
        self.assertEqual(lines[-1]['snapshot']['status'], 'centering_readback_verified')

    def test_actual_anchor_does_not_overwrite_historical_open_observation(self):
        port, control, _ = self.make()
        port.put(6, 56, 196)
        after = control.run(True)['after']
        self.assertEqual(after['position'], 2048)
        self.assertEqual(after['predicted_observed_closed']['new_count'], 1709)
        self.assertEqual(after['predicted_observed_open']['new_count'], 2046)
        self.assertEqual(after['predicted_observed_open']['new_count']-
                         after['predicted_observed_closed']['new_count'], 337)
        self.assertTrue(after['predicted_observed_open']['not_for_motion'])
        self.assertFalse(after['predicted_observed_open']['measured_after_centering'])

    def test_quantized_center_and_translation_within_two_counts_are_allowed(self):
        for position, extra in ((2047, 0), (2048, 1), (2048, -2)):
            with self.subTest(position=position, extra=extra):
                port, control, _ = self.make()
                port.center_position, port.center_offset_extra = position, extra
                self.assertEqual(control.run(True)['status'], 'centering_readback_verified')
                self.assert_released(port, control)

    def test_preflight_rejects_every_existing_enabled_joint_without_writes(self):
        for sid in center.ALL_IDS:
            with self.subTest(sid=sid):
                port, control, _ = self.make()
                port.regs[sid][40] = 1
                with self.assertRaises(center.CenterError):
                    control.run(True)
                self.assertEqual(port.writes, [])
                self.assertEqual(port.regs[sid][40], 1)

    def test_preflight_configuration_rejections_are_zero_write(self):
        cases = [(0, 4), (3, 8), (2, 1), (5, 7), (8, 0), (18, 0),
                 (30, 2), (33, 1), (9, 1), (11, 0), (31, 1), (32, 0x10)]
        for address, value in cases:
            with self.subTest(address=address, value=value):
                port, control, _ = self.make()
                port.regs[6][address] = value
                with self.assertRaises(center.CenterError):
                    control.run(True)
                self.assertEqual(port.writes, [])

    def test_preflight_health_speed_output_and_open_window_rejections(self):
        cases = [(62, 59, 1), (62, 75, 1), (63, 55, 1), (65, 1, 1),
                 (58, 1, 2), (58, 0x8001, 2), (60, 1, 2), (60, 0x401, 2),
                 (56, 185, 2), (56, 203, 2), (56, 4096, 2), (56, 0x8001, 2)]
        for address, value, width in cases:
            with self.subTest(address=address, value=value):
                port, control, _ = self.make()
                if width == 2:
                    port.put(6, address, value)
                else:
                    port.regs[6][address] = value
                with self.assertRaises(center.CenterError):
                    control.run(True)
                self.assertEqual(port.writes, [])

    def test_stability_range_over_two_refuses_without_writes(self):
        port, control, _ = self.make()
        def drift(seconds):
            port.clock.sleep(seconds)
            port.put(6, 56, 197)
        control.sleep = drift
        with self.assertRaisesRegex(center.CenterError, '稳定'):
            control.run(True)
        self.assertEqual(port.writes, [])

    def test_stable_open_window_boundaries_allowed(self):
        for position in (186, 202):
            with self.subTest(position=position):
                port, control, _ = self.make()
                port.put(6, 56, position)
                self.assertEqual(control.run(True)['status'], 'centering_readback_verified')

    def test_bad_response_checks_are_zero_write(self):
        def checksum(p):
            p.pending = p.pending[:-1]+bytes([p.pending[-1]^1])
        for damage in (lambda p: setattr(p, 'pending', b''), checksum,
                       lambda p: setattr(p, 'pending', b'\x00'+p.pending[1:]),
                       lambda p: setattr(p, 'pending', p.pending[:2]+b'\x07'+p.pending[3:])):
            with self.subTest(damage=damage):
                port, control, _ = self.make(lambda p, *args: damage(p))
                with self.assertRaises(center.CenterError):
                    control.run(True)
                self.assertEqual(port.writes, [])

    def test_durable_before_state_and_intent_exist_when_128_is_transmitted(self):
        inspected = []
        def inspect(p, ins, sid, address, params):
            if (ins, sid, address, params) == (3, 6, 40, b'\x80'):
                lines = self.journal_lines(p.control)
                self.assertEqual(lines[0]['stage'], 'before_any_write')
                self.assertEqual(lines[0]['snapshot']['before']['config']['offset_raw'], 0)
                self.assertEqual(lines[-1]['stage'], 'centering_write_intent_before_transport')
                self.assertTrue(lines[-1]['snapshot']['center_write_intent']['attempted'])
                inspected.append(True)
        port, control, _ = self.make(inspect)
        control.run(True)
        self.assertEqual(inspected, [True])

    def test_initial_fsync_failure_never_enables_or_writes(self):
        port, control, _ = self.make()
        with patch.object(center.os, 'fsync', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                control.run(True)
        self.assertEqual(port.writes, [])
        self.assertFalse(control.bus.apply_enabled)

    def failing_factory(self, stages):
        def create(directory, snapshot):
            journal = center.DurableJournal.create(directory, snapshot)
            original = journal.append
            def append(stage, report):
                if stage in stages:
                    with patch.object(center.os, 'fsync', side_effect=OSError('disk unavailable')):
                        original(stage, report)
                else:
                    original(stage, report)
            journal.append = append
            return journal
        return create

    def test_fsync_before_128_blocks_128_but_cleanup_runs(self):
        port, control, _ = self.make(factory=self.failing_factory(
            {'centering_write_intent_before_transport', 'final_readback_and_cleanup'}))
        with self.assertRaises(OSError):
            control.run(True)
        self.assert_safe_writes(port, 0)
        self.assert_released(port, control)
        self.assertIsNotNone(control.journal_error)

    def test_fsync_failure_after_effect_never_prevents_cleanup_or_retries(self):
        for stages in ({'centering_command_acknowledged', 'final_readback_and_cleanup'},
                       {'final_readback_and_cleanup'}):
            with self.subTest(stages=stages):
                port, control, messages = self.make(factory=self.failing_factory(stages))
                with self.assertRaises((OSError, center.CenterError)):
                    control.run(True)
                self.assert_safe_writes(port, 1)
                self.assert_released(port, control)
                self.assertTrue(control.readback_verified)
                self.assertFalse(any('读回通过' in msg for msg in messages))
                if stages == {'final_readback_and_cleanup'}:
                    self.assertEqual(control.status, 'journal_save_failed')

    def test_lost_128_ack_is_uncertain_even_if_after_readback_matches(self):
        def lose(p, ins, sid, address, params):
            if (ins, address, params) == (3, 40, b'\x80'):
                p.pending = b''
        port, control, messages = self.make(lose)
        with self.assertRaisesRegex(center.CenterError, '超时'):
            control.run(True)
        self.assert_safe_writes(port, 1)
        self.assert_released(port, control)
        self.assertEqual(control.status, 'centering_uncertain')
        self.assertTrue(control.readback_verified)
        self.assertFalse(control.center_acknowledged)
        self.assertFalse(any('读回通过' in msg for msg in messages))

    def test_lost_first_mutation_ack_still_cleans_up_and_never_centers(self):
        lost = []
        def lose(p, ins, sid, address, params):
            if ins == 3 and not lost:
                lost.append(True)
                p.pending = b''
        port, control, _ = self.make(lose)
        with self.assertRaises(center.CenterError):
            control.run(True)
        self.assert_safe_writes(port, 0)
        self.assert_released(port, control)

    def test_interrupt_after_mutation_always_cleans_up(self):
        for after128 in (False, True):
            with self.subTest(after128=after128):
                fired = []
                def interrupt(p, ins, sid, address, params):
                    if ins == 3 and not fired and (not after128 or params == b'\x80'):
                        fired.append(True)
                        raise KeyboardInterrupt
                port, control, _ = self.make(interrupt)
                with self.assertRaises(KeyboardInterrupt):
                    control.run(True)
                self.assert_safe_writes(port, int(after128))
                self.assert_released(port, control)

    def test_cleanup_attempts_both_writes_and_reads_after_first_failure(self):
        def fail(p, ins, sid, address, params):
            if p.control.bus.phase == 'release' and address == 40:
                p.pending = b''
        port, control, _ = self.make(fail)
        with self.assertRaisesRegex(center.CenterError, '清零未确认'):
            control.run(True)
        release = [event for event in control.bus.events if event['phase'] == 'release']
        self.assertEqual([(e['instruction'], bytes.fromhex(e['parameters_hex'])[0]) for e in release],
                         [(3, 40), (3, 48), (2, 40), (2, 48)])
        self.assertEqual(control.status, 'release_unconfirmed')
        self.assertEqual(control.release_result['readback']['48'], 0)
        self.assert_safe_writes(port, 1)

    def test_changed_non_offset_configuration_is_rejected(self):
        def alter(p, ins, sid, address, params):
            if (ins, address, params) == (3, 40, b'\x80'):
                p.regs[6][21] = 99
        port, control, _ = self.make(alter)
        with self.assertRaisesRegex(center.CenterError, '非offset'):
            control.run(True)
        self.assert_released(port, control)
        self.assertEqual(control.after['unexpected_changed_config_addresses'], [21])
        self.assertFalse(control.readback_verified)

    def test_wrong_offset_translation_or_no_offset_is_rejected(self):
        for extra in (3, -3):
            with self.subTest(extra=extra):
                port, control, _ = self.make()
                port.center_offset_extra = extra
                with self.assertRaisesRegex(center.CenterError, '平移不一致'):
                    control.run(True)
                self.assert_released(port, control)
        def no_offset(p, ins, sid, address, params):
            if (ins, address, params) == (3, 40, b'\x80'):
                p.put(6, 31, 0)
        port, control, _ = self.make(no_offset)
        with self.assertRaisesRegex(center.CenterError, 'offset 未改变'):
            control.run(True)
        self.assert_released(port, control)

    def test_after_center_outside_two_counts_rejected_and_not_restored(self):
        port, control, _ = self.make()
        port.center_position = 2045
        with self.assertRaisesRegex(center.CenterError, '2048±2'):
            control.run(True)
        self.assert_released(port, control)
        self.assertNotEqual(port.get(6, 31), 0)
        self.assert_safe_writes(port, 1)

    def test_rerun_nonzero_offset_rejects_without_any_new_write(self):
        port, control, _ = self.make()
        control.run(True)
        writes_before = list(port.writes)
        new = center.GripperCenterController(center.CenterBus(port, clock=port.clock.time),
            journal_dir=control.journal_dir, emit=lambda x: None,
            clock=port.clock.time, sleep=port.clock.sleep)
        with self.assertRaisesRegex(center.CenterError, 'offset 已非0'):
            new.run(True)
        self.assertEqual(port.writes, writes_before)

    def test_bus_rejects_all_unlisted_write_capabilities_and_double_128(self):
        port, control, _ = self.make()
        bus = control.bus
        with self.assertRaises(center.CenterError):
            bus.write(6, 40, 0)
        bus.enable_apply()
        for sid, address, value in ((1,40,0), (5,48,0), (6,40,1), (6,48,1),
                                    (6,42,194), (6,31,3902), (6,55,0), (6,33,0), (6,9,0)):
            with self.subTest(sid=sid, address=address, value=value):
                with self.assertRaises(center.CenterError):
                    bus.write(sid, address, value)
        for ins, params in ((4,[40,128]), (5,[]), (3,[40,0,0]), (3,[48,0]), (3,[40,1])):
            with self.assertRaises(center.CenterError):
                bus._exchange(6, ins, params, 0)
        self.assertEqual(port.writes, [])
        bus.write(6,40,128)
        with self.assertRaisesRegex(center.CenterError, '禁止重试'):
            bus.write(6,40,128)
        self.assert_safe_writes(port,1)

    def test_recheck_detects_prepare_drift_before_128(self):
        def drift(p, ins, sid, address, params):
            if ins == 3 and address == 48:
                p.put(6,56,197)
        port, control, _ = self.make(drift)
        with self.assertRaisesRegex(center.CenterError, '准备期间漂移'):
            control.run(True)
        self.assert_safe_writes(port,0)
        self.assert_released(port, control)

    def test_help_never_imports_pyserial_and_cli_has_no_motion_options(self):
        actual_import = builtins.__import__
        def guard(name, *args, **kwargs):
            self.assertNotEqual(name, 'serial')
            return actual_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guard), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                center.main(['--help'])
            self.assertEqual(raised.exception.code,0)
        for option in ('--enable', '--joint', '--delta', '--limit'):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                center.main([option, '1'])

    def test_cli_preview_uses_exclusive_bounded_serial_and_prints_before(self):
        port, control, _ = self.make()
        options = []
        def serial_factory(*args, **kwargs):
            options.append(kwargs)
            return port
        def controller_factory(bus, **kwargs):
            # Retain injected clock while exercising the actual CLI and bus.
            bus.clock, bus.started = port.clock.time, port.clock.time()
            control.bus = bus
            return control
        output = io.StringIO()
        with patch.dict(sys.modules, {'serial': types.SimpleNamespace(Serial=serial_factory)}), \
             patch.object(center,'GripperCenterController', side_effect=controller_factory), \
             redirect_stdout(output):
            self.assertEqual(center.main([]),0)
        self.assertEqual(options[0], {'baudrate':1000000, 'timeout':.15,
                                     'write_timeout':.15, 'exclusive':True})
        self.assertEqual(port.writes,[])
        self.assertIn('"before": {',output.getvalue())
        self.assertIn('"offset_raw": 0',output.getvalue())


if __name__ == '__main__':
    unittest.main()
