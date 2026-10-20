"""Optional fast shoulder telemetry tested only against an in-memory servo bus."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_pan_step import PanStepController, main
from test_so101_hold_current import Clock, FakeSerial


class MotionTraceTests(unittest.TestCase):
    def setup_trace(self, *, move=True):
        clock = Clock()

        def plant(port, instruction, sid, address, params):
            if (move and instruction == 3 and sid == 2 and address == GOAL
                    and port.regs[sid][TORQUE] == 1 and port.get(sid, OUTPUT)):
                port.put(sid, 56, int.from_bytes(params, "little"))

        port = FakeSerial(clock, plant)
        port.allow_live_goal_writes = True
        for sid, position in enumerate((58, 3698, 4077, 845, 3301, 3951), 1):
            port.put(sid, 56, position)
        controller = PanStepController(
            Bus(port), moving_id=2, probe_shoulder=True, trace_motion=True,
            emit=lambda message: None, clock=clock.time, sleep=clock.sleep,
        )
        return port, controller

    @staticmethod
    def is_trace(instruction, sid, address, params):
        return (instruction, sid, address, bytes(params)) == (2, 2, 40, b"\x1f")

    @staticmethod
    def rewrite_trace_reply(port, updates):
        """Change only this response; ordinary feedback remains unchanged."""
        payload = bytearray(port.regs[2][40:71])
        for address, raw in updates.items():
            payload[address-40:address-40+len(raw)] = raw
        fields = bytes([2, len(payload)+2, 0]) + payload
        port.pending = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])

    def add_hook(self, port, hook):
        original = port.hook

        def chained(*args):
            original(*args)
            hook(*args)

        port.hook = chained

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    @staticmethod
    def goals(port, sid=2):
        return [value for servo, address, value in port.writes
                if servo == sid and address == GOAL]

    def assert_immediate_release_after_trace(self, port):
        last_trace = max(index for index, event in enumerate(port.events)
                         if self.is_trace(*event))
        self.assertEqual(port.events[last_trace+1][:3], (3, 1, TORQUE))
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assert_released(port)

    def test_read_only_preview_checks_long_packet_and_preserves_raw_fields(self):
        port, controller = self.setup_trace()
        port.put(2, 58, 0x8000 | 50)
        port.put(2, 60, 0x0400 | 68)
        port.regs[2][66] = 1
        port.put(2, 67, 3697)
        port.put(2, 69, 123)
        controller.run(False)
        self.assertEqual(port.writes, [])
        self.assertTrue(any(self.is_trace(*event) for event in port.events))
        sample = controller.motion_trace_reads[-1]
        self.assertEqual((sample["servo_id"], sample["phase"]), (2, "preflight"))
        self.assertEqual((sample["position"], sample["speed_raw_signed"], sample["output_raw_signed"]),
                         (3698, -50, -68))
        self.assertEqual((sample["torque_enabled"], sample["moving_flag"], sample["internal_target_raw"], sample["current_raw"]),
                         (0, 1, 3697, 123))
        self.assertEqual(sample["raw_hex"], bytes(port.regs[2][40:71]).hex())
        self.assertEqual(sample["feedback_raw_hex"], bytes(port.regs[2][56:66]).hex())
        matching = [entry for entry in controller.feedback_reads
                    if entry["servo_id"] == 2 and entry["elapsed_s"] == sample["elapsed_s"]]
        self.assertTrue(matching)
        self.assertEqual(matching[-1]["raw_hex"], sample["feedback_raw_hex"])
        self.assertFalse(controller.outbound_complete)

    def test_fake_cli_success_keeps_other_axes_monitored_and_reports_sampling(self):
        port, _ = self.setup_trace()
        controllers = []
        initial = {sid: bytes(registers) for sid, registers in port.regs.items()}
        real_controller = PanStepController

        def make_controller(*args, **kwargs):
            controller = real_controller(*args, **kwargs, clock=port.clock.time, sleep=port.clock.sleep)
            controllers.append(controller)
            return controller

        fake_serial = SimpleNamespace(Serial=None)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {"serial": fake_serial}), \
                patch.object(fake_serial, "Serial", return_value=contextlib.nullcontext(port)) as opening, \
                patch("so101_pan_step.PanStepController", side_effect=make_controller), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main(["--joint", "2", "--probe-shoulder", "--trace-motion", "--report-dir", folder])
            paths = list(Path(folder).glob("joint2_*.json"))
            self.assertEqual(len(paths), 1)
            report = json.loads(paths[0].read_text())
        self.assertEqual(result, 0)
        opening.assert_called_once()
        self.assertEqual(self.goals(port), [3698, 3702])
        for sid in (1, 3, 4, 5):
            self.assertEqual(self.goals(port, sid), [int.from_bytes(initial[sid][56:58], "little")])
        self.assertFalse(any(sid == 6 or address < 40 for sid, address, _ in port.writes))
        self.assertEqual(bytes(port.regs[6]), initial[6])
        self.assertLessEqual(max(value for _, address, value in port.writes if address == OUTPUT), 100)
        controller = controllers[0]
        for phase in ("outbound", "probe_hold"):
            self.assertEqual({entry["servo_id"] for entry in controller.feedback_reads if entry["phase"] == phase},
                             set(range(1, 7)))
        self.assertEqual(report["schema_version"], 7)
        self.assertTrue(report["trace_motion"] and report["probe_hold_complete"])
        self.assertFalse(report["return_complete"] or report["final_anchor_hold_complete"])
        self.assertGreater(len(report["motion_trace_reads"]), 100)
        timing = report["motion_trace_timing"]
        self.assertGreater(timing["sample_count"], 100)
        self.assertGreater(timing["min_interval_s"], 0)
        self.assertLessEqual(timing["min_interval_s"], timing["mean_interval_s"])
        self.assertLessEqual(timing["mean_interval_s"], timing["max_interval_s"])
        self.assert_released(port)

    def test_transient_overshoot_between_normal_reads_stops_on_trace_reply(self):
        port, controller = self.setup_trace()
        full_motion_reads = 0

        def transient(port, instruction, sid, address, params):
            nonlocal full_motion_reads
            if (controller.phase == "outbound" and port.get(2, GOAL) == 3702
                    and (instruction, sid, address) == (2, 2, 56)):
                full_motion_reads += 1
            if (controller.phase == "outbound" and full_motion_reads
                    and self.is_trace(instruction, sid, address, params)):
                self.rewrite_trace_reply(port, {56: (3705).to_bytes(2, "little"),
                                                58: (100).to_bytes(2, "little")})

        self.add_hook(port, transient)
        with self.assertRaises(ControlError):
            controller.run(True)
        self.assertEqual(port.get(2, 56), 3702)  # All normal packets would pass.
        trace = controller.motion_trace_reads[-1]
        self.assertEqual((trace["position"], trace["expected_target"]), (3705, 3702))
        self.assertEqual(controller.fault_context["last_feedback_by_id"][2]["position"], 3705)
        self.assertFalse(controller.outbound_complete or controller.return_complete)
        self.assert_immediate_release_after_trace(port)

    def test_trace_detects_torque_control_and_health_failures_without_retry(self):
        cases = ({40: b"\x00"}, {41: b"\x02"}, {42: (3703).to_bytes(2, "little")},
                 {46: b"\x00\x00"}, {48: (200).to_bytes(2, "little")},
                 {62: b"\x32"}, {63: b"\x3c"}, {65: b"\x01"})
        for changed_registers in cases:
            with self.subTest(changed_registers=changed_registers):
                port, controller = self.setup_trace()

                def corrupt_state(port, instruction, sid, address, params):
                    if controller.phase == "outbound" and self.is_trace(instruction, sid, address, params):
                        self.rewrite_trace_reply(port, changed_registers)

                self.add_hook(port, corrupt_state)
                with self.assertRaises(ControlError):
                    controller.run(True)
                self.assertFalse(controller.probe_hold_complete)
                self.assert_immediate_release_after_trace(port)

    def test_fast_and_normal_samples_share_continuous_saturation_timer(self):
        port, controller = self.setup_trace(move=False)

        def saturate(port, instruction, sid, address, params):
            if (instruction == 3 and sid == 2 and address == GOAL
                    and int.from_bytes(params, "little") == 3702):
                port.put(2, 60, 0x0400 | 100)

        self.add_hook(port, saturate)
        with self.assertRaisesRegex(ControlError, "输出持续接近本次上限"):
            controller.run(True)
        saturated = [entry for entry in controller.feedback_reads
                     if entry["servo_id"] == 2 and entry["output_raw_signed"] == -100]
        self.assertGreaterEqual(saturated[-1]["elapsed_s"] - saturated[0]["elapsed_s"], 1)
        self.assertLess(saturated[-1]["elapsed_s"] - saturated[0]["elapsed_s"], 1.1)
        self.assertGreater(len([entry for entry in controller.motion_trace_reads
                               if entry["phase"] == "outbound"]), 20)
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assertEqual(port.get(2, 56), 3698)
        self.assertFalse(controller.outbound_complete or controller.return_complete)
        self.assert_released(port)

    def test_trace_slow_reply_timeout_or_checksum_failure_releases(self):
        for fault in ("slow_valid", "timeout", "checksum"):
            with self.subTest(fault=fault):
                port, controller = self.setup_trace()

                def fail_read(port, instruction, sid, address, params):
                    if controller.phase == "outbound" and self.is_trace(instruction, sid, address, params):
                        if fault == "slow_valid":
                            port.clock.now += .051
                        elif fault == "timeout":
                            port.pending = b""
                        else:
                            port.pending = port.pending[:-1] + bytes([port.pending[-1] ^ 1])

                self.add_hook(port, fail_read)
                with self.assertRaises(ControlError):
                    controller.run(True)
                self.assertFalse(controller.probe_hold_complete or controller.return_complete)
                self.assert_immediate_release_after_trace(port)

    def test_unsupported_long_packet_is_rejected_before_any_control_write(self):
        for failure in ("timeout", "checksum"):
            with self.subTest(failure=failure):
                port, controller = self.setup_trace()

                def unsupported(port, instruction, sid, address, params):
                    if self.is_trace(instruction, sid, address, params):
                        if failure == "timeout":
                            port.pending = b""
                        else:
                            port.pending = port.pending[:-1] + bytes([port.pending[-1] ^ 1])

                self.add_hook(port, unsupported)
                with self.assertRaises(ControlError):
                    controller.run(True)
                self.assertEqual(port.writes, [])
                self.assertFalse(controller.touched)

    def test_out_of_arrival_band_fast_samples_reset_three_cycle_arrival(self):
        port, controller = self.setup_trace()
        full_cycles = 0

        def oscillate(port, instruction, sid, address, params):
            nonlocal full_cycles
            if controller.phase != "outbound" or port.get(2, GOAL) != 3702:
                return
            if (instruction, sid, address) == (2, 2, 56):
                full_cycles += 1
            if self.is_trace(instruction, sid, address, params) and full_cycles <= 4:
                self.rewrite_trace_reply(port, {56: (3699).to_bytes(2, "little")})

        self.add_hook(port, oscillate)
        controller.run(True)
        self.assertGreaterEqual(full_cycles, 7)
        self.assertTrue(controller.probe_hold_complete)
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assert_released(port)

    def test_other_axis_drift_remains_monitored_during_fast_sampling(self):
        port, controller = self.setup_trace()

        def other_joint_drift(port, instruction, sid, address, params):
            if controller.phase == "outbound" and self.is_trace(instruction, sid, address, params):
                port.put(3, 56, 4064)  # Thirteen counts away from its hold anchor.

        self.add_hook(port, other_joint_drift)
        with self.assertRaisesRegex(ControlError, "ID 3:.*偏离保持点"):
            controller.run(True)
        self.assertEqual(self.goals(port, 3), [4077])
        self.assertFalse(controller.probe_hold_complete or controller.return_complete)
        self.assert_released(port)

    def test_trace_api_and_cli_require_explicit_shoulder_probe_before_io(self):
        cases = ({}, {"moving_id": 2, "delta": 8},
                 {"moving_id": 1, "probe_negative": True},
                 {"moving_id": 1, "probe_shoulder": True})
        for parameters in cases:
            with self.subTest(parameters=parameters):
                port = FakeSerial(Clock())
                with self.assertRaises(ValueError):
                    PanStepController(Bus(port), trace_motion=True, **parameters)
                self.assertEqual(port.events, [])
        fake_serial = SimpleNamespace(Serial=None)
        cli_cases = (["--trace-motion"], ["--joint", "2", "--trace-motion"],
                     ["--joint", "2", "--enable", "--trace-motion"],
                     ["--joint", "2", "--diagnose", "--trace-motion"],
                     ["--probe-negative", "--trace-motion"])
        with patch.dict(sys.modules, {"serial": fake_serial}), \
                patch.object(fake_serial, "Serial", side_effect=AssertionError("no serial open")) as opening:
            for arguments in cli_cases:
                with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(arguments)
                    self.assertEqual(error.exception.code, 2)
            opening.assert_not_called()


if __name__ == "__main__":
    unittest.main()
