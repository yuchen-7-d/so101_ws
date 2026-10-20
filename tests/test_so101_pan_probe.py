"""Negative-only pan probe tests. All serial traffic goes to FakeSerial."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_pan_step import PanStepController, main, save_report
from test_so101_hold_current import Clock, FakeSerial


class PanProbeTests(unittest.TestCase):
    def setup_probe(self, position=4082, delta=16, fault=None, move=True):
        clock = Clock()

        def plant(port, instruction, sid, address, params):
            if (move and instruction == 3 and address == GOAL
                    and port.regs[sid][TORQUE] == 1
                    and port.get(sid, OUTPUT)):
                port.put(sid, 56, int.from_bytes(params, "little"))
            if fault:
                fault(port, instruction, sid, address, params)

        port = FakeSerial(clock, plant)
        port.allow_live_goal_writes = True
        port.put(1, 56, position)
        controller = PanStepController(
            Bus(port), delta=delta, probe_negative=True,
            emit=lambda message: None, clock=clock.time, sleep=clock.sleep,
        )
        return port, controller

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def test_probe_preview_has_no_writes_and_forces_negative_eight(self):
        for requested_delta in (16, -16):
            with self.subTest(requested_delta=requested_delta):
                port, controller = self.setup_probe(delta=requested_delta)
                controller.run(False)
                self.assertEqual(port.writes, [])
                self.assertEqual(controller.delta, -8)
                self.assertEqual(controller.path, [4078, 4074])
                self.assertEqual(controller.return_path, [])
                self.assertFalse(controller.outbound_complete)
                self.assertFalse(controller.probe_hold_complete)

    def test_probe_moves_only_negative_and_never_returns(self):
        port, controller = self.setup_probe()
        controller.run(True)
        self.assertEqual(
            [value for sid, address, value in port.writes
             if sid == 1 and address == GOAL],
            [4082, 4078, 4074],
        )
        self.assertEqual(port.get(1, 56), 4074)
        self.assertTrue(controller.outbound_complete)
        self.assertTrue(controller.probe_hold_complete)
        self.assertFalse(controller.return_complete)
        self.assertFalse(controller.final_anchor_hold_complete)
        self.assert_released(port)

    def test_other_joint_goals_and_gripper_are_unchanged(self):
        port, controller = self.setup_probe()
        initial_positions = {sid: port.get(sid, 56) for sid in ARM_IDS}
        original_registers = {
            sid: bytes(registers) for sid, registers in port.regs.items()
        }
        controller.run(True)
        for sid in ARM_IDS[1:]:
            self.assertEqual(
                [value for written_sid, address, value in port.writes
                 if written_sid == sid and address == GOAL],
                [initial_positions[sid]],
            )
            self.assertEqual(port.get(sid, 56), initial_positions[sid])
        self.assertEqual(bytes(port.regs[6]), original_registers[6])
        self.assertFalse(any(sid == 6 for sid, _, _ in port.writes))
        for sid in ARM_IDS:
            self.assertEqual(bytes(port.regs[sid][:40]), original_registers[sid][:40])
        self.assertLessEqual(
            max(value for _, address, value in port.writes if address == OUTPUT), 100
        )
        self.assert_released(port)

    def test_first_step_overshoot_keeps_signed_fault_frame_and_releases(self):
        def overshoot(port, instruction, sid, address, params):
            if (instruction == 3 and sid == 1 and address == GOAL
                    and int.from_bytes(params, "little") == 4078):
                port.put(1, 56, 4075)  # First-step corridor starts at 4076.
                port.put(1, 58, 0x8000 | 17)
                port.put(1, 60, 0x0400 | 68)

        port, controller = self.setup_probe(fault=overshoot)
        with self.assertRaises(ControlError):
            controller.run(True)
        sample = controller.fault_context["last_feedback_by_id"][1]
        self.assertEqual(sample["position"], 4075)
        self.assertEqual(sample["expected_target"], 4078)
        self.assertEqual(sample["speed_raw_signed"], -17)
        self.assertEqual(sample["output_raw_signed"], -68)
        self.assertEqual(
            [value for sid, address, value in port.writes
             if sid == 1 and address == GOAL],
            [4082, 4078],
        )
        self.assertFalse(controller.outbound_complete)
        self.assertFalse(controller.probe_hold_complete)
        self.assertFalse(controller.return_complete)
        # The bad pan reply must abort before reading another joint.
        last_feedback_index = max(
            index for index, event in enumerate(port.events)
            if event[:3] == (2, 1, 56)
        )
        self.assertEqual(port.events[last_feedback_index + 1][:3], (3, 1, TORQUE))
        self.assert_released(port)

    def test_every_probe_target_requires_boundary_margin_before_any_write(self):
        # Include an unsafe intermediate target with a safe final target.
        cases = (
            (23, 0, 4095),       # Final target 15 is too close to lower bound.
            (4084, 0, 4095),    # First target 4080 has only 15 counts above it.
            (1023, 1000, 1095),
            (1084, 1000, 1095),
        )
        for position, lower, upper in cases:
            with self.subTest(position=position, lower=lower, upper=upper):
                port, controller = self.setup_probe(position=position)
                port.put(1, 9, lower)
                port.put(1, 11, upper)
                with self.assertRaises(ControlError):
                    controller.run(True)
                self.assertEqual(port.writes, [])

    def test_exact_sixteen_count_target_margin_is_allowed(self):
        port, controller = self.setup_probe(position=24)
        controller.run(False)
        self.assertEqual(controller.path, [20, 16])
        self.assertEqual(port.writes, [])

    def test_probe_hold_drift_does_not_claim_success_or_command_return(self):
        port, controller = self.setup_probe()
        previous_hook = port.hook

        def drift(port, instruction, sid, address, params):
            previous_hook(port, instruction, sid, address, params)
            if (controller.phase == "probe_hold" and instruction == 2
                    and sid == 1 and address == 56):
                port.put(1, 56, 4077)  # Inside overall range, 3 from endpoint.
                payload = bytes(port.regs[1][56:66])
                fields = bytes([1, len(payload) + 2, 0]) + payload
                port.pending = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])

        port.hook = drift
        with self.assertRaises(ControlError):
            controller.run(True)
        self.assertTrue(controller.outbound_complete)
        self.assertFalse(controller.probe_hold_complete)
        self.assertFalse(controller.return_complete)
        self.assertFalse(controller.final_anchor_hold_complete)
        self.assertEqual(controller.fault_context["phase"], "probe_hold")
        self.assertEqual(
            controller.fault_context["last_feedback_by_id"][1]["position"], 4077
        )
        self.assertEqual(
            [value for sid, address, value in port.writes
             if sid == 1 and address == GOAL],
            [4082, 4078, 4074],
        )
        self.assert_released(port)

    def test_probe_completion_report_does_not_claim_roundtrip(self):
        port, controller = self.setup_probe()
        controller.run(True)
        with tempfile.TemporaryDirectory() as folder:
            path = save_report(controller, "completed_and_released", None, Path(folder))
            record = json.loads(path.read_text())
        self.assertEqual(record["planned_path"], [4078, 4074])
        self.assertTrue(record["outbound_complete"])
        self.assertTrue(record["probe_hold_complete"])
        self.assertFalse(record["return_complete"])
        self.assertFalse(record["final_anchor_hold_complete"])
        self.assertEqual(record["release_unconfirmed_ids"], [])
        self.assertFalse(record["model_mapping_verified"])

    def test_probe_mode_is_mutually_exclusive_with_enable_and_diagnose(self):
        for conflicting_mode in ("--enable", "--diagnose"):
            with self.subTest(conflicting_mode=conflicting_mode):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        main(["--probe-negative", conflicting_mode])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("not allowed with", stderr.getvalue())

    def test_probe_fixed_delta_cannot_be_overridden_from_cli(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                main(["--probe-negative", "--delta", "-16"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("固定为负向8计数", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
