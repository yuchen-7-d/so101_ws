"""One-step shoulder probe acceptance and failure tests; no real serial access."""

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


class ShoulderProbeTests(unittest.TestCase):
    def setup_probe(self, *, position=3698, delta=16, move=True, fault=None):
        clock = Clock()

        def plant(port, instruction, sid, address, params):
            if (move and instruction == 3 and sid == 2 and address == GOAL
                    and port.regs[sid][TORQUE] == 1 and port.get(sid, OUTPUT)):
                port.put(sid, 56, int.from_bytes(params, "little"))
            if fault:
                fault(port, instruction, sid, address, params)

        port = FakeSerial(clock, plant)
        port.allow_live_goal_writes = True
        for sid, count in enumerate((58, position, 4077, 845, 3301, 3951), 1):
            port.put(sid, 56, count)
        controller = PanStepController(
            Bus(port), moving_id=2, probe_shoulder=True, delta=delta,
            emit=lambda message: None, clock=clock.time, sleep=clock.sleep,
        )
        return port, controller

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    @staticmethod
    def goals(port, servo_id=2):
        return [value for sid, address, value in port.writes
                if sid == servo_id and address == GOAL]

    @staticmethod
    def replace_feedback_reply(port, sid):
        payload = bytes(port.regs[sid][56:66])
        fields = bytes([sid, len(payload) + 2, 0]) + payload
        port.pending = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])

    def test_preview_uses_fresh_pose_fixed_four_counts_and_never_writes(self):
        for requested_delta, position in ((16, 3698), (-16, 800)):
            with self.subTest(delta=requested_delta, position=position):
                port, controller = self.setup_probe(position=position, delta=requested_delta)
                controller.run(False)
                self.assertEqual(port.writes, [])
                self.assertEqual(controller.anchor, position)
                self.assertEqual(controller.delta, 4)
                self.assertEqual(controller.path, [position + 4])
                self.assertEqual(controller.return_path, [])
                self.assertEqual(controller.start_position_counts[2], position)
                self.assertFalse(controller.outbound_complete or controller.probe_hold_complete)

    def test_success_changes_only_one_shoulder_target_holds_and_releases(self):
        port, controller = self.setup_probe()
        original = {sid: bytes(registers) for sid, registers in port.regs.items()}
        initial_positions = {sid: port.get(sid, 56) for sid in ARM_IDS}
        controller.run(True)
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assertEqual(port.get(2, 56), 3702)
        for sid in (1, 3, 4, 5):
            self.assertEqual(self.goals(port, sid), [initial_positions[sid]])
            self.assertEqual(port.get(sid, 56), initial_positions[sid])
        for sid in ARM_IDS:
            self.assertEqual(bytes(port.regs[sid][:40]), original[sid][:40])
        self.assertEqual(bytes(port.regs[6]), original[6])
        self.assertFalse(any(sid == 6 for sid, _, _ in port.writes))
        self.assertLessEqual(max(value for _, address, value in port.writes if address == OUTPUT), 100)
        self.assertEqual([entry["target"] for entry in controller.reached], [3698, 3702])
        hold_times = [entry["elapsed_s"] for entry in controller.feedback_reads
                      if entry["servo_id"] == 2 and entry["phase"] == "probe_hold"]
        self.assertGreaterEqual(hold_times[-1] - hold_times[0], 1.9)
        self.assert_released(port)
        report = json.loads(json.dumps(controller.report("completed_and_released", None)))
        self.assertEqual(report["schema_version"], 6)
        self.assertEqual(report["test"], "joint_single_step_probe")
        self.assertTrue(report["probe_shoulder"])
        self.assertTrue(report["outbound_complete"] and report["probe_hold_complete"])
        self.assertFalse(report["return_complete"] or report["final_anchor_hold_complete"])
        self.assertEqual(report["planned_path"], [3702])
        self.assertEqual(report["release_unconfirmed_ids"], [])
        self.assertFalse(report["model_mapping_verified"])

    def test_reported_stuck_shoulder_saturates_then_stops_without_retry(self):
        def saturate(port, instruction, sid, address, params):
            if (instruction == 3 and sid == 2 and address == GOAL
                    and int.from_bytes(params, "little") == 3702):
                port.put(2, 60, 0x0400 | 100)

        port, controller = self.setup_probe(move=False, fault=saturate)
        with self.assertRaisesRegex(ControlError, "输出持续接近本次上限"):
            controller.run(True)
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assertEqual(port.get(2, 56), 3698)
        self.assertLessEqual(max(value for _, address, value in port.writes if address == OUTPUT), 100)
        saturated = [entry for entry in controller.feedback_reads
                     if entry["servo_id"] == 2 and entry["output_raw_signed"] == -100]
        self.assertGreaterEqual(saturated[-1]["elapsed_s"] - saturated[0]["elapsed_s"], 1)
        self.assertLess(saturated[-1]["elapsed_s"] - saturated[0]["elapsed_s"], 1.2)
        sample = controller.fault_context["last_feedback_by_id"][2]
        self.assertEqual((sample["position"], sample["expected_target"], sample["output_raw_signed"]),
                         (3698, 3702, -100))
        self.assertFalse(controller.outbound_complete or controller.probe_hold_complete)
        self.assertFalse(controller.return_complete)
        self.assert_released(port)

    def test_three_count_overshoot_aborts_on_first_bad_reply_and_releases(self):
        def overshoot(port, instruction, sid, address, params):
            if (instruction == 3 and sid == 2 and address == GOAL
                    and int.from_bytes(params, "little") == 3702):
                port.put(2, 56, 3705)
                port.put(2, 58, 100)
                port.put(2, 60, 0x0400 | 28)

        port, controller = self.setup_probe(fault=overshoot)
        with self.assertRaisesRegex(ControlError, "ID2 当前小步反馈越界"):
            controller.run(True)
        sample = controller.fault_context["last_feedback_by_id"][2]
        self.assertEqual((sample["position"], sample["speed_raw_signed"], sample["output_raw_signed"]),
                         (3705, 100, -28))
        last_feedback_index = max(index for index, event in enumerate(port.events)
                                  if event[:3] == (2, 2, 56))
        self.assertEqual(port.events[last_feedback_index + 1][:3], (3, 1, TORQUE))
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assertFalse(controller.outbound_complete or controller.return_complete)
        self.assert_released(port)

    def test_final_hold_drift_inside_overall_corridor_still_fails(self):
        port, controller = self.setup_probe()
        original_hook = port.hook

        def drift(port, instruction, sid, address, params):
            original_hook(port, instruction, sid, address, params)
            if (controller.phase == "probe_hold" and instruction == 2
                    and sid == 2 and address == 56):
                port.put(2, 56, 3699)  # In the full corridor, 3 short of endpoint.
                self.replace_feedback_reply(port, 2)

        port.hook = drift
        with self.assertRaisesRegex(ControlError, "偏离末目标超过2计数"):
            controller.run(True)
        self.assertTrue(controller.outbound_complete)
        self.assertFalse(controller.probe_hold_complete or controller.return_complete)
        self.assertEqual(controller.fault_context["phase"], "probe_hold")
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assert_released(port)

    def test_applied_goal_with_lost_ack_or_interrupt_releases_without_return(self):
        for fault_kind in ("lost_ack", "interrupt"):
            with self.subTest(fault=fault_kind):
                def fail(port, instruction, sid, address, params):
                    if (instruction == 3 and sid == 2 and address == GOAL
                            and int.from_bytes(params, "little") == 3702):
                        if fault_kind == "lost_ack":
                            port.pending = b""
                        else:
                            raise KeyboardInterrupt

                port, controller = self.setup_probe(fault=fail)
                error_class = ControlError if fault_kind == "lost_ack" else KeyboardInterrupt
                with self.assertRaises(error_class):
                    controller.run(True)
                self.assertEqual(port.get(2, 56), 3702)
                self.assertEqual(self.goals(port), [3698, 3702])
                self.assertEqual(controller.fault_context["targets"][2], 3702)
                self.assertFalse(controller.outbound_complete or controller.return_complete)
                self.assert_released(port)

    def test_probe_target_requires_sixteen_count_device_margin_before_write(self):
        for position, lower, upper in ((4076, 0, 4095), (1011, 1000, 1100),
                                       (3731, 0, 3750), (4094, 0, 4095)):
            with self.subTest(position=position, limits=(lower, upper)):
                port, controller = self.setup_probe(position=position)
                port.put(2, 9, lower)
                port.put(2, 11, upper)
                with self.assertRaises(ControlError):
                    controller.run(True)
                self.assertEqual(port.writes, [])
                self.assertFalse(controller.touched)

    def test_exact_sixteen_count_target_margin_is_allowed_read_only(self):
        for position in (12, 4075):
            with self.subTest(position=position):
                port, controller = self.setup_probe(position=position)
                controller.run(False)
                self.assertEqual(controller.path, [position + 4])
                self.assertEqual(port.writes, [])

    def test_invalid_probe_parameters_are_rejected_without_bus_access(self):
        cases = ({"moving_id": 1}, {"moving_id": 2, "probe_negative": True},
                 {"moving_id": 2, "start_range": (3600, 3800)})
        for parameters in cases:
            with self.subTest(parameters=parameters):
                port = FakeSerial(Clock())
                with self.assertRaises(ValueError):
                    PanStepController(Bus(port), probe_shoulder=True, **parameters)
                self.assertEqual(port.events, [])

    def test_cli_rejects_incompatible_modes_or_parameters_before_serial_open(self):
        cases = (["--probe-shoulder"],
                 ["--joint", "1", "--probe-shoulder"],
                 *[["--joint", "2", "--probe-shoulder", flag]
                   for flag in ("--enable", "--diagnose", "--probe-negative")],
                 ["--joint", "2", "--probe-shoulder", "--delta", "8"],
                 ["--joint", "2", "--probe-shoulder", "--delta", "4"],
                 ["--joint", "2", "--probe-shoulder", "--start-range", "3600", "3800"])
        fake_serial = SimpleNamespace(Serial=None)
        with patch.dict(sys.modules, {"serial": fake_serial}), \
                patch.object(fake_serial, "Serial", side_effect=AssertionError("serial must not open")) as opening:
            for arguments in cases:
                with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(arguments)
                    self.assertEqual(error.exception.code, 2)
            opening.assert_not_called()

    def test_cli_probe_enables_one_step_and_saves_truthful_report_using_fake_serial(self):
        port, _ = self.setup_probe()
        real_controller = PanStepController

        def make_controller(*args, **kwargs):
            return real_controller(*args, **kwargs, clock=port.clock.time, sleep=port.clock.sleep)

        fake_serial = SimpleNamespace(Serial=None)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"serial": fake_serial}), \
                patch.object(fake_serial, "Serial", return_value=contextlib.nullcontext(port)) as opening, \
                patch("so101_pan_step.PanStepController", side_effect=make_controller), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main(["--joint", "2", "--probe-shoulder", "--report-dir", directory])
            paths = list(Path(directory).glob("joint2_*.json"))
            self.assertEqual(len(paths), 1)
            report = json.loads(paths[0].read_text())
        self.assertEqual(result, 0)
        opening.assert_called_once()
        self.assertEqual(self.goals(port), [3698, 3702])
        self.assertEqual(report["outcome"], "completed_and_released")
        self.assertEqual(report["test"], "joint_single_step_probe")
        self.assertTrue(report["probe_shoulder"] and report["probe_hold_complete"])
        self.assertFalse(report["return_complete"] or report["final_anchor_hold_complete"])
        self.assert_released(port)


if __name__ == "__main__":
    unittest.main()
