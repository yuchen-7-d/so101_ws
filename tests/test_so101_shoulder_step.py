"""Selected shoulder movement tests; all bus traffic uses FakeSerial."""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_pan_step import PanStepController, main
from test_so101_hold_current import Clock, FakeSerial


class ShoulderStepTests(unittest.TestCase):
    def setup_control(self, move=True, fault=None, start_range=None):
        clock = Clock()

        def plant(port, instruction, sid, address, params):
            if (move and instruction == 3 and sid == 2 and address == GOAL
                    and port.regs[sid][TORQUE] == 1 and port.get(sid, OUTPUT)):
                port.put(sid, 56, int.from_bytes(params, "little"))
            if fault:
                fault(port, instruction, sid, address, params)

        port = FakeSerial(clock, plant)
        port.allow_live_goal_writes = True
        for sid, position in enumerate((20, 720, 3166, 2022, 3249, 4013), 1):
            port.put(sid, 56, position)
        controller = PanStepController(
            Bus(port), moving_id=2, delta=8, start_range=start_range,
            emit=lambda message: None, clock=clock.time, sleep=clock.sleep,
        )
        return port, controller

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def goals(self, port, servo_id):
        return [value for sid, address, value in port.writes
                if sid == servo_id and address == GOAL]

    def test_shoulder_preview_is_read_only_and_uses_shoulder_anchor(self):
        port, controller = self.setup_control()
        controller.run(False)
        self.assertEqual(controller.anchor, 720)
        self.assertEqual(controller.path, [724, 728, 724, 720])
        self.assertEqual(port.writes, [])

    def test_only_selected_shoulder_changes_goal_and_report_names_it(self):
        port, controller = self.setup_control()
        initial = {sid: port.get(sid, 56) for sid in ARM_IDS}
        registers = {sid: bytes(values) for sid, values in port.regs.items()}
        controller.run(True)
        self.assertEqual(self.goals(port, 2), [720, 724, 728, 724, 720])
        for sid in (1, 3, 4, 5):
            self.assertEqual(self.goals(port, sid), [initial[sid]])
            self.assertEqual(port.get(sid, 56), initial[sid])
        for sid in ARM_IDS:
            self.assertEqual(bytes(port.regs[sid][:40]), registers[sid][:40])
        self.assertEqual(bytes(port.regs[6]), registers[6])
        self.assertFalse(any(sid == 6 for sid, _, _ in port.writes))
        self.assertTrue(controller.return_complete)
        self.assertTrue(controller.final_anchor_hold_complete)
        report = json.loads(json.dumps(
            controller.report("completed_and_released", None)
        ))
        self.assertEqual(report["moving_id"], 2)
        self.assertEqual(report["anchor"], 720)
        self.assertEqual(report["planned_path"], [724, 728, 724, 720])
        self.assertIsNone(report["start_range"])
        self.assertIsNone(report["start_pose_window"])
        self.assertEqual(report["start_pose_source"], "fresh_stable_feedback")
        self.assertEqual(report["start_position_counts"]["2"], 720)
        self.assertFalse(report["model_mapping_verified"])
        self.assert_released(port)

    def test_still_pan_cannot_make_stuck_shoulder_pass(self):
        port, controller = self.setup_control(move=False)
        with self.assertRaisesRegex(ControlError, "ID2.*未在3秒内连续到位"):
            controller.run(True)
        self.assertEqual(self.goals(port, 2), [720, 724])
        self.assertEqual(self.goals(port, 1), [20])
        self.assertEqual(port.get(1, 56), 20)
        self.assertEqual(port.get(2, 56), 720)
        self.assertEqual([entry["target"] for entry in controller.reached], [720])
        self.assertFalse(controller.outbound_complete or controller.return_complete)
        self.assertEqual(controller.fault_context["targets"][2], 724)
        self.assert_released(port)

    def test_shoulder_overshoot_records_reply_then_immediately_releases_all(self):
        def overshoot(port, instruction, sid, address, params):
            if (instruction == 3 and sid == 2 and address == GOAL
                    and int.from_bytes(params, "little") == 724):
                port.put(2, 56, 727)  # First segment permits only 718..726.
                port.put(2, 58, 50)
                port.put(2, 60, 0x0400 | 68)

        port, controller = self.setup_control(fault=overshoot)
        with self.assertRaisesRegex(ControlError, "ID2 当前小步反馈越界"):
            controller.run(True)
        sample = controller.fault_context["last_feedback_by_id"][2]
        self.assertEqual(sample["position"], 727)
        self.assertEqual(sample["expected_target"], 724)
        self.assertEqual(sample["speed_raw_signed"], 50)
        self.assertEqual(sample["output_raw_signed"], -68)
        last_read_index = max(
            index for index, event in enumerate(port.events)
            if event[:3] == (2, 2, 56)
        )
        self.assertEqual(port.events[last_read_index + 1][:3], (3, 1, TORQUE))
        self.assertEqual(self.goals(port, 2), [720, 724])
        self.assertFalse(controller.return_complete)
        self.assert_released(port)

    def test_shoulder_device_limits_reject_path_before_write(self):
        port, controller = self.setup_control()
        port.put(2, 11, 725)  # Anchor and first step legal; second step is not.
        with self.assertRaisesRegex(ControlError, "完整往返路径"):
            controller.run(True)
        self.assertEqual(port.writes, [])
        self.assertFalse(controller.touched)

    def test_pan_limits_do_not_replace_shoulder_limits(self):
        port, controller = self.setup_control()
        port.put(1, 11, 52)  # Legal for the stationary pan, not shoulder goals.
        controller.run(True)
        self.assertTrue(controller.final_anchor_hold_complete)
        self.assertEqual(self.goals(port, 1), [20])
        self.assertEqual(self.goals(port, 2), [720, 724, 728, 724, 720])
        self.assert_released(port)

    def test_current_pose_is_held_without_requiring_historical_window(self):
        port, controller = self.setup_control()
        port.put(1, 56, 100)
        port.put(2, 56, 800)
        port.put(3, 56, 3482)
        controller.run(True)
        self.assertEqual(controller.anchor, 800)
        self.assertEqual(self.goals(port, 2), [800, 804, 808, 804, 800])
        self.assertEqual(self.goals(port, 1), [100])
        self.assertEqual(self.goals(port, 3), [3482])
        self.assertEqual(controller.start_position_counts[2], 800)
        self.assertEqual(controller.start_position_counts[3], 3482)
        self.assertTrue(controller.final_anchor_hold_complete)
        self.assert_released(port)

    def test_explicit_optional_range_still_rejects_before_writes(self):
        port, controller = self.setup_control(start_range=(704, 736))
        port.put(2, 56, 800)
        with self.assertRaisesRegex(ControlError, "起点=800.*指定起点范围"):
            controller.run(True)
        self.assertEqual(port.writes, [])
        self.assertTrue(all(event[0] == 2 for event in port.events))
        self.assertFalse(controller.touched)

    def test_current_pose_still_rejects_encoder_boundary_without_writes(self):
        port, controller = self.setup_control()
        port.put(2, 56, 4090)
        with self.assertRaisesRegex(ControlError, "完整往返路径"):
            controller.run(True)
        self.assertEqual(port.writes, [])
        self.assertFalse(controller.touched)

    def test_shoulder_target_ack_loss_does_not_command_return(self):
        def lose_ack(port, instruction, sid, address, params):
            if (instruction == 3 and sid == 2 and address == GOAL
                    and int.from_bytes(params, "little") == 724):
                port.pending = b""

        port, controller = self.setup_control(fault=lose_ack)
        with self.assertRaisesRegex(ControlError, "超时"):
            controller.run(True)
        self.assertEqual(port.get(2, 56), 724)
        self.assertEqual(self.goals(port, 2), [720, 724])
        self.assertEqual(self.goals(port, 1), [20])
        self.assertEqual(controller.fault_context["targets"][2], 724)
        self.assertEqual(controller.start_position_counts[2], 720)
        self.assertEqual(controller.fault_context["phase"], "outbound")
        self.assertFalse(controller.return_complete)
        self.assert_released(port)

    def test_return_hold_checks_shoulder_instead_of_stationary_pan(self):
        port, controller = self.setup_control()
        previous_hook = port.hook

        def drift(port, instruction, sid, address, params):
            previous_hook(port, instruction, sid, address, params)
            if (controller.phase == "before_release" and instruction == 2
                    and sid == 2 and address == 56):
                port.put(2, 56, 723)
                payload = bytes(port.regs[2][56:66])
                fields = bytes([2, len(payload) + 2, 0]) + payload
                port.pending = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])

        port.hook = drift
        with self.assertRaisesRegex(ControlError, "ID2 返回后再次偏离起点"):
            controller.run(True)
        self.assertEqual(port.get(1, 56), 20)
        self.assertTrue(controller.return_complete)
        self.assertFalse(controller.final_anchor_hold_complete)
        self.assertEqual(
            controller.fault_context["last_feedback_by_id"][2]["position"], 723
        )
        self.assert_released(port)

    def test_invalid_selected_joint_or_shoulder_parameters_have_no_bus_access(self):
        cases = (
            {"moving_id": 0}, {"moving_id": 3}, {"moving_id": True},
            {"moving_id": 2, "delta": -8, "start_range": (704, 736)},
            {"moving_id": 2, "delta": 16, "start_range": (704, 736)},
            {"moving_id": 2, "delta": 8, "start_range": (704, 736),
             "probe_negative": True},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                port = FakeSerial(Clock())
                with self.assertRaises(ValueError):
                    PanStepController(Bus(port), **parameters)
                self.assertEqual(port.events, [])

    def test_invalid_shoulder_cli_modes_are_rejected_before_serial_import(self):
        cases = (
            ["--joint", "2", "--enable", "--start-range", "704", "736",
             "--delta", "16"],
            ["--joint", "2", "--enable", "--start-range", "704", "736",
             "--delta", "-8"],
            ["--joint", "2", "--probe-negative"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_shoulder_cli_without_start_range_reaches_serial_open(self):
        # Stop at the injected serial open, with no real device access.
        fake_serial = type("FakeSerialModule", (), {})()
        with patch.dict(sys.modules, {"serial": fake_serial}):
            with patch.object(fake_serial, "Serial", create=True,
                              side_effect=OSError("mock: no device access")) as opening:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = main(["--joint", "2", "--enable", "--delta", "8"])
        self.assertEqual(result, 1)
        opening.assert_called_once()


if __name__ == "__main__":
    unittest.main()
