"""Explicit single-step probe: protocol fakes only, no real serial device."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from so101_elbow_probe import main
from so101_hold_current import GOAL, OUTPUT, ControlError
import test_so101_elbow_probe as probe_tests


START = probe_tests.START


class SingleStepTests(unittest.TestCase):
    def make(self, **kwargs):
        return probe_tests.ProbeTests().setup_control(single_step_eight=True, bias=kwargs.pop("bias", 1), **kwargs)

    def goals(self, port):
        return probe_tests.ProbeTests().goals(port)

    def released(self, port):
        probe_tests.ProbeTests().assert_released(port)

    def test_single_step_plus_one_bias_moves_negative_eight(self):
        port, control, _, messages = self.make()
        control.run(True)
        self.released(port)
        self.assertEqual(control.baseline, START[3]+1)
        self.assertEqual(self.goals(port), [START[3], START[3]-8])
        self.assertEqual(len(control.steps), 1)
        self.assertEqual(control.steps[0]["position"]-control.baseline, -8)
        self.assertEqual(control.steps[0]["absolute_error"], 1)
        self.assertEqual(control.report()["command_offsets"], [-8])
        self.assertEqual(control.report()["mode"], "single_step_eight")
        self.assertEqual(control.report()["baseline_bias_limit"], 2)
        self.assertEqual(control.precommand_checks[0]["command_gap"], 9)
        self.assertTrue(control.precommand_checks[0]["accepted"])
        self.assertTrue(any("单步相对位移达标" in msg for msg in messages))
        self.assertFalse(any(sid == 6 for sid, _, _ in port.writes))
        self.assertEqual(max(v for _, addr, v in port.writes if addr == OUTPUT), 100)

    def test_one_count_or_zero_motion_cannot_pass(self):
        for displacement in (-1, 0):
            with self.subTest(displacement=displacement):
                def insufficient(port, instruction, sid, address, params):
                    if instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                        port.put(3, 56, START[3]+1+displacement)
                port, control, _, _ = self.make(extra_hook=insufficient)
                with self.assertRaisesRegex(ControlError, "相对位移未在 3 秒内达标"):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [START[3], START[3]-8])
                self.assertEqual(control.steps, [])
                self.assertFalse(control.motion_response_complete)
                self.assertEqual(control.fault["phase"], "step_-8")

    def test_baseline_plus_three_or_six_rejected_before_motion(self):
        for bias in (3, 6, -3, -6):
            with self.subTest(bias=bias):
                port, control, _, _ = self.make(bias=bias)
                with self.assertRaisesRegex(ControlError, "偏差超过 2"):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [START[3]])
                self.assertEqual(control.precommand_checks, [])

    def test_old_two_step_mode_still_allows_six_count_bias(self):
        port, control, _, _ = probe_tests.ProbeTests().setup_control(bias=6)
        control.run(True)
        self.released(port)
        self.assertEqual(self.goals(port), [START[3], START[3]-4, START[3]-8])
        self.assertEqual(control.report()["command_offsets"], [-4, -8])
        self.assertEqual(control.report()["baseline_bias_limit"], 6)
        self.assertEqual(control.report()["mode"], "two_steps_four")

    def test_fresh_three_count_drift_rejected_before_new_goal(self):
        for drift in (3, -3):
            with self.subTest(drift=drift):
                port, control, _, _ = self.make()
                old_monitor = control.monitor
                def monitor(*args):
                    if control.phase == "step_-8" and control.targets[3] == START[3]:
                        port.put(3, 56, START[3]+drift)
                    return old_monitor(*args)
                control.monitor = monitor
                with self.assertRaisesRegex(ControlError, "写前检查未通过"):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [START[3]])
                self.assertEqual(control.baseline, START[3]+1)
                self.assertFalse(control.precommand_checks[0]["accepted"])
                self.assertEqual(control.precommand_checks[0]["delta_from_g0"], drift)

    def test_two_count_baseline_and_fresh_feedback_boundaries_are_inclusive(self):
        for bias, fresh_delta in ((2, 2), (-2, -2), (0, 2), (0, -2)):
            with self.subTest(bias=bias, fresh_delta=fresh_delta):
                port, control, _, _ = self.make(bias=bias)
                old_monitor = control.monitor
                def monitor(*args):
                    if control.phase == "step_-8" and control.targets[3] == START[3]:
                        port.put(3, 56, START[3]+fresh_delta)
                    return old_monitor(*args)
                control.monitor = monitor
                control.run(True)
                self.released(port)
                self.assertEqual(control.baseline, START[3]+bias)
                self.assertEqual(self.goals(port), [START[3], START[3]-8])
                self.assertTrue(control.motion_response_complete)
                self.assertEqual(control.steps[0]["position"]-control.baseline, -8)
                self.assertEqual(len(control.precommand_checks), 1)
                check = control.precommand_checks[0]
                self.assertTrue(check["accepted"])
                self.assertEqual(check["position"], START[3]+fresh_delta)
                self.assertEqual(check["delta_from_g0"], fresh_delta)
                self.assertEqual(check["goal"], START[3]-8)
                self.assertEqual(check["command_gap"], 8+fresh_delta)

    def test_acceptance_requires_at_least_six_counts_of_real_motion(self):
        for displacement, succeeds in ((-6, True), (-5, False)):
            with self.subTest(displacement=displacement):
                def move(port, instruction, sid, address, params):
                    if instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                        port.put(3, 56, START[3]+1+displacement)
                port, control, _, _ = self.make(extra_hook=move)
                if succeeds:
                    control.run(True)
                    self.assertEqual(control.steps[0]["position"]-control.baseline, -6)
                else:
                    with self.assertRaisesRegex(ControlError, "相对位移未在"):
                        control.run(True)
                self.released(port)
                self.assertEqual(control.motion_response_complete, succeeds)
                self.assertEqual(self.goals(port), [START[3], START[3]-8])

    def test_wrist_four_count_drift_stops_on_its_feedback_reply(self):
        def wrist(port, instruction, sid, address, params):
            if instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                port.put(4, 56, START[4]+4)
        port, control, _, _ = self.make(extra_hook=wrist)
        with self.assertRaisesRegex(ControlError, "ID4.*3 计数"):
            control.run(True)
        self.released(port)
        last = [r for r in control.reads if r["phase"] == "step_-8"][-1]
        self.assertEqual((last["servo_id"], last["address"]), (4, 56))
        self.assertEqual(self.goals(port), [START[3], START[3]-8])

    def test_saturation_and_corridor_violations_still_release(self):
        for fault in ("saturation", "overshoot", "support_loss"):
            with self.subTest(fault=fault):
                def inject(port, instruction, sid, address, params):
                    if instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                        if fault == "saturation":
                            port.put(3, 60, 100)
                        elif fault == "overshoot":
                            port.put(3, 56, START[3]+1-11)
                        else:
                            port.put(3, 56, START[3]+13)
                port, control, _, _ = self.make(extra_hook=inject)
                pattern = "输出持续接近" if fault == "saturation" else "越界"
                with self.assertRaisesRegex(ControlError, pattern):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [START[3], START[3]-8])
                self.assertLessEqual(max(v for _, addr, v in port.writes if addr == OUTPUT), 100)

    def test_restore_support_is_allowed_after_single_step(self):
        port, control, _, _ = self.make()
        old_sleep = control.sleep
        def restore(seconds):
            old_sleep(seconds)
            if control.phase == "final_monitor":
                port.put(3, 56, START[3]-8)
        control.sleep = restore
        control.run(True)
        self.released(port)
        self.assertTrue(control.motion_response_complete)
        self.assertEqual(self.goals(port), [START[3], START[3]-8])

    def test_cli_mode_without_enable_is_only_read_only_preview(self):
        port, control, _, _ = self.make()
        fake_serial = SimpleNamespace(Serial=lambda *a, **kw: nullcontext(port))
        with tempfile.TemporaryDirectory() as folder:
            with (patch.dict("sys.modules", {"serial": fake_serial}),
                  patch("so101_elbow_probe.ElbowProbeController", return_value=control) as create,
                  patch("builtins.print")):
                result = main(["--single-step-eight", "--log-dir", folder])
            self.assertEqual(result, 0)
            self.assertTrue(create.call_args.kwargs["single_step_eight"])
            self.assertEqual(port.writes, [])
            record = json.loads(next(Path(folder).glob("*.json")).read_text())
            self.assertEqual(record["status"], "read_only_preview")
            self.assertEqual(record["command_offsets"], [-8])


if __name__ == "__main__":
    unittest.main()
