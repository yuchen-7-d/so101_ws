"""Timeout diagnostics use existing validated packets; no added bus traffic."""

import itertools
import re
import unittest
from unittest.mock import patch

from so101_hold_current import GOAL, OUTPUT, TORQUE, Bus, ControlError
import test_so101_shoulder_relative_probe as shoulder_tests


class TimeoutDiagnosticTests(unittest.TestCase):
    def run_timeout(self, displacement, *, controller_class=None, skip_feedback=False):
        def incomplete_motion(port, instruction, sid, address, params):
            if instruction == 3 and sid == 2 and address == GOAL and port.get(2, OUTPUT) == 100:
                port.put(2, 56, 115+displacement)
                port.put(2, 60, 0x400 | 40)  # Signed raw output -40, magnitude 40.
        with patch.dict(shoulder_tests.START, {2: 115}):
            port, control, clock, messages = shoulder_tests.ShoulderRelativeTests().make(
                bias=0, hook=incomplete_motion)
            if controller_class is not None:
                control = controller_class(
                    Bus(port), profile="shoulder_single_eight", ready_check=lambda: True,
                    emit=messages.append, clock=clock.time, sleep=clock.sleep)
            last_completed_monitor = None
            original_monitor = control.monitor
            def monitor(*args):
                nonlocal last_completed_monitor
                states = original_monitor(*args)
                if control.phase == "step_+8" and control.targets[2] == 123:
                    last_completed_monitor = len(port.events)
                return states
            control.monitor = monitor
            if skip_feedback:
                def emit(message):
                    messages.append(message)
                    if message.startswith("ID2 命令="):
                        ticks = itertools.chain((0.0,), itertools.repeat(3.0))
                        control.clock = lambda: next(ticks)
                control.emit = emit
            with self.assertRaisesRegex(ControlError, "相对位移未在 3 秒内达标") as caught:
                control.run(True)
        shoulder_tests.ShoulderRelativeTests().released(port)
        self.assertEqual(control.status, "failed")
        self.assertEqual(control.steps, [])
        self.assertFalse(control.motion_response_complete)
        self.assertEqual(control.fault["phase"], "step_+8")
        self.assertEqual(shoulder_tests.ShoulderRelativeTests().goals(port), [115, 123])
        if not skip_feedback:
            self.assertIsNotNone(last_completed_monitor)
            # No diagnostic serial reads are inserted between the last valid
            # movement monitor and the first torque-release command.
            self.assertEqual(port.events[last_completed_monitor], (3, 1, TORQUE, b"\x00"))
        return port, control, str(caught.exception)

    def test_timeout_reports_partial_or_zero_motion_without_protocol_changes(self):
        for displacement in (4, 0):
            with self.subTest(displacement=displacement):
                port, control, message = self.run_timeout(displacement)
                self.assertIn(f"实际={115+displacement}，P0=115", message)
                self.assertIn(f"相对位移={displacement:+d}", message)
                self.assertIn(f"目标=123，目标误差={displacement-8:+d}", message)
                self.assertIn("输出幅值=40/1000", message)
                self.assertIn("固定预期=123±2", message)
                self.assertIn("停止，不更新基线、不执行回程", message)
                age = float(re.search(r"样本距今=([0-9.]+)s", message).group(1))
                latest = [r for r in control.reads if r["phase"] == "step_+8"
                          and r["servo_id"] == 2 and r["address"] == 56][-1]
                expected_age = control.fault["elapsed_s"]-latest["elapsed_s"]
                self.assertAlmostEqual(age, expected_age, delta=.00051)
                self.assertEqual(latest["output_raw_signed"], -40)

    def test_timeout_without_new_valid_feedback_explicitly_reports_no_sample(self):
        _, _, message = self.run_timeout(4, skip_feedback=True)
        self.assertIn("本小步无已完成整轮校验的反馈样本", message)
        self.assertNotIn("最近有效反馈", message)
        self.assertNotIn("样本距今", message)


if __name__ == "__main__":
    unittest.main()
