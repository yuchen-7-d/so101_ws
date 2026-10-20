"""Fresh-start pan range checks with FakeSerial only."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_pan_step import PanStepController
from test_so101_hold_current import Clock, FakeSerial


class PanStartRangeTests(unittest.TestCase):
    def setup_control(self, position, start_range=(16, 32)):
        clock = Clock()

        def plant(port, instruction, sid, address, params):
            if (instruction == 3 and address == GOAL
                    and port.regs[sid][TORQUE] == 1
                    and port.get(sid, OUTPUT)):
                port.put(sid, 56, int.from_bytes(params, "little"))

        port = FakeSerial(clock, plant)
        port.allow_live_goal_writes = True
        port.put(1, 56, position)
        controller = PanStepController(
            Bus(port), delta=8, start_range=start_range,
            emit=lambda message: None, clock=clock.time, sleep=clock.sleep,
        )
        return port, controller

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def test_low_start_allows_roundtrip_and_reports_requested_range(self):
        port, controller = self.setup_control(17)
        controller.run(True)
        self.assertEqual(controller.anchor, 17)
        self.assertEqual(controller.path, [21, 25, 21, 17])
        self.assertEqual(
            [value for sid, address, value in port.writes
             if sid == 1 and address == GOAL],
            [17, 21, 25, 21, 17],
        )
        self.assertTrue(controller.return_complete)
        self.assertTrue(controller.final_anchor_hold_complete)
        self.assertEqual(port.get(1, 56), 17)
        report = json.loads(json.dumps(
            controller.report("completed_and_released", None)
        ))
        self.assertEqual(report["start_range"], [16, 32])
        self.assertFalse(report["model_mapping_verified"])
        self.assert_released(port)

    def test_high_start_is_rejected_before_any_control_write(self):
        port, controller = self.setup_control(4082)
        with self.assertRaises(ControlError):
            controller.run(True)
        self.assertEqual(port.writes, [])
        self.assertTrue(all(event[0] == 2 for event in port.events))
        self.assertFalse(controller.touched)

    def test_inclusive_start_boundaries_allow_normal_roundtrip(self):
        for position in (16, 32):
            with self.subTest(position=position):
                port, controller = self.setup_control(position)
                controller.run(True)
                self.assertEqual(controller.anchor, position)
                self.assertEqual(
                    controller.path,
                    [position + 4, position + 8, position + 4, position],
                )
                self.assertTrue(controller.final_anchor_hold_complete)
                self.assertEqual(port.get(1, 56), position)
                self.assert_released(port)

    def test_reversed_and_out_of_encoder_range_limits_are_rejected(self):
        for invalid_range in ((32, 16), (-1, 32), (16, 4096)):
            with self.subTest(invalid_range=invalid_range):
                clock = Clock()
                port = FakeSerial(clock)
                with self.assertRaises(ValueError):
                    PanStepController(
                        Bus(port), delta=8, start_range=invalid_range,
                        emit=lambda message: None,
                        clock=clock.time, sleep=clock.sleep,
                    )
                self.assertEqual(port.events, [])
                self.assertEqual(port.writes, [])


if __name__ == "__main__":
    unittest.main()
