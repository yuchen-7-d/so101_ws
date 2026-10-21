"""ID2 +8 fixed-profile protocol tests; never opens a real serial port."""

from contextlib import nullcontext, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_relative_probe import FixedRelativeProbeController
from so101_shoulder_probe import ShoulderProbeController, main
from test_so101_hold_current import Clock, FakeSerial
import test_so101_elbow_probe as elbow_tests


START = {1: 67, 2: 113, 3: 3941, 4: 967, 5: 3299, 6: 3951}


class ShoulderRelativeTests(unittest.TestCase):
    def make(self, *, bias=1, response="follow", hook=None, ready_check=lambda: True):
        clock, messages = Clock(), []
        def respond(port, instruction, sid, address, params):
            value = int.from_bytes(params, "little")
            if instruction == 3 and sid == 5 and address == OUTPUT and value == 100:
                port.put(2, 56, START[2]+bias)
                port.put(2, 60, 40)
                port.put(3, 56, START[3]+5)  # ID3 is held, never retargeted.
                port.put(3, 60, 48)
            if instruction == 3 and sid == 2 and address == GOAL and port.get(2, OUTPUT) == 100:
                positions = {"follow": value+bias,
                             "one": START[2]+bias+1,
                             "still": START[2]+bias,
                             "reverse": START[2]+bias-3,
                             "historical_overshoot": START[2]+14}
                port.put(2, 56, positions[response])
            if hook:
                hook(port, instruction, sid, address, params)
        port = FakeSerial(clock, respond)
        port.allow_live_goal_writes = True
        for sid, position in START.items():
            port.put(sid, 56, position)
        control = ShoulderProbeController(Bus(port), ready_check=ready_check,
                                          emit=messages.append, clock=clock.time,
                                          sleep=clock.sleep)
        return port, control, clock, messages

    def goals(self, port, sid=2):
        return [value for motor, address, value in port.writes if motor == sid and address == GOAL]

    def released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def test_positive_eight_with_stable_baseline_and_correct_physical_id(self):
        port, control, _, messages = self.make()
        control.run(True)
        self.released(port)
        self.assertEqual(self.goals(port), [113, 121])
        for sid in (1, 3, 4, 5):
            self.assertEqual(self.goals(port, sid), [START[sid]])
        self.assertFalse(any(sid == 6 for sid, _, _ in port.writes))
        self.assertEqual(control.baseline, 114)
        self.assertEqual(control.steps[0]["position"], 122)
        self.assertEqual(control.steps[0]["position"]-control.baseline, 8)
        self.assertEqual(control.steps[0]["servo_id"], 2)
        self.assertEqual(control.steps[0]["offset"], 8)
        self.assertEqual(control.steps[0]["absolute_error"], 1)
        self.assertGreaterEqual(control.steps[0]["consecutive_samples"], 3)
        self.assertGreaterEqual(control.steps[0]["span_s"], .2)
        report = control.report()
        self.assertEqual(report["servo_id"], 2)
        self.assertEqual(report["test"], "shoulder_relative_positive_probe")
        self.assertEqual(report["profile"], "shoulder_single_eight")
        self.assertEqual(report["command_offsets"], [8])
        self.assertTrue(report["motion_response_complete"])
        self.assertEqual(control.total_corridor, (111, 124))
        self.assertTrue(any("ID2 相对位移达标" in msg for msg in messages))
        self.assertFalse(any("ID3 相对位移达标" in msg for msg in messages))

    def test_baseline_and_fresh_feedback_plus_minus_two_are_accepted(self):
        for bias, fresh in ((2, 2), (-2, -2), (0, 2), (0, -2)):
            with self.subTest(bias=bias, fresh=fresh):
                port, control, _, _ = self.make(bias=bias)
                original = control.monitor
                def monitor(*args):
                    if control.phase == "step_+8" and control.targets[2] == START[2]:
                        port.put(2, 56, START[2]+fresh)
                    return original(*args)
                control.monitor = monitor
                control.run(True)
                self.released(port)
                self.assertEqual(control.baseline, START[2]+bias)
                self.assertEqual(self.goals(port), [113, 121])
                self.assertEqual(control.precommand_checks[0]["delta_from_g0"], fresh)
                self.assertEqual(control.precommand_checks[0]["command_gap"], fresh-8)
                self.assertEqual(control.precommand_checks[0]["servo_id"], 2)

    def test_baseline_plus_minus_three_rejects_without_motion(self):
        for bias in (3, -3):
            with self.subTest(bias=bias):
                port, control, _, _ = self.make(bias=bias)
                with self.assertRaisesRegex(ControlError, "ID2.*偏差超过 2"):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [113])
                self.assertIsNone(control.baseline)

    def test_fresh_drift_rejects_without_motion_or_baseline_change(self):
        for drift in (3, -3):
            with self.subTest(drift=drift):
                # Match baseline sign so +/-3 is inside the current feedback
                # corridor; the distinct fresh-command gate must reject it.
                bias = 1 if drift > 0 else -1
                port, control, _, _ = self.make(bias=bias)
                original = control.monitor
                def monitor(*args):
                    if control.phase == "step_+8" and control.targets[2] == START[2]:
                        port.put(2, 56, START[2]+drift)
                    return original(*args)
                control.monitor = monitor
                with self.assertRaisesRegex(ControlError, "ID2.*写前检查未通过"):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [113])
                self.assertEqual(control.baseline, START[2]+bias)
                self.assertFalse(control.precommand_checks[0]["accepted"])

    def test_no_motion_and_only_one_count_cannot_pass(self):
        for response in ("still", "one"):
            with self.subTest(response=response):
                port, control, _, _ = self.make(response=response)
                with self.assertRaisesRegex(ControlError, "ID2 相对位移未在 3 秒内达标"):
                    control.run(True)
                self.released(port)
                self.assertEqual(self.goals(port), [113, 121])
                self.assertEqual(control.steps, [])
                self.assertFalse(control.motion_response_complete)

    def test_reverse_three_stops_on_id2_reply_before_next_axis(self):
        port, control, _, _ = self.make(response="reverse")
        with self.assertRaisesRegex(ControlError, "ID2 .*越界"):
            control.run(True)
        self.released(port)
        last = [r for r in control.reads if r["phase"] == "step_+8"][-1]
        self.assertEqual((last["servo_id"], last["address"]), (2, 56))
        self.assertEqual(last["position"], control.baseline-3)
        self.assertEqual(self.goals(port), [113, 121])

    def test_historical_fourteen_count_overshoot_stops_on_id2_reply(self):
        port, control, _, _ = self.make(bias=0, response="historical_overshoot")
        with self.assertRaisesRegex(ControlError, "ID2 .*越界"):
            control.run(True)
        self.released(port)
        phase_reads = [r for r in control.reads if r["phase"] == "step_+8"]
        self.assertEqual((phase_reads[-1]["servo_id"], phase_reads[-1]["address"]), (2, 56))
        self.assertEqual(phase_reads[-1]["position"], START[2]+14)
        self.assertEqual(phase_reads[-1]["absolute_error"], 6)
        self.assertLess(abs(phase_reads[-1]["absolute_error"]), 12)
        self.assertEqual(self.goals(port), [113, 121])

    def test_id4_drift_immediately_releases(self):
        def drift(port, ins, sid, addr, params):
            if ins == 3 and sid == 2 and addr == GOAL and port.get(2, OUTPUT) == 100:
                port.put(4, 56, START[4]+4)
        port, control, _, _ = self.make(hook=drift)
        with self.assertRaisesRegex(ControlError, "ID4.*3 计数"):
            control.run(True)
        self.released(port)
        last = [r for r in control.reads if r["phase"] == "step_+8"][-1]
        self.assertEqual((last["servo_id"], last["address"]), (4, 56))

    def test_saturation_lost_ack_and_interrupt_release_without_return(self):
        for fault in ("saturation", "lost_ack", "interrupt"):
            with self.subTest(fault=fault):
                fired = False
                def inject(port, ins, sid, addr, params):
                    nonlocal fired
                    if not fired and ins == 3 and sid == 2 and addr == GOAL and port.get(2, OUTPUT) == 100:
                        fired = True
                        if fault == "saturation":
                            port.put(2, 60, 100)
                        elif fault == "lost_ack":
                            port.pending = b""
                        else:
                            raise KeyboardInterrupt
                port, control, _, _ = self.make(hook=inject)
                error = KeyboardInterrupt if fault == "interrupt" else ControlError
                with self.assertRaises(error):
                    control.run(True)
                self.released(port)
                self.assertTrue(fired)
                self.assertEqual(self.goals(port), [113, 121])
                self.assertLessEqual(max(v for _, addr, v in port.writes if addr == OUTPUT), 100)

    def test_handover_eof_releases_without_motion(self):
        def eof():
            raise ControlError("承重交接输入 EOF，停止")
        port, control, _, _ = self.make(ready_check=eof)
        with self.assertRaisesRegex(ControlError, "EOF"):
            control.run(True)
        self.released(port)
        self.assertEqual(self.goals(port), [113])
        self.assertEqual(control.fault["phase"], "support_handover")

    def test_readonly_and_boundary_preflight_never_write(self):
        port, control, _, _ = self.make()
        control.run()
        self.assertEqual(port.writes, [])
        self.assertEqual(control.report()["command_offsets"], [8])
        self.assertEqual(control.preflight_corridors[2], (101, 125))
        for position in (27, 4068):
            with self.subTest(position=position):
                port, control, _, _ = self.make()
                port.put(2, 56, position)
                with self.assertRaisesRegex(ControlError, "ID2 完整反馈走廊"):
                    control.run(True)
                self.assertEqual(port.writes, [])

    def test_restored_support_is_allowed_after_relative_motion_completes(self):
        port, control, _, _ = self.make(bias=2)
        original = control.sleep
        def restore(duration):
            original(duration)
            if control.phase == "final_monitor":
                port.put(2, 56, START[2]+8)
        control.sleep = restore
        control.run(True)
        self.released(port)
        self.assertEqual(control.baseline, START[2]+2)
        self.assertTrue(control.motion_response_complete)
        self.assertEqual(self.goals(port), [113, 121])

    def test_final_envelope_allows_support_restoration_but_rejects_retreat(self):
        cases = [("shoulder", bias, destination) for bias in (-2, 2)
                 for destination in ("goal", "g0", "p0")]
        cases += [("elbow", bias, destination) for bias in (-6, 6)
                  for destination in ("goal",)]
        for kind, bias, destination in cases:
            with self.subTest(kind=kind, bias=bias, destination=destination):
                if kind == "shoulder":
                    port, control, _, _ = self.make(bias=bias)
                else:
                    port, control, _, _ = elbow_tests.ProbeTests().setup_control(bias=bias)
                original = control.sleep
                sid = control.moving_id
                def support_change(duration):
                    original(duration)
                    if control.phase == "final_monitor":
                        locations = {"goal": control.targets[sid],
                                     "g0": control.initial_targets[sid],
                                     "p0": control.baseline}
                        port.put(sid, 56, locations[destination])
                control.sleep = support_change
                if destination == "goal":
                    control.run(True)
                    self.assertEqual(control.status, "relative_probe_complete")
                else:
                    with self.assertRaisesRegex(ControlError, "恢复承托越界"):
                        control.run(True)
                    self.assertEqual(control.status, "failed")
                    self.assertEqual(control.fault["phase"], "final_monitor")
                    last = [r for r in control.reads if r["phase"] == "final_monitor"][-1]
                    self.assertEqual((last["servo_id"], last["address"]), (sid, 56))
                self.released(port)
                self.assertTrue(control.motion_response_complete)
                expected_band = (min(control.targets[sid], control.expected_position)-2,
                                 max(control.targets[sid], control.expected_position)+2)
                self.assertEqual(control.report()["final_feedback_corridor"], expected_band)
                goals = [value for motor, addr, value in port.writes if motor == sid and addr == GOAL]
                self.assertEqual(goals, [control.initial_targets[sid]+o for o in (0, *control.offsets)])

    def test_cli_readonly_log_and_no_arbitrary_parameters(self):
        port, control, _, _ = self.make()
        fake_serial = SimpleNamespace(Serial=lambda *a, **kw: nullcontext(port))
        with tempfile.TemporaryDirectory() as folder:
            with (patch.dict("sys.modules", {"serial": fake_serial}),
                  patch("so101_shoulder_probe.ShoulderProbeController", return_value=control),
                  patch("builtins.print")):
                result = main(["--log-dir", folder])
            self.assertEqual(result, 0)
            self.assertEqual(port.writes, [])
            path = next(Path(folder).glob("shoulder_relative_probe_*.json"))
            record = json.loads(path.read_text())
            self.assertEqual(record["servo_id"], 2)
            self.assertEqual(record["command_offsets"], [8])
            self.assertIn("ID2 向计数增大方向", record["cli_action"])
        for option in ("--joint", "--delta", "--limit", "--seconds", "--single-step-eight"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([option, "8"])
        with (patch("so101_shoulder_probe.sys.stdin", io.StringIO("\n")),
              redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            main(["--enable"])

    def test_common_constructor_only_accepts_fixed_profiles(self):
        port, _, _, _ = self.make()
        for profile in ("ID2+16", "arbitrary", (2, 8), None):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                FixedRelativeProbeController(Bus(port), profile=profile)
        self.assertEqual(port.writes, [])


if __name__ == "__main__":
    unittest.main()
