"""Protocol-level probe tests. Fake serial and clock only; no device opens."""

import io
import json
from contextlib import nullcontext, redirect_stderr
import fcntl
import os
from pathlib import Path
import pty
import select
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_elbow_probe import ElbowProbeController, TerminalReady, main
from test_so101_hold_current import Clock, FakeSerial


START = {1: 67, 2: 113, 3: 3961, 4: 967, 5: 3299, 6: 3951}


class ProbeTests(unittest.TestCase):
    def setup_control(self, *, response="follow", ready_check=lambda: True,
                      extra_hook=None, bias=5):
        clock, messages = Clock(), []
        def hook(port, instruction, sid, address, params):
            value = int.from_bytes(params, "little")
            if instruction == 3 and sid == 5 and address == OUTPUT and value == 100:
                port.put(2, 56, START[2]+4)
                port.put(3, 56, START[3]+bias)
                port.put(2, 60, 40)
                port.put(3, 60, 48)
            if (instruction == 3 and sid == 3 and address == GOAL
                    and port.get(3, OUTPUT) == 100):
                if response == "follow":
                    port.put(3, 56, value+bias)
                elif response == "reverse":
                    port.put(3, 56, START[3]+bias+3)
                elif response == "overshoot":
                    port.put(3, 56, value+bias-3)
                elif response == "still":
                    pass
                else:
                    raise AssertionError(response)
            if extra_hook is not None:
                extra_hook(port, instruction, sid, address, params)
        port = FakeSerial(clock, hook)
        port.allow_live_goal_writes = True
        for sid, position in START.items():
            port.put(sid, 56, position)
        controller = ElbowProbeController(Bus(port), ready_check=ready_check,
                                          emit=messages.append, clock=clock.time,
                                          sleep=clock.sleep)
        return port, controller, clock, messages

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def goals(self, port, sid=3):
        return [value for servo, address, value in port.writes
                if servo == sid and address == GOAL]

    def test_preview_never_writes_or_requests_handover(self):
        def no_ready():
            self.fail("preview must not wait for input")
        port, control, _, _ = self.setup_control(ready_check=no_ready)
        control.run()
        self.assertEqual(port.writes, [])
        self.assertEqual(control.report()["write_attempts"], [])
        self.assertEqual(control.status, "read_only_preview")
        self.assertIsNone(control.baseline)

    def test_enable_without_readiness_is_rejected_without_writes(self):
        port, control, _, _ = self.setup_control(ready_check=None)
        with self.assertRaisesRegex(ControlError, "交接就绪输入"):
            control.run(True)
        self.assertEqual(port.writes, [])

    def test_loaded_bias_is_not_added_to_commands(self):
        port, control, clock, messages = self.setup_control()
        control.run(True)
        self.assert_released(port)
        self.assertEqual(control.baseline, 3966)
        self.assertEqual(self.goals(port), [3961, 3957, 3953])
        self.assertEqual([s["position"] for s in control.steps], [3962, 3958])
        self.assertTrue(all(s["absolute_error"] == 5 for s in control.steps))
        self.assertTrue(all(s["relative_error"] == 0 for s in control.steps))
        self.assertTrue(all(s["span_s"] >= .2 and s["consecutive_samples"] >= 3
                            for s in control.steps))
        self.assertGreaterEqual(control.baseline_samples[-1]["elapsed_s"]
                                - control.baseline_samples[0]["elapsed_s"], .4)
        self.assertGreater(clock.time(), 6)
        self.assertEqual(control.status, "relative_probe_complete")
        self.assertTrue(control.motion_response_complete)
        self.assertTrue(any("相对位移达标" in msg for msg in messages))
        self.assertFalse(any(sid == 6 for sid, _, _ in port.writes))
        for sid in (1, 2, 4, 5):
            self.assertEqual(self.goals(port, sid), [START[sid]])
        self.assertEqual([value for sid, address, value in port.writes
                          if sid == 3 and address == OUTPUT and value], [25, 50, 75, 100])
        self.assertTrue(control.report()["release"]["confirmed"])
        json.dumps(control.report(), allow_nan=False)

    def test_still_within_absolute_tolerance_does_not_pass_relative(self):
        port, control, _, _ = self.setup_control(response="still")
        with self.assertRaisesRegex(ControlError, "相对位移未在 3 秒内达标"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(self.goals(port), [3961, 3957])
        self.assertEqual(control.steps, [])
        self.assertEqual(port.get(3, 56)-3957, 9)  # Below inherited 12-count limit.
        self.assertEqual(control.fault["phase"], "step_-4")
        self.assertFalse(control.motion_response_complete)

    def test_reverse_and_overshoot_release_without_return(self):
        for response in ("reverse", "overshoot"):
            with self.subTest(response=response):
                port, control, _, _ = self.setup_control(response=response)
                with self.assertRaisesRegex(ControlError, "越界"):
                    control.run(True)
                self.assert_released(port)
                self.assertEqual(self.goals(port), [3961, 3957])
                self.assertEqual(control.fault["phase"], "step_-4")

    def test_wrist_drift_is_rejected_on_same_feedback_packet(self):
        def wrist(port, instruction, sid, address, params):
            if instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                port.put(4, 56, START[4]+4)
        port, control, _, _ = self.setup_control(extra_hook=wrist)
        with self.assertRaisesRegex(ControlError, "ID4.*3 计数"):
            control.run(True)
        self.assert_released(port)
        fault_reads = [r for r in control.reads if r["phase"] == "step_-4"]
        self.assertEqual((fault_reads[-1]["servo_id"], fault_reads[-1]["address"]), (4, 56))
        self.assertEqual(fault_reads[-1]["position"], START[4]+4)
        self.assertEqual(self.goals(port), [3961, 3957])

    def test_saturation_is_inherited_and_does_not_increase_limit(self):
        def saturate(port, instruction, sid, address, params):
            if instruction == 3 and sid == 5 and address == OUTPUT and params == b"d\0":
                port.put(3, 60, 100)
        port, control, _, _ = self.setup_control(extra_hook=saturate)
        with self.assertRaisesRegex(ControlError, "输出持续接近"):
            control.run(True)
        self.assert_released(port)
        self.assertLessEqual(max(value for _, address, value in port.writes if address == OUTPUT), 100)
        self.assertEqual(self.goals(port).count(START[3]), 1)

    def test_lost_ack_after_applied_motion_still_releases_without_return(self):
        fired = False
        def lost_ack(port, instruction, sid, address, params):
            nonlocal fired
            if not fired and instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                fired = True
                port.pending = b""
        port, control, _, _ = self.setup_control(extra_hook=lost_ack)
        with self.assertRaisesRegex(ControlError, "应答超时"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(self.goals(port), [3961, 3957])
        lost = [w for w in control.writes if w["address"] == GOAL and not w["acknowledged"]]
        self.assertEqual(len(lost), 1)
        self.assertEqual(lost[0]["value"], 3957)

    def test_interrupt_after_motion_releases_without_return(self):
        fired = False
        def interrupt(port, instruction, sid, address, params):
            nonlocal fired
            if not fired and instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                fired = True
                raise KeyboardInterrupt
        port, control, _, _ = self.setup_control(extra_hook=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(self.goals(port), [3961, 3957])
        self.assertEqual(control.fault["type"], "KeyboardInterrupt")
        self.assertEqual(control.fault["phase"], "step_-4")

    def test_preflight_rejects_entire_corridor_near_both_boundaries(self):
        for start in (24, 4070):
            with self.subTest(start=start):
                port, control, _, _ = self.setup_control()
                port.put(3, 56, start)
                with self.assertRaisesRegex(ControlError, "完整反馈走廊"):
                    control.run(True)
                self.assertEqual(port.writes, [])

    def test_preflight_checks_device_limits_not_only_single_turn(self):
        port, control, _, _ = self.setup_control()
        port.put(3, 9, 3940)
        with self.assertRaisesRegex(ControlError, "完整反馈走廊"):
            control.run(True)
        self.assertEqual(port.writes, [])

    def test_baseline_bias_over_six_fails_without_motion(self):
        port, control, _, _ = self.setup_control(bias=7)
        with self.assertRaisesRegex(ControlError, "偏差超过 6"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(self.goals(port), [3961])
        self.assertIsNone(control.baseline)

    def test_loaded_baseline_is_not_updated_during_failed_motion(self):
        count = 0
        def change_load(port, instruction, sid, address, params):
            nonlocal count
            if instruction == 3 and sid == 3 and address == GOAL and port.get(3, OUTPUT) == 100:
                count += 1
                # Offset changes after the baseline; all feedback stays static.
                port.put(3, 56, 3966)
        port, control, _, _ = self.setup_control(extra_hook=change_load)
        with self.assertRaisesRegex(ControlError, "相对位移未"):
            control.run(True)
        self.assertEqual(control.baseline, 3966)
        self.assert_released(port)
        self.assertEqual(count, 1)

    def test_handover_timeout_releases_without_motion(self):
        port, control, clock, _ = self.setup_control(ready_check=lambda: False)
        with self.assertRaisesRegex(ControlError, "交接在 15 秒内未就绪"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(self.goals(port), [3961])
        self.assertGreaterEqual(clock.time(), 15)
        self.assertIsNone(control.baseline)
        reads = [r for r in control.reads if r["phase"] == "support_handover" and r["address"] == 56]
        self.assertGreater(len(reads), 500)

    def test_handover_eof_or_interrupt_releases_without_motion(self):
        for error in (ControlError("承重交接输入 EOF，停止"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                def ready():
                    raise error
                port, control, _, _ = self.setup_control(ready_check=ready)
                with self.assertRaises(type(error)):
                    control.run(True)
                self.assert_released(port)
                self.assertEqual(self.goals(port), [3961])
                self.assertEqual(control.fault["phase"], "support_handover")

    def test_restoring_support_in_final_five_seconds_is_allowed(self):
        port, control, _, _ = self.setup_control()
        original_sleep = control.sleep
        def restore(duration):
            original_sleep(duration)
            if control.phase == "final_monitor":
                port.put(2, 56, START[2])
                port.put(3, 56, START[3]-8)
        control.sleep = restore
        control.run(True)
        self.assert_released(port)
        self.assertTrue(control.motion_response_complete)
        self.assertEqual(control.status, "relative_probe_complete")
        self.assertEqual(self.goals(port), [3961, 3957, 3953])

    def test_cli_enable_requires_tty_before_serial_import_or_open(self):
        with patch("so101_elbow_probe.sys.stdin", io.StringIO("\n")), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["--enable"])
        self.assertEqual(result.exception.code, 1)

    def test_no_arbitrary_motion_or_output_cli_options(self):
        for option in ("--delta", "--limit", "--seconds", "--return"):
            with self.subTest(option=option), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                main([option, "8"])

    def test_ready_arriving_after_callback_deadline_is_not_accepted(self):
        port, control, clock, _ = self.setup_control()
        def late_ready():
            clock.now += 15
            return True
        control.ready_check = late_ready
        with self.assertRaisesRegex(ControlError, "交接在 15 秒内未就绪"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(self.goals(port), [3961])

    def test_monitor_crossing_handover_deadline_never_checks_enter(self):
        calls = []
        port, control, clock, _ = self.setup_control(ready_check=lambda: calls.append(1) or True)
        original_monitor = control.monitor
        def late_monitor(*args):
            states = original_monitor(*args)
            if control.phase == "support_handover":
                clock.now += 15
            return states
        control.monitor = late_monitor
        with self.assertRaisesRegex(ControlError, "交接在 15 秒内未就绪"):
            control.run(True)
        self.assert_released(port)
        self.assertEqual(calls, [])
        self.assertEqual(self.goals(port), [3961])

    def test_late_baseline_and_late_motion_feedback_cannot_pass(self):
        for phase, text in (("loaded_baseline", "反馈在 3 秒内未稳定"),
                            ("step_-4", "相对位移未在 3 秒内达标")):
            with self.subTest(phase=phase):
                port, control, clock, _ = self.setup_control()
                original_monitor = control.monitor
                calls = 0
                def late_monitor(*args):
                    nonlocal calls
                    states = original_monitor(*args)
                    if (control.phase == phase
                            and (phase == "loaded_baseline" or control.targets[3] == 3957)):
                        calls += 1
                        if calls == 3:
                            clock.now += 3
                    return states
                control.monitor = late_monitor
                with self.assertRaisesRegex(ControlError, text):
                    control.run(True)
                self.assert_released(port)
                self.assertEqual(control.steps, [])

    def test_terminal_restore_failure_cannot_skip_record_or_release(self):
        port, control, _, _ = self.setup_control()
        def fail_close():
            raise OSError("injected terminal restore failure")
        fake_ready = SimpleNamespace(close=fail_close)
        fake_serial = SimpleNamespace(Serial=lambda *a, **kw: nullcontext(port))
        with tempfile.TemporaryDirectory() as folder:
            with (patch("so101_elbow_probe.TerminalReady", return_value=fake_ready),
                  patch("so101_elbow_probe.ElbowProbeController", return_value=control),
                  patch.dict("sys.modules", {"serial": fake_serial}),
                  patch("builtins.print")):
                result = main(["--enable", "--log-dir", folder])
            self.assertEqual(result, 1)
            files = list(Path(folder).glob("*.json"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text())
            self.assertTrue(record["release"]["confirmed"])
            self.assertEqual(record["status"], "relative_probe_complete")
        self.assert_released(port)


class TerminalReadyTests(unittest.TestCase):
    """Use a local pseudo-terminal, never a robot serial port."""

    def setUp(self):
        self.master, slave = pty.openpty()
        self.stream = os.fdopen(slave, "r")
        self.ready = TerminalReady(self.stream)
        self.original_flags = fcntl.fcntl(slave, fcntl.F_GETFL)

    def tearDown(self):
        self.ready.close()
        self.stream.close()
        os.close(self.master)

    def send_line(self, data):
        os.write(self.master, data)
        self.assertTrue(select.select([self.stream], [], [], .2)[0])

    def test_old_enter_is_flushed_new_enter_is_accepted_and_flags_restore(self):
        self.send_line(b"\n")
        self.ready.begin()
        self.assertFalse(self.ready())
        self.assertTrue(fcntl.fcntl(self.stream.fileno(), fcntl.F_GETFL) & os.O_NONBLOCK)
        self.send_line(b"\n")
        self.assertTrue(self.ready())
        self.ready.close()
        self.assertEqual(fcntl.fcntl(self.stream.fileno(), fcntl.F_GETFL), self.original_flags)

    def test_partial_input_does_not_block_and_eof_is_rejected(self):
        self.ready.begin()
        os.write(self.master, b" ")
        self.assertFalse(self.ready())
        self.send_line(b"\n")
        self.assertTrue(self.ready())
        self.ready.close()
        self.ready.begin()
        self.send_line(b"\x04")
        with self.assertRaisesRegex(ControlError, "EOF"):
            self.ready()


if __name__ == "__main__":
    unittest.main()
