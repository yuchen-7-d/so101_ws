"""Read-only diagnostic routing at protocol and CLI level; no real serial port."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import Bus
from so101_pan_step import main, read_pan_diagnostics
from test_so101_hold_current import Clock, FakeSerial


class JointDiagnosticTests(unittest.TestCase):
    def test_selected_joint_data_and_all_read_packets(self):
        for sid in (1, 2):
            with self.subTest(sid=sid):
                clock = Clock()
                port = FakeSerial(clock)
                port.regs[sid][21:28] = bytes((32, 16, 2, 20, 0, 1, 2))
                port.regs[sid][40] = 1  # Read-only even if torque is already on.
                port.put(sid, 31, 1822)
                port.put(sid, 56, 3698)
                port.put(sid, 58, 0x8000 | 50)
                port.put(sid, 60, 0x0400 | 60)
                before = {key: bytes(value) for key, value in port.regs.items()}
                kwargs = {} if sid == 1 else {"servo_id": sid}
                result = read_pan_diagnostics(
                    Bus(port), sleep=clock.sleep, clock=clock.time, **kwargs
                )
                self.assertEqual(result["servo_id"], sid)
                self.assertEqual((result["P"], result["D"], result["I"]), (32, 16, 2))
                self.assertEqual(result["offset_raw"], 1822)
                self.assertEqual(result["torque_enabled"], 1)
                self.assertEqual(result["writes_sent"], 0)
                self.assertEqual(result["configuration_raw_hex"], before[sid][:40].hex())
                self.assertEqual(result["control_raw_hex"], before[sid][40:50].hex())
                self.assertEqual(len(result["feedback"]), 5)
                for sample in result["feedback"]:
                    self.assertEqual(sample["position"], 3698)
                    self.assertEqual(sample["speed_raw_signed"], -50)
                    self.assertEqual(sample["output_raw_signed"], -60)
                    self.assertEqual(sample["raw_hex"], before[sid][56:66].hex())
                expected = [(2, sid, 0, b"\x28"), (2, sid, 40, b"\x0a")]
                expected += [(2, sid, 56, b"\x0a")] * 5
                self.assertEqual(port.events, expected)
                self.assertEqual(port.writes, [])
                self.assertEqual({key: bytes(value) for key, value in port.regs.items()}, before)

    def run_cli(self, port, folder, arguments):
        serial_module = SimpleNamespace(
            Serial=Mock(return_value=contextlib.nullcontext(port))
        )
        with patch.dict(sys.modules, {"serial": serial_module}):
            with patch("so101_pan_step.PanStepController") as controller:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        return main(arguments + ["--report-dir", folder])
                    finally:
                        controller.assert_not_called()
                        serial_module.Serial.assert_called_once()

    def test_cli_id2_diagnosis_saves_id2_report_without_control(self):
        port = FakeSerial(Clock())
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.run_cli(port, folder, ["--joint", "2", "--diagnose"]), 0)
            reports = list(Path(folder).glob("*.json"))
            self.assertEqual(len(reports), 1)
            self.assertTrue(reports[0].name.startswith("joint2_readback_"))
            result = json.loads(reports[0].read_text())
        self.assertEqual(result["servo_id"], 2)
        self.assertEqual(result["feedback"][0]["position"], 841)
        self.assertEqual(result["test"], "joint_read_only_diagnostics")
        self.assertTrue(all(event[:2] == (2, 2) for event in port.events))
        self.assertEqual(port.writes, [])

    def test_default_cli_keeps_id1_report_prefix_and_type(self):
        port = FakeSerial(Clock())
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.run_cli(port, folder, ["--diagnose"]), 0)
            reports = list(Path(folder).glob("pan_readback_*.json"))
            self.assertEqual(len(reports), 1)
            result = json.loads(reports[0].read_text())
        self.assertEqual(result["servo_id"], 1)
        self.assertEqual(result["test"], "pan_read_only_diagnostics")
        self.assertTrue(all(event[:2] == (2, 1) for event in port.events))
        self.assertEqual(port.writes, [])

    def test_read_failure_or_interrupt_never_sends_cleanup_writes(self):
        for failure in ("timeout", "interrupt"):
            with self.subTest(failure=failure):
                def fail(port, instruction, sid, address, params):
                    if address == 56:
                        if failure == "interrupt":
                            raise KeyboardInterrupt
                        port.pending = b""
                port = FakeSerial(Clock(), fail)
                with tempfile.TemporaryDirectory() as folder:
                    if failure == "interrupt":
                        with self.assertRaises(KeyboardInterrupt):
                            self.run_cli(port, folder, ["--joint", "2", "--diagnose"])
                    else:
                        self.assertEqual(self.run_cli(port, folder, ["--joint", "2", "--diagnose"]), 1)
                    self.assertEqual(list(Path(folder).iterdir()), [])
                self.assertEqual(port.writes, [])
                self.assertTrue(all(event[:2] == (2, 2) for event in port.events))

    def test_invalid_diagnostic_id_has_no_bus_access(self):
        for sid in (0, 3, 6, True, "2"):
            port = FakeSerial(Clock())
            with self.subTest(sid=sid), self.assertRaises(ValueError):
                read_pan_diagnostics(Bus(port), servo_id=sid)
            self.assertEqual(port.events, [])

    def test_diagnostics_rejects_motion_options_before_opening_serial(self):
        cases = (
            ["--enable"], ["--probe-negative"], ["--delta", "8"],
            ["--start-range", "0", "4095"],
        )
        serial_module = SimpleNamespace(Serial=Mock())
        for extra in cases:
            with self.subTest(extra=extra), patch.dict(sys.modules, {"serial": serial_module}):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(["--joint", "2", "--diagnose"] + extra)
                self.assertEqual(error.exception.code, 2)
                serial_module.Serial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
