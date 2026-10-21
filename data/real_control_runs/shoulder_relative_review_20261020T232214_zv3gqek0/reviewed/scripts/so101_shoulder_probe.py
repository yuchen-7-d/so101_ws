"""ID2 fixed +8-count relative probe; default read-only, no arbitrary motion."""

import argparse
from pathlib import Path
import sys
import time

from so101_relative_probe import FixedRelativeProbeController, TerminalReady, run_probe_cli


class ShoulderProbeController(FixedRelativeProbeController):
    def __init__(self, bus, *, ready_check=None, emit=print,
                 clock=time.monotonic, sleep=time.sleep):
        super().__init__(bus, profile="shoulder_single_eight", ready_check=ready_check,
                         emit=emit, clock=clock, sleep=sleep)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ID2 固定正向 8 计数单程诊断；默认只读预览。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--enable", action="store_true", help="实际使能 ID1–5 并执行 ID2 固定 +8 计数")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]/"data"/"real_control_runs")
    args = parser.parse_args(argv)
    return run_probe_cli(parser, args, controller_factory=ShoulderProbeController,
                         ready_factory=TerminalReady, controller_kwargs={},
                         log_prefix="shoulder_relative_probe",
                         action_description="ID2 向计数增大方向单次 8 计数")


if __name__ == "__main__":
    raise SystemExit(main())
