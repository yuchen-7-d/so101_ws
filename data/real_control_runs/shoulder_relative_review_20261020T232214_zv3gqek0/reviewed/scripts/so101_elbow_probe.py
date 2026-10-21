"""ID3 fixed negative probes; the existing interface remains backward compatible."""

import argparse
from pathlib import Path
import sys
import time

from so101_relative_probe import (
    BASELINE_BIAS, BOUNDARY_MARGIN, OFFSETS, RELATIVE_TOLERANCE,
    WRIST_TOLERANCE, FixedRelativeProbeController, JournalBus, TerminalReady,
    run_probe_cli,
)


class ElbowProbeController(FixedRelativeProbeController):
    def __init__(self, bus, *, single_step_eight=False, ready_check=None, emit=print,
                 clock=time.monotonic, sleep=time.sleep):
        if type(single_step_eight) is not bool:
            raise ValueError("single_step_eight 必须为布尔值")
        profile = "elbow_single_eight" if single_step_eight else "elbow_two_steps"
        super().__init__(bus, profile=profile, ready_check=ready_check, emit=emit,
                         clock=clock, sleep=sleep)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ID3 固定负向 8 计数单程诊断；默认只读预览。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--enable", action="store_true", help="实际使能并运行所选固定诊断")
    parser.add_argument("--single-step-eight", action="store_true",
                        help="显式选择单次 -8 计数；仅允许有载基线及写前反馈偏差 ≤2，仍需 --enable 才运动")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]/"data"/"real_control_runs")
    args = parser.parse_args(argv)
    steps = "单次 8 计数" if args.single_step_eight else "两步共 8 计数"
    return run_probe_cli(parser, args, controller_factory=ElbowProbeController,
                         ready_factory=TerminalReady,
                         controller_kwargs={"single_step_eight": args.single_step_eight},
                         log_prefix="elbow_probe",
                         action_description=f"ID3 向计数减小方向{steps}")


if __name__ == "__main__":
    raise SystemExit(main())
