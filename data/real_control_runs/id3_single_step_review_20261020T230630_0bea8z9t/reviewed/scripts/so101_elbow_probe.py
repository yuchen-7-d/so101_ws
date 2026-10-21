"""Fixed ID3 -8-count probe; raw live feedback only, no calibrated model motion.

G0 is the initial (unpowered) command reference. P0 is a separately measured,
fixed loaded feedback reference. The default commands are G0-4 and G0-8.
Explicit single-step mode commands G0-8 only, with a stricter +/-2 loaded
baseline and fresh pre-command gate. Commands never incorporate the P0 bias.
Acceptance checks relative feedback displacement from P0, not absolute arrival
at the command. No EEPROM writes, wrap, retargeting, or automatic return.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import select
import statistics
import sys
import tempfile
import termios
import time

from so101_hold_current import (
    ARM_IDS, GOAL, HOLD_TOLERANCE, OUTPUT, Bus, ControlError, HoldController,
    u16,
)
from so101_pan_step import decode_motion_feedback


OFFSETS = (-4, -8)
BASELINE_BIAS = 6
RELATIVE_TOLERANCE = 2
WRIST_TOLERANCE = 3
BOUNDARY_MARGIN = 16


class JournalBus:
    """Record replies before validation, including applied writes with lost ACKs."""

    def __init__(self, inner, controller):
        self.inner, self.controller = inner, controller

    def read(self, sid, address, size=1, tolerate_alarm=False):
        raw = self.inner.read(sid, address, size, tolerate_alarm)
        self.controller.record_read(sid, address, raw)
        return raw

    def value(self, sid, address, size=1, tolerate_alarm=False):
        return u16(self.read(sid, address, size, tolerate_alarm))

    def write(self, sid, address, value, tolerate_alarm=False):
        entry = {**self.controller.context(), "servo_id": sid,
                 "address": address, "value": value, "acknowledged": False}
        self.controller.writes.append(entry)
        try:
            self.inner.write(sid, address, value, tolerate_alarm)
            entry["acknowledged"] = True
        except BaseException as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            raise

    def set_checked(self, sid, address, value):
        from so101_hold_current import WRITE_WIDTHS
        self.write(sid, address, value)
        if self.value(sid, address, WRITE_WIDTHS[address]) != value:
            raise ControlError(f"ID {sid}: 寄存器 {address} 写后读回不一致")


class ElbowProbeController(HoldController):
    """Limited standalone diagnostic. Parameters intentionally are not tunable."""

    def __init__(self, bus, *, single_step_eight=False, ready_check=None, emit=print,
                 clock=time.monotonic, sleep=time.sleep):
        if type(single_step_eight) is not bool:
            raise ValueError("single_step_eight 必须为布尔值")
        super().__init__(bus, limit=100, seconds=5, emit=emit,
                         clock=clock, sleep=sleep)
        self.single_step_eight = single_step_eight
        self.offsets = (-8,) if single_step_eight else OFFSETS
        self.baseline_bias = 2 if single_step_eight else BASELINE_BIAS
        self.mode = "single_step_eight" if single_step_eight else "two_steps_four"
        self.step_description = "单步" if single_step_eight else "两步"
        self.started = self.clock()
        self.phase = "preflight"
        self.initial_targets = {}
        self.device_limits = {}
        self.baseline = None
        self.expected_position = None
        self.total_corridor = None
        self.step_corridor = None
        self.reads, self.writes, self.steps = [], [], []
        self.baseline_samples = []
        self.precommand_checks = []
        self.fault = None
        self.release_result = None
        self.status = "not_started"
        self.ready_check = ready_check
        self.motion_response_complete = False
        self.bus = JournalBus(bus, self)

    def context(self):
        return {"elapsed_s": self.clock() - self.started, "phase": self.phase}

    def record_read(self, sid, address, raw):
        entry = {**self.context(), "servo_id": sid, "address": address,
                 "raw_hex": raw.hex()}
        self.reads.append(entry)  # Preserve the exact failing reply, too.
        if address == 0 and len(raw) == 34:
            self.device_limits[sid] = (u16(raw[9:11]), u16(raw[11:13]))
        elif address == 56 and len(raw) == 10:
            entry.update(decode_motion_feedback(raw))
            goal = self.targets.get(sid)
            entry["command_target"] = goal
            entry["absolute_error"] = (None if goal is None
                                       else entry["position"] - goal)
            if sid == 3:
                entry["loaded_baseline"] = self.baseline
                entry["expected_position"] = self.expected_position
                entry["relative_error"] = (
                    None if self.expected_position is None
                    else entry["position"] - self.expected_position)
            self.check_position(sid, entry["position"])

    def check_position(self, sid, position):
        # This runs on each feedback reply, including preparation reads.
        if self.phase == "release" or not self.initial_targets:
            return
        if sid == 4 and abs(position - self.initial_targets[4]) > WRIST_TOLERANCE:
            raise ControlError("ID4 相对本次初始目标漂移超过 3 计数，停止")
        if sid == 3:
            for label, corridor in (("总反馈", self.total_corridor),
                                    ("当前小步", self.step_corridor)):
                if corridor is not None and not corridor[0] <= position <= corridor[1]:
                    raise ControlError(
                        f"ID3 {label}越界：实际={position}, "
                        f"允许={corridor[0]}–{corridor[1]}；不执行故障回程")

    def preflight(self):
        super().preflight()
        self.initial_targets = dict(self.targets)
        g0 = self.initial_targets[3]
        # Before any writes, include every permitted loaded baseline, command,
        # +/-2 feedback band, ramp/handover guard and other axes' guards.
        # Enforce device limits AND single-turn limits, with an extra 16 counts.
        self.preflight_corridors = {}
        for sid, initial in self.initial_targets.items():
            if sid == 3:
                corridor = (min(g0 - HOLD_TOLERANCE,
                                g0 - self.baseline_bias + self.offsets[-1] - RELATIVE_TOLERANCE),
                            g0 + max(HOLD_TOLERANCE,
                                     self.baseline_bias + RELATIVE_TOLERANCE))
            else:
                tolerance = WRIST_TOLERANCE if sid == 4 else HOLD_TOLERANCE
                corridor = (initial - tolerance, initial + tolerance)
            lower, upper = self.device_limits[sid]
            allowed = (max(0, lower) + BOUNDARY_MARGIN,
                       min(4095, upper) - BOUNDARY_MARGIN)
            if not allowed[0] <= corridor[0] <= corridor[1] <= allowed[1]:
                raise ControlError(
                    f"ID{sid} 完整反馈走廊 {corridor} 距设备/单圈边界不足 "
                    f"{BOUNDARY_MARGIN} 计数，未发送写入")
            self.preflight_corridors[sid] = corridor
        path = " → ".join(str(g0 + offset) for offset in (0, *self.offsets))
        self.emit(f"ID3 固定{self.step_description}计划：{path}，单程不返回；"
                  "其余关节保持本次反馈目标，ID6 不使能。")
        self.emit(f"模式={self.mode}；有载基线相对 G0 偏差上限="
                  f"±{self.baseline_bias} 计数，输出上限仍为 100/1000。")
        self.emit("有载基线 P0 只用于核验真实相对位移；不会加到目标或修改零位。")

    def prepare(self):
        self.phase = "prepare"
        try:
            super().prepare()
        except BaseException as error:
            self.capture_fault(error)
            raise

    def capture_fault(self, error):
        if self.fault is None:
            self.fault = {**self.context(), "type": type(error).__name__,
                          "message": str(error)}

    def monitor(self, expected_torque, tolerance, expected_output):
        # Preserve absolute error, health, control readbacks, saturation duration,
        # ID6 torque prohibition and bounded whole-cycle communication checks.
        return super().monitor(expected_torque, tolerance, expected_output)

    def ramp(self):
        self.phase = "ramp"
        previous = 0
        for level in (25, 50, 75, 100):
            self.monitor(1, HOLD_TOLERANCE, previous)
            started = self.clock()
            for sid in ARM_IDS:
                self.bus.set_checked(sid, OUTPUT, level)
                if self.clock() - started > .5:
                    raise ControlError("输出上限写入累计超过 0.5 秒")
            self.monitor(1, HOLD_TOLERANCE, level)
            self.sleep(.1)
            previous = level

    def collect_baseline(self):
        self.phase = "loaded_baseline"
        self.emit("开始采集有载基线 P0；手在下方保持承接，不主动推拉。"
                  f"稳定后自动执行{self.step_description}，异常立即停止，不回程。")
        g0 = self.initial_targets[3]
        deadline = self.clock() + 3.0
        window = deque()
        while self.clock() < deadline:
            states = self.monitor(1, HOLD_TOLERANCE, self.limit)
            now, position = self.clock(), states[3].position
            if now >= deadline:
                break
            self.baseline_samples.append({"elapsed_s": now-self.started,
                                          "position": position})
            if abs(position - g0) > self.baseline_bias:
                raise ControlError(f"ID3 有载基线相对 G0 偏差超过 "
                                   f"{self.baseline_bias} 计数，停止")
            window.append((now, position))
            while window and now - window[0][0] > .6:
                window.popleft()
            positions = [value for _, value in window]
            if (len(window) >= 3 and now - window[0][0] >= .4
                    and max(positions) - min(positions) <= 2):
                self.baseline = int(round(statistics.median(positions)))
                self.expected_position = self.baseline
                self.total_corridor = (
                    min(g0 - 8, self.baseline - 8) - RELATIVE_TOLERANCE,
                    max(g0, self.baseline) + RELATIVE_TOLERANCE)
                self.step_corridor = (self.baseline - 2, self.baseline + 2)
                self.check_position(3, position)
                self.emit(f"固定基线：G0={g0}，P0={self.baseline}，"
                          f"有载偏差={self.baseline-g0:+d}；只检查相对运动。")
                return
            self.sleep(.05)
        raise ControlError("ID3 有载反馈在 3 秒内未稳定，停止；不发送运动目标")

    def move_step(self, offset):
        self.phase = f"step_{offset}"
        previous_expected = self.expected_position
        expected = self.baseline + offset
        goal = self.initial_targets[3] + offset
        self.step_corridor = (min(previous_expected, expected) - 2,
                              max(previous_expected, expected) + 2)
        # Check the current state under the old target before sending a new one.
        states = self.monitor(1, HOLD_TOLERANCE, self.limit)
        if self.single_step_eight:
            # P0 alone cannot rule out drift after the stable window. Use the
            # latest full-axis monitor, before changing the target register.
            current = states[3].position
            delta_from_g0 = current - self.initial_targets[3]
            command_gap = current - goal
            accepted = (abs(delta_from_g0) <= self.baseline_bias
                        and abs(command_gap) <= HOLD_TOLERANCE)
            self.precommand_checks.append({**self.context(), "position": current,
                                           "goal": goal, "delta_from_g0": delta_from_g0,
                                           "command_gap": command_gap, "accepted": accepted})
            if not accepted:
                raise ControlError(
                    "ID3 单步 8 计数写前检查未通过："
                    f"实际相对 G0={delta_from_g0:+d}（允许 ±2），"
                    f"相对新目标={command_gap:+d}（允许 ±{HOLD_TOLERANCE}）；"
                    "未发送运动目标，不更新基线")
        self.targets[3] = goal
        self.expected_position = expected
        self.bus.set_checked(3, GOAL, goal)
        self.emit(f"ID3 命令={goal}；固定基线的预期反馈={expected}±2。")
        deadline = self.clock() + 3.0
        consecutive, first_in_band = 0, None
        while self.clock() < deadline:
            states = self.monitor(1, HOLD_TOLERANCE, self.limit)
            now, state = self.clock(), states[3]
            if now >= deadline:
                break
            error = state.position - expected
            if abs(error) <= RELATIVE_TOLERANCE:
                consecutive += 1
                if first_in_band is None:
                    first_in_band = now
                if consecutive >= 3 and now - first_in_band >= .2:
                    self.steps.append({"offset": offset, "goal": goal,
                                       "expected_feedback": expected,
                                       "position": state.position,
                                       "relative_error": error,
                                       "absolute_error": state.position-goal,
                                       "consecutive_samples": consecutive,
                                       "span_s": now-first_in_band})
                    self.step_corridor = (expected-2, expected+2)
                    self.emit(f"ID3 相对位移达标：实际={state.position}，"
                              f"相对 P0={state.position-self.baseline:+d}，"
                              f"命令误差={state.position-goal:+d}；"
                              "此处不表示绝对目标到位。")
                    return
            else:
                consecutive, first_in_band = 0, None
            self.sleep(.05)
        raise ControlError(f"ID3 相对位移未在 3 秒内达标：固定预期={expected}±2；"
                           "停止，不更新基线、不执行回程")

    def wait_for_handover(self):
        self.phase = "support_handover"
        if hasattr(self.ready_check, "begin"):
            self.ready_check.begin()
        self.emit("15 秒内完成承重交接：从下方承接前臂，让纸箱脱离接触，"
                  "逐渐减到本次运动要保持的托力，稳定后按 Enter。"
                  f"按 Enter 后直到{self.step_description}运动完成，不再改变托力、不主动推拉。"
                  "等待期间持续监测；超时、EOF 或异常会释放扭矩。")
        end = self.clock()+15.0
        while self.clock() < end:
            self.monitor(1, HOLD_TOLERANCE, self.limit)
            if self.clock() >= end:
                break
            ready = self.ready_check()
            if self.clock() >= end:
                break
            if ready:
                return
            self.sleep(.05)
        raise ControlError("承重交接在 15 秒内未就绪，停止；未发送运动目标")

    def _hold(self):
        self.ramp()
        self.wait_for_handover()
        self.collect_baseline()
        for offset in self.offsets:
            self.move_step(offset)
        self.motion_response_complete = True
        self.phase = "final_monitor"
        # Restore physical support before release. That changes the static
        # load bias, so keep the fixed GLOBAL guard, not the relative-arrival
        # band. The completed movement evidence remains separate in the log.
        self.step_corridor = None
        self.emit(f"{self.step_description}相对位移达标；继续承托，监测 5 秒后释放扭矩，不返回起点。")
        end, next_print = self.clock()+5.0, self.clock()
        while self.clock() < end:
            states = self.monitor(1, HOLD_TOLERANCE, self.limit)
            now, state = self.clock(), states[3]
            if now >= next_print:
                self.emit(f"剩余 {max(0, end-now):.1f}s  ID3={state.position}, "
                          f"相对 P0={state.position-self.baseline:+d}, "
                          f"命令误差={state.position-self.targets[3]:+d}, "
                          f"输出={state.load}/1000")
                next_print = now+.5
            self.sleep(.05)

    def hold(self):
        try:
            self._hold()
        except BaseException as error:
            self.capture_fault(error)
            raise

    def release(self):
        self.phase = "release"
        unconfirmed = super().release()
        self.release_result = {"attempted_ids": sorted(self.touched),
                               "unconfirmed_ids": unconfirmed,
                               "confirmed": bool(self.touched) and not unconfirmed}
        return unconfirmed

    def run(self, enable=False):
        try:
            if enable and self.ready_check is None:
                raise ControlError("实际运行必须提供承重交接就绪输入；默认只读")
            if enable:
                super().run(True)
            else:
                self.preflight()
                self.emit("只读预览完成，未发送写入；--enable 才会实际使能，"
                          f"并在承重交接就绪后执行 ID3 固定负向{self.step_description}诊断。")
        except BaseException as error:
            self.status = "failed"
            self.capture_fault(error)
            raise
        else:
            self.status = "relative_probe_complete" if enable else "read_only_preview"

    def report(self):
        return {"schema_version": 1, "test": "elbow_relative_negative_probe",
                "status": self.status, "joint_mapping_used": False,
                "mode": self.mode, "baseline_bias_limit": self.baseline_bias,
                "automatic_return": False, "limit": self.limit,
                "original_targets": self.initial_targets,
                "last_command_targets": dict(self.targets),
                "loaded_baseline": self.baseline,
                "motion_response_complete": self.motion_response_complete,
                "command_offsets": list(self.offsets),
                "precommand_checks": self.precommand_checks,
                "preflight_corridors": getattr(self, "preflight_corridors", {}),
                "total_feedback_corridor": self.total_corridor,
                "baseline_samples": self.baseline_samples,
                "completed_steps": self.steps, "reads": self.reads,
                "write_attempts": self.writes, "fault": self.fault,
                "release": self.release_result,
                "note": "Only relative feedback displacement is checked; not calibration or grasp acceptance."}


class TerminalReady:
    """An attended terminal line, polled without blocking feedback checks."""

    def __init__(self, stream):
        if not stream.isatty():
            raise ControlError("--enable 必须在交互终端运行，不能通过管道预先发送 Enter")
        self.stream = stream
        self.old_flags = None
        self.buffer = b""

    def begin(self):
        fd = self.stream.fileno()
        termios.tcflush(fd, termios.TCIFLUSH)
        self.buffer = b""
        self.old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, self.old_flags | os.O_NONBLOCK)

    def __call__(self):
        if not select.select([self.stream], [], [], 0)[0]:
            return False
        try:
            chunk = os.read(self.stream.fileno(), 4096)
        except BlockingIOError:
            return False
        if chunk == b"":
            raise ControlError("承重交接输入 EOF，停止")
        self.buffer += chunk
        if len(self.buffer) > 4096:
            raise ControlError("承重交接输入过长，停止")
        if b"\n" not in self.buffer:
            return False
        line, _ = self.buffer.split(b"\n", 1)
        if line.strip():
            raise ControlError("承重交接只接受就绪后的空行 Enter，停止")
        return True

    def close(self):
        if self.old_flags is not None:
            fcntl.fcntl(self.stream.fileno(), fcntl.F_SETFL, self.old_flags)
            self.old_flags = None


def main(argv=None):
    parser = argparse.ArgumentParser(description="ID3 固定负向 8 计数单程诊断；默认只读预览。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--enable", action="store_true", help="实际使能并运行所选固定诊断")
    parser.add_argument("--single-step-eight", action="store_true",
                        help="显式选择单次 -8 计数；仅允许有载基线及写前反馈偏差 ≤2，仍需 --enable 才运动")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]/"data"/"real_control_runs")
    args = parser.parse_args(argv)
    try:
        ready = TerminalReady(sys.stdin) if args.enable else None
    except ControlError as error:
        parser.exit(1, str(error)+"\n")
    try:
        import serial
    except ImportError:
        parser.exit(1, "请使用 /home/yu/so101_ws/.venv_vision/bin/python（已安装 pyserial）。\n")
    # Verify log destination before opening the port or sending any writes.
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_")
    fd, filename = tempfile.mkstemp(prefix=f"elbow_probe_{stamp}", suffix=".json", dir=args.log_dir)
    controller, result, outer_fault = None, 0, None
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        if args.enable:
            steps = "单次 8 计数" if args.single_step_eight else "两步共 8 计数"
            print(f"ID3 向计数减小方向{steps}，不自动返回；手从下方保持承接，"
                  "不推拉、不夹紧连杆。退出或异常会释放扭矩。关闭其他串口控制程序。", flush=True)
        else:
            print("只读预览：不改变目标、扭矩、限位或 PID；不会使机械臂运动。", flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            controller = ElbowProbeController(Bus(port), ready_check=ready,
                                              single_step_eight=args.single_step_eight,
                                              emit=lambda msg: print(msg, flush=True))
            controller.run(args.enable)
    except KeyboardInterrupt:
        outer_fault, result = "KeyboardInterrupt", 130
        print("已中止；请以上方逐 ID 释放读回结果为准。", flush=True)
    except (ControlError, OSError, ValueError) as error:
        outer_fault, result = f"{type(error).__name__}: {error}", 1
        print(f"控制未完成：{error}", file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if ready is not None:
            try:
                ready.close()
            except OSError as error:
                print(f"终端输入状态恢复失败：{error}；继续保存本次记录。",
                      file=sys.stderr, flush=True)
                if result == 0:
                    result = 1
        record = controller.report() if controller is not None else {"status": "not_started"}
        record["cli_fault"] = outer_fault
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(record, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            print(f"本次记录：{filename}", flush=True)
        except OSError as error:
            print(f"日志保存失败：{error}；硬件释放状态以上方逐 ID 读回为准。",
                  file=sys.stderr, flush=True)
            if result == 0:
                result = 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
