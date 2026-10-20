"""One selected joint's local movement. No model mapping or old-pose targets."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

from so101_hold_current import (
    ARM_IDS, GOAL, HOLD_TOLERANCE, OUTPUT, Bus, ControlError, HoldController,
    Feedback, check_health, signed_magnitude, u16,
)


def decode_motion_feedback(raw):
    """Preserve signed speed/output from the same feedback packet as position."""
    return {
        "raw_hex": raw.hex(),
        "position": signed_magnitude(u16(raw[:2]), 15),
        "speed_raw_signed": signed_magnitude(u16(raw[2:4]), 15),
        "output_raw_signed": signed_magnitude(u16(raw[4:6]), 10),
        "voltage": raw[6]/10, "temperature": raw[7], "status": raw[9],
    }


class RecordingBus:
    """Observe successful READ responses before controller validation can raise."""
    def __init__(self, inner, record):
        self.inner, self.record = inner, record

    def read(self, sid, address, size=1, tolerate_alarm=False):
        raw = self.inner.read(sid, address, size, tolerate_alarm)
        self.record(sid, address, raw)
        return raw

    def __getattr__(self, name):
        return getattr(self.inner, name)


def read_pan_diagnostics(bus, sleep=time.sleep, clock=time.monotonic, *, servo_id=1):
    """Read the selected joint only; default ID1 preserves the original API."""
    if type(servo_id) is not int or servo_id not in (1, 2):
        raise ValueError("只读诊断仅支持 ID1 或 ID2")
    config = bus.read(servo_id, 0, 40)
    control = bus.read(servo_id, 40, 10)
    result = {
        "schema_version": 1,
        "test": "pan_read_only_diagnostics" if servo_id == 1 else "joint_read_only_diagnostics",
        "servo_id": servo_id, "writes_sent": 0,
        "register_reference": "https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation",
        "firmware_version_bytes": list(config[:2]),
        "model_version_bytes": list(config[3:5]),
        "mode": config[33], "phase": config[18], "resolution": config[30],
        "position_limits": [u16(config[9:11]), u16(config[11:13])],
        "offset_raw": u16(config[31:33]),
        "P": config[21], "D": config[22], "I": config[23],
        "minimum_starting_output": config[24], "integral_limit": config[25],
        "positive_dead_zone": config[26], "negative_dead_zone": config[27],
        "maximum_output": u16(config[16:18]),
        "torque_enabled": control[0], "acceleration": control[1],
        "goal_position_raw": u16(control[2:4]), "speed_raw": u16(control[6:8]),
        "output_limit": u16(control[8:10]),
        "configuration_raw_hex": config.hex(), "control_raw_hex": control.hex(),
        "feedback": [],
        "note": "Post-stop readback only. Current registers do not prove their values throughout the previous movement.",
    }
    started = clock()
    for index in range(5):
        raw = bus.read(servo_id, 56, 10)
        result["feedback"].append({"elapsed_s": clock()-started, **decode_motion_feedback(raw)})
        if index < 4:
            sleep(.1)
    return result


class PanStepController(HoldController):
    def __init__(self, bus, delta=16, probe_negative=False, start_range=None,
                 moving_id=1, probe_shoulder=False, trace_motion=False, **kwargs):
        if type(moving_id) is not int or moving_id not in (1, 2):
            raise ValueError("本次只允许选择 ID1 或 ID2")
        if probe_shoulder and (moving_id != 2 or probe_negative):
            raise ValueError("肩部单步仅支持 ID2，不能与负向探测同用")
        if trace_motion and not probe_shoulder:
            raise ValueError("运动采样仅支持 ID2 肩部单步模式")
        if moving_id == 2:
            if (not probe_shoulder and delta != 8) or probe_negative:
                raise ValueError("ID2 仅允许 +8 计数往返或固定 +4 计数单步，不能负向探测")
        self.moving_id = moving_id
        self.one_way_probe = probe_negative or probe_shoulder
        if start_range is not None:
            if (len(start_range) != 2 or any(type(value) is not int for value in start_range)
                    or not 0 <= start_range[0] <= start_range[1] <= 4095):
                raise ValueError("起点范围须为整数，且满足 0 ≤ 下界 ≤ 上界 ≤ 4095")
            if self.one_way_probe:
                raise ValueError("起点范围仅用于普通往返或其只读预览")
        self.start_range = tuple(start_range) if start_range is not None else None
        if probe_negative:
            delta = -8
        elif probe_shoulder:
            delta = 4
        if not probe_shoulder and delta not in (-16, -12, -8, 8, 12, 16):
            raise ValueError("位移只允许 ±8、±12、±16 个计数")
        super().__init__(bus, limit=100, seconds=2, **kwargs)
        self.probe_negative = probe_negative
        self.probe_shoulder = probe_shoulder
        self.trace_motion = trace_motion
        self.probe_hold_complete = False
        self.delta = delta
        self.anchor = None
        self.start_position_counts = {}
        self.path = []
        self.samples = []
        self.reached = []
        self.phase = "preflight"
        self.outbound_complete = False
        self.return_complete = False
        self.final_anchor_hold_complete = False
        self.release_unconfirmed = None
        self.started = self.clock()
        self.feedback_reads = []
        self.control_reads = []
        self.motion_trace_reads = []
        self.fault_context = None
        self.step_corridor = None
        self.corridor = None
        self.bus = RecordingBus(self.bus, self.record_read)

    def record_read(self, sid, address, raw):
        context = {"elapsed_s": self.clock()-self.started, "phase": self.phase,
                   "servo_id": sid, "expected_target": self.targets.get(sid)}
        if self.trace_motion and sid == self.moving_id and address == 40 and len(raw) == 31:
            feedback = decode_motion_feedback(raw[16:26])
            trace = {
                **context, **feedback, "raw_hex": raw.hex(),
                "feedback_raw_hex": feedback["raw_hex"],
                "torque_enabled": raw[0], "acceleration": raw[1],
                "goal_readback": u16(raw[2:4]), "speed_readback": u16(raw[6:8]),
                "output_limit_readback": u16(raw[8:10]), "moving_flag": raw[26],
                "internal_target_raw": u16(raw[27:29]), "current_raw": u16(raw[29:31]),
            }
            # Keep the same reply even if a position/health/control check fails.
            self.motion_trace_reads.append(trace)
            self.feedback_reads.append({**context, **feedback, "source": "motion_trace"})
            self.control_reads.append({
                **context, "source": "motion_trace", "raw_hex": raw[1:10].hex(),
                "acceleration": trace["acceleration"], "goal_readback": trace["goal_readback"],
                "speed_readback": trace["speed_readback"],
                "output_limit_readback": trace["output_limit_readback"],
            })
            self.check_pan_position(trace["position"])
        elif address == 56 and len(raw) == 10:
            entry = {**context, **decode_motion_feedback(raw)}
            self.feedback_reads.append(entry)
            # Stop on the pan reply itself; do not wait for the remaining axes.
            if sid == self.moving_id:
                self.check_pan_position(entry["position"])
        elif address == 41 and len(raw) == 9:
            self.control_reads.append({**context, "raw_hex": raw.hex(),
                                       "acceleration": raw[0], "goal_readback": u16(raw[1:3]),
                                       "speed_readback": u16(raw[5:7]),
                                       "output_limit_readback": u16(raw[7:9])})

    def read_motion_trace(self, validate=True):
        try:
            return self.read_motion_trace_checked(validate)
        except BaseException as error:
            self.capture_fault(error)
            raise

    def read_motion_trace_checked(self, validate):
        started = self.clock()
        self.bus.read(self.moving_id, 40, 31)
        trace = self.motion_trace_reads[-1]
        if self.clock()-started > .05:
            raise ControlError("ID2 扩展反馈读取超过50毫秒，停止")
        state = Feedback(trace["position"], abs(trace["output_raw_signed"]),
                         trace["voltage"], trace["temperature"], trace["status"])
        check_health(self.moving_id, state, *self.configs[self.moving_id])
        if trace["torque_enabled"] != (1 if validate else 0):
            raise ControlError("ID2 扩展反馈：扭矩状态意外改变")
        if validate:
            if (trace["acceleration"] != 1 or trace["goal_readback"] != self.targets[self.moving_id]
                    or trace["speed_readback"] != 50 or trace["output_limit_readback"] != self.limit):
                raise ControlError("ID2 扩展反馈：目标/速度/输出上限被改变")
            # Share the full monitor's timer so extra reads cannot reset saturation.
            now = self.clock()
            if state.load >= self.limit * .95:
                self.high_load_since.setdefault(self.moving_id, now)
                if now-self.high_load_since[self.moving_id] > 1.0:
                    raise ControlError("ID2 输出持续接近本次上限，停止；不会自动加大输出")
            else:
                self.high_load_since.pop(self.moving_id, None)
        return trace

    def pause_with_motion_trace(self, duration):
        if not self.trace_motion or self.phase not in ("outbound", "probe_hold"):
            self.sleep(duration)
            return True
        deadline = self.clock()+duration
        all_in_target_band = True
        while self.clock() < deadline:
            trace = self.read_motion_trace()
            if abs(trace["position"]-self.targets[self.moving_id]) > 2:
                all_in_target_band = False
            remaining = deadline-self.clock()
            if remaining > 0:
                self.sleep(min(.005, remaining))
        return all_in_target_band

    def check_pan_position(self, position):
        if self.corridor is None or self.phase in ("preflight", "release"):
            return
        if self.phase in ("outbound", "return") and self.step_corridor is not None:
            lower, upper = self.step_corridor
            if not lower <= position <= upper:
                raise ControlError(f"ID{self.moving_id} 当前小步反馈越界：目标={self.targets[self.moving_id]}, "
                                   f"实际={position}, 允许={lower}–{upper}；停止，不执行故障回程")
        if not self.corridor[0] <= position <= self.corridor[1]:
            raise ControlError(f"ID{self.moving_id} 反馈超出固定往返范围（含2计数反馈余量），停止")
        if self.phase == "before_release" and abs(position-self.anchor) > 2:
            raise ControlError(f"ID{self.moving_id} 返回后再次偏离起点超过2计数，停止")
        if self.phase == "probe_hold" and abs(position-self.outbound[-1]) > 2:
            raise ControlError(f"ID{self.moving_id} 单程诊断到位后偏离末目标超过2计数，停止")

    def capture_fault(self, error):
        self.fault_context = {
            "elapsed_s": self.clock()-self.started, "phase": self.phase,
            "reason": str(error), "targets": dict(self.targets),
            "step_corridor": self.step_corridor,
            "last_motion_trace": self.motion_trace_reads[-1] if self.motion_trace_reads else None,
            "last_feedback_by_id": {sid: next((item for item in reversed(self.feedback_reads)
                                                if item["servo_id"] == sid), None)
                                    for sid in range(1, 7)},
        }

    def preflight(self):
        super().preflight()
        self.start_position_counts = dict(self.targets)
        self.anchor = self.targets[self.moving_id]
        if self.start_range is not None:
            lower_start, upper_start = self.start_range
            if not lower_start <= self.anchor <= upper_start:
                raise ControlError(f"ID{self.moving_id} 本次起点={self.anchor}，不在指定起点范围 "
                                   f"{lower_start}–{upper_start} 内；未发送任何控制写入")
        raw_limits = self.bus.read(self.moving_id, 9, 4)
        lower, upper = u16(raw_limits[:2]), u16(raw_limits[2:])
        sign = 1 if self.delta > 0 else -1
        self.outbound = [self.anchor + sign * n for n in range(4, abs(self.delta) + 1, 4)]
        self.return_path = [] if self.one_way_probe else list(reversed([self.anchor] + self.outbound[:-1]))
        self.path = self.outbound + self.return_path
        if any(not 0 <= target <= 4095 or not lower <= target <= upper for target in self.path):
            raise ControlError("本次完整往返路径越过计数边界/设备限位；未发送任何控制写入")
        if any(abs(target - self.anchor) > 16 for target in self.path):
            raise ControlError("累计位移超过本次起点 ±16 计数")
        if self.one_way_probe and any(not lower+16 <= target <= upper-16 for target in self.path):
            raise ControlError("单程诊断目标距设备计数边界不足16计数；未发送任何控制写入")
        self.corridor = (max(lower, min(self.anchor, self.anchor+self.delta)-2),
                         min(upper, max(self.anchor, self.anchor+self.delta)+2))
        destination = f"{self.anchor+self.delta}" if self.one_way_probe else f"{self.anchor+self.delta} → {self.anchor}"
        suffix = "本次单程诊断不自动返回。" if self.one_way_probe else ""
        self.emit(f"ID{self.moving_id} 计划: {self.anchor} → {destination}; "
                  f"单程 {self.delta:+d} 计数（{self.delta*360/4096:+.2f}°），每步4计数。{suffix}")
        if self.trace_motion:
            # Prove the extended READ works before prepare() sends any writes.
            self.read_motion_trace(validate=False)
            self.emit("运动采样已就绪：单步运动和末端保持时，在全轴检查间隙加采ID2反馈。")

    def monitor(self, expected_torque, tolerance, expected_output):
        try:
            return self.monitor_recorded(expected_torque, tolerance, expected_output)
        except Exception as error:
            self.capture_fault(error)
            raise

    def monitor_recorded(self, expected_torque, tolerance, expected_output):
        states = super().monitor(expected_torque, tolerance, expected_output)
        self.samples.append({
            "elapsed_s": self.clock() - self.started, "phase": self.phase,
            "targets": dict(self.targets),
            "feedback": {sid: asdict(state) for sid, state in states.items()},
        })
        return states

    def settle(self, target):
        deadline = self.clock() + 3.0
        good = 0
        while self.clock() < deadline:
            states = self.monitor(1, HOLD_TOLERANCE, self.limit)
            if self.clock() >= deadline:
                break
            actual = states[self.moving_id].position
            good = good + 1 if abs(actual - target) <= 2 else 0
            if good >= 3:
                self.reached.append({"target": target, "actual": actual,
                                     "elapsed_s": self.clock()-self.started})
                self.emit(f"ID{self.moving_id} 到位: 目标={target}, 实际={actual}, 误差={actual-target:+d}, 连续3次")
                return
            if not self.pause_with_motion_trace(.05):
                good = 0
        raise ControlError(f"ID{self.moving_id} 目标 {target} 未在3秒内连续到位，停止，不执行故障回程")

    def move_step(self, target):
        self.monitor(1, HOLD_TOLERANCE, self.limit)
        if target not in self.path or abs(target-self.targets[self.moving_id]) > 4:
            raise ControlError("禁止执行预计算路径之外的目标")
        previous_target = self.targets[self.moving_id]
        self.step_corridor = (max(self.corridor[0], min(previous_target, target)-2),
                              min(self.corridor[1], max(previous_target, target)+2))
        self.targets[self.moving_id] = target  # Even lost ACK may mean the target took effect.
        self.bus.set_checked(self.moving_id, GOAL, target)
        if self.trace_motion:
            self.read_motion_trace()
        self.settle(target)

    def hold(self):
        try:
            self.hold_recorded()
        except BaseException as error:
            # Also preserve timeout, lost write ACK and interruption context.
            if self.fault_context is None:
                self.capture_fault(error)
            raise

    def hold_recorded(self):
        self.phase = "ramp"
        previous = 0
        for level in (25, 50, 75, 100):
            self.monitor(1, HOLD_TOLERANCE, previous)
            start = self.clock()
            for sid in ARM_IDS:
                self.bus.set_checked(sid, OUTPUT, level)
                if self.clock()-start > .5:
                    raise ControlError("输出上限写入累计超过0.5秒")
            self.monitor(1, HOLD_TOLERANCE, level)
            self.sleep(.1)
            previous = level
        self.phase = "settle_anchor"
        self.settle(self.anchor)
        self.phase = "outbound"
        for target in self.outbound:
            self.move_step(target)
        self.outbound_complete = True
        if self.one_way_probe:
            self.phase = "probe_hold"
            self.emit("ID2单步到位，不自动返回；2秒后释放扭矩，请继续承托。" if self.probe_shoulder
                      else "负向单程诊断到位，不自动返回；2秒后释放扭矩，请托住机械臂。")
        else:
            self.emit("单程到位，开始沿已计算的小步路径返回本次起点。")
            self.phase = "return"
            for target in self.return_path:
                self.move_step(target)
            self.return_complete = True
            self.phase = "before_release"
            self.emit("已返回本次起点；2秒后释放扭矩，请托住机械臂。")
        deadline = self.clock()+2
        while self.clock() < deadline:
            self.monitor(1, HOLD_TOLERANCE, self.limit)
            self.pause_with_motion_trace(.05)
        if self.one_way_probe:
            self.probe_hold_complete = True
        else:
            self.final_anchor_hold_complete = True

    def release(self):
        self.phase = "release"
        self.release_unconfirmed = super().release()
        return self.release_unconfirmed

    def report(self, outcome, reason):
        trace_times = [entry["elapsed_s"] for entry in self.motion_trace_reads
                       if entry["phase"] in ("outbound", "probe_hold")]
        intervals = [later-earlier for earlier, later in zip(trace_times, trace_times[1:])]
        return {
            "schema_version": 7 if self.trace_motion else 6, "source": "controller_feedback",
            "test": ("joint_single_step_probe" if self.probe_shoulder else
                     "joint_local_step_and_return" if self.moving_id != 1 else
                     "pan_negative_probe" if self.probe_negative else "pan_local_step_and_return"),
            "moving_id": self.moving_id,
            "start_pose_window": None,
            "start_pose_source": "fresh_stable_feedback",
            "start_position_counts": dict(self.start_position_counts),
            "probe_negative": self.probe_negative, "probe_shoulder": self.probe_shoulder,
            "probe_hold_complete": self.probe_hold_complete,
            "outcome": outcome,
            "stop_reason": reason, "anchor": self.anchor, "delta_counts": self.delta,
            "start_range": list(self.start_range) if self.start_range is not None else None,
            "planned_path": self.path, "output_limit": self.limit,
            "outbound_complete": self.outbound_complete,
            "return_complete": self.return_complete,
            "final_anchor_hold_complete": self.final_anchor_hold_complete,
            "touched_ids": sorted(self.touched),
            "release_unconfirmed_ids": self.release_unconfirmed,
            "reached": self.reached, "samples": self.samples,
            "feedback_reads": self.feedback_reads, "control_reads": self.control_reads,
            "trace_motion": self.trace_motion, "motion_trace_reads": self.motion_trace_reads,
            "motion_trace_timing": {
                "sample_count": len(trace_times),
                "min_interval_s": min(intervals) if intervals else None,
                "max_interval_s": max(intervals) if intervals else None,
                "mean_interval_s": sum(intervals)/len(intervals) if intervals else None,
            },
            "fault_context": self.fault_context,
            "model_mapping_verified": False,
            "note": "Local encoder motion only. Complete monitor cycles are in samples; successful raw feedback READs, including position-fault samples, are in feedback_reads. A failed communication cannot provide a valid physical sample. Mechanical ranges and automatic grasp are not validated.",
        }


def save_report(controller, outcome, reason, directory):
    # Called only after run() has completed all cleanup attempts.
    stamp = datetime.now(timezone.utc)
    report = controller.report(outcome, reason)
    report["saved_at_utc"] = stamp.isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "pan" if controller.moving_id == 1 else f"joint{controller.moving_id}"
    path = directory / f"{prefix}_{stamp.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def save_diagnostics(result, directory):
    stamp = datetime.now(timezone.utc)
    result = {**result, "saved_at_utc": stamp.isoformat()}
    directory.mkdir(parents=True, exist_ok=True)
    servo_id = result.get("servo_id", 1)
    prefix = "pan" if servo_id == 1 else f"joint{servo_id}"
    path = directory / f"{prefix}_readback_{stamp.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="选定单关节局部控制，其余臂关节原位保持，ID6不使能。默认只读预览。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--joint", type=int, choices=(1, 2), default=1, help="运动或只读诊断关节，默认ID1；ID2支持+8往返或专用+4单步")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enable", action="store_true", help="实际执行小幅往返")
    mode.add_argument("--diagnose", action="store_true", help="只读所选关节控制参数和反馈，不使能、不写寄存器")
    mode.add_argument("--probe-negative", action="store_true", help="实际执行ID1固定负向8计数单程诊断，不自动返回")
    mode.add_argument("--probe-shoulder", action="store_true", help="实际执行ID2固定正向4计数单步，不自动返回；需承托减轻肩部负担并允许微动")
    parser.add_argument("--trace-motion", action="store_true", help="仅用于肩部单步，在全轴检查间隙加采ID2控制及反馈寄存器")
    parser.add_argument("--delta", type=int, choices=(-16, -12, -8, 8, 12, 16), default=None,
                        help="普通往返单程计数，默认16；不能与单程诊断模式同用")
    parser.add_argument("--start-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="可选的起点计数范围；默认以本次稳定反馈起步，不要求历史姿态")
    parser.add_argument("--report-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]/"data"/"real_control_runs")
    args = parser.parse_args(argv)
    if args.trace_motion and not args.probe_shoulder:
        parser.error("--trace-motion 仅能与 --joint 2 --probe-shoulder 同用")
    if args.probe_shoulder:
        if args.joint != 2:
            parser.error("--probe-shoulder 必须指定 --joint 2")
        if args.delta is not None:
            parser.error("--probe-shoulder 固定为正向4计数，不能指定 --delta")
    if args.diagnose and args.delta is not None:
        parser.error("--diagnose 为只读模式，不能指定 --delta")
    if args.joint == 2:
        if args.probe_negative:
            parser.error("ID2 不支持 ID1 负向探测模式")
        if args.delta not in (None, 8):
            parser.error("ID2 只允许 --delta 8；起点使用本次稳定反馈")
    if args.probe_negative and args.delta is not None:
        parser.error("--probe-negative 固定为负向8计数，不能指定 --delta")
    if args.start_range is not None:
        if args.diagnose or args.probe_negative or args.probe_shoulder:
            parser.error("--start-range 仅用于普通往返或其只读预览")
        if not 0 <= args.start_range[0] <= args.start_range[1] <= 4095:
            parser.error("起点范围须满足 0 ≤ MIN ≤ MAX ≤ 4095")
    try:
        import serial
    except ImportError:
        parser.exit(1, "请使用 /home/yu/so101_ws/.venv_vision/bin/python。\n")
    if args.diagnose:
        print(f"只读 ID{args.joint} 参数和反馈；不改变扭矩、目标、PID或限位。", flush=True)
        try:
            with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                               write_timeout=.15, exclusive=True) as port:
                result = read_pan_diagnostics(Bus(port), servo_id=args.joint)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            try:
                print(f"只读记录: {save_diagnostics(result, args.report_dir)}", flush=True)
            except OSError as error:
                print(f"读取完成，但记录保存失败: {error}", file=sys.stderr)
            return 0
        except (ControlError, OSError, ValueError) as error:
            print(f"只读诊断未完成: {error}", file=sys.stderr)
            return 1
    def stop(signum, frame):
        raise KeyboardInterrupt
    old_handler = signal.signal(signal.SIGTERM, stop)
    controller = None
    outcome, reason, exit_code = "not_started", None, 0
    action = ("ID2将向计数增大方向移动4计数，本次不自动返回。\n"
              "请从下方承托前臂减轻肩部负担，允许微动；不要夹紧连杆或主动推向目标。" if args.probe_shoulder
              else "ID1将向计数减小方向移动8计数，本次不自动返回。" if args.probe_negative
              else f"ID{args.joint}将小幅转动再返回。" if args.enable else "只读预览，不执行运动。")
    print(action + "确保连杆周围留有余量；支撑不得妨碍关节转动。\n"
          "正常退出/异常均会尝试断扭矩；手保持可承接位置，退出前托住。", flush=True)
    try:
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            controller = PanStepController(Bus(port), delta=args.delta if args.delta is not None else (8 if args.joint == 2 else 16),
                                           probe_negative=args.probe_negative,
                                           probe_shoulder=args.probe_shoulder,
                                           trace_motion=args.trace_motion,
                                           start_range=args.start_range,
                                           moving_id=args.joint,
                                           emit=lambda text: print(text, flush=True))
            enabled = args.enable or args.probe_negative or args.probe_shoulder
            controller.run(enabled)
            outcome = "completed_and_released" if enabled else "read_only_preview"
    except KeyboardInterrupt:
        outcome, reason, exit_code = "interrupted", "KeyboardInterrupt", 130
        print("已中止；释放情况以逐ID读回为准。", flush=True)
    except (ControlError, OSError, ValueError) as error:
        outcome, reason, exit_code = "failed", str(error), 1
        print(f"控制未完成：{error}", file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    if controller is not None:
        try:
            path = save_report(controller, outcome, reason, args.report_dir)
            print(f"本次记录: {path}", flush=True)
        except (OSError, ValueError) as error:
            print(f"控制结果={outcome}；记录保存失败：{error}", file=sys.stderr)
    if outcome == "completed_and_released":
        message = ("ID2单步对照完成并已释放；未执行回程。此结果不代表独立承重或自动抓取通过。" if args.probe_shoulder
                   else "ID1负向单程诊断完成并已释放；未执行回程。" if args.probe_negative
                   else f"ID{args.joint}小幅往返完成并已释放；请同时确认实物确有小幅转动。")
        print(message, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
