"""ID6-only, torque-disabled count recentering; default strictly read-only.

References:
https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation
https://github.com/ftservo/FTServo_Arduino/blob/main/src/SMS_STS.cpp

40=128 calibrates the CURRENT position as the middle. It changes the position
offset; it is not a goal command. No register 55 unlock, explicit EEPROM write,
goal, PID, phase, mode or limit change is exposed. Persistence after a power
cycle is deliberately not claimed by this procedure.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


ALL_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_ID = 6
TORQUE, OUTPUT = 40, 48
OPEN_OBSERVED, CLOSED_OBSERVED = 194, 3953
EXPECTED_FIRMWARE = (3, 10)
EXPECTED_MODEL = (9, 3)
REGISTER_REFERENCE = "https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation"


class CenterError(RuntimeError):
    pass


def u16(data):
    return int.from_bytes(data, "little")


def signed_offset(raw):
    """ST position offset: 11 magnitude bits plus sign bit 11."""
    if not 0 <= raw <= 0xFFF:
        raise CenterError(f"位置修正原始值超出 12 位符号原码：0x{raw:04X}")
    magnitude = raw & 0x7FF
    return -magnitude if raw & 0x800 else magnitude


def circular_difference(left, right):
    """For readback comparison only; never used to generate a motion goal."""
    return (left-right+2048) % 4096 - 2048


def signed_magnitude(raw, sign_bit):
    magnitude = raw & ((1 << sign_bit)-1)
    return -magnitude if raw & (1 << sign_bit) else magnitude


class CenterBus:
    """Dedicated capability: ID6 OUTPUT=0 and TORQUE=0/128, nothing else."""

    def __init__(self, port, *, clock=time.monotonic):
        self.port, self.clock = port, clock
        self.started = clock()
        self.phase = "preview"
        self.events = []
        self.apply_enabled = False
        self.center_attempted = False

    def enable_apply(self):
        self.apply_enabled = True

    def _exchange(self, sid, instruction, params, size, tolerate_alarm=False):
        if sid not in ALL_IDS:
            raise CenterError("只允许访问 ID1–6")
        if instruction == 2:
            if (len(params) != 2 or not 1 <= size <= 40 or params[1] != size
                    or not 0 <= params[0] <= 255 or params[0]+size > 256):
                raise CenterError("读取范围不合法")
        elif instruction == 3:
            allowed = ([TORQUE, 0], [TORQUE, 128], [OUTPUT, 0, 0])
            if not self.apply_enabled or sid != GRIPPER_ID or list(params) not in allowed or size != 0:
                raise CenterError("禁止写入：仅 --apply 后允许 ID6 的 40=0/128 或 48=0")
            if list(params) == [TORQUE, 128]:
                if self.center_attempted:
                    raise CenterError("128 已尝试一次，禁止重试")
                # Mark before transport: the command can take effect without ACK.
                self.center_attempted = True
        else:
            raise CenterError("禁止未列入白名单的串口指令")
        fields = bytes([sid, len(params)+2, instruction]) + bytes(params)
        request = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])
        event = {"elapsed_s": self.clock()-self.started, "phase": self.phase,
                 "servo_id": sid, "instruction": instruction,
                 "parameters_hex": bytes(params).hex(), "request_hex": request.hex(),
                 "acknowledged": False}
        self.events.append(event)
        try:
            self.port.reset_input_buffer()
            sent = self.port.write(request)
            event["bytes_sent"] = sent
            if sent != len(request):
                raise CenterError(f"ID{sid} 请求未完整发送")
            reply = self.port.read(size+6)
            event["response_hex"] = reply.hex()
            if len(reply) != size+6:
                raise CenterError(f"ID{sid} 应答超时或长度错误")
            if reply[:4] != bytes([255, 255, sid, size+2]):
                raise CenterError(f"ID{sid} 应答包头/ID/长度错误")
            if reply[-1] != (~sum(reply[2:-1])) & 255:
                raise CenterError(f"ID{sid} 应答校验和错误")
            event["alarm"] = reply[4]
            if reply[4] and not tolerate_alarm:
                raise CenterError(f"ID{sid} 舵机报警 0x{reply[4]:02X}")
            event["acknowledged"] = True
            return reply[5:-1]
        except BaseException as error:
            event["error"] = f"{type(error).__name__}: {error}"
            raise

    def read(self, sid, address, size=1, tolerate_alarm=False):
        return self._exchange(sid, 2, [address, size], size, tolerate_alarm)

    def value(self, sid, address, size=1, tolerate_alarm=False):
        return u16(self.read(sid, address, size, tolerate_alarm))

    def write(self, sid, address, value, tolerate_alarm=False):
        if address == TORQUE and value in (0, 128):
            params = [address, value]
        elif address == OUTPUT and value == 0:
            params = [address, 0, 0]
        else:
            raise CenterError("禁止目标、使能1、EEPROM、PID、模式、限位或其他寄存器写入")
        self._exchange(sid, 3, params, 0, tolerate_alarm)

    def set_zero_checked(self, address):
        width = 1 if address == TORQUE else 2
        self.write(GRIPPER_ID, address, 0)
        if self.value(GRIPPER_ID, address, width) != 0:
            raise CenterError(f"ID6 寄存器 {address} 清零读回失败")


class DurableJournal:
    """Append-only snapshots: never truncate the durable pre-change state."""

    def __init__(self, path, stream):
        self.path, self.stream = Path(path), stream

    @classmethod
    def create(cls, directory, snapshot):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_")
        fd, path = tempfile.mkstemp(prefix=f"gripper_center_{stamp}", suffix=".jsonl", dir=directory)
        journal = cls(path, os.fdopen(fd, "w", encoding="utf-8"))
        try:
            journal.append("before_any_write", snapshot)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            journal.close()
            raise
        return journal

    def append(self, stage, snapshot):
        line = json.dumps({"stage": stage, "snapshot": snapshot}, ensure_ascii=False, allow_nan=False)
        self.stream.write(line+"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def decode_config(raw):
    return {"raw_hex": raw.hex(), "firmware_version_bytes": list(raw[:2]),
            "model_version_bytes": list(raw[3:5]), "endianness": raw[2],
            "id": raw[5], "reply_level": raw[8], "mode": raw[33],
            "phase": raw[18], "resolution": raw[30],
            "minimum": u16(raw[9:11]), "maximum": u16(raw[11:13]),
            "offset_raw": u16(raw[31:33]), "offset_signed": signed_offset(u16(raw[31:33]))}


class GripperCenterController:
    def __init__(self, bus, *, journal_dir=None, journal_factory=DurableJournal.create,
                 emit=print, clock=time.monotonic, sleep=time.sleep):
        self.bus, self.emit, self.clock, self.sleep = bus, emit, clock, sleep
        self.journal_dir, self.journal_factory = journal_dir, journal_factory
        self.journal = None
        self.status = "not_started"
        self.before = self.before_center = self.after = None
        self.fault = self.after_error = self.journal_error = None
        self.release_result = None
        self.center_intent = None
        self.center_acknowledged = False
        self.readback_verified = False
        self.mutation_started = False

    def torques_off(self):
        values = {sid: self.bus.value(sid, TORQUE) for sid in ALL_IDS}
        if any(values.values()):
            raise CenterError(f"ID1–6 必须均为扭矩0；实际={values}，未开始中位重定位")
        return values

    def read_config(self, require_uncentered=True):
        raw = self.bus.read(GRIPPER_ID, 0, 40)
        config = decode_config(raw)
        if tuple(raw[:2]) != EXPECTED_FIRMWARE or tuple(raw[3:5]) != EXPECTED_MODEL:
            raise CenterError(f"ID6 固件/型号字节不符合已核实设备：{config}")
        if (raw[2] != 0 or raw[5] != 6 or raw[8] != 1 or raw[33] != 0
                or raw[18] != 0x0C or raw[30] != 1
                or u16(raw[9:11]) != 0 or u16(raw[11:13]) != 4095):
            raise CenterError(f"ID6 配置不符合本次固定重定位条件：{config}")
        if require_uncentered and config["offset_raw"] != 0:
            raise CenterError(f"ID6 offset 已非0（{config['offset_raw']}），拒绝重复重定位")
        return config

    def read_feedback(self, config):
        raw = self.bus.read(GRIPPER_ID, 56, 10)
        position_raw = u16(raw[:2])
        position = -(position_raw & 0x7FFF) if position_raw & 0x8000 else position_raw
        data = {"elapsed_s": self.clock()-self.bus.started, "raw_hex": raw.hex(),
                "position": position, "voltage": raw[6]/10,
                "temperature": raw[7], "status": raw[9],
                "speed_raw_signed": signed_magnitude(u16(raw[2:4]), 15),
                "output_raw_signed": signed_magnitude(u16(raw[4:6]), 10)}
        config_raw = bytes.fromhex(config["raw_hex"])
        vmin, vmax = max(6.0, config_raw[15]/10), min(7.4, config_raw[14]/10)
        temperature_limit = min(55, config_raw[13])
        if (not 0 <= position <= 4095 or not vmin <= data["voltage"] <= vmax
                or data["temperature"] >= temperature_limit or data["status"]
                or u16(raw[2:4]) != 0 or u16(raw[4:6]) != 0):
            raise CenterError(f"ID6 健康/位置检查失败：{data}")
        return data

    def stable_feedback(self, config, *, centered=False):
        center, tolerance = (2048, 2) if centered else (OPEN_OBSERVED, 8)
        samples = []
        for index in range(3):
            if index:
                self.sleep(.1)
            state = self.read_feedback(config)
            if abs(state["position"]-center) > tolerance:
                raise CenterError(f"ID6 实际={state['position']}，须在 {center}±{tolerance} 内；不发运动目标")
            samples.append(state)
        positions = [sample["position"] for sample in samples]
        if samples[-1]["elapsed_s"]-samples[0]["elapsed_s"] < .2 or max(positions)-min(positions) > 2:
            raise CenterError("ID6 三样本未满足跨度至少0.2秒、计数范围不超过2的稳定要求")
        return samples

    def read_control(self):
        raw = self.bus.read(GRIPPER_ID, 40, 16)
        return {"raw_hex": raw.hex(), "torque_enabled": raw[0],
                "acceleration": raw[1], "goal_position_raw": u16(raw[2:4]),
                "time_raw": u16(raw[4:6]), "speed_raw": u16(raw[6:8]),
                "output_limit": u16(raw[8:10]), "lock": raw[15]}

    def preflight(self):
        torques = self.torques_off()
        config = self.read_config()
        samples = self.stable_feedback(config)
        self.torques_off()
        control = self.read_control()
        if control["torque_enabled"] != 0:
            raise CenterError("ID6 控制区复读发现扭矩非0，停止")
        return {"config": config, "samples": samples, "position": samples[-1]["position"],
                "torques": torques, "control": control, "output_limit": control["output_limit"]}

    def persist(self, stage):
        self.journal.append(stage, self.report())

    def cleanup(self):
        self.bus.phase = "release"
        errors = []
        for address in (TORQUE, OUTPUT):
            try:
                self.bus.write(GRIPPER_ID, address, 0, tolerate_alarm=True)
            except BaseException as error:
                errors.append({"operation": "write", "address": address,
                               "error": f"{type(error).__name__}: {error}"})
        values = {}
        for address, size in ((TORQUE, 1), (OUTPUT, 2)):
            try:
                values[str(address)] = self.bus.value(GRIPPER_ID, address, size, tolerate_alarm=True)
            except BaseException as error:
                errors.append({"operation": "read", "address": address,
                               "error": f"{type(error).__name__}: {error}"})
        self.release_result = {"servo_id": 6, "readback": values, "errors": errors,
                               "confirmed": values.get(str(TORQUE)) == 0 and values.get(str(OUTPUT)) == 0}
        return self.release_result["confirmed"]

    def verify_after(self):
        self.bus.phase = "post_readback"
        config = self.read_config(require_uncentered=False)
        # Save partial after-state before checking so an altered byte is visible.
        self.after = {"config": config}
        before_config = self.before_center["config"]
        before_raw, after_raw = bytes.fromhex(before_config["raw_hex"]), bytes.fromhex(config["raw_hex"])
        changed = [index for index in range(40) if index not in (31, 32) and before_raw[index] != after_raw[index]]
        self.after["unexpected_changed_config_addresses"] = changed
        if changed:
            raise CenterError(f"中位重定位后非offset配置发生变化：地址{changed}")
        if config["offset_raw"] == before_config["offset_raw"]:
            raise CenterError("中位重定位后 offset 未改变")
        samples = self.stable_feedback(config, centered=True)
        control = self.read_control()
        self.after.update({"samples": samples, "position": samples[-1]["position"],
                           "torques": self.torques_off(),
                           "control": control, "output_limit": control["output_limit"]})
        if control["torque_enabled"] != 0 or control["output_limit"] != 0:
            raise CenterError("重定位后 ID6 扭矩或输出上限不是0")
        delta_offset = config["offset_signed"]-before_config["offset_signed"]
        predicted_anchor = (self.before_center["position"]-delta_offset) % 4096
        error = circular_difference(self.after["position"], predicted_anchor)
        self.after["translation_check"] = {"offset_change": delta_offset,
                                           "predicted_anchor_after": predicted_anchor,
                                           "actual_anchor_after": self.after["position"],
                                           "error_counts": error}
        if abs(error) > 2:
            raise CenterError(f"offset解释与前后编号平移不一致：误差={error:+d}计数")
        self.after["predicted_observed_closed"] = {
            "old_count": CLOSED_OBSERVED, "new_count": (CLOSED_OBSERVED-delta_offset) % 4096,
            "measured_after_centering": False, "not_for_motion": True}
        self.after["predicted_observed_open"] = {
            "old_count": OPEN_OBSERVED, "new_count": (OPEN_OBSERVED-delta_offset) % 4096,
            "measured_after_centering": False, "not_for_motion": True}
        self.readback_verified = True

    def run(self, apply=False):
        self.bus.phase = "preflight"
        try:
            self.before = self.preflight()
        except BaseException as error:
            self.status = "preflight_rejected"
            self.fault = {"phase": self.bus.phase, "error": f"{type(error).__name__}: {error}"}
            raise
        if not apply:
            self.status = "read_only_preview"
            self.emit(f"只读预览通过：ID6={self.before['position']}，offset=0，ID1–6 扭矩均0；未发送写指令。")
            return self.report()
        if self.journal_dir is None:
            raise CenterError("--apply 必须先指定持久journal目录")
        # Creation, flush/fsync and directory durability must succeed before
        # this bus gains its narrow write capability.
        self.journal = self.journal_factory(self.journal_dir, self.report())
        self.bus.enable_apply()
        primary = None
        try:
            self.bus.phase = "prepare_zero_output"
            self.mutation_started = True  # Before any write, even a lost ACK.
            self.bus.set_zero_checked(OUTPUT)
            self.persist("output_zero_readback")
            self.bus.set_zero_checked(TORQUE)
            self.persist("torque_zero_readback")
            self.bus.phase = "recheck_before_center"
            self.before_center = self.preflight()
            if self.before_center["config"]["raw_hex"] != self.before["config"]["raw_hex"]:
                raise CenterError("首次检查与重定位前的完整配置不同，停止")
            if abs(self.before_center["position"]-self.before["position"]) > 2:
                raise CenterError("原位位置在准备期间漂移超过2计数，停止")
            if self.before_center["output_limit"] != 0:
                raise CenterError("重定位前输出上限不是0")
            self.bus.phase = "centering_command"
            self.center_intent = {"servo_id": 6, "address": 40, "value": 128,
                                  "attempted": True, "maximum_attempts": 1}
            self.persist("centering_write_intent_before_transport")
            self.bus.write(GRIPPER_ID, TORQUE, 128)
            self.center_acknowledged = True
            self.persist("centering_command_acknowledged")
        except BaseException as error:
            primary = error
            self.fault = {"phase": self.bus.phase, "error": f"{type(error).__name__}: {error}"}
        finally:
            saved = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                if self.mutation_started:
                    self.cleanup()
                if self.bus.center_attempted:
                    try:
                        self.verify_after()
                    except BaseException as error:
                        self.after_error = f"{type(error).__name__}: {error}"
                if not self.release_result or not self.release_result["confirmed"]:
                    self.status = "release_unconfirmed"
                elif self.bus.center_attempted and not self.center_acknowledged:
                    self.status = "centering_uncertain"
                elif primary is not None:
                    self.status = "failed_after_write_started"
                elif not self.readback_verified:
                    self.status = "centering_readback_failed"
                else:
                    self.status = "centering_readback_verified"
                try:
                    self.persist("final_readback_and_cleanup")
                except BaseException as error:
                    self.journal_error = f"{type(error).__name__}: {error}"
                try:
                    self.journal.close()
                except BaseException as error:
                    self.journal_error = self.journal_error or f"{type(error).__name__}: {error}"
                if self.journal_error and self.status == "centering_readback_verified":
                    self.status = "journal_save_failed"
            finally:
                for sig, handler in saved.items():
                    signal.signal(sig, handler)
        if self.status == "release_unconfirmed":
            raise CenterError(f"ID6 扭矩/输出清零未确认，保持夹爪不受外力并关闭舵机电源；原错误={primary}") from primary
        if primary is not None:
            raise primary
        if self.after_error:
            raise CenterError(self.after_error)
        if self.journal_error:
            raise CenterError(f"硬件清理已执行，但最终journal保存失败：{self.journal_error}")
        self.emit("ID6 中位重定位读回通过；未执行开闭运动。旧ID6原始目标已失效，断电持久性仍待复读。")
        return self.report()

    def report(self):
        return {"schema_version": 1, "test": "id6_torque_disabled_centering",
                "register_reference": REGISTER_REFERENCE, "status": self.status,
                "servo_id": 6, "before": self.before, "before_center": self.before_center,
                "after": self.after, "center_write_intent": self.center_intent,
                "center_command_attempted": self.bus.center_attempted,
                "center_command_acknowledged": self.center_acknowledged,
                "centering_readback_verified": self.readback_verified,
                "release": self.release_result, "fault": self.fault,
                "after_error": self.after_error, "journal_error": self.journal_error,
                "writes_sent": sum(event["instruction"] == 3 for event in self.bus.events),
                "events": self.bus.events,
                "old_id6_raw_targets_invalid_if_centered": self.bus.center_attempted,
                "power_cycle_persistence_verified": False,
                "mechanical_limits_verified": False, "grasp_verified": False,
                "note": "Predicted historical endpoints are not new measurements and are not motion commands."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="ID6 原位编号中位重定位；默认只读，不发开闭目标。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--apply", action="store_true", help="持久保存前态后，仅尝试一次 ID6 40=128")
    parser.add_argument("--log-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]/"data"/"real_control_runs")
    args = parser.parse_args(argv)
    try:
        import serial
    except ImportError:
        parser.exit(1, "请使用 /home/yu/so101_ws/.venv_vision/bin/python（已安装 pyserial）。\n")
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    controller = None
    try:
        print("保持当前张开姿态，取出参照纸盒，手不要接触夹爪运动部件；关闭其他串口程序。", flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            controller = GripperCenterController(CenterBus(port), journal_dir=args.log_dir,
                                                  emit=lambda msg: print(msg, flush=True))
            result = controller.run(args.apply)
            print(json.dumps({key: result[key] for key in (
                "status", "before", "before_center", "after", "release", "writes_sent")}, ensure_ascii=False, indent=2), flush=True)
    except KeyboardInterrupt:
        print("已中止；若已尝试128，不要重发，检查journal中的读回和清理结果。", flush=True)
        return 130
    except (CenterError, OSError, ValueError) as error:
        print(f"本次未完成：{error}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if controller is not None and controller.journal is not None:
            print(f"持久journal：{controller.journal.path}", flush=True)
            if controller.release_result:
                print(f"ID6 清理读回：{controller.release_result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
