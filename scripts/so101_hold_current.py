"""Short, supported SO101 hold test; no model mapping or EEPROM writes.

Register reference:
https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation
Output limit is a register/duty limit, not a calibrated force measurement.
"""

import argparse
from dataclasses import dataclass
import math
import signal
import sys
import time


ARM_IDS = (1, 2, 3, 4, 5)
ALL_IDS = (*ARM_IDS, 6)
TORQUE, ACCEL, GOAL, SPEED, OUTPUT = 40, 41, 42, 46, 48
WRITE_WIDTHS = {TORQUE: 1, ACCEL: 1, GOAL: 2, SPEED: 2, OUTPUT: 2}
START_TOLERANCE = 3
HOLD_TOLERANCE = 12  # ~1.05 degrees, raw differences; never wrap a command.


class ControlError(RuntimeError):
    pass


def u16(data):
    return int.from_bytes(data, "little")


def signed_magnitude(raw, bit):
    magnitude = raw & ((1 << bit) - 1)
    return -magnitude if raw & (1 << bit) else magnitude


class Bus:
    """Only five whitelisted SRAM register writes are exposed."""

    def __init__(self, port):
        self.port = port

    def exchange(self, sid, instruction, params, size, tolerate_alarm=False):
        if sid not in ALL_IDS:
            raise ControlError("ID 必须为 1–6")
        fields = bytes([sid, len(params) + 2, instruction]) + bytes(params)
        packet = b"\xff\xff" + fields + bytes([(~sum(fields)) & 255])
        self.port.reset_input_buffer()
        if self.port.write(packet) != len(packet):
            raise ControlError(f"ID {sid}: 请求未完整发送")
        reply = self.port.read(size + 6)
        if len(reply) != size + 6:
            raise ControlError(f"ID {sid}: 应答超时或长度错误")
        if reply[:4] != bytes([255, 255, sid, size + 2]):
            raise ControlError(f"ID {sid}: 应答包头/ID/长度错误")
        if reply[-1] != (~sum(reply[2:-1])) & 255:
            raise ControlError(f"ID {sid}: 应答校验和错误")
        if reply[4] and not tolerate_alarm:
            raise ControlError(f"ID {sid}: 舵机报警 0x{reply[4]:02X}")
        return reply[5:-1]

    def read(self, sid, address, size=1, tolerate_alarm=False):
        return self.exchange(sid, 2, [address, size], size, tolerate_alarm)

    def value(self, sid, address, size=1, tolerate_alarm=False):
        return u16(self.read(sid, address, size, tolerate_alarm))

    def write(self, sid, address, value, tolerate_alarm=False):
        if sid not in ARM_IDS or address not in WRITE_WIDTHS:
            raise ControlError("禁止写入夹爪、EEPROM 或未列入白名单的寄存器")
        permitted = {
            TORQUE: (0, 1), ACCEL: (1,), SPEED: (50,),
        }
        if address in permitted and value not in permitted[address]:
            raise ControlError("控制参数不在本次保持测试范围内")
        if address == GOAL and not 0 <= value <= 4095:
            raise ControlError("目标必须为单圈计数")
        if address == OUTPUT and not 0 <= value <= 300:
            raise ControlError("本次测试输出上限不得超过 300/1000")
        payload = [address, *value.to_bytes(WRITE_WIDTHS[address], "little")]
        self.exchange(sid, 3, payload, 0, tolerate_alarm)

    def set_checked(self, sid, address, value):
        self.write(sid, address, value)
        if self.value(sid, address, WRITE_WIDTHS[address]) != value:
            raise ControlError(f"ID {sid}: 寄存器 {address} 写后读回不一致")


@dataclass
class Feedback:
    position: int
    load: int
    voltage: float
    temperature: int
    status: int


def feedback(bus, sid):
    raw = bus.read(sid, 56, 10)  # Position through Status in one response.
    return Feedback(signed_magnitude(u16(raw[:2]), 15),
                    abs(signed_magnitude(u16(raw[4:6]), 10)),
                    raw[6] / 10, raw[7], raw[9])


def check_health(sid, state, temp_limit=55, voltage_min=6.0, voltage_max=7.4):
    if not 0 <= state.position <= 4095:
        raise ControlError(f"ID {sid}: 位置不符合单圈模式")
    if not voltage_min <= state.voltage <= voltage_max:
        raise ControlError(f"ID {sid}: 电压 {state.voltage:.1f}V 不在 {voltage_min:.1f}–{voltage_max:.1f}V 内")
    if state.temperature >= temp_limit or state.status:
        raise ControlError(f"ID {sid}: 温度 {state.temperature}℃ / 状态 0x{state.status:02X} 异常")


class HoldController:
    def __init__(self, bus, limit=100, seconds=10, emit=print,
                 clock=time.monotonic, sleep=time.sleep):
        if not isinstance(limit, int) or not 1 <= limit <= 300:
            raise ValueError("输出上限必须为 1–300")
        if not math.isfinite(seconds) or not 0 < seconds <= 30:
            raise ValueError("持续时间必须大于 0 且不超过 30 秒")
        self.bus, self.limit, self.seconds = bus, limit, seconds
        self.emit, self.clock, self.sleep = emit, clock, sleep
        self.targets, self.configs, self.touched = {}, {}, set()
        self.high_load_since = {}

    def preflight(self):
        initial = {}
        # Entire preflight is read-only, including failure with existing torque ON.
        for sid in ALL_IDS:
            config = self.bus.read(sid, 0, 34)
            state = feedback(self.bus, sid)
            if config[2] != 0 or config[5] != sid or config[8] != 1:
                raise ControlError(f"ID {sid}: 字节序、ID 或应答等级不匹配")
            if config[33] != 0 or config[30] != 1 or config[18] != 0x0C:
                raise ControlError(f"ID {sid}: 模式/分辨率/相位与已读配置不同")
            if self.bus.value(sid, TORQUE) != 0:
                raise ControlError(f"ID {sid}: 扭矩已开启；本程序没有写入，请结束占用它的控制程序")
            lower, upper = u16(config[9:11]), u16(config[11:13])
            if not 0 <= lower < upper <= 4095 or not lower <= state.position <= upper:
                raise ControlError(f"ID {sid}: 当前位置或设备限位不适用于本次保持")
            hw_limit = u16(config[16:18])
            if sid in ARM_IDS and not self.limit <= hw_limit <= 1000:
                raise ControlError(f"ID {sid}: 请求上限超过设备最大输出 {hw_limit}/1000")
            temp_limit = min(55, config[13])
            vmin, vmax = max(6.0, config[15] / 10), min(7.4, config[14] / 10)
            check_health(sid, state, temp_limit, vmin, vmax)
            self.configs[sid] = (temp_limit, vmin, vmax)
            initial[sid] = state.position
            self.emit(f"ID {sid}: 当前位置={state.position}, 扭矩=0, 电压={state.voltage:.1f}V, 温度={state.temperature}℃")
        self.sleep(0.2)
        for sid in ALL_IDS:
            state = feedback(self.bus, sid)
            check_health(sid, state, *self.configs[sid])
            if abs(state.position - initial[sid]) > START_TOLERANCE:
                raise ControlError(f"ID {sid}: 托住后仍有位置漂移，未写入任何控制指令")
            if self.bus.value(sid, TORQUE) != 0:
                raise ControlError(f"ID {sid}: 检查期间扭矩状态改变，未开始控制")
            if sid in ARM_IDS:
                self.targets[sid] = state.position

    def monitor(self, expected_torque, tolerance, expected_output):
        started = self.clock()
        def bounded_read(function, *args):
            if self.clock() - started > .5:
                raise ControlError("反馈累计超过 0.5 秒，停止本次保持")
            result = function(*args)
            if self.clock() - started > .5:
                raise ControlError("反馈累计超过 0.5 秒，停止本次保持")
            return result
        states = {}
        for sid in ALL_IDS:
            state = bounded_read(feedback, self.bus, sid)
            check_health(sid, state, *self.configs[sid])
            if sid == 6:
                if bounded_read(self.bus.value, sid, TORQUE) != 0:
                    raise ControlError("ID 6 夹爪被其他控制源使能")
                continue
            if abs(state.position - self.targets[sid]) > tolerance:
                raise ControlError(f"ID {sid}: 位置偏离保持点 {state.position - self.targets[sid]:+d} 计数，停止输出")
            if bounded_read(self.bus.value, sid, TORQUE) != expected_torque:
                raise ControlError(f"ID {sid}: 扭矩状态意外改变")
            control = bounded_read(self.bus.read, sid, 41, 9)
            if (control[0] != 1 or u16(control[1:3]) != self.targets[sid]
                    or u16(control[5:7]) != 50 or u16(control[7:9]) != expected_output):
                raise ControlError(f"ID {sid}: 目标/速度/输出上限被改变")
            now = self.clock()
            if expected_output and state.load >= expected_output * .95:
                self.high_load_since.setdefault(sid, now)
                if now - self.high_load_since[sid] > 1.0:
                    raise ControlError(f"ID {sid}: 输出持续接近本次上限，停止；不会自动加大输出")
            else:
                self.high_load_since.pop(sid, None)
            states[sid] = state
        if self.clock() - started > .5:
            raise ControlError("整轮反馈超过 0.5 秒，停止本次保持")
        return states

    def prepare(self):
        # Mark before sending: a write may take effect even if its ACK is lost.
        for sid in ARM_IDS:
            if self.bus.value(sid, TORQUE) != 0:
                raise ControlError(f"ID {sid}: 预置前扭矩状态改变")
            self.touched.add(sid)
            self.bus.set_checked(sid, OUTPUT, 0)
        for sid in ARM_IDS:
            self.bus.set_checked(sid, ACCEL, 1)
            self.bus.set_checked(sid, SPEED, 50)
            if abs(feedback(self.bus, sid).position - self.targets[sid]) > START_TOLERANCE:
                raise ControlError(f"ID {sid}: 预置前位置漂移")
            self.bus.set_checked(sid, GOAL, self.targets[sid])
            # Some firmware can enable on a goal write. Output is still zero.
            self.bus.set_checked(sid, TORQUE, 0)
        self.monitor(0, START_TOLERANCE, 0)
        for sid in ARM_IDS:
            self.bus.set_checked(sid, TORQUE, 1)
        self.monitor(1, START_TOLERANCE, 0)

    def hold(self):
        self.emit(f"目标仅取本次反馈: {self.targets}；全程承托，结束或异常会释放扭矩。")
        # Fixed ramp, never increased in response to sagging or failure.
        levels = list(range(min(25, self.limit), self.limit, 25)) + [self.limit]
        previous = 0
        for level in levels:
            self.monitor(1, HOLD_TOLERANCE, previous)
            ramp_started = self.clock()
            for sid in ARM_IDS:
                self.bus.set_checked(sid, OUTPUT, level)
                if self.clock() - ramp_started > .5:
                    raise ControlError("输出上限写入累计超过 0.5 秒")
            self.monitor(1, HOLD_TOLERANCE, level)
            self.sleep(.1)
            previous = level
        end = self.clock() + self.seconds
        next_print = self.clock()
        self.emit(f"受限原位保持开始：{self.seconds:g} 秒，输出上限={self.limit}/1000，ID6 不使能。")
        while self.clock() < end:
            states = self.monitor(1, HOLD_TOLERANCE, self.limit)
            now = self.clock()
            if now >= next_print:
                detail = "  ".join(f"ID{sid}:{s.position}(差{s.position-self.targets[sid]:+d},输出{s.load}/1000)" for sid, s in states.items())
                self.emit(f"剩余 {max(0, end-now):.1f}s  {detail}")
                next_print = now + .5
            self.sleep(.05)

    def release(self):
        if not self.touched:
            return []
        def report(message):
            # A broken stdout/terminal must never interrupt hardware cleanup.
            try:
                self.emit(message)
            except Exception:
                pass
        report("正在释放本次控制的关节；继续承托机械臂。")
        # Try every ID even if a different servo is unreachable/alarming.
        for address in (TORQUE, OUTPUT):
            for sid in sorted(self.touched):
                try:
                    self.bus.write(sid, address, 0, tolerate_alarm=True)
                except Exception as error:
                    report(f"ID {sid}: 关闭写入未收到有效应答: {error}")
        unconfirmed = []
        for sid in sorted(self.touched):
            try:
                torque = self.bus.value(sid, TORQUE, tolerate_alarm=True)
                limit = self.bus.value(sid, OUTPUT, 2, tolerate_alarm=True)
                if torque != 0 or limit != 0:
                    raise ControlError(f"扭矩={torque},输出上限={limit}")
                report(f"ID {sid}: 已读回扭矩=0、输出上限=0")
            except Exception as error:
                unconfirmed.append(sid)
                report(f"ID {sid}: 释放未确认: {error}")
        return unconfirmed

    def run(self, enable=False):
        self.preflight()
        if not enable:
            self.emit("只读检查完成。未发送写入；使用 --enable 才会执行短时原位保持。")
            return
        primary = None
        try:
            self.prepare()
            self.hold()
        except BaseException as error:
            primary = error
        finally:
            # A second Ctrl+C must not skip the rest of the release attempts.
            saved = {}
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.signal(sig, signal.SIG_IGN)
            try:
                unconfirmed = self.release()
            finally:
                for sig, handler in saved.items():
                    signal.signal(sig, handler)
        if unconfirmed:
            raise ControlError(f"停止原因={primary!s}; 未确认释放 ID={unconfirmed}。继续支撑，关闭舵机电源。") from primary
        if primary is not None:
            raise primary
        self.emit("保持时段完成并已释放。此结果不代表模型映射或自动抓取通过。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="SO101 短时原位保持。全程承托；正常结束/Ctrl+C/异常会释放扭矩。默认只读。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--enable", action="store_true", help="实际使能 ID1–5，目标为本次稳定反馈")
    parser.add_argument("--seconds", type=float, default=10, help="保持秒数，0–30，默认10")
    parser.add_argument("--limit", type=int, default=100, help="RAM输出上限，1–300，默认100/1000；不能保证承重")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error("--seconds 必须大于0且不超过30")
    if not 1 <= args.limit <= 300:
        parser.error("--limit 必须为1–300")
    try:
        import serial
    except ImportError:
        parser.exit(1, "请使用 /home/yu/so101_ws/.venv_vision/bin/python（已安装 pyserial）。\n")
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    print("全程托住/支撑连杆；低输出不保证承重，退出会断扭矩。关闭其他串口控制程序。", flush=True)
    try:
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            controller = HoldController(Bus(port), args.limit, args.seconds,
                                        emit=lambda msg: print(msg, flush=True))
            controller.run(args.enable)
    except KeyboardInterrupt:
        print("已中止；如已开始控制，请以上方逐 ID 释放读回结果为准。", flush=True)
        return 130
    except (ControlError, OSError, ValueError) as error:
        print(f"控制未完成：{error}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
