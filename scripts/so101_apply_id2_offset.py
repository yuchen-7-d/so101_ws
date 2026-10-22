"""ID2-only offset change, default read-only. No goal or torque-enable command.

Requires the reviewed offline proposal. --apply writes offset 0x0C9D once,
using the EPROM unlock/write/lock sequence, with torque disabled. Software
mapping is NEVER changed here. Power-cycle persistence remains unverified.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

from so101_preview_id2_offset import analyze as analyze_proposal


ROOT = Path('/home/yu/so101_ws')
IDS = range(1, 7)
NEW_OFFSET = 0x0C9D
SHIFT = 1181
ALLOWED_WRITES = {b'\x28\x00', b'\x30\x00\x00', b'\x37\x00',
                  b'\x37\x01', b'\x1f\x9d\x0c'}
REGISTER_REFERENCE = 'https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation'


class OffsetError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise OffsetError(message)


def u16(raw):
    return int.from_bytes(raw, 'little')


def delta(a, b):
    # Readback comparison only. This program cannot generate motion commands.
    return (a-b+2048) % 4096 - 2048


def read_json(path):
    raw = Path(path).read_bytes()
    def bad(value):
        raise OffsetError(f'JSON 含非有限值：{value}')
    return json.loads(raw, parse_constant=bad), hashlib.sha256(raw).hexdigest()


def load_proposal(path):
    path = Path(path).resolve()
    preview, sha = read_json(path)
    require(preview.get('kind') == 'so101_id2_offset_proposal'
            and preview.get('status') == 'offline_offset_preview'
            and preview.get('execution_available') is False
            and type(preview.get('register_writes')) is int and preview['register_writes'] == 0
            and preview.get('hardware_offset_applied') is False
            and preview.get('software_mapping_applied') is False, '输入不是未应用的离线偏移预览')
    paths = [Path(p) for p in preview['input_files']
             if Path(p).name.startswith('so101_approach_check_')]
    require(len(paths) == 1, '不能唯一确定接近检查报告')
    args = argparse.Namespace(approach=paths[0], workspace=ROOT,
                              observed_count=3871, observed_offset_raw=0)
    recomputed = {}
    analyze_proposal(args, recomputed)
    require(preview['proposal'] == recomputed['proposal']
            and preview['input_files'] == recomputed['input_files'], '重新计算与预览报告不一致')
    p = preview['proposal']
    require(p['servo_id'] == 2 and p['offset_raw'] == NEW_OFFSET
            and p['offset_signed'] == -SHIFT and p['new_reference_count'] == 2005
            and p['old_reference_count'] == 824 and p['start_count_predicted'] == 956,
            '本程序只允许已复核的 ID2 固定偏移方案')
    approach, _ = read_json(paths[0])
    mapping = ROOT / 'data/so101_joint_mapping.json'
    return {'path': str(path), 'sha256': sha, 'preview': preview,
            'expected_counts': {str(s['servo_id']): s['position_counts']
                                for s in approach['state_snapshot']},
            'mapping_before_utf8': mapping.read_text(encoding='utf-8')}


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Journal:
    def __init__(self, directory, proof):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='id2_offset_', suffix='.jsonl', dir=self.directory)
        self.path = Path(name)
        self.stream = os.fdopen(fd, 'w', encoding='utf-8')
        try:
            self.append('proposal_and_original_mapping', proof)
            sync_directory(self.directory)
        except BaseException:
            self.stream.close()
            raise

    def append(self, stage, data):
        self.stream.write(json.dumps({'stage': stage, 'data': data},
                                    ensure_ascii=False, allow_nan=False)+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def mark_attempt(self, proof):
        # A persistent marker prevents a second --apply for this proposal,
        # including after a lost ACK or a partially saved report.
        path = self.directory / ('id2_offset_attempt_'+proof['sha256']+'.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'journal': str(self.path), 'proposal_sha256': proof['sha256']}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(self.directory)
        return str(path)


class OffsetBus:
    """Only reads, plus five exact ID2 write payloads after local authorization."""
    def __init__(self, port, journal, clock=time.monotonic):
        self.port, self.journal, self.clock = port, journal, clock
        self.started = clock()
        self.phase = 'read_only'
        self.apply_enabled = False
        self.cleanup_mode = False
        self.offset_attempted = False
        self.offset_acknowledged = False
        self.write_attempts = 0
        self.events = []
        self.journal_errors = []

    def record(self, stage, value):
        try:
            self.journal.append(stage, value)
        except BaseException as error:
            self.journal_errors.append(f'{stage}: {type(error).__name__}: {error}')
            if not self.cleanup_mode:
                raise

    def exchange(self, sid, instruction, params, size):
        params = bytes(params)
        require(type(sid) is int and sid in IDS, '仅允许 ID1–6')
        if instruction == 2:
            require(len(params) == 2 and params[1] == size and 1 <= size <= 40
                    and params[0]+size <= 256, '读取范围无效')
        elif instruction == 3:
            require(self.apply_enabled and sid == 2 and params in ALLOWED_WRITES and size == 0,
                    '禁止白名单之外的写入')
            if self.cleanup_mode:
                require(params in (b'\x28\x00', b'\x30\x00\x00', b'\x37\x01'),
                        '收尾只允许 ID2 扭矩/输出清零及锁定')
            if params[0] == 31:
                require(not self.offset_attempted, '偏移写入已尝试；禁止重复发送')
        else:
            raise OffsetError('禁止其他协议指令')
        fields = bytes([sid, len(params)+2, instruction])+params
        request = b'\xff\xff'+fields+bytes([(~sum(fields)) & 255])
        event = {'elapsed_s': self.clock()-self.started, 'phase': self.phase,
                 'servo_id': sid, 'instruction': instruction, 'parameters_hex': params.hex(),
                 'request_hex': request.hex(), 'acknowledged': False}
        self.events.append(event)
        if instruction == 3:
            self.record('write_intent_before_transport', event)
            self.write_attempts += 1
            if params[0] == 31:
                self.offset_attempted = True
        try:
            self.port.reset_input_buffer()
            event['bytes_sent'] = self.port.write(request)
            require(event['bytes_sent'] == len(request), '请求未完整发送')
            reply = self.port.read(size+6)
            event['response_hex'] = reply.hex()
            require(len(reply) == size+6 and reply[:4] == bytes([255, 255, sid, size+2]),
                    f'ID{sid} 应答长度/ID/包头错误')
            require(reply[-1] == (~sum(reply[2:-1])) & 255, f'ID{sid} 应答校验和错误')
            event['alarm'] = reply[4]
            require(reply[4] == 0, f'ID{sid} 应答报警：{reply[4]}')
            event['acknowledged'] = True
            if instruction == 3 and params[0] == 31:
                self.offset_acknowledged = True
            return reply[5:-1]
        except BaseException as error:
            event['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.record('packet_result', event)

    def read(self, sid, address, size=1):
        return self.exchange(sid, 2, [address, size], size)

    def value(self, sid, address, size=1):
        return u16(self.read(sid, address, size))

    def write(self, address, value, size=1):
        self.exchange(2, 3, bytes([address])+int(value).to_bytes(size, 'little'), 0)

    def set_checked(self, address, value, size=1):
        self.write(address, value, size)
        require(self.value(2, address, size) == value, f'ID2 地址 {address} 写后读回不符')


def frame_fields(raw):
    require(len(raw) == 26, '控制/反馈帧长度错误')
    return {'raw_hex': raw.hex(), 'torque': raw[0], 'output_limit': u16(raw[8:10]),
            'lock': raw[15], 'position': u16(raw[16:18]), 'speed_raw': u16(raw[18:20]),
            'output_raw': u16(raw[20:22]), 'voltage': raw[22]/10,
            'temperature': raw[23], 'status': raw[25]}


@contextmanager
def uninterrupted_cleanup():
    saved = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)


class Controller:
    def __init__(self, bus, proof, sleep=time.sleep):
        self.bus, self.proof, self.sleep = bus, proof, sleep
        self.report = {'schema_version': 1, 'test': 'id2_fixed_offset_change',
                       'status': 'not_started', 'execution_available': False,
                       'software_mapping_updated': False, 'power_cycle_persistence_verified': False,
                       'register_reference': REGISTER_REFERENCE,
                       'proposed_offset_raw': NEW_OFFSET, 'offset_readback_verified': False,
                       'release_confirmed': False, 'lock_confirmed': False,
                       'journal_path': str(bus.journal.path)}

    def frame(self, sid):
        f = frame_fields(self.bus.read(sid, 40, 26))
        f['elapsed_s'] = self.bus.clock()-self.bus.started
        return f

    def healthy(self, sid, frame, config):
        require(frame['torque'] == 0 and frame['speed_raw'] == 0 and frame['output_raw'] == 0
                and frame['status'] == 0 and 0 <= frame['position'] <= 4095,
                f'ID{sid} 非扭矩关闭且静止无报警状态：{frame}')
        low, high = max(6.0, config[15]/10), min(7.4, config[14]/10)
        require(low <= frame['voltage'] <= high and frame['temperature'] < min(55, config[13]),
                f'ID{sid} 电压/温度异常：{frame}')

    def snapshot(self):
        state = {}
        for sid in IDS:
            raw = self.bus.read(sid, 0, 40)
            state[str(sid)] = {'config_raw_hex': raw.hex(), 'frame': self.frame(sid)}
        return state

    def validate_initial(self, state):
        for sid in IDS:
            s = state[str(sid)]
            raw = bytes.fromhex(s['config_raw_hex'])
            require(raw[:5] == bytes([3, 10, 0, 9, 3]) and raw[5] == sid and raw[8] == 1
                    and raw[18] == 12 and raw[30] == 1 and raw[33] == 0
                    and u16(raw[9:11]) == 0 and u16(raw[11:13]) == 4095,
                    f'ID{sid} 配置与已复核设备条件不一致')
            self.healthy(sid, s['frame'], raw)
            tolerance = 4 if sid == 2 else 8
            require(abs(delta(s['frame']['position'], self.proof['expected_counts'][str(sid)])) <= tolerance,
                    f'ID{sid} 姿态与保存的起点不同；不发任何位置目标')
        raw = bytes.fromhex(state['2']['config_raw_hex'])
        require(u16(raw[31:33]) == 0, f'ID2 offset 已不是0（{u16(raw[31:33])}），拒绝重复修改')
        require(state['2']['frame']['lock'] == 1, 'ID2 配置锁不是1；先核对之前的配置操作')

    def stable_window(self, before, count=6, after=False):
        frames = []
        anchor = before['2']['frame']['position']
        for index in range(count):
            if index:
                self.sleep(.1)
            # Monitor all six axes, not just the changed shoulder.
            row = {str(sid): self.frame(sid) for sid in IDS}
            frames.append(row)
            self.bus.journal.append('after_window' if after else 'before_window', row)
            for sid in IDS:
                old = before[str(sid)]
                self.healthy(sid, row[str(sid)], bytes.fromhex(old['config_raw_hex']))
                expected = old['frame']['position']
                if after and sid == 2:
                    expected = (anchor+SHIFT) % 4096  # Only expected READBACK.
                tolerance = 4 if sid == 2 else 8
                require(abs(delta(row[str(sid)]['position'], expected)) <= tolerance,
                        f'ID{sid} 前后读数不符合原位平移/保持条件')
                if after and sid == 2:
                    require(row['2']['output_limit'] == 0 and row['2']['lock'] == 1,
                            'ID2 输出上限/配置锁读回异常')
        positions = [r['2']['position'] for r in frames]
        require(frames[-1]['2']['elapsed_s']-frames[0]['2']['elapsed_s'] >= (count-1)*.1-1e-4,
                'ID2 稳定窗口的采样时间不足')
        require(max(positions)-min(positions) <= 2, 'ID2 采样窗口不稳定，未通过原位核对')
        return frames

    def cleanup(self):
        b = self.bus
        b.phase, b.cleanup_mode = 'cleanup', True
        result = {'writes_errors': [], 'readback': {}, 'read_errors': []}
        # These are idempotent stop/lock writes; the offset is never retried or rolled back.
        for addr, value, width in ((40, 0, 1), (48, 0, 2), (55, 1, 1)):
            try:
                b.write(addr, value, width)
            except BaseException as error:
                result['writes_errors'].append(f'{addr}: {type(error).__name__}: {error}')
        for addr, width in ((40, 1), (48, 2), (55, 1), (31, 2)):
            try:
                result['readback'][str(addr)] = b.value(2, addr, width)
            except BaseException as error:
                result['read_errors'].append(f'{addr}: {type(error).__name__}: {error}')
        self.report['cleanup'] = result
        self.report['release_confirmed'] = result['readback'].get('40') == 0 and result['readback'].get('48') == 0
        self.report['lock_confirmed'] = result['readback'].get('55') == 1
        b.cleanup_mode = False

    def verify_after(self, before):
        b = self.bus
        b.phase = 'read_only_after_write'
        after = self.snapshot()
        self.report['after'] = after
        for sid in IDS:
            old = bytes.fromhex(before[str(sid)]['config_raw_hex'])
            new = bytes.fromhex(after[str(sid)]['config_raw_hex'])
            if sid == 2:
                require(old[:31]+old[33:] == new[:31]+new[33:] and u16(new[31:33]) == NEW_OFFSET,
                        'ID2 offset 未正确改变，或非offset配置发生变化')
            else:
                require(old == new, f'ID{sid} 配置发生变化')
            a = bytes.fromhex(before[str(sid)]['frame']['raw_hex'])[:16]
            z = bytes.fromhex(after[str(sid)]['frame']['raw_hex'])[:16]
            keep = [i for i in range(16) if sid != 2 or i not in (0, 8, 9, 15)]
            require(all(a[i] == z[i] for i in keep), f'ID{sid} 非预期控制寄存器发生变化')
        # Preserve immediate post-write feedback; allow a documented settling interval
        # before checking a separate 2-second stable window. No earlier transient is hidden.
        self.sleep(.2)
        frames = self.stable_window(before, count=21, after=True)
        self.report['after_stable_window'] = frames
        self.report['expected_position_after'] = (before['2']['frame']['position']+SHIFT) % 4096
        self.report['actual_position_after'] = frames[-1]['2']['position']
        final_config = b.read(2, 0, 40)
        require(final_config.hex() == after['2']['config_raw_hex'], '后续复读配置发生变化')
        self.report['offset_readback_verified'] = True

    def run(self, apply=False):
        b, r = self.bus, self.report
        try:
            b.phase = 'initial_preflight'
            r['before'] = self.snapshot()
            self.validate_initial(r['before'])
            r['before_stable_window'] = self.stable_window(r['before'])
            b.journal.append('before_any_write', r)
            if not apply:
                r['status'] = 'read_only_preflight'
                return r
            # Revalidate all file hashes before granting the narrow write capability.
            fresh = load_proposal(self.proof['path'])
            require(fresh['sha256'] == self.proof['sha256'], '预览文件发生变化')
            r['attempt_marker'] = b.journal.mark_attempt(self.proof)
            b.apply_enabled = True
            b.phase = 'prepare_disabled_output'
            b.set_checked(40, 0)
            b.set_checked(48, 0, 2)
            b.phase = 'recheck_before_offset'
            second = self.snapshot()
            r['before_offset'] = second
            self.validate_initial(second)
            for sid in IDS:
                require(second[str(sid)]['config_raw_hex'] == r['before'][str(sid)]['config_raw_hex'],
                        f'ID{sid} 准备阶段配置变化')
                require(abs(delta(second[str(sid)]['frame']['position'],
                                  r['before'][str(sid)]['frame']['position'])) <= (2 if sid == 2 else 8),
                        f'ID{sid} 准备阶段位置变化')
            require(second['2']['frame']['output_limit'] == 0, 'ID2 输出上限未清零')
            b.journal.append('pre_offset_full_snapshot', second)
            b.phase = 'unlock_offset_configuration'
            b.set_checked(55, 0)
            require(all(b.value(sid, 40) == 0 for sid in IDS), '偏移写入前扭矩状态变化')
            last = self.frame(2)
            self.healthy(2, last, bytes.fromhex(second['2']['config_raw_hex']))
            require(last['output_limit'] == 0 and last['lock'] == 0
                    and abs(last['position']-second['2']['frame']['position']) <= 2,
                    '偏移写入前 ID2 状态变化')
            second['2']['frame'] = last
            b.journal.append('offset_anchor_and_intent', second)
            b.phase = 'single_offset_write'
            b.write(31, NEW_OFFSET, 2)
            self.sleep(.05)
            b.phase = 'relock'
            b.set_checked(55, 1)
        except BaseException as error:
            r['error'] = f'{type(error).__name__}: {error}'
            r['error_phase'] = b.phase
        finally:
            # Even lost ACKs count as writes attempted; cleanup must still run.
            if b.write_attempts:
                with uninterrupted_cleanup():
                    self.cleanup()
                    if b.offset_attempted:
                        try:
                            self.verify_after(r.get('before_offset', r['before']))
                        except BaseException as error:
                            r['verification_error'] = f'{type(error).__name__}: {error}'
                r['status'] = ('offset_readback_verified_mapping_pending'
                    if not r.get('error') and not r.get('verification_error')
                    and b.offset_acknowledged and r['offset_readback_verified']
                    and r['release_confirmed'] and r['lock_confirmed']
                    else 'failed_after_write_started')
            elif r['status'] != 'read_only_preflight':
                r['status'] = 'preflight_rejected_zero_writes'
            r.update(register_write_attempts=b.write_attempts, offset_write_attempted=b.offset_attempted,
                     offset_write_acknowledged=b.offset_acknowledged, events=b.events,
                     journal_errors=b.journal_errors)
            # Verify evidence is saved before declaring overall success.
            if b.journal_errors and r['status'] == 'offset_readback_verified_mapping_pending':
                r['status'] = 'journal_failed_mapping_pending'
            try:
                b.journal.append('final', r)
            except BaseException as error:
                r['final_journal_error'] = f'{type(error).__name__}: {error}'
                r['status'] = 'journal_failed_mapping_pending' if b.write_attempts else 'journal_failed_zero_writes'
        return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--log-dir', type=Path, default=ROOT/'data/real_control_runs')
    parser.add_argument('--apply', action='store_true', help='实际写入固定 ID2 偏移；否则全程只读')
    args = parser.parse_args()
    journal = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('收到停止信号')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        proof = load_proposal(args.proposal)
        journal = Journal(args.log_dir, proof)
        import serial
        print('保持机械臂稳定支撑；关闭其他串口/运动控制程序。全程不使能任何关节。', flush=True)
        print('模式：'+('修改 ID2 offset=0x0C9D，含配置解锁/锁定；软件映射保持原文件。'
                       if args.apply else '只读预检；不会修改寄存器。'), flush=True)
        with serial.Serial(args.port, baudrate=1_000_000, timeout=.15,
                           write_timeout=.15, exclusive=True) as port:
            result = Controller(OffsetBus(port, journal), proof).run(args.apply)
        print('状态：', result['status'], flush=True)
        print('寄存器写入尝试：', result['register_write_attempts'], flush=True)
        print('ID2 偏移读回核对：', result['offset_readback_verified'], flush=True)
        print('ID2 扭矩/输出清零确认：', result['release_confirmed'], flush=True)
        print('ID2 配置锁定确认：', result['lock_confirmed'], flush=True)
        if 'actual_position_after' in result:
            print('新位置：', result['actual_position_after'], '；预期：', result['expected_position_after'], flush=True)
        for key in ('error', 'verification_error', 'final_journal_error'):
            if key in result:
                print(key+': '+result[key], flush=True)
        if result['offset_write_attempted']:
            print('偏移写入已尝试；不要重复 --apply。旧 ID2 原始计数不可直接使用，软件映射待更新。', flush=True)
        print('本结果不代表断电持久性、模型标定或抓取验证通过。', flush=True)
        return 0 if result['status'] in ('read_only_preflight', 'offset_readback_verified_mapping_pending') else 1
    except BaseException as error:
        print(f'未完成：{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        if journal is not None:
            print('持久记录：', journal.path, flush=True)
            journal.stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
