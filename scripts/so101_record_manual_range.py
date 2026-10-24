"""手动摆动单个关节，连续只读记录两侧边界；不修改设备或生成运动命令。"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import select
import sys
import tempfile
import time

ROOT = Path('/home/yu/so101_ws')
MAPPING_SHA = 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8'
OFFSETS = [0, 3229, 0, 1822, 0, 3901]
MAX_GAP_S = 0.75
MAX_SAMPLE_DELTA = 128


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Recorder:
    def __init__(self, joint):
        require(type(joint) is int and 1 <= joint <= 6, '关节ID应为1–6')
        self.joint = joint
        self.last = None
        self.window = deque(maxlen=12)
        self.markers = {}
        self.count = 0
        self.baseline = None
        self.minimum = self.maximum = None
        self.invalid = False

    def ingest(self, states, timestamp):
        try:
            require(not self.invalid, '采样连续性已经失效，不能继续标记')
            require(math.isfinite(timestamp), '采样时间无效')
            require(len(states) == 6 and [s['servo_id'] for s in states] == list(range(1, 7)),
                    '需要按ID1–6排列的完整反馈')
            for s in states:
                require(type(s['position']) is int and 0 <= s['position'] <= 4095,
                        f"ID{s['servo_id']}位置不在单圈计数范围")
                require(s['torque'] == 0 and s['output_raw_signed'] == 0
                        and s['status'] == 0 and s['lock'] == 1,
                        f"ID{s['servo_id']}扭矩/输出/状态异常；停止读取，不执行清理写入")
                require(6.0 <= s['voltage'] <= 7.4 and s['temperature'] < 55,
                        f"ID{s['servo_id']}电压或温度异常")
            raw = [s['position'] for s in states]
            if self.last is None:
                continuous = raw.copy()
                self.baseline = raw.copy()
            else:
                require(0 < timestamp-self.last['t'] <= MAX_GAP_S,
                        '采样间隔过长或时间不递增，不能可靠衔接计数；本次停止')
                delta = [(a-b+2048) % 4096-2048 for a,b in zip(raw, self.last['raw_counts'])]
                require(all(abs(d) <= MAX_SAMPLE_DELTA for d in delta),
                        '相邻样本变化过大；请缓慢手动摆动，本次停止记录')
                continuous = [a+d for a,d in zip(self.last['continuous_counts'], delta)]
            sample = {'index': self.count, 't': timestamp, 'raw_counts': raw,
                      'continuous_counts': continuous, 'states': states}
            self.last = sample
            self.window.append(sample)
            self.count += 1
            value = continuous[self.joint-1]
            self.minimum = value if self.minimum is None else min(self.minimum, value)
            self.maximum = value if self.maximum is None else max(self.maximum, value)
            return sample
        except Exception:
            self.invalid = True
            raise

    def mark(self, label, note=''):
        require(not self.invalid and self.last is not None, '没有有效连续反馈')
        require(label in ('a', 'b') and label not in self.markers, '标记无效或已经记录；不覆盖旧边界')
        recent = list(self.window)[-7:]
        require(len(recent) == 7 and recent[-1]['t']-recent[0]['t'] >= 0.5,
                '尚未有足够稳定样本，请停稳约1秒后再标记')
        require(all(s['speed_raw_signed'] == 0 for sample in recent for s in sample['states']),
                '仍检测到运动，请停稳约1秒后再标记')
        intervals = [[min(r['continuous_counts'][i] for r in recent),
                      max(r['continuous_counts'][i] for r in recent)] for i in range(6)]
        require(all(hi-lo <= 4 for lo,hi in intervals), '六轴尚未稳定，请保持支撑后再标记')
        result = {'label': label, 'user_note': note, 'physical_clearance_verified': False,
                  'sample_index': self.last['index'], 'raw_counts': self.last['raw_counts'].copy(),
                  'continuous_counts': self.last['continuous_counts'].copy(),
                  'stable_intervals_counts': intervals, 'stable_samples': recent,
                  'scope': 'user-marked endpoint in this full-arm pose; not an approved limit'}
        self.markers[label] = result
        return result

    def summary(self):
        result = {'joint_id': self.joint, 'sample_count': self.count,
                  'continuous_tracking_valid': not self.invalid,
                  'markers': self.markers, 'two_distinct_endpoints_recorded': False,
                  'sampled_count_span': None if self.minimum is None else [self.minimum,self.maximum],
                  'endpoint_interval_counts': None, 'endpoint_span_encoder_deg': None}
        if not self.invalid and set(self.markers) == {'a','b'}:
            i = self.joint-1
            first,second = sorted(self.markers.values(), key=lambda m:m['continuous_counts'][i])
            if first['stable_intervals_counts'][i][1] < second['stable_intervals_counts'][i][0]:
                lo,hi = first['continuous_counts'][i],second['continuous_counts'][i]
                result.update(two_distinct_endpoints_recorded=True, endpoint_interval_counts=[lo,hi],
                              endpoint_span_encoder_deg=(hi-lo)*360/4096,
                              other_axes_difference_between_endpoints={str(j+1):second['continuous_counts'][j]-first['continuous_counts'][j]
                                                                      for j in range(6) if j != i})
        return result


def configuration(bus, decode):
    result = {}
    for sid,offset in enumerate(OFFSETS, 1):
        raw = bus.read(sid, 0, 40)
        require(len(raw) == 40, '配置长度错误')
        c = decode(raw)
        require(c['id'] == sid and c['offset_raw'] == offset and c['endianness'] == 0
                and c['mode'] == 0 and c['phase'] == 12 and c['resolution'] == 1
                and c['minimum'] == 0 and c['maximum'] == 4095,
                f'ID{sid}配置与当前映射不匹配；停止，不自动修复')
        result[str(sid)] = c
    return result


def acquire(bus, decode):
    states = []
    for sid in range(1,7):
        raw = bus.read(sid, 40, 26)
        require(len(raw) == 26, '反馈长度错误')
        frame = decode(raw)
        frame.update(servo_id=sid, sampled_at_ns=time.time_ns())
        states.append(frame)
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint', type=int, choices=range(1,7), required=True)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    args = parser.parse_args(argv)
    report = {'schema_version':1, 'kind':'so101_manual_joint_range', 'joint_id':args.joint,
              'status':'not_completed', 'measurement_method':'manual_torque_disabled',
              'execution_available':False, 'software_limits_verified':False,
              'register_writes':0, 'read_requests':0, 'configuration_unchanged':False,
              'error':None, 'started_at_ns':time.time_ns(),
              'sampling_assumption':'slow manual motion; gaps >0.75s or deltas >128 counts invalidate tracking',
              'scope':'Observed endpoints in two saved arm poses; not a collision-free joint-space box. No model or device limits modified.'}
    recorder = Recorder(args.joint)
    directory = ROOT/'data/joint_range_records'
    directory.mkdir(parents=True, exist_ok=True)
    fd,path = tempfile.mkstemp(prefix=f'joint{args.joint}_manual_', suffix='.jsonl', dir=directory)
    journal_path = Path(path)
    report_path = journal_path.with_suffix('.json')
    report['journal_path'] = str(journal_path)
    bus = None
    with os.fdopen(fd,'w',encoding='utf-8') as journal:
        def write(kind, payload, durable=False):
            journal.write(json.dumps({'kind':kind, 'data':payload},ensure_ascii=False,allow_nan=False)+'\n')
            journal.flush()
            if durable:
                os.fsync(journal.fileno())

        def events():
            if bus is not None and bus.events:
                batch = bus.events.copy()
                write('read_events', batch)
                report['read_requests'] += sum(e['instruction']==2 for e in batch)
                report['register_writes'] += sum(e['instruction']!=2 for e in batch)
                bus.events.clear()

        write('start',report,True)
        print('只读手动范围记录；支撑机械臂，六轴扭矩必须关闭；关闭其他串口程序。',flush=True)
        print(f'本次只测ID{args.joint}。缓慢摆动，停稳约1秒后输入 a 回车记录一侧，b 回车记录另一侧；q 回车结束。',flush=True)
        print('可在a或b后补充停止原因；不要顶紧结构或拉紧线缆。最多记录5分钟。',flush=True)
        print('过程记录：',journal_path,flush=True)
        try:
            require(sys.stdin.isatty(), '请直接在交互终端运行，不要通过管道输入')
            mapping_path = ROOT/'data/so101_joint_mapping.json'
            mapping_bytes = mapping_path.read_bytes()
            require(hashlib.sha256(mapping_bytes).hexdigest()==MAPPING_SHA, '当前映射改变，请先核对')
            report['mapping_sha256'] = MAPPING_SHA
            sys.path.insert(0,str(ROOT/'scripts'))
            import serial
            from so101_verify_gripper_center import ReadOnlyBus
            from so101_center_gripper import decode_config
            from so101_gripper_cycle import frame_fields
            finished_by_q = False
            with serial.Serial(args.port,baudrate=1_000_000,timeout=.15,write_timeout=.15,exclusive=True) as port:
                bus = ReadOnlyBus(port)
                bus.phase = 'manual_range_configuration_before'
                before = configuration(bus,decode_config)
                report['config_before'] = before
                write('configuration_before',before,True)
                events()
                started = time.monotonic()
                last_display = -1.0
                while time.monotonic()-started < 300:
                    tick = time.monotonic()
                    bus.phase = 'manual_range_feedback'
                    states = acquire(bus,frame_fields)
                    elapsed = time.monotonic()-started
                    write('raw_feedback',{'t':elapsed,'states':states})
                    events()
                    sample = recorder.ingest(states,elapsed)
                    write('sample',sample)
                    if elapsed-last_display >= .5:
                        print(f"ID{args.joint}：原始={sample['raw_counts'][args.joint-1]}，"
                              f"连续计数={sample['continuous_counts'][args.joint-1]}；"
                              f"六轴={sample['raw_counts']}；已标记={list(recorder.markers)}",flush=True)
                        last_display = elapsed
                        os.fsync(journal.fileno())
                    if select.select([sys.stdin],[],[],0)[0]:
                        line = sys.stdin.readline()
                        require(line != '', '输入关闭，本次不算完整记录')
                        command, _, note = line.strip().partition(' ')
                        if command == 'q':
                            finished_by_q = True
                            break
                        if command in ('a','b'):
                            try:
                                marker = recorder.mark(command,note)
                                write('endpoint',marker,True)
                                print(f"已记录{command.upper()}：六轴={marker['raw_counts']}；ID{args.joint}连续计数={marker['continuous_counts'][args.joint-1]}",flush=True)
                            except ValueError as exc:
                                print('未标记：',exc,flush=True)
                        else:
                            print('输入 a、b 或 q 后回车。',flush=True)
                    time.sleep(max(0,.1-(time.monotonic()-tick)))
                bus.phase = 'manual_range_configuration_after'
                after = configuration(bus,decode_config)
                events()
                report['config_after'] = after
                write('configuration_after',after,True)
                require(before == after, '采集前后配置发生变化；端点不验收')
                require(mapping_path.read_bytes()==mapping_bytes, '采集期间软件映射改变')
                report['configuration_unchanged'] = True
                require(finished_by_q, '已到5分钟时限；本次记录不算完整')
                require(recorder.summary()['two_distinct_endpoints_recorded'], '尚未记录两个分离的稳定端点')
                require(report['register_writes'] == 0, '发现非读取指令，本次记录不验收')
                report['status'] = 'manual_endpoints_recorded_not_limits_verified'
        except (Exception, KeyboardInterrupt) as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            print('记录停止：',report['error'],flush=True)
        finally:
            events()
            report['observation'] = recorder.summary()
            report['finished_at_ns'] = time.time_ns()
            write('final',report,True)
    with report_path.open('w',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print('状态：',report['status'],flush=True)
    print('寄存器写入次数：',report['register_writes'],flush=True)
    print('两端连续计数：',report['observation']['endpoint_interval_counts'],flush=True)
    print('完整报告：',report_path,flush=True)
    print('这只是手动边界观察；未写入设备限位，未验证多关节碰撞或抓取。',flush=True)
    return 0 if report['error'] is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
