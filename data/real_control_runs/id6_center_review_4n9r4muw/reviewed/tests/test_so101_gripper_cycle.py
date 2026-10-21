"""Fixed ID6 cycle tests: packet mocks only, never opens serial hardware."""
import copy
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from so101_gripper_cycle import (GripperBus,GripperCycle,ControlError,fixed_path,main,
                                opening_path,stopped_evidence,STOPPED_SHA256)
from test_so101_center_gripper import FakeSerial,Clock

AFTER_CONFIG=bytes.fromhex('030a000903060000010000ff0f465028e8030c2c2f20200010000101f401013d0f0014c8500ac8c8')

class CycleSerial(FakeSerial):
    def __init__(self,clock):
        super().__init__(clock)
        self.regs[6][:40]=AFTER_CONFIG
        self.put(6,48,0);self.put(6,42,0);self.put(6,56,2048)
        self.follow=True;self.before_hook=None;self.auto_enable=True
        self.pre_goal_states=[]
    def write(self,packet):
        sid,ins,address=packet[2],packet[4],packet[5]
        params=packet[6:-1]
        if sid==6 and ins==3 and address==42:
            self.pre_goal_states.append((self.regs[6][40],self.get(6,48),int.from_bytes(params,'little')))
        if sid==6 and ins==2 and address==40 and params==b'\x1a':
            if self.follow and self.regs[6][40]==1 and self.get(6,48)>0:
                self.put(6,56,self.get(6,42))
        if self.before_hook:self.before_hook(self,ins,sid,address,params)
        result=super().write(packet)
        if sid==6 and ins==3 and address==42 and self.auto_enable:
            self.regs[6][40]=1
        return result

class MemoryJournal:
    def __init__(self,initial,fail_stage=None):
        self.initial=copy.deepcopy(initial);self.fail_stage=fail_stage
        self.path=Path('/tmp/mock_only_no_file.jsonl');self.stages=[]
    def append(self,stage,snapshot):
        self.stages.append(stage)
        if stage==self.fail_stage:raise OSError('mock disk failure')
    def close(self):pass

class CycleTests(unittest.TestCase):
    def make(self,fail_stage=None):
        clock=Clock();port=CycleSerial(clock);messages=[]
        proof={'calibration':{'after_config':{'raw_hex':AFTER_CONFIG.hex()}}}
        def factory(directory,snapshot):
            self.assertEqual(port.writes,[])
            return MemoryJournal(snapshot,fail_stage)
        control=GripperCycle(GripperBus(port,clock=clock.time),proof,journal_dir='/tmp',
            clock=clock.time,sleep=clock.sleep,emit=messages.append,journal_factory=factory)
        port.control=control
        return port,control,messages
    def assert_released(self,p,c):
        self.assertEqual(p.regs[6][40],0);self.assertEqual(p.get(6,48),0)
        self.assertTrue(c.release_result['confirmed'])
    def goals(self,p):return [value for sid,addr,value in p.writes if addr==42]
    def assert_no_return(self,p):
        goals=self.goals(p)
        self.assertTrue(all(b<=a for a,b in zip(goals,goals[1:])),goals)

    def make_open(self,start=1891):
        p,c,messages=self.make();p.put(6,56,start);c.open_only=True
        c.proof['open_only_recovery']={'sha256':STOPPED_SHA256,'fresh_start_bounds':[1888,2050],
                                      'open_target':2048}
        return p,c,messages

    def test_open_only_preview_is_zero_write_with_monotone_path(self):
        p,c,_=self.make_open();before={sid:bytes(r) for sid,r in p.regs.items()}
        result=c.run()
        self.assertEqual(p.writes,[]);self.assertIsNone(c.journal)
        self.assertEqual(result['status'],'read_only_preview');self.assertEqual(result['mode'],'open_only')
        self.assertEqual(result['path'],[1891,1907,1923,1939,1955,1971,1987,2003,2019,2035,2048])
        self.assertEqual(before,{sid:bytes(r) for sid,r in p.regs.items()})

    def test_open_only_success_moves_from_fresh_1891_to_2048_without_closing(self):
        p,c,messages=self.make_open();others={sid:bytes(p.regs[sid]) for sid in range(1,6)}
        result=c.run(True)
        self.assertEqual(result['status'],'empty_gripper_open_only_complete')
        self.assertEqual(result['test'],'fixed_id6_empty_gripper_open_only')
        self.assertTrue(result['open_response_complete']);self.assertFalse(result['cycle_response_complete'])
        self.assertEqual(self.goals(p),opening_path(1891))
        self.assertTrue(all(0<b-a<=16 for a,b in zip(self.goals(p),self.goals(p)[1:])))
        self.assertEqual(p.get(6,56),2048)
        self.assertEqual(c.journal.initial['start'],1891)
        self.assertEqual(c.journal.initial['proof']['open_only_recovery']['sha256'],STOPPED_SHA256)
        self.assertEqual(others,{sid:bytes(p.regs[sid]) for sid in range(1,6)})
        self.assertTrue(all(sid==6 for sid,_,_ in p.writes));self.assert_released(p,c)
        self.assertEqual(p.pre_goal_states[0],(0,0,1891))
        self.assertTrue(all(torque==1 and cap==100 for torque,cap,_ in p.pre_goal_states[1:]))
        self.assertFalse(any(s['phase'] in ('closing','returning','precontact_observation') for s in c.samples))

    def test_open_only_2048_2049_2050_only_hold_current_and_never_negative_step(self):
        for start in (2048,2049,2050):
            with self.subTest(start=start):
                p,c,_=self.make_open(start);result=c.run(True)
                self.assertEqual(self.goals(p),[start]);self.assertEqual(result['path'],[start])
                self.assertEqual(p.get(6,56),start);self.assert_released(p,c)

    def test_open_only_accepts_lower_bound_but_rejects_outside_or_missing_proof_before_write(self):
        p,c,_=self.make_open(1888);c.run(True)
        self.assertEqual(self.goals(p),opening_path(1888));self.assert_released(p,c)
        for start in (1887,2051):
            with self.subTest(start=start):
                p,c,_=self.make_open(start)
                with self.assertRaises(ControlError):c.run(True)
                self.assertEqual(p.writes,[])
        p,c,_=self.make_open();del c.proof['open_only_recovery']
        with self.assertRaises(RuntimeError):c.run(True)
        self.assertEqual(p.writes,[])

    def test_open_only_timeout_stops_without_further_goal_or_fault_return(self):
        p,c,_=self.make_open();p.follow=False
        with self.assertRaisesRegex(ControlError,'3秒'):c.run(True)
        self.assertEqual(self.goals(p),[1891,1907]);self.assert_released(p,c)
        self.assertEqual(c.status,'failed_no_return');self.assertFalse(c.open_response_complete)

    def test_open_only_global_and_current_step_corridors_stop_overshoot(self):
        for actual in (1910,2051,1887):
            with self.subTest(actual=actual):
                p,c,_=self.make_open()
                def alter(port,ins,sid,addr,params):
                    if c.bus.phase=='opening_only' and port.get(6,42)==1907 and (ins,sid,addr)==(2,6,40):
                        port.put(6,56,actual)
                p.before_hook=alter
                with self.assertRaisesRegex(ControlError,'越界'):c.run(True)
                self.assertEqual(self.goals(p),[1891,1907]);self.assert_released(p,c)

    def test_open_only_support_drift_reports_details_and_preserves_limit(self):
        p,c,_=self.make_open()
        def drift(port,ins,sid,addr,params):
            if c.bus.phase=='opening_only' and port.get(6,42)==1907:port.put(1,56,113)
        p.before_hook=drift
        with self.assertRaisesRegex(ControlError,r'初始=100.*实际=113.*差值=\+13.*允许±12'):c.run(True)
        self.assertEqual(self.goals(p),[1891,1907]);self.assert_released(p,c)
        self.assertTrue(all(sid==6 for sid,_,_ in p.writes))

    def test_open_only_id5_low_voltage_remains_a_stop_condition(self):
        for during_motion in (False,True):
            with self.subTest(during_motion=during_motion):
                p,c,_=self.make_open()
                if during_motion:
                    def low(port,ins,sid,addr,params):
                        if c.bus.phase=='opening_only' and port.get(6,42)==1907:port.regs[5][62]=58
                    p.before_hook=low
                else:p.regs[5][62]=58
                with self.assertRaisesRegex(ControlError,'ID5健康'):c.run(True)
                if during_motion:
                    self.assertEqual(self.goals(p),[1891,1907]);self.assert_released(p,c)
                else:self.assertEqual(p.writes,[])

    def test_stopped_evidence_is_digest_pinned_and_binds_original_sources(self):
        proof={'calibration':{'sha256':'calibration-sha','after_config':{'raw_hex':AFTER_CONFIG.hex()}},
               'verification_sha256':'verification-sha','gap_record_sha256':'gap-sha'}
        initial={'proof':proof,'initial_configs':{'6':AFTER_CONFIG.hex()},'start':2050}
        final={**copy.deepcopy(initial),'test':'fixed_id6_empty_gripper_cycle','status':'failed_no_return',
               'fault':{'phase':'closing','error':'ControlError: ID1支撑姿态漂移超过允许值'},
               'release':{'servo_id':6,'readback':{'40':0,'48':0},'errors':[],'confirmed':True},
               'journal_error':None,'cycle_response_complete':False,'path':fixed_path(2050),
               'current_goal':1890,'events':[{'instruction':2,'parameters_hex':'0028','servo_id':6,
                 'acknowledged':True,'response_payload_hex':AFTER_CONFIG.hex()}],
               'samples':[{'servo_id':6,'goal':1890,'position':1891,'status':0},
                          {'servo_id':1,'position':3998}],'initial_positions':{'1':4013}}
        rows=[{'stage':'before_any_write','snapshot':initial},{'stage':'final_cleanup','snapshot':final}]
        def encode(value):return ('\n'.join(json.dumps(row) for row in value)+'\n').encode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stop.jsonl';raw=encode(rows);path.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError,'原始文件'):stopped_evidence(path,proof)
            with patch('so101_gripper_cycle.STOPPED_SHA256',hashlib.sha256(raw).hexdigest()):
                result=stopped_evidence(path,proof)
                self.assertEqual(result['fresh_start_bounds'],[1888,2050]);self.assertTrue(result['independent_transaction'])
                badproof=copy.deepcopy(proof);badproof['verification_sha256']='different'
                with self.assertRaisesRegex(RuntimeError,'来源不一致'):stopped_evidence(path,badproof)
                path.write_bytes(raw+b' ')
                with self.assertRaisesRegex(RuntimeError,'原始文件'):stopped_evidence(path,proof)
            for mutate in (
                lambda r:r[-1]['snapshot']['release']['readback'].__setitem__('40',1),
                lambda r:r[-1]['snapshot'].__setitem__('current_goal',1874),
                lambda r:r[-1]['snapshot']['fault'].__setitem__('phase','returning'),
                lambda r:r[-1]['snapshot']['events'][0].__setitem__('response_payload_hex','00'*40),
                lambda r:r[-1]['snapshot']['samples'][0].__setitem__('position',1800),
            ):
                changed=copy.deepcopy(rows);mutate(changed);badraw=encode(changed);path.write_bytes(badraw)
                with patch('so101_gripper_cycle.STOPPED_SHA256',hashlib.sha256(badraw).hexdigest()):
                    with self.assertRaises(RuntimeError):stopped_evidence(path,proof)

    def test_open_only_wrong_stopped_source_is_rejected_before_opening_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrong.jsonl';path.write_text('{}\n')
            with patch('so101_gripper_cycle.evidence',return_value={}),patch('serial.Serial') as serial_open:
                with redirect_stderr(io.StringIO()):
                    code=main(['--open-only','--enable','--stopped-journal',str(path)])
                self.assertEqual(code,1);serial_open.assert_not_called()

    def test_preview_zero_write_and_fixed_path(self):
        p,c,_=self.make();before={sid:bytes(r) for sid,r in p.regs.items()}
        result=c.run()
        self.assertEqual(result['status'],'read_only_preview');self.assertEqual(p.writes,[])
        self.assertEqual(result['path'],fixed_path(2048))
        self.assertEqual(before,{sid:bytes(r) for sid,r in p.regs.items()})
        self.assertIsNone(c.journal)

    def test_success_fixed_cycle_never_targets_closed_prediction_or_other_axes(self):
        p,c,_=self.make();others={sid:bytes(p.regs[sid]) for sid in range(1,6)}
        result=c.run(True)
        self.assertEqual(result['status'],'empty_gripper_cycle_complete')
        self.assertEqual(self.goals(p),fixed_path(2048))
        self.assertEqual(min(self.goals(p)),1727);self.assertNotIn(1710,self.goals(p))
        self.assertTrue(all(abs(b-a)<=16 for a,b in zip(self.goals(p),self.goals(p)[1:])))
        self.assertTrue(all(sid==6 for sid,_,_ in p.writes))
        self.assertEqual(others,{sid:bytes(p.regs[sid]) for sid in range(1,6)})
        self.assert_released(p,c)
        self.assertEqual(c.journal.initial['writes_sent'],0)
        self.assertEqual(c.journal.initial['start'],2048)
        self.assertFalse(result['grasp_verified'])

    def test_returns_to_actual_fresh_start_not_hardcoded_2048(self):
        for start in (2046,2050):
            with self.subTest(start=start):
                p,c,_=self.make();p.put(6,56,start)
                c.run(True)
                self.assertEqual(self.goals(p),fixed_path(start))
                self.assertEqual(p.get(6,56),start)
                self.assert_released(p,c)

    def test_goal_autoenable_is_cleared_before_output_is_nonzero(self):
        p,c,_=self.make();c.run(True)
        self.assertEqual(p.pre_goal_states[0],(0,0,2048))
        self.assertTrue(all(torque==1 and output==100 for torque,output,_ in p.pre_goal_states[1:]))
        writes=p.writes
        firstgoal=writes.index((6,42,2048));firstpower=writes.index((6,48,100))
        self.assertEqual(writes[firstgoal+1],(6,40,0))
        self.assertIn((6,40,1),writes[firstgoal+2:firstpower])
        self.assertTrue(all(value in (0,100) for _,addr,value in writes if addr==48))

    def test_preflight_bad_offset_start_torque_health_are_zero_write(self):
        cases=((6,31,3902,2),(6,56,2051,2),(2,40,1,1),(6,58,1,2),(6,60,1,2),
               (6,62,75,1),(4,63,55,1),(6,48,100,2))
        for sid,addr,value,width in cases:
            with self.subTest(sid=sid,addr=addr,value=value):
                p,c,_=self.make()
                if width==2:p.put(sid,addr,value)
                else:p.regs[sid][addr]=value
                with self.assertRaises(ControlError):c.run(True)
                self.assertEqual(p.writes,[])

    def test_preflight_unstable_position_has_zero_writes(self):
        p,c,_=self.make()
        original=c.sleep
        def drift(duration):original(duration);p.put(6,56,2050)
        p.put(6,56,2047);c.sleep=drift
        with self.assertRaisesRegex(ControlError,'三帧'):c.run(True)
        self.assertEqual(p.writes,[])

    def test_no_movement_times_out_and_does_not_return_or_increase_output(self):
        p,c,_=self.make();p.follow=False
        with self.assertRaisesRegex(ControlError,'3秒'):c.run(True)
        self.assertEqual(self.goals(p),[2048,2032])
        self.assert_no_return(p);self.assert_released(p,c)

    def test_small_step_overshoot_stops_on_id6_packet_before_reading_other_axes(self):
        p,c,_=self.make()
        def overshoot(port,ins,sid,address,params):
            if c.bus.phase=='closing' and (ins,sid,address,params)==(2,6,40,b'\x1a') and port.get(6,42)<2048:
                port.put(6,56,port.get(6,42)-3)
        p.before_hook=overshoot
        with self.assertRaisesRegex(ControlError,'越界'):c.run(True)
        eventindex=next(i for i,e in enumerate(c.bus.events) if e['phase']=='release')
        self.assertEqual(c.bus.events[eventindex-1]['servo_id'],6)
        self.assertEqual(self.goals(p),[2048,2032]);self.assert_released(p,c)

    def test_health_id4_drift_and_control_corruption_abort_without_return(self):
        for sid,address,value,width in ((6,63,55,1),(4,56,404,2),(6,41,2,1)):
            with self.subTest(sid=sid,address=address):
                p,c,_=self.make()
                def alter(port,ins,read_sid,addr,params):
                    if c.bus.phase=='closing' and port.get(6,42)<2048:
                        if width==2:port.put(sid,address,value)
                        else:port.regs[sid][address]=value
                p.before_hook=alter
                with self.assertRaises(ControlError):c.run(True)
                self.assert_released(p,c);self.assert_no_return(p)

    def test_saturation_for_one_second_aborts_not_contact_success(self):
        p,c,_=self.make();p.follow=False
        def loaded(port,ins,sid,address,params):
            if c.bus.phase=='closing' and port.get(6,42)<2048:port.put(6,60,95)
        p.before_hook=loaded
        with self.assertRaisesRegex(ControlError,'1秒'):c.run(True)
        self.assertEqual(self.goals(p),[2048,2032]);self.assert_released(p,c)
        self.assertFalse(c.cycle_response_complete)

    def test_lost_goal_ack_and_interrupt_still_release_without_return(self):
        for fault in ('ack','interrupt','first_ack'):
            with self.subTest(fault=fault):
                p,c,_=self.make();fired=[]
                def fail(port,ins,sid,address,params):
                    relevant=(c.bus.phase=='prepare' and address==48) if fault=='first_ack' else (c.bus.phase=='closing' and address==42)
                    if not fired and relevant and ins==3:
                        fired.append(True)
                        if fault in ('ack','first_ack'):port.pending=b''
                        else:raise KeyboardInterrupt
                p.hook=fail
                with self.assertRaises((ControlError,KeyboardInterrupt)):c.run(True)
                self.assert_released(p,c);self.assert_no_return(p)
                self.assertEqual(len(self.goals(p)),0 if fault=='first_ack' else 2)

    def test_configuration_change_blocks_next_goal(self):
        p,c,_=self.make()
        def alter(port,ins,sid,addr,params):
            if c.bus.phase=='closing' and port.get(6,42)==2032:port.regs[6][31]=0x3c
        p.hook=alter
        with self.assertRaisesRegex(ControlError,'配置'):c.run(True)
        self.assertEqual(self.goals(p),[2048,2032]);self.assert_released(p,c)

    def test_cleanup_both_paths_are_attempted_after_one_fails(self):
        p,c,_=self.make();p.follow=False
        def fail(port,ins,sid,addr,params):
            if c.bus.phase=='release' and addr==40:port.pending=b''
        p.hook=fail
        with self.assertRaisesRegex(ControlError,'清零未确认'):c.run(True)
        release=[e for e in c.bus.events if e['phase']=='release']
        self.assertEqual([(e['instruction'],bytes.fromhex(e['parameters_hex'])[0]) for e in release],
                         [(3,40),(3,48),(2,40),(2,48)])
        self.assertEqual(c.release_result['readback']['48'],0)
        self.assert_no_return(p)

    def test_journal_creation_failure_is_zero_write_and_later_failures_release(self):
        p,c,_=self.make()
        c.journal_factory=lambda *a: (_ for _ in ()).throw(OSError('disk'))
        with self.assertRaises(OSError):c.run(True)
        self.assertEqual(p.writes,[])
        p,c,_=self.make(fail_stage='final_cleanup')
        with self.assertRaises(ControlError):c.run(True)
        self.assert_released(p,c)
        self.assertEqual(c.status,'journal_save_failed')

    def test_no_disk_append_while_output_is_enabled(self):
        p,c,_=self.make()
        original=c.journal_factory
        def factory(directory,snapshot):
            journal=original(directory,snapshot)
            append=journal.append
            def guarded(stage,report):
                self.assertEqual(p.regs[6][40],0)
                self.assertEqual(p.get(6,48),0)
                append(stage,report)
            journal.append=guarded
            return journal
        c.journal_factory=factory
        c.run(True)
        self.assertEqual(c.journal.stages,['final_cleanup'])

    def test_recheck_after_journal_never_changes_start_or_writes_on_drift(self):
        for sid,address,value,width in ((6,56,2051,2),(6,31,3902,2),(4,56,404,2),(2,40,1,1)):
            with self.subTest(sid=sid,address=address):
                p,c,_=self.make();original=c.journal_factory
                def factory(directory,snapshot):
                    journal=original(directory,snapshot)
                    if width==2:p.put(sid,address,value)
                    else:p.regs[sid][address]=value
                    return journal
                c.journal_factory=factory
                with self.assertRaises(ControlError):c.run(True)
                self.assertEqual(p.writes,[])
                self.assertEqual(c.status,'failed_before_write')
                self.assertEqual(c.start,2048)
                self.assertEqual(c.path,fixed_path(2048))
                self.assertIsNone(c.release_result)

    def test_two_second_precontact_observation_is_monitored_and_fault_prevents_return(self):
        p,c,messages=self.make()
        c.run(True)
        observed=[s for s in c.samples if s['phase']=='precontact_observation' and s['servo_id']==6]
        self.assertGreaterEqual(observed[-1]['elapsed_s']-observed[0]['elapsed_s'],1.9)
        self.assertTrue(any('保持2秒' in m for m in messages))
        p,c,_=self.make()
        def fault(port,ins,sid,addr,params):
            if c.bus.phase=='precontact_observation':port.put(6,60,95)
        p.before_hook=fault
        with self.assertRaisesRegex(ControlError,'1秒'):c.run(True)
        self.assertEqual(self.goals(p)[-1],1727)
        self.assert_no_return(p);self.assert_released(p,c)

    def test_settled_span_uses_id6_sample_time_not_late_other_axis_response(self):
        p,c,_=self.make()
        # First three ID6 samples span .112s; only the third round's ID5 is slow.
        count=[]
        def slow(port,ins,sid,addr,params):
            if c.bus.phase=='closing' and port.get(6,42)==2032:
                if ins==2 and sid==6 and addr==40:count.append(1)
                if ins==2 and sid==5 and addr==40 and len(count)==3:port.clock.now+=.15
        p.hook=slow
        c.run(True)
        first=next(step for step in c.steps if step['target']==2032)
        self.assertGreaterEqual(first['settled_samples'],4)
        self.assertGreaterEqual(first['settled_span_s'],.2)

    def test_feedback_and_step_deadlines_include_io(self):
        for mode in ('round','step'):
            with self.subTest(mode=mode):
                p,c,_=self.make();fired=[]
                def delay(port,ins,sid,addr,params):
                    if not fired and c.bus.phase=='closing' and ins==2:
                        fired.append(True)
                        port.clock.now+=.6 if mode=='round' and addr==40 else (3.1 if mode=='step' else 0)
                p.hook=delay
                # For the round check delay its first feedback, not config read.
                if mode=='round':
                    def delay_round(port,ins,sid,addr,params):
                        if not fired and c.bus.phase=='closing' and ins==2 and addr==40:
                            fired.append(True);port.clock.now+=.6
                    p.hook=delay_round
                with self.assertRaises(ControlError):c.run(True)
                self.assert_released(p,c);self.assert_no_return(p)
                self.assertEqual(self.goals(p),[2048])

    def test_total_90_second_deadline_stops_before_any_further_goal(self):
        p,c,_=self.make()
        def delay(port,ins,sid,addr,params):
            if c.bus.phase=='closing' and ins==3 and addr==42:
                port.clock.now=c.active_since+90.001
        p.hook=delay
        with self.assertRaisesRegex(ControlError,'90秒'):c.run(True)
        self.assertEqual(self.goals(p),[2048,2032])
        self.assert_released(p,c);self.assert_no_return(p)

    def test_bus_whitelist_rejects_other_axes_128_offsets_and_unplanned_goal(self):
        p,c,_=self.make();bus=c.bus
        with self.assertRaises(ControlError):bus.write(6,40,0)
        bus.authorize_path(fixed_path(2048))
        for sid,addr,value in ((1,40,0),(5,48,0),(6,40,128),(6,31,3901),(6,55,0),
                               (6,48,101),(6,41,2),(6,46,51),(6,42,1710),(6,42,2001)):
            with self.subTest(sid=sid,addr=addr,value=value),self.assertRaises(ControlError):
                bus.write(sid,addr,value)
        for ins,params in ((4,[40,1]),(3,[40,128]),(3,[48,100]),(3,[42,174,6])):
            with self.assertRaises(ControlError):bus.exchange(6,ins,params,0)
        self.assertEqual(p.writes,[])

    def test_help_does_not_open_serial_and_no_arbitrary_control_flags(self):
        with redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as result:main(['--help'])
        self.assertEqual(result.exception.code,0)
        for option in ('--limit','--delta','--joint','--target'):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main([option,'1'])

if __name__=='__main__':unittest.main()
