"""Offline packet mocks; no pyserial/device access."""
import builtins
import copy
import hashlib
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import so101_verify_gripper_center as verify
import so101_center_gripper as center
from test_so101_center_gripper import FakeSerial, Clock


class VerifyTests(unittest.TestCase):
    def setup_case(self, hook=None):
        temp=tempfile.TemporaryDirectory(prefix='center_verify_mock_')
        self.addCleanup(temp.cleanup)
        clock=Clock()
        port=FakeSerial(clock)
        port.put(6,56,195)
        old=center.GripperCenterController(center.CenterBus(port,clock=clock.time),
            journal_dir=temp.name,clock=clock.time,sleep=clock.sleep,emit=lambda x:None)
        port.control=old
        old.run(True)
        source_path=old.journal.path
        source=verify.load_source(source_path)
        port.events.clear()
        port.writes.clear()
        port.hook=hook
        control=verify.VerifyController(verify.ReadOnlyBus(port,clock=clock.time),source,
            clock=clock.time,sleep=clock.sleep,emit=lambda x:None)
        port.control=control
        return port,control,source_path

    def frame_response(self,p,relative_address,value,width=2):
        response=bytearray(p.pending)
        response[5+relative_address:5+relative_address+width]=value.to_bytes(width,'little')
        response[-1]=(~sum(response[2:-1]))&255
        p.pending=bytes(response)

    def assert_read_only(self,port,report):
        self.assertEqual(port.writes,[])
        self.assertTrue(all(event[0]==2 for event in port.events))
        self.assertEqual(report['writes_sent'],0)

    def test_success_keeps_21_full_frames_and_distinguishes_historical_predictions(self):
        port,control,source_path=self.setup_case()
        original=source_path.read_bytes()
        report=control.run()
        self.assertEqual(report['status'],'read_only_center_verified')
        self.assertEqual(len(report['samples']),21)
        self.assertGreaterEqual(report['sample_span_s'],2)
        self.assertEqual(report['position_range_counts'],0)
        self.assertTrue(all(len(bytes.fromhex(s['raw_hex']))==26 for s in report['samples']))
        self.assertEqual(report['historical_predictions']['observed_closed']['predicted_count'],1710)
        self.assertEqual(report['historical_predictions']['observed_open']['predicted_count'],2047)
        self.assertEqual(report['historical_predictions']['current_calibration_anchor']['old_count'],195)
        self.assertEqual(report['historical_predictions']['current_calibration_anchor']['predicted_count'],2048)
        self.assertEqual(report['source']['sha256'],hashlib.sha256(original).hexdigest())
        self.assertFalse(report['motion_authorized'])
        self.assertFalse(report['power_cycle_persistence_verified'])
        self.assert_read_only(port,report)
        paths=[verify.save_report(source_path.parent,report) for _ in range(2)]
        self.assertNotEqual(*paths)
        self.assertEqual(source_path.read_bytes(),original)

    def test_current_offset_change_refuses_without_sampling_or_writes(self):
        port,control,_=self.setup_case()
        port.put(6,31,3902)
        report=control.run()
        self.assertEqual(report['status'],'not_verified')
        self.assertEqual(report['samples'],[])
        self.assertTrue(any('offset' in s for s in report['reasons']))
        self.assert_read_only(port,report)

    def test_late_nonzero_speed_or_output_is_preserved_and_all_samples_collected(self):
        for address,value in ((18,27114),(20,1)):
            with self.subTest(address=address):
                count=[]
                def change(p,ins,sid,addr,params):
                    if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                        count.append(1)
                        if len(count)>=12:
                            self.frame_response(p,address,value)
                port,control,_=self.setup_case(change)
                report=control.run()
                self.assertEqual(report['status'],'not_verified')
                self.assertEqual(len(report['samples']),21)
                self.assertEqual(len(count),21)
                self.assertTrue(report['samples'][11]['verification_failures'])
                self.assertEqual(report['samples'][11]['speed_raw' if address==18 else 'output_raw'],value)
                self.assertIn('final',report)
                self.assert_read_only(port,report)

    def test_same_frame_torque_or_output_limit_cannot_hide_between_end_reads(self):
        for address,value,width in ((0,1,1),(8,100,2)):
            with self.subTest(address=address):
                def change(p,ins,sid,addr,params):
                    if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                        self.frame_response(p,address,value,width)
                port,control,_=self.setup_case(change)
                report=control.run()
                self.assertEqual(report['status'],'not_verified')
                self.assertEqual(len(report['samples']),21)
                self.assertTrue(any('同帧' in s for s in report['reasons']))
                self.assert_read_only(port,report)

    def test_config_or_control_changes_during_window_are_rejected(self):
        for address,value in ((21,31),(55,0)):
            with self.subTest(address=address):
                def change(p,ins,sid,addr,params):
                    if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                        p.regs[6][address]=value
                port,control,_=self.setup_case(change)
                report=control.run()
                self.assertEqual(report['status'],'not_verified')
                self.assertEqual(len(report['samples']),21)
                self.assertTrue(any('配置' in s or '控制' in s for s in report['reasons']))
                self.assert_read_only(port,report)

    def test_transient_control_change_cannot_hide_between_identical_end_states(self):
        count=[]
        def change(p,ins,sid,addr,params):
            if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                count.append(1)
                if len(count)==12:
                    self.frame_response(p,2,1001)  # Feedback packet only; restore at once.
        port,control,_=self.setup_case(change)
        report=control.run()
        self.assertEqual(report['status'],'not_verified')
        self.assertEqual(len(report['samples']),21)
        self.assertEqual(report['initial']['control'],report['final']['control'])
        self.assertTrue(any('控制40..55' in s for s in report['samples'][11]['verification_failures']))
        self.assert_read_only(port,report)

    def test_position_window_and_anchor_deviation_are_checked_over_all_samples(self):
        for positions in ((2046,2049),(2051,2051)):
            with self.subTest(positions=positions):
                count=[]
                def change(p,ins,sid,addr,params):
                    if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                        count.append(1)
                        self.frame_response(p,16,positions[len(count)>10])
                port,control,_=self.setup_case(change)
                report=control.run()
                self.assertEqual(report['status'],'not_verified')
                self.assertEqual(len(report['samples']),21)
                self.assertTrue(any('位置' in s for s in report['reasons']))
                self.assert_read_only(port,report)

    def test_nonzero_health_preserved_for_full_window(self):
        def change(p,ins,sid,addr,params):
            if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                self.frame_response(p,23,55,1)
        port,control,_=self.setup_case(change)
        report=control.run()
        self.assertEqual(report['status'],'not_verified')
        self.assertEqual(len(report['samples']),21)
        self.assertTrue(any('健康' in s for s in report['reasons']))
        self.assert_read_only(port,report)

    def test_read_failure_keeps_partial_frames_attempts_final_reads_and_never_cleans_up(self):
        count=[]
        def fail(p,ins,sid,addr,params):
            if (ins,sid,addr,params)==(2,6,40,b'\x1a'):
                count.append(1)
                if len(count)==7:
                    p.pending=b''
        port,control,_=self.setup_case(fail)
        report=control.run()
        self.assertEqual(report['status'],'not_verified')
        self.assertEqual(len(report['samples']),6)
        self.assertIn('fault',report)
        self.assertIn('final',report)
        self.assert_read_only(port,report)

    def test_source_wrong_ack_duplicate_128_or_unconfirmed_release_is_rejected(self):
        port,control,path=self.setup_case()
        records=[json.loads(line) for line in path.read_text().splitlines()]
        for case in ('ack','checksum','duplicate','release','config'):
            with self.subTest(case=case):
                altered=copy.deepcopy(records)
                snapshot=altered[-1]['snapshot']
                command=next(e for e in snapshot['events'] if e['instruction']==3 and e['parameters_hex']=='2880')
                if case=='ack':
                    command['acknowledged']=False
                elif case=='checksum':
                    command['response_hex']='ffff060200f6'
                elif case=='duplicate':
                    snapshot['events'].append(copy.deepcopy(command))
                elif case=='release':
                    snapshot['release']['confirmed']=False
                else:
                    raw=bytearray.fromhex(snapshot['after']['config']['raw_hex'])
                    raw[21]^=1
                    snapshot['after']['config']['raw_hex']=raw.hex()
                data=''.join(json.dumps(r)+'\n' for r in altered).encode()
                with self.assertRaises(center.CenterError):
                    verify.load_source(path,data)
        self.assertEqual(port.events,[])

    def test_source_requires_complete_final_record_no_bad_tail_fallback(self):
        _,_,path=self.setup_case()
        original=path.read_bytes()
        for data in (original+b'{"stage":', b'\n'.join(original.splitlines()[:-1])+b'\n',
                     original+b'{"stage":"x","stage":"y","snapshot":{}}\n'):
            with self.subTest(tail=data[-50:]):
                with self.assertRaises(center.CenterError):
                    verify.load_source(path,data)

    def test_readonly_bus_cannot_gain_or_use_write_capability(self):
        port,control,_=self.setup_case()
        for action in (lambda:control.bus.enable_apply(),lambda:control.bus.write(6,40,128),
                       lambda:control.bus._exchange(6,3,[40,0],0)):
            with self.assertRaises(center.CenterError):
                action()
        control.bus.apply_enabled=True
        with self.assertRaises(center.CenterError):
            control.bus._exchange(6,3,[40,128],0)
        self.assertEqual(port.events,[])

    def test_help_never_imports_serial_and_apply_option_is_absent(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            self.assertNotEqual(name,'serial')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded),redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                verify.main(['--help'])
            self.assertEqual(result.exception.code,0)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            verify.main(['--journal','missing','--apply'])

    def test_invalid_source_cli_saves_failure_with_sha_before_serial_import(self):
        _,_,path=self.setup_case()
        path.write_bytes(path.read_bytes()+b'{')
        with patch.dict(sys.modules,{'serial':types.SimpleNamespace(Serial=lambda *a,**k:self.fail('serial opened'))}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(verify.main(['--journal',str(path),'--log-dir',str(path.parent)]),1)
        reports=list(path.parent.glob('gripper_center_readonly_*.json'))
        self.assertEqual(len(reports),1)
        result=json.loads(reports[0].read_text())
        self.assertEqual(result['source_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result['status'],'not_verified')
        self.assertEqual(result['writes_sent'],0)


if __name__=='__main__':
    unittest.main()
