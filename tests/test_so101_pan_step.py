"""Local movement sequencing and abort tests; no hardware access."""

from pathlib import Path
import sys
import tempfile
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from so101_hold_current import ARM_IDS, GOAL, OUTPUT, TORQUE, Bus, ControlError
from so101_pan_step import PanStepController, read_pan_diagnostics, save_report, main
from test_so101_hold_current import Clock, FakeSerial


class PanTests(unittest.TestCase):
    def setup_control(self, delta=16, fault=None, move=True):
        clock = Clock()
        def plant(port, ins, sid, addr, params):
            if (move and ins == 3 and addr == GOAL
                    and port.regs[sid][TORQUE] == 1 and port.get(sid, OUTPUT)):
                port.put(sid, 56, int.from_bytes(params, "little"))
            if fault:
                fault(port, ins, sid, addr, params)
        port = FakeSerial(clock, plant)
        port.allow_live_goal_writes = True
        controller = PanStepController(Bus(port), delta=delta, emit=lambda text: None,
                                       clock=clock.time, sleep=clock.sleep)
        return port, controller

    def assert_released(self, port):
        for sid in ARM_IDS:
            self.assertEqual(port.regs[sid][TORQUE], 0)
            self.assertEqual(port.get(sid, OUTPUT), 0)

    def test_preview_has_no_writes(self):
        port, c = self.setup_control()
        c.run(False)
        self.assertEqual(port.writes, [])
        self.assertEqual(c.path, [33,37,41,45,41,37,33,29])

    def test_complete_positive_and_negative_roundtrip(self):
        for delta in (16, -16, 8, -8):
            with self.subTest(delta=delta):
                port, c = self.setup_control(delta)
                initial = {sid: port.get(sid,56) for sid in ARM_IDS}
                c.run(True)
                self.assertTrue(c.outbound_complete and c.return_complete)
                self.assertTrue(c.final_anchor_hold_complete)
                self.assertEqual(port.get(1,56), initial[1])
                self.assertEqual([v for sid,addr,v in port.writes if sid==1 and addr==GOAL],
                                 [initial[1]]+c.path)
                self.assertTrue(all(abs(v-initial[1]) <= 16 for v in c.path))
                for sid in ARM_IDS[1:]:
                    self.assertEqual([v for s,addr,v in port.writes if s==sid and addr==GOAL], [initial[sid]])
                self.assertFalse(any(sid==6 for sid,addr,v in port.writes))
                self.assert_released(port)

    def test_full_plan_rejected_before_writes_at_encoder_boundary(self):
        for position,delta in ((3,-16),(4090,16)):
            port, c = self.setup_control(delta)
            port.put(1,56,position)
            with self.assertRaisesRegex(ControlError,"完整往返路径"):
                c.run(True)
            self.assertEqual(port.writes,[])

    def test_device_limit_rejected_before_writes(self):
        port,c=self.setup_control()
        port.put(1,11,40)
        with self.assertRaisesRegex(ControlError,"完整往返路径"):
            c.run(True)
        self.assertEqual(port.writes,[])

    def test_stuck_pan_does_not_pass_or_command_return(self):
        port,c=self.setup_control(move=False)
        with self.assertRaisesRegex(ControlError,"未在3秒内连续到位"):
            c.run(True)
        self.assertFalse(c.outbound_complete or c.return_complete)
        self.assertEqual(c.fault_context["phase"], "outbound")
        self.assertIn("未在3秒内连续到位", c.fault_context["reason"])
        self.assertEqual([v for sid,addr,v in port.writes if sid==1 and addr==GOAL],[29,33])
        self.assert_released(port)

    def test_only_two_good_samples_are_not_accepted(self):
        reads=0
        def unstable(port,ins,sid,addr,params):
            nonlocal reads
            if ins==2 and sid==1 and addr==56 and port.get(1,GOAL)==33:
                reads+=1
                # Fault modifies the next reading. Never three successive good samples.
                port.put(1,56,29 if reads%3==0 else 33)
        port,c=self.setup_control(fault=unstable,move=False)
        with self.assertRaisesRegex(ControlError,"未在3秒内连续到位"):
            c.run(True)
        self.assertFalse(c.outbound_complete)
        self.assert_released(port)

    def test_lost_ack_after_live_target_applied_releases_without_return(self):
        fired=False
        def lose(port,ins,sid,addr,params):
            nonlocal fired
            if not fired and ins==3 and sid==1 and addr==GOAL and int.from_bytes(params,"little")==33:
                fired=True
                port.pending=b""
        port,c=self.setup_control(fault=lose)
        with self.assertRaisesRegex(ControlError,"超时"):
            c.run(True)
        self.assertEqual(port.get(1,56),33)
        self.assertFalse(c.return_complete)
        self.assertEqual(c.fault_context["targets"][1],33)
        self.assertEqual(c.fault_context["phase"],"outbound")
        self.assertIn("超时",c.fault_context["reason"])
        self.assertEqual([v for sid,addr,v in port.writes if sid==1 and addr==GOAL],[29,33])
        self.assert_released(port)

    def test_other_joint_drift_aborts(self):
        def drift(port,ins,sid,addr,params):
            if ins==3 and sid==1 and addr==GOAL and int.from_bytes(params,"little")==33:
                port.put(2,56,855)
        port,c=self.setup_control(fault=drift)
        with self.assertRaisesRegex(ControlError,"ID 2: 位置偏离"):
            c.run(True)
        self.assertFalse(c.outbound_complete)
        self.assert_released(port)

    def test_fixed_corridor_not_moved_with_target(self):
        def overshoot(port,ins,sid,addr,params):
            if ins==3 and sid==1 and addr==GOAL and int.from_bytes(params,"little")==45:
                port.put(1,56,48) # Only 3 from current target, but exceeds fixed end+2.
        port,c=self.setup_control(fault=overshoot)
        with self.assertRaisesRegex(ControlError,"小步反馈越界"):
            c.run(True)
        self.assert_released(port)

    def test_real_overshoot_trace_aborts_first_step_and_saves_fault_reply(self):
        sequence=iter((4026,4026,4028,4032,4038,4040,4047))
        feedback_reads_after_first_target=[]
        def replay(p,ins,sid,addr,params):
            if ins==2 and sid==1 and addr==56 and p.get(1,GOAL)==4030:
                position=next(sequence)
                p.put(1,56,position)
                p.put(1,58,50)
                p.put(1,60,68)
                payload=bytes(p.regs[1][56:66])
                fields=bytes([1,len(payload)+2,0])+payload
                p.pending=b"\xff\xff"+fields+bytes([(~sum(fields))&255])
                feedback_reads_after_first_target.append(position)
        port,c=self.setup_control(fault=replay,move=False)
        port.put(1,56,4026)
        with self.assertRaisesRegex(ControlError,"实际=4038"):
            c.run(True)
        self.assertEqual(feedback_reads_after_first_target,[4026,4026,4028,4032,4038])
        self.assertEqual(c.fault_context["last_feedback_by_id"][1]["position"],4038)
        self.assertEqual(c.fault_context["last_feedback_by_id"][1]["expected_target"],4030)
        self.assertEqual(c.fault_context["last_feedback_by_id"][1]["speed_raw_signed"],50)
        self.assertEqual([v for sid,addr,v in port.writes if sid==1 and addr==GOAL],[4026,4030])
        self.assertFalse(c.outbound_complete or c.return_complete)
        self.assert_released(port)

    def test_failed_other_axis_frame_survives_base_validation(self):
        def drift(port,ins,sid,addr,params):
            if ins==3 and sid==1 and addr==GOAL and int.from_bytes(params,"little")==33:
                port.put(2,56,860)
        port,c=self.setup_control(fault=drift)
        with self.assertRaisesRegex(ControlError,"ID 2: 位置偏离"):
            c.run(True)
        self.assertEqual(c.fault_context["last_feedback_by_id"][2]["position"],860)
        self.assert_released(port)

    def test_diagnostics_reads_pid_and_signed_feedback_without_writes(self):
        port,c=self.setup_control()
        port.regs[1][21:28]=bytes([16,8,0,20,0,1,1])
        port.put(1,58,0x8000|17)
        port.put(1,60,0x0400|68)
        result=read_pan_diagnostics(Bus(port),sleep=c.sleep,clock=c.clock)
        self.assertEqual(port.writes,[])
        self.assertTrue(all(event[0]==2 for event in port.events))
        self.assertEqual((result['P'],result['D'],result['I']),(16,8,0))
        self.assertEqual(result['minimum_starting_output'],20)
        self.assertEqual(len(result['feedback']),5)
        self.assertEqual(result['feedback'][0]['speed_raw_signed'],-17)
        self.assertEqual(result['feedback'][0]['output_raw_signed'],-68)

    def test_cli_diagnostics_cannot_be_combined_with_enable(self):
        with self.assertRaises(SystemExit) as error:
            main(['--diagnose','--enable'])
        self.assertEqual(error.exception.code,2)

    def test_mid_motion_interrupt_releases(self):
        def interrupt(port,ins,sid,addr,params):
            if ins==3 and sid==1 and addr==GOAL and int.from_bytes(params,"little")==37:
                raise KeyboardInterrupt
        port,c=self.setup_control(fault=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            c.run(True)
        self.assert_released(port)
        self.assertFalse(c.return_complete)

    def test_report_has_real_path_feedback_and_release_outcome(self):
        port,c=self.setup_control()
        c.run(True)
        with tempfile.TemporaryDirectory() as folder:
            path=save_report(c,"completed_and_released",None,Path(folder))
            record=json.loads(path.read_text())
        self.assertEqual(record["anchor"],29)
        self.assertTrue(record["outbound_complete"] and record["return_complete"])
        self.assertEqual(record["release_unconfirmed_ids"],[])
        self.assertGreater(len(record["samples"]),0)
        self.assertFalse(record["model_mapping_verified"])

    def test_drift_after_return_does_not_claim_final_anchor_hold(self):
        port,c=self.setup_control()
        previous_hook=port.hook
        def drift_after_return(p,ins,sid,addr,params):
            previous_hook(p,ins,sid,addr,params)
            if c.phase == "before_release" and ins == 2 and sid == 1:
                p.put(1,56,c.anchor+8)
        port.hook=drift_after_return
        with self.assertRaisesRegex(ControlError,"返回后再次偏离起点"):
            c.run(True)
        self.assertTrue(c.return_complete)
        self.assertFalse(c.final_anchor_hold_complete)
        self.assert_released(port)


if __name__ == "__main__":
    unittest.main()
