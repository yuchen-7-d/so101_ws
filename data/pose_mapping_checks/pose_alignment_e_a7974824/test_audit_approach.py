import json
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np

from audit_approach import JointSegmentAuditor, linear_joint_samples, main


ROOT = Path('/home/yu/so101_ws')
HYPOTHESES = ROOT / 'data/pose_mapping_checks/wrist_model_e_ebbd4505/wrist_model_review.json'


class InterpolationTests(unittest.TestCase):
    def test_both_endpoints_and_one_degree_max_step(self):
        start = np.deg2rad([-1, -10, 20, -90.382, .5])
        goal = np.deg2rad([-2, 40, -40, 90.866, 10])
        samples = linear_joint_samples(start, goal)
        np.testing.assert_array_equal(samples[0], start)
        np.testing.assert_array_equal(samples[-1], goal)
        self.assertEqual(len(samples), 183)
        self.assertLessEqual(np.max(np.abs(np.rad2deg(np.diff(samples, axis=0)))), 1 + 1e-12)

    def test_no_shortest_wrap_or_coordinate_clipping(self):
        start, goal = np.zeros(5), np.zeros(5)
        start[0], goal[0] = math.radians(170), math.radians(-170)
        samples = linear_joint_samples(start, goal)
        self.assertEqual(len(samples), 341)
        self.assertAlmostEqual(samples[len(samples)//2, 0], 0)
        self.assertAlmostEqual(float(np.rad2deg(np.diff(samples[:, 0]).sum())), -340)

    def test_stationary_segment_still_has_one_sample(self):
        q = np.arange(5) / 10
        np.testing.assert_array_equal(linear_joint_samples(q, q), q[None, :])

    def test_invalid_sampling_resolution_rejected(self):
        for value in [0, -1, 1.01, float('nan'), float('inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                linear_joint_samples(np.zeros(5), np.zeros(5), value)


class ModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hypotheses = json.loads(HYPOTHESES.read_text())
        replay = hypotheses['offline_geometry_replay']
        cls.pre = np.asarray(replay['samples'][0]['model_q_rad'])
        target = json.loads((ROOT / replay['source_archived_plan']).read_text())['input_target']
        base = json.loads((ROOT / 'data/base_reference_initial.json').read_text())
        axes = np.column_stack([base[k] for k in ['x_axis_table', 'y_axis_table', 'z_axis_table']])
        cls.top = axes.T @ (np.asarray(target['center_m']) - base['origin_table_m']) + [.064635292, 0, 0]
        yaw = math.radians(target['square_yaw_deg'])
        closing = axes.T @ [math.cos(yaw), math.sin(yaw), 0]
        cls.yaw = (math.atan2(closing[1], closing[0]) + math.pi/2) % math.pi - math.pi/2
        cls.starts = {int(c['assumed_wrist_axis_gamma_deg']): np.deg2rad(c['q_deg'])
                      for c in hypotheses['models']['active_shoulder_elbow_mapping']['scenarios']}

    def setUp(self):
        self.auditor = JointSegmentAuditor(ROOT / 'simulation/so101/scene_grasp.xml', self.top, self.yaw)

    def test_malformed_nonfinite_start_blocked_before_sampling(self):
        invalid = [[0]*4, [[0]*5], [0, 0, float('nan'), 0, 0], [0, float('inf'), 0, 0, 0]]
        with patch.object(self.auditor, 'sample') as sample:
            for start in invalid:
                with self.subTest(start=start):
                    result = self.auditor.audit(start, self.pre)
                    self.assertEqual(result['status'], 'blocked_start')
                    self.assertEqual(result['sample_count'], 0)
                    self.assertEqual(result['samples'], [])
            sample.assert_not_called()
            self.assertIsNone(self.auditor.data)

    def test_nominal_and_100_degree_starts_blocked_without_clipping(self):
        with patch.object(self.auditor, 'sample') as sample:
            for gamma in [90, 100]:
                result = self.auditor.audit(self.starts[gamma], self.pre)
                self.assertEqual(result['status'], 'blocked_start')
                self.assertEqual(result['sample_count'], 0)
                error = result['endpoint_error']['joints'][0]
                self.assertEqual(error['servo_id'], 4)
                self.assertAlmostEqual(error['angle_deg'], float(np.rad2deg(self.starts[gamma][3])))
            sample.assert_not_called()

    def test_invalid_goal_blocked_before_sampling(self):
        goal = self.pre.copy()
        goal[0] = self.auditor.limits[0, 1] + .01
        with patch.object(self.auditor, 'sample') as sample:
            result = self.auditor.audit(self.starts[80], goal)
            self.assertEqual(result['status'], 'blocked_goal')
            self.assertEqual(result['sample_count'], 0)
            sample.assert_not_called()

    def test_box_penetration_caught_on_archived_80_degree_direct_segment(self):
        result = self.auditor.audit(self.starts[80], self.pre)
        self.assertEqual(result['status'], 'unsafe_segment')
        self.assertEqual(result['sample_count'], 183)
        self.assertGreater(result['colliding_sample_count'], 0)
        self.assertGreater(result['maximum_sampled_penetration_mm'], .01)
        pair = {'fixed_jaw_tip_collision', 'a_box_geom'}
        self.assertTrue(any(set(contact['geoms']) == pair for sample in result['samples']
                            for contact in sample['penetrating_contacts']))
        self.assertLessEqual(result['realized_max_sample_joint_step_deg'], 1 + 1e-12)
        np.testing.assert_array_equal(result['samples'][0]['q_rad'], self.starts[80])
        np.testing.assert_array_equal(result['samples'][-1]['q_rad'], self.pre)
        self.assertFalse(result['execution_available'])

    def test_stationary_pregrasp_does_not_flag_resting_box(self):
        result = self.auditor.audit(self.pre, self.pre)
        self.assertEqual(result['status'], 'no_sampled_penetration')
        self.assertEqual(result['sample_count'], 1)
        self.assertEqual(result['colliding_sample_count'], 0)
        self.assertEqual(result['model_gripper_angle_rad'], .6)
        self.assertFalse(result['execution_available'])


class OutputProtectionTests(unittest.TestCase):
    def test_existing_hardlink_and_active_target_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            source = root / 'source.json'
            source.write_text('preserve input')
            hardlink = root / 'hardlink.json'
            os.link(source, hardlink)
            target = root / 'data/box_grasp_target.json'
            target.write_text('preserve live target')
            plan = root / 'data/box_grasp_plan.json'
            plan.write_text('preserve live plan')
            report = {'source_hashes': {}, 'code_hashes': {}, 'cases': []}
            with patch('audit_approach.make_review', return_value=report):
                for output in [hardlink, target, plan]:
                    with self.subTest(output=output), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                        main(['--workspace', str(root), '--hypotheses', str(source), '--output', str(output)])
                    self.assertEqual(exc.exception.code, 2)
            self.assertEqual(source.read_text(), 'preserve input')
            self.assertEqual(target.read_text(), 'preserve live target')
            self.assertEqual(plan.read_text(), 'preserve live plan')


if __name__ == '__main__':
    unittest.main()
