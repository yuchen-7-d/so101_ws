import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plan_grasp', ROOT / 'scripts/so101_plan_grasp.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class TargetContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calibration = Path(self.tmp.name) / 'calibration.npz'
        self.calibration.write_bytes(b'calibration-fixture')
        self.target = {'schema_version': 1, 'kind': 'so101_box_grasp_target',
                       'valid': True, 'coordinate_system': 'table', 'source_mode': 'live',
                       'session_id': 'test-session', 'frame_id': 3,
                       'captured_at_unix_s': 99.5, 'published_at_unix_s': 99.9,
                       'center_m': [.1, .2, -.074], 'size_m': [.04, .04, .074],
                       'max_edge_error_mm': 1.5, 'square_yaw_deg': 15.,
                       'calibration': {'sha256': hashlib.sha256(self.calibration.read_bytes()).hexdigest()}}

    def check(self, target=None, **kwargs):
        return plan.validate_target(self.target if target is None else target, 100.,
                                    kwargs.get('max_age', 2.), kwargs.get('allow_offline', False),
                                    self.calibration)

    def test_fresh_target_accepted(self):
        center, size, yaw = self.check()
        np.testing.assert_allclose(center, [.1, .2, -.074])
        self.assertEqual(yaw, 15.)

    def test_old_live_target_uses_capture_not_publish_time(self):
        self.target['captured_at_unix_s'] = 97
        with self.assertRaisesRegex(ValueError, '过期'):
            self.check()

    def test_invalid_target_rejects_even_if_old_center_remains(self):
        self.target['valid'] = False
        with self.assertRaisesRegex(ValueError, '没有有效目标'):
            self.check()

    def test_old_offline_requires_explicit_preview_option(self):
        self.target.update(source_mode='offline', captured_at_unix_s=1)
        with self.assertRaisesRegex(ValueError, '离线预览'):
            self.check()
        self.check(allow_offline=True)

    def test_calibration_change_rejected(self):
        self.calibration.write_bytes(b'new-calibration')
        with self.assertRaisesRegex(ValueError, '不一致'):
            self.check()

    def test_geometry_and_nonfinite_rejected(self):
        for key, value in [('center_m', [float('nan'), .2, -.074]),
                           ('center_m', [.1, .2, 0]), ('size_m', [.04, .04, .04]),
                           ('square_yaw_deg', float('nan')),
                           ('max_edge_error_mm', 3.1), ('captured_at_unix_s', float('nan')),
                           ('captured_at_unix_s', 101), ('published_at_unix_s', 99)]:
            with self.subTest(key=key, value=value):
                target = copy.deepcopy(self.target)
                target[key] = value
                with self.assertRaises(ValueError):
                    self.check(target)

    def test_failed_cli_replaces_prior_plan(self):
        path = Path(self.tmp.name) / 'target.json'
        output = Path(self.tmp.name) / 'plan.json'
        path.write_text('{bad json')
        output.write_text('{"status":"not_for_motion","samples":[1]}')
        self.assertEqual(plan.main(['--target', str(path), '--output', str(output)]), 2)
        record = json.loads(output.read_text())
        self.assertEqual(record['status'], 'invalid')
        self.assertFalse(record['execution_available'])
        self.assertNotIn('samples', record)

    def test_model_runtime_failure_also_invalidates_previous_path(self):
        path = Path(self.tmp.name) / 'target.json'
        output = Path(self.tmp.name) / 'plan.json'
        path.write_text(json.dumps(self.target))
        output.write_text('{"status":"not_for_motion","samples":[1]}')
        with patch.object(plan, 'build_report', side_effect=RuntimeError('model load failed')):
            self.assertEqual(plan.main(['--target', str(path), '--output', str(output)]), 2)
        self.assertEqual(json.loads(output.read_text())['status'], 'invalid')
        self.assertNotIn('samples', json.loads(output.read_text()))

    def test_output_cannot_replace_initial_pose(self):
        initial = Path(self.tmp.name) / 'data/so101_initial_pose.json'
        initial.parent.mkdir()
        initial.write_text('{"original_pose":true}')
        with self.assertRaises(SystemExit):
            plan.main(['--workspace', self.tmp.name, '--output', str(initial)])
        self.assertEqual(initial.read_text(), '{"original_pose":true}')

    def test_transform_depends_on_detected_position(self):
        reference = dict(x_axis_table=[1, 0, 0], y_axis_table=[0, -1, 0],
                         z_axis_table=[0, 0, -1], origin_table_m=[0, 0, 0])
        a, rotation = plan.candidate_transform(np.array([.1, .2, -.074]), 0, reference)
        b, _ = plan.candidate_transform(np.array([.12, .2, -.074]), 0, reference)
        np.testing.assert_allclose(b - a, [.02, 0, 0])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_left_handed_reference_rejected(self):
        reference = dict(x_axis_table=[1, 0, 0], y_axis_table=[0, 1, 0],
                         z_axis_table=[0, 0, -1], origin_table_m=[0, 0, 0])
        with self.assertRaisesRegex(ValueError, '右手'):
            plan.candidate_transform(np.zeros(3), 0, reference)

    def test_tilted_table_rejected_even_with_one_horizontal_edge(self):
        a = np.deg2rad(30)
        reference = dict(x_axis_table=[1, 0, 0], y_axis_table=[0, -np.cos(a), np.sin(a)],
                         z_axis_table=[0, -np.sin(a), -np.cos(a)], origin_table_m=[0, 0, 0])
        with self.assertRaisesRegex(ValueError, '桌面法向'):
            plan.candidate_transform(np.array([.1, .2, -.074]), 0, reference)

    def test_snapshot_invalidated_or_expired_during_planning_rejected(self):
        for latest, now in ((dict(self.target, valid=False, reason='lost'), 100),
                            (dict(self.target, session_id='restarted'), 100),
                            (dict(self.target), 102),
                            (dict(self.target, frame_id=2), 100)):
            with self.subTest(latest=latest, now=now):
                with self.assertRaises(ValueError):
                    plan.check_live_snapshot(self.target, latest, now, 2, self.calibration)
        newer = dict(self.target, frame_id=4)
        self.assertEqual(plan.check_live_snapshot(self.target, newer, 100, 2, self.calibration),
                         'snapshot_not_latest_frame')

    def test_existing_shoulder_contradiction_not_hidden_by_wrap(self):
        mapping = {'joint_references': {}, 'joint_directions': {}}
        pose = {'position_counts': {}}
        for i, name in enumerate(plan.ARM_NAMES, 1):
            key = str(i)
            mapping['joint_references'][key] = {'position_count': 824, 'model_angle_deg': -13.97}
            mapping['joint_directions'][key] = {'joint_name': name, 'direction': 1}
            pose['position_counts'][key] = 3698
        rows = plan.mapping_diagnostics(mapping, pose, np.deg2rad([[-100, 100]] * 5))
        self.assertAlmostEqual(rows[1]['model_angle_nearest_branch_deg'], -121.37234375)
        self.assertFalse(rows[1]['nearest_branch_in_model_range'])


if __name__ == '__main__':
    unittest.main()
