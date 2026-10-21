import math
import unittest

import numpy as np

from search_route import build_case, search_routes


class BoxObstacleAuditor:
    """Small geometric oracle: a box blocks the diagonal in a two-joint plane."""
    def __init__(self, block_every_motion=False):
        self.block_every_motion = block_every_motion
        self.calls = []

    def audit(self, start, goal, max_step_deg=1):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        self.calls.append((start.copy(), goal.copy(), max_step_deg))
        for label, q in [('start', start), ('goal', goal)]:
            if q.shape != (5,) or not np.isfinite(q).all() or np.any(np.abs(q) > 2):
                return {'status': f'blocked_{label}', 'sample_count': 0, 'samples': []}
        count = max(1, math.ceil(float(np.max(np.abs(goal-start))) / math.radians(max_step_deg)) + 1)
        qs = np.linspace(start, goal, count)
        degrees = np.rad2deg(qs)
        inside = ((degrees[:, 0] > .45) & (degrees[:, 0] < .55)
                  & (degrees[:, 1] > .45) & (degrees[:, 1] < .55))
        blocked = bool(np.any(inside) or (self.block_every_motion and not np.array_equal(start, goal)))
        return {'status': 'unsafe_segment' if blocked else 'no_sampled_penetration',
                'sample_count': count, 'samples': [{'q_rad': q.tolist()} for q in qs]}


class CatalogSearchTests(unittest.TestCase):
    def test_fine_collision_rejects_coarse_clear_diagonal_and_finds_detour(self):
        start = np.zeros(5)
        goal = np.deg2rad([1, 1, 0, 0, 0])
        result = search_routes(BoxObstacleAuditor(), start, goal)
        self.assertEqual(result['attempts'][0]['status'], 'rejected_fine')
        self.assertEqual(result['attempts'][0]['coarse_segments'][0]['status'], 'no_sampled_penetration')
        self.assertEqual(result['status'], 'found_sampled_model_route')
        self.assertNotEqual(result['route']['name'], 'direct')
        self.assertEqual(len(result['route']['waypoints_rad']), 3)
        self.assertTrue(all(s['status'] == 'no_sampled_penetration' for s in result['route']['fine_segments']))
        np.testing.assert_array_equal(result['route']['waypoints_rad'][0], start)
        np.testing.assert_array_equal(result['route']['waypoints_rad'][-1], goal)

    def test_invalid_start_not_repaired_or_searched(self):
        start = np.array([0, 0, 0, -3., 0])
        auditor = BoxObstacleAuditor()
        result = search_routes(auditor, start, np.zeros(5))
        self.assertEqual(result['status'], 'blocked_start')
        self.assertEqual(len(auditor.calls), 1)
        self.assertEqual(result['attempts'], [])
        self.assertIsNone(result['route'])
        self.assertEqual(start[3], -3)

    def test_invalid_goal_blocks_before_catalog(self):
        auditor = BoxObstacleAuditor()
        result = search_routes(auditor, np.zeros(5), [0, 0, float('nan'), 0, 0])
        self.assertEqual(result['status'], 'blocked_goal')
        self.assertEqual(len(auditor.calls), 2)
        self.assertEqual(result['attempts'], [])

    def test_exhaustion_is_bounded_and_does_not_claim_global_impossibility(self):
        result = search_routes(BoxObstacleAuditor(block_every_motion=True), np.zeros(5), np.deg2rad([1, 2, 3, 4, 5]))
        self.assertEqual(result['status'], 'blocked_no_route_in_finite_catalog')
        self.assertEqual(len(result['attempts']), 31)
        self.assertIsNone(result['route'])

    def test_catalog_is_deterministic(self):
        args = np.zeros(5), np.deg2rad([1, 1, 0, 0, 0])
        self.assertEqual(search_routes(BoxObstacleAuditor(), *args), search_routes(BoxObstacleAuditor(), *args))


class ActualModelRegressionTests(unittest.TestCase):
    def test_reported_f_route_has_fine_sampled_clearance_without_changing_limits(self):
        auditor, start, goal, context = build_case()
        original_limits = auditor.limits.copy()
        result = search_routes(auditor, start, goal)
        self.assertEqual(result['status'], 'found_sampled_model_route')
        self.assertEqual(result['route']['name'], 'goal_first_ids_1_2_3_5')
        self.assertEqual(result['attempts'][0]['status'], 'rejected_coarse')
        self.assertFalse(context['nominal_mapping_verified'])
        self.assertFalse(context['current_target_freshness_claimed'])
        self.assertFalse(result['execution_available'])
        np.testing.assert_array_equal(auditor.limits, original_limits)
        np.testing.assert_array_equal(result['route']['waypoints_rad'][0], start)
        np.testing.assert_array_equal(result['route']['waypoints_rad'][-1], goal)
        for segment in result['route']['fine_segments']:
            self.assertEqual(segment['colliding_sample_count'], 0)
            self.assertLessEqual(segment['realized_max_sample_joint_step_deg'], .25 + 1e-12)
            qs = np.array([s['q_rad'] for s in segment['samples']])
            self.assertTrue(np.all(qs >= auditor.limits[:, 0]))
            self.assertTrue(np.all(qs <= auditor.limits[:, 1]))


if __name__ == '__main__':
    unittest.main()
