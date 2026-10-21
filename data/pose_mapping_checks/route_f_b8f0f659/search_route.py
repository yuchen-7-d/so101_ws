"""Bounded offline joint-order search. No camera, serial or robot execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/yu/so101_ws')
AUDITOR_RELATIVE = 'data/pose_mapping_checks/pose_alignment_e_a7974824/audit_approach.py'
HYPOTHESES_RELATIVE = 'data/pose_mapping_checks/wrist_model_e_ebbd4505/wrist_model_review.json'
MAPPING_RELATIVE = 'data/pose_mapping_checks/wrist_model_e_ebbd4505/wrist_candidate_mapping.json'
ARCHIVE_RELATIVE = 'data/pose_mapping_checks/grasp_plan_before_id3_direction_4mfmwcx_.json'
POSE_RELATIVE = 'data/pose_mapping_checks/pose_f_supported_start_glgoy1eh.json'


def candidate_waypoints(start, goal):
    """At most 31 routes: direct, then one intermediate assembled from endpoints."""
    yield 'direct', [start.copy(), goal.copy()]
    preferred = [(3,), (0, 1, 2, 4)]  # wrist first, then arm/pan/roll first
    masks = preferred + [ids for length in range(1, 5)
                         for ids in itertools.combinations(range(5), length) if ids not in preferred]
    seen = set()
    for mask in masks:
        mid = start.copy()
        mid[list(mask)] = goal[list(mask)]
        key = tuple(mid)
        if key in seen or np.array_equal(mid, start) or np.array_equal(mid, goal):
            continue
        seen.add(key)
        yield 'goal_first_ids_' + '_'.join(str(i + 1) for i in mask), [start.copy(), mid, goal.copy()]


def audit_summary(audit):
    return {k: v for k, v in audit.items() if k != 'samples'}


def search_routes(auditor, start, goal, coarse_step_deg=1., fine_step_deg=.25):
    """Return only a route whose every segment passes the finer sampled audit."""
    if not (0 < fine_step_deg <= coarse_step_deg <= 1):
        raise ValueError('Require 0 < fine step <= coarse step <= 1 degree')
    result = {'status': 'not_evaluated', 'execution_available': False, 'hardware_writes_sent': 0,
              'search_kind': 'deterministic_endpoint_joint_order_catalog',
              'candidate_catalog_limit': 31, 'attempts': [], 'route': None,
              'coarse_step_deg': coarse_step_deg, 'final_step_deg': fine_step_deg,
              'scope': 'finite catalog only; sampled model collision checks, not continuous safety or an exhaustive planner'}
    start_check = auditor.audit(start, start, max_step_deg=fine_step_deg)
    result['start_check'] = start_check
    if start_check['status'] != 'no_sampled_penetration':
        result['status'] = 'blocked_start'
        return result
    goal_check = auditor.audit(goal, goal, max_step_deg=fine_step_deg)
    result['goal_check'] = goal_check
    if goal_check['status'] != 'no_sampled_penetration':
        result['status'] = 'blocked_goal'
        return result
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    for name, waypoints in candidate_waypoints(start, goal):
        attempt = {'name': name, 'waypoints_rad': [q.tolist() for q in waypoints],
                   'coarse_segments': [], 'fine_segments': []}
        result['attempts'].append(attempt)
        coarse_ok = True
        for a, b in zip(waypoints, waypoints[1:]):
            audit = auditor.audit(a, b, max_step_deg=coarse_step_deg)
            attempt['coarse_segments'].append(audit_summary(audit))
            if audit['status'] != 'no_sampled_penetration':
                coarse_ok = False
                break
        if not coarse_ok:
            attempt['status'] = 'rejected_coarse'
            continue
        fine = []
        for a, b in zip(waypoints, waypoints[1:]):
            audit = auditor.audit(a, b, max_step_deg=fine_step_deg)
            fine.append(audit)
            attempt['fine_segments'].append(audit_summary(audit))
            if audit['status'] != 'no_sampled_penetration':
                break
        if len(fine) != len(waypoints) - 1 or any(x['status'] != 'no_sampled_penetration' for x in fine):
            attempt['status'] = 'rejected_fine'
            continue
        attempt['status'] = 'accepted_sampled_model_only'
        result.update(status='found_sampled_model_route', route={
            'name': name, 'waypoints_rad': [q.tolist() for q in waypoints],
            'fine_segments': fine, 'unique_sample_count': sum(x['sample_count'] for x in fine) - len(fine) + 1,
            'interpolation': 'linear joint coordinates, no clipping, angle wrap or shortcut',
            'time_parameterization_available': False})
        return result
    result['status'] = 'blocked_no_route_in_finite_catalog'
    return result


def build_case(workspace=ROOT):
    workspace = Path(workspace)
    sys.path.insert(0, str(workspace / Path(AUDITOR_RELATIVE).parent))
    sys.path.insert(0, str(workspace / 'scripts'))
    from audit_approach import JointSegmentAuditor
    from so101_plan_grasp import candidate_transform, validate_target
    mapping_path, hypotheses_path = workspace / MAPPING_RELATIVE, workspace / HYPOTHESES_RELATIVE
    mapping = json.loads(mapping_path.read_text())
    hypotheses = json.loads(hypotheses_path.read_text())
    pose_path = workspace / POSE_RELATIVE
    pose = json.loads(pose_path.read_text())
    if pose['kind'] != 'so101_supported_start_pose_observation' or pose['pose_label'] != 'F':
        raise ValueError('Expected the recorded supported pose F')
    counts = [pose['position_counts'][str(i)] for i in range(1, 7)]
    if any(type(c) is not int or not 0 <= c <= 4095 for c in counts):
        raise ValueError('Pose counts must be integers in the single-turn register range')
    for record in [hypotheses, pose]:
        for relative, expected in record['source_hashes'].items():
            if hashlib.sha256((workspace / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Archived input changed: {relative}')
    archived = json.loads((workspace / ARCHIVE_RELATIVE).read_text())
    target = archived['input_target']
    calibration_path = workspace / 'data/calibration/table_plane_pose.npz'
    center, size, yaw = validate_target(target, now=target['published_at_unix_s'], max_age=2,
                                        allow_offline=False, calibration_path=calibration_path)
    base_path = workspace / 'data/base_reference_initial.json'
    center, rotation = candidate_transform(center, yaw, json.loads(base_path.read_text()))
    model_path = workspace / 'simulation/so101/scene_grasp.xml'
    auditor = JointSegmentAuditor(model_path, center, math.atan2(rotation[1, 2], rotation[0, 2]), size)
    qdeg = []
    for i, count in enumerate(counts[:5], 1):
        ref = mapping['joint_references'][str(i)]
        direction = mapping['joint_directions'][str(i)]['direction']
        qdeg.append(ref['model_angle_deg'] + direction * (count - ref['position_count']) * 360 / 4096)
    start = np.deg2rad(qdeg)
    if not np.allclose(qdeg, pose['candidate_model_q_deg'], atol=1e-8, rtol=0):
        raise ValueError('Recorded pose F and the candidate mapping no longer agree')
    first = hypotheses['offline_geometry_replay']['samples'][0]
    if first['phase'] != 'pregrasp':
        raise ValueError('Archived first goal is not a pregrasp')
    goal = np.asarray(first['model_q_rad'])
    paths = [workspace / AUDITOR_RELATIVE, mapping_path, hypotheses_path, pose_path, workspace / ARCHIVE_RELATIVE,
             calibration_path, base_path, model_path, workspace / 'simulation/so101/so101_new_calib.xml',
             workspace / 'scripts/so101_plan_grasp.py', Path(__file__).resolve()]
    context = {'schema_version': 1, 'kind': 'so101_pose_f_offline_route_search',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'not_for_motion', 'execution_available': False, 'hardware_writes_sent': 0,
        'reported_pose': {'label': 'F', 'source_path': str(pose_path), 'source': pose['source'],
                          'position_counts': pose['position_counts'],
                          'reported_torque_enabled': pose['torque_enabled'],
                          'device_sample_time': pose['device_sample_time'],
                          'assistant_device_read_performed': False,
                          'heights_measured_this_pose_mm': pose['heights_measured_this_pose_mm']},
        'support_box': pose['support_box'],
        'current_feedback_pose_held': False,
        'gripper_start_transition_available': False,
        'gripper_assumption': {
            'model_q_rad': .6, 'fixed_throughout_model_route': True,
            'reported_physical_ID6_count': counts[5],
            'physical_photo_appearance': 'appears_nearly_closed; qualitative_observation_only',
            'count_to_model_opening_calibrated': False,
            'opening_transition_validated': False},
        'execution_blocked_reasons': [
            'The user-confirmed supporting cardboard box is absent from collision geometry',
            'Current live joint feedback is not acquired or held by this offline analysis',
            'Nominal wrist calibration and base transform remain unverified',
            'Nominal ID4 lower-limit margin is only about 0.594 degrees',
            'Model gripper q=0.6 is not the measured opening at real ID6 count 3951; opening transition is not planned',
            'Goal is a historical image target, not a current live target'],
        'nominal_mapping_verified': False, 'candidate_mapping_source': str(mapping_path),
        'reported_pose_model_q_deg': qdeg,
        'reported_pose_model_limit_margins_deg': np.rad2deg(np.minimum(start - auditor.limits[:, 0], auditor.limits[:, 1] - start)).tolist(),
        'start_q_rad': start.tolist(), 'pregrasp_q_rad': goal.tolist(),
        'archived_target': target, 'current_target_freshness_claimed': False,
        'historical_target_check': 'schema, dimensions, calibration hash and age at ORIGINAL publication time passed',
        'assumptions': ['E near-upright wrist zero remains an unverified visual hypothesis',
                        'Shoulder/elbow mapping remains a rough height estimate',
                        'Base transform still contains an unverified 64.635292 mm offset',
                        'Gripper is fixed at model q=0.6, not calibrated to real ID6 count 3951',
                        'Large supporting cardboard, cables and unmodeled scene objects are absent',
                        'No physical motion, speed, force, grasp closure, carrying or dynamic validation'],
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    return auditor, start, goal, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'route_review.json')
    args = parser.parse_args(argv)
    auditor, start, goal, report = build_case(args.workspace)
    # Keep generated artifacts beside this standalone analysis, never in active
    # project data or over an input script/record.
    if args.output.resolve().parent != Path(__file__).resolve().parent:
        parser.error('Output must remain beside this standalone analysis script')
    if args.output.exists() and any(args.output.samefile(p) for p in map(Path, report['source_hashes'])):
        parser.error('Output cannot overwrite an input file')
    report['search'] = search_routes(auditor, start, goal)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    result = report['search']
    print(f"{result['status']}; attempted routes={len(result['attempts'])}")
    if result['route']:
        print(f"route={result['route']['name']}; final model samples={result['route']['unique_sample_count']}")
    print(f'Offline hypothesis only: {args.output}')
    return 0 if result['route'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
