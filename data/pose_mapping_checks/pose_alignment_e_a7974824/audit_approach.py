"""Audit one explicit joint-space segment offline; never executes robot motion."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

ARM_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
PENETRATION_TOLERANCE_M = 1e-5
RESTING_BOX_TOLERANCE_M = 5e-4


def vector(value, length, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must contain exactly {length} finite values')
    return result


def endpoint_error(value, limits, label):
    try:
        q = vector(value, len(ARM_NAMES), label)
    except (TypeError, ValueError) as exc:
        return None, {'reason': 'invalid_vector', 'detail': str(exc)}
    outside = np.flatnonzero((q < limits[:, 0]) | (q > limits[:, 1]))
    if len(outside):
        return None, {'reason': 'outside_model_limits', 'joints': [
            {'servo_id': int(i + 1), 'joint_name': ARM_NAMES[i],
             'angle_deg': float(np.rad2deg(q[i])),
             'limits_deg': np.rad2deg(limits[i]).tolist()} for i in outside]}
    return q, None


def linear_joint_samples(start, goal, max_step_deg=1.0):
    """Direct coordinates: never wrap angles or clip endpoints."""
    if (isinstance(max_step_deg, bool) or not math.isfinite(max_step_deg)
            or not 0 < max_step_deg <= 1):
        raise ValueError('max_step_deg must be finite, positive and at most 1 degree')
    start = vector(start, len(ARM_NAMES), 'start')
    goal = vector(goal, len(ARM_NAMES), 'goal')
    delta = goal - start
    maximum = float(np.max(np.abs(delta)))
    if maximum == 0:
        return start[None, :].copy()
    segments = math.ceil(maximum / math.radians(max_step_deg))
    return np.linspace(start, goal, segments + 1)


class JointSegmentAuditor:
    def __init__(self, model_path, box_top_center_m, box_yaw_rad,
                 box_size_m=(.04, .04, .074), gripper_q_rad=.6):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        joint_ids = [self.lookup(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES]
        actuators = [self.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_NAMES]
        if not np.all(self.model.jnt_limited[joint_ids]) or not np.all(self.model.actuator_ctrllimited[actuators]):
            raise ValueError('All five arm joints and actuators must have explicit model limits')
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.limits = np.column_stack([
            np.maximum(self.model.jnt_range[joint_ids, 0], self.model.actuator_ctrlrange[actuators, 0]),
            np.minimum(self.model.jnt_range[joint_ids, 1], self.model.actuator_ctrlrange[actuators, 1])])
        self.top = vector(box_top_center_m, 3, 'box top center')
        self.size = vector(box_size_m, 3, 'box dimensions')
        if not np.allclose(self.size, [.04, .04, .074], atol=1e-6, rtol=0):
            raise ValueError('This scene supports the 40x40x74 mm box only')
        if not math.isfinite(box_yaw_rad) or not math.isfinite(gripper_q_rad):
            raise ValueError('Box yaw and model gripper angle must be finite')
        self.yaw, self.gripper_q = box_yaw_rad, gripper_q_rad
        grip = self.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
        if not self.model.jnt_range[grip, 0] <= self.gripper_q <= self.model.jnt_range[grip, 1]:
            raise ValueError('Model gripper angle is outside joint limits')
        self.gripper_qadr = self.model.jnt_qposadr[grip]
        self.box_qadr = self.model.jnt_qposadr[self.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'a_box_free')]
        self.box_geom = self.lookup(mujoco.mjtObj.mjOBJ_GEOM, 'a_box_geom')
        if not np.allclose(self.model.geom_size[self.box_geom], self.size / 2, atol=1e-8, rtol=0):
            raise ValueError('Box geometry dimensions do not match the archived target')
        self.tcp = self.lookup(mujoco.mjtObj.mjOBJ_SITE, 'grasp_center_40mm')
        self.data = None

    def lookup(self, kind, name):
        found = mujoco.mj_name2id(self.model, kind, name)
        if found < 0:
            raise ValueError(f'Model is missing {name}')
        return found

    def sample(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        collisions = []
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist >= -PENETRATION_TOLERANCE_M:
                continue
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f'geom{int(g)}'
                     for g in (contact.geom1, contact.geom2)]
            if set(names) == {'floor', 'a_box_geom'} and contact.dist >= -RESTING_BOX_TOLERANCE_M:
                continue
            collisions.append({'geoms': names, 'penetration_mm': float(-contact.dist * 1000)})
        return {'tcp_m': self.data.site_xpos[self.tcp].tolist(), 'penetrating_contacts': collisions}

    def audit(self, start_q_rad, goal_q_rad, max_step_deg=1.0):
        result = {'status': 'not_evaluated', 'execution_available': False, 'hardware_writes_sent': 0,
                  'model_gripper_angle_rad': self.gripper_q,
                  'gripper_assumption': 'model opening only; no real ID6 count-to-opening calibration',
                  'sample_count': 0, 'samples': [],
                  'scope': 'one direct joint-space segment; not a path search, continuous collision proof or hardware clearance guarantee',
                  'penetration_tolerance_mm': PENETRATION_TOLERANCE_M * 1000,
                  'resting_box_floor_tolerance_mm': RESTING_BOX_TOLERANCE_M * 1000,
                  'max_sample_joint_step_deg': max_step_deg,
                  'model_limits_deg': np.rad2deg(self.limits).tolist()}
        start, error = endpoint_error(start_q_rad, self.limits, 'start')
        if error:
            result.update(status='blocked_start', endpoint_error=error)
            return result
        goal, error = endpoint_error(goal_q_rad, self.limits, 'goal')
        if error:
            result.update(status='blocked_goal', endpoint_error=error)
            return result
        qs = linear_joint_samples(start, goal, max_step_deg)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.gripper_qadr] = self.gripper_q
        self.data.qpos[self.box_qadr:self.box_qadr + 3] = self.top - [0, 0, self.size[2] / 2]
        self.data.qpos[self.box_qadr + 3:self.box_qadr + 7] = [math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)]
        for index, q in enumerate(qs):
            sample = self.sample(q)
            sample.update(index=index, fraction=index / (len(qs) - 1) if len(qs) > 1 else 0.0,
                          q_rad=q.tolist())
            result['samples'].append(sample)
        bad = [s for s in result['samples'] if s['penetrating_contacts']]
        result.update(status='unsafe_segment' if bad else 'no_sampled_penetration',
                      sample_count=len(qs), colliding_sample_count=len(bad),
                      first_collision_sample=bad[0] if bad else None,
                      maximum_sampled_penetration_mm=max(
                          (c['penetration_mm'] for s in bad for c in s['penetrating_contacts']), default=0),
                      direct_joint_delta_deg=np.rad2deg(goal - start).tolist(),
                      realized_max_sample_joint_step_deg=float(np.rad2deg(np.max(np.abs(np.diff(qs, axis=0)))))
                      if len(qs) > 1 else 0.0)
        return result


def make_review(workspace, hypotheses_path):
    workspace, hypotheses_path = Path(workspace), Path(hypotheses_path)
    hypotheses = json.loads(hypotheses_path.read_text())
    for relative, expected in hypotheses['source_hashes'].items():
        if hashlib.sha256((workspace / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archived hypothesis input changed: {relative}')
    sys.path.insert(0, str(workspace / 'scripts'))
    from so101_plan_grasp import candidate_transform, validate_target
    source = workspace / hypotheses['offline_geometry_replay']['source_archived_plan']
    archived = json.loads(source.read_text())
    target = archived['input_target']
    calibration_path = workspace / 'data/calibration/table_plane_pose.npz'
    # Historical validation uses the ORIGINAL publication time. No timestamp or
    # source-mode field is edited, and current target freshness is never claimed.
    center, size, yaw = validate_target(target, now=target['published_at_unix_s'], max_age=2,
                                        allow_offline=False, calibration_path=calibration_path)
    reference_path = workspace / 'data/base_reference_initial.json'
    center_model, rotation = candidate_transform(center, yaw, json.loads(reference_path.read_text()))
    model_path = workspace / 'simulation/so101/scene_grasp.xml'
    auditor = JointSegmentAuditor(model_path, center_model,
                                  math.atan2(rotation[1, 2], rotation[0, 2]), size)
    replay = hypotheses['offline_geometry_replay']
    if not replay['samples'] or replay['samples'][0]['phase'] != 'pregrasp':
        raise ValueError('Hypothesis report lacks an archived model pregrasp goal')
    goal = replay['samples'][0]['model_q_rad']
    cases = []
    for case in hypotheses['models']['active_shoulder_elbow_mapping']['scenarios']:
        result = auditor.audit(np.deg2rad(case['q_deg']), goal)
        result['assumed_wrist_axis_gamma_deg'] = case['assumed_wrist_axis_gamma_deg']
        cases.append(result)
    return {'schema_version': 1, 'kind': 'so101_offline_explicit_start_segment_review',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'not_for_motion', 'execution_available': False, 'hardware_writes_sent': 0,
            'current_start_feedback_available': False, 'current_target_validity_claimed': False,
            'assumption_scope': 'E is an archived user-reported pose; gamma80/90/100 are hypotheses, not measured bounds',
            'historical_target_check': {'schema_geometry_calibration_passed': True,
                'time_basis': 'original_published_at_unix_s',
                'original_target': target, 'source_archived_plan': str(source)},
            'model_coordinate_conversion_verified': False,
            'assumed_model_offset_m': [.064635292, 0, 0],
            'archived_model_pregrasp_goal_rad': goal,
            'source_hashes': {**hypotheses['source_hashes'], str(hypotheses_path): hashlib.sha256(hypotheses_path.read_bytes()).hexdigest(),
                             str(calibration_path): hashlib.sha256(calibration_path.read_bytes()).hexdigest()},
            'code_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (Path(__file__).resolve(), workspace / 'scripts/so101_plan_grasp.py')},
            'cases': cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/yu/so101_ws'))
    parser.add_argument('--hypotheses', type=Path, default=Path('/home/yu/so101_ws/data/pose_mapping_checks/wrist_model_e_ebbd4505/wrist_model_review.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'approach_review.json')
    args = parser.parse_args(argv)
    try:
        report = make_review(args.workspace, args.hypotheses)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Offline review blocked: {exc}. No new report was written; any existing '
              f'output at {args.output} was not refreshed.', file=sys.stderr)
        return 2
    protected = [args.hypotheses, Path(__file__).resolve(),
                 args.workspace / 'data/box_grasp_target.json',
                 args.workspace / 'data/box_grasp_plan.json']
    protected += [args.workspace / p for p in report['source_hashes']] + [Path(p) for p in report['code_hashes']]
    if any(args.output.resolve() == p.resolve() or
           (args.output.exists() and p.exists() and os.path.samefile(args.output, p)) for p in protected):
        parser.error('Output must not overwrite an input file')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for case in report['cases']:
        print(f"gamma={case['assumed_wrist_axis_gamma_deg']:g}: {case['status']}; "
              f"samples={case['sample_count']}; collision_samples={case.get('colliding_sample_count', 0)}")
    print(f'Offline segment review only: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
