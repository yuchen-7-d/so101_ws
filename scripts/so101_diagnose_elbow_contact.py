"""离线复现接触并比较虚拟肘部起点；不连接串口，不修改模型或生成执行目标。"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path('/home/yu/so101_ws')
CHECKER = ROOT / 'scripts/so101_check_elbow_fresh_negative_nine.py'
CHECKER_SHA = '6cf7112f1d98d49e8439e7e30f1e7c9510b37dfc153091fa4a6dd7a129cb9690'
FAILED = Path('/tmp/so101_elbow_fresh_negative9_step_c097paq3.json')
FAILED_SHA = '34a97b83cf2c34e46e3ad0259d5bc6b778f08fdcfb41f4615640319ae768cd52'
VIRTUAL_SHIFTS = (0, -8, -16, -32)
DISTANCE_CAP_M = .05


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_bytes(path, sha):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, '输入文件改变：' + str(path))
    return raw


def load_evidence():
    checked_bytes(CHECKER, CHECKER_SHA)
    spec = importlib.util.spec_from_file_location('fresh_elbow9_source', CHECKER)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    failed = json.loads(checked_bytes(FAILED, FAILED_SHA))
    require(failed['kind'] == 'so101_fresh_snapshot_elbow_negative_nine_geometry'
            and failed['sampled_geometry_complete'] is False
            and failed['status'] == 'not_for_motion' and failed['execution_available'] is False
            and failed['start_counts'] == base.EXPECTED_START
            and failed['sample_counts'] == {'nominal': 10, 'holding_error_grid': 12},
            '来源不是本次几何失败记录')
    sample = failed['first_penetrating_sample']
    require(sample['diagnostic_counts'] == [5, 953, 4077, 1947, 3284, 2143]
            and sample['group'] == 'holding_error_grid' and sample['index'] == 11
            and len(sample['penetrating_contacts']) == 1, '首个失败样本不匹配')
    labels = sample['penetrating_contacts'][0]['geoms']
    require(labels == ['geom_6', 'geom_18'], '来源碰撞对已改变')
    pair = [int(re.fullmatch(r'geom_(\d+)', label).group(1)) for label in labels]
    for path, sha in {**failed['input_files'], **failed['model_and_reference_files']}.items():
        checked_bytes(path, sha)
    return base, failed, pair


def geom_info(mujoco, model, gid):
    require(0 <= gid < model.ngeom, '几何编号超出当前模型')
    body = int(model.geom_bodyid[gid])
    mesh = int(model.geom_dataid[gid])
    is_mesh = int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    return {'geom_id': gid,
            'name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            'body_id': body, 'body_name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            'mesh_name': (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh)
                          if is_mesh and mesh >= 0 else None),
            'type': int(model.geom_type[gid]), 'contype': int(model.geom_contype[gid]),
            'conaffinity': int(model.geom_conaffinity[gid])}


def build_evaluator(report, base, pair):
    start, mapping, candidate, plan = base.load_inputs(report)
    # No model is loaded when this file is imported for offline code checks.
    import mujoco
    import numpy as np
    sys.path.insert(0, str(ROOT / 'scripts'))
    import so101_plan_grasp as planning
    planner = planning.CandidatePlanner(ROOT / 'simulation/so101/scene_grasp.xml', mapping)
    model, data = planner.model, planner.data
    reference = json.loads((ROOT / 'data/base_reference_initial.json').read_bytes())
    target = plan['input_target']
    center, rotation = planning.candidate_transform(np.asarray(target['center_m']),
                                                   target['square_yaw_deg'], reference)
    size = np.asarray(target['size_m'])
    box = planner.lookup(mujoco.mjtObj.mjOBJ_GEOM, 'a_box_geom')
    require(np.allclose(model.geom_size[box], size / 2, atol=1e-9, rtol=0), '盒子尺寸发生变化')
    data.qpos[planner.box_qadr:planner.box_qadr + 3] = center - [0, 0, size[2] / 2]
    yaw = math.atan2(rotation[1, 2], rotation[0, 2])
    data.qpos[planner.box_qadr + 3:planner.box_qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    grip = candidate['mapping_candidate']
    require(grip['offset_raw'] == base.OFFSETS[5] and grip['reference_counts'] == 1913
            and grip['direction'] == 1 and abs(grip['radians_per_count'] - base.SCALE) < 1e-15
            and grip['measured_count_interval'][0] <= start[5] <= grip['measured_count_interval'][1],
            '夹爪候选映射不匹配或需要外推')
    g = grip['reference_model_rad'] + (start[5] - 1913) * base.SCALE
    gj = planner.lookup(mujoco.mjtObj.mjOBJ_JOINT, 'gripper')
    ga = planner.lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper')
    require(max(model.jnt_range[gj, 0], model.actuator_ctrlrange[ga, 0]) <= g <=
            min(model.jnt_range[gj, 1], model.actuator_ctrlrange[ga, 1]), '夹爪角超限')
    report['gripper_angle_deg'] = math.degrees(g)

    report['mujoco_version'] = mujoco.__version__
    report['collision_geoms'] = [geom_info(mujoco, model, gid) for gid in pair]
    require(all(g['name'] is None for g in report['collision_geoms']), '编号不再对应未命名几何体')
    report['model_disableflags'] = int(model.opt.disableflags)
    flag = getattr(mujoco.mjtDisableBit, 'mjDSBL_NATIVECCD', None)
    if flag is not None:
        require(not (int(model.opt.disableflags) & int(flag)), '当前使用legacy CCD，停止距离诊断')
    for item in report['collision_geoms']:
        print(f"geom_{item['geom_id']}：body={item['body_name']}，mesh={item['mesh_name']}", flush=True)

    def evaluate_pose(angles):
        q = np.asarray(angles)
        require(np.all(q >= planner.limits[:, 0]) and np.all(q <= planner.limits[:, 1]),
                '样本超出原模型关节限位，停止检查')
        data.qpos[planner.qadr], data.qpos[planner.gripper_qadr] = q, g
        mujoco.mj_forward(model, data)
        named, contacts = [], []
        for contact in data.contact[:data.ncon]:
            ids = [int(contact.geom1), int(contact.geom2)]
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                     or f'geom_{gid}' for gid in ids]
            named.append((names, float(contact.dist)))
            if set(ids) == set(pair):
                contacts.append({'geoms': names, 'distance_mm': float(contact.dist)*1000,
                                 'position_m': contact.pos.copy().tolist()})
        segment = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, pair[0], pair[1], DISTANCE_CAP_M, segment))
        require(math.isfinite(distance) and np.all(np.isfinite(segment)), '部件距离不是有限值')
        return {'tcp_m': data.site_xpos[planner.tcp].copy().tolist(),
                'joint_limit_margin_deg': np.rad2deg(np.minimum(q-planner.limits[:, 0],
                                                               planner.limits[:, 1]-q)).tolist(),
                'penetrating_contacts': base.penetrating_contacts(named),
                'source_pair_contacts': contacts,
                'pair_signed_distance_mm': distance*1000,
                'pair_distance_capped': distance >= DISTANCE_CAP_M,
                'pair_nearest_segment_m': segment.tolist()}

    return start, mapping, evaluate_pose


def verify_replay(base, failed, mapping, evaluate_pose):
    saved = failed['first_penetrating_sample']
    q = base.initial_angles(saved['diagnostic_counts'], mapping)
    require(max(abs(math.degrees(x)-y) for x,y in zip(q,saved['model_angles_deg'])) < 1e-9,
            '失败样本角度复现不一致')
    replay = evaluate_pose(q)
    expected = saved['penetrating_contacts'][0]
    matches = [c for c in replay['penetrating_contacts'] if c['geoms'] == expected['geoms']]
    require(matches and abs(matches[0]['penetration_mm']-expected['penetration_mm']) <= .001,
            '原穿透未按同一模型复现，停止后续比较')
    return replay


def compare(report, base, start, mapping, evaluate_pose):
    for shift in VIRTUAL_SHIFTS:
        virtual = start.copy()
        virtual[2] += shift
        case = base.new_report()
        case.update(virtual_start_counts=virtual, virtual_elbow_shift_counts=shift,
                    source_start_is_measured=(shift == 0),
                    approach_from_actual_start_checked=False,
                    scope='hypothetical start plus local nine-count step; not a hardware target')
        report['cases'].append(case)
        try:
            base.evaluate_samples(case, virtual, mapping, evaluate_pose)
        except ValueError as error:
            case['sampled_geometry_complete'] = False
            case['error'] = str(error)
        samples = case['checked_samples']
        distances = [s['pair_signed_distance_mm'] for s in samples]
        case['source_pair_min_sampled_distance_mm'] = min(distances) if distances else None
        case['all_pair_distances_capped'] = bool(samples) and all(s['pair_distance_capped'] for s in samples)
        case['distance_scope'] = 'only evaluated samples; early-stop cases are incomplete'
        distance_text = ('>=' if case['all_pair_distances_capped'] else '') + str(case['source_pair_min_sampled_distance_mm'])
        print(f"虚拟ID3起点变化{shift:+d}计数：起点={virtual[2]}；"
              f"局部路径及误差采样完成={case['sampled_geometry_complete']}；"
              f"样本={len(samples)}/1225；"
              f"本碰撞对已采样最小带符号距离(mm)={distance_text}", flush=True)
        if case.get('first_penetrating_sample'):
            first = case['first_penetrating_sample']
            print('  首个穿透：' + json.dumps(first['penetrating_contacts'], ensure_ascii=False), flush=True)
        elif case.get('error'):
            print('  检查停止：' + case['error'], flush=True)


def new_report():
    return {'schema_version': 1, 'kind': 'so101_elbow_contact_diagnosis',
            'status': 'not_for_motion', 'execution_available': False, 'register_writes': 0,
            'diagnostics_completed': False, 'error': None, 'cases': [],
            'failed_source': {'path': str(FAILED), 'sha256': FAILED_SHA},
            'checker': {'path': str(CHECKER), 'sha256': CHECKER_SHA},
            'distance_cap_mm': DISTANCE_CAP_M*1000,
            'limitations': ['Hypothetical starts are not hardware goals or validated approaches.',
                            'Discrete model samples; real mechanical clearance is not measured.',
                            'Source pair distance does not replace all-contact collision checking.',
                            'Hands, cables and external supports are not in this model.']}


def main():
    report = new_report()
    print('离线接触诊断；不连接串口，不修改模型或碰撞阈值。', flush=True)
    try:
        base, failed, pair = load_evidence()
        start, mapping, evaluate_pose = build_evaluator(report, base, pair)
        report['replay'] = verify_replay(base, failed, mapping, evaluate_pose)
        print('已复现来源失败样本；开始比较虚拟起点。', flush=True)
        compare(report, base, start, mapping, evaluate_pose)
        checked_bytes(FAILED, FAILED_SHA)
        checked_bytes(CHECKER, CHECKER_SHA)
        for path, sha in {**report['input_files'], **report['model_and_reference_files']}.items():
            checked_bytes(path, sha)
        report['diagnostics_completed'] = True
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
        print('诊断停止：', report['error'], flush=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='so101_elbow_contact_diagnosis_',
                                     suffix='.json', delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        saved = stream.name
    print('状态：not_for_motion；虚拟起点不是实机调整指令。', flush=True)
    print('完整记录：', saved, flush=True)
    return 0 if report['diagnostics_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
