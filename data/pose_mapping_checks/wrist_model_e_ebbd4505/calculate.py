"""Offline geometric hypotheses only. No camera/serial access or active-map writes."""
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco
import numpy as np

ROOT = Path('/home/yu/so101_ws')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from so101_plan_grasp import CandidatePlanner, ARM_NAMES, candidate_transform, mapping_diagnostics, validate_target

paths = ['data/so101_joint_mapping.json',
         'data/pose_mapping_checks/pose_e_wrist_observation_sl8kds01.json',
         'data/pose_mapping_checks/grasp_plan_before_id3_direction_4mfmwcx_.json',
         'data/base_reference_initial.json', 'simulation/so101/scene_grasp.xml',
         'simulation/so101/so101_new_calib.xml', 'data/so101_initial_pose.json',
         'data/calibration/table_plane_pose.npz']
hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
read = lambda rel: json.loads((ROOT / rel).read_text())
mapping, pose, archived, base = [read(p) for p in paths[:4]]
p = CandidatePlanner(ROOT / paths[4], mapping)
jids = [p.lookup(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES]
diagnostics = mapping_diagnostics(mapping, pose, p.limits)
q = np.array([c['model_angle_nearest_branch_deg'] for c in diagnostics])

def state(qdeg):
    p.data.qpos[p.qadr] = np.deg2rad(qdeg)
    p.data.qpos[p.gripper_qadr] = .6  # visualization opening; not calibrated real ID6
    mujoco.mj_forward(p.model, p.data)
    return p.data.xanchor[jids].copy()*1000, p.data.site_xpos[p.tcp].copy()*1000

zero, _ = state(np.zeros(5))
alpha0 = np.rad2deg(np.arctan2(*(zero[2]-zero[1])[[2, 0]]))
beta0 = np.rad2deg(np.arctan2(*(zero[3]-zero[2])[[2, 0]]))
L2 = np.linalg.norm(zero[2]-zero[1])
L3 = np.linalg.norm(zero[3]-zero[2])
wrist_planar = np.linalg.norm((zero[4]-zero[3])[[0, 2]])
wrist_lateral = abs((zero[4]-zero[3])[1])
assert np.allclose([L2,L3,wrist_planar,wrist_lateral],[116,135,61.1,18.1],atol=.001)
count_delta = (pose['position_counts']['4'] - mapping['joint_references']['4']['position_count'])*360/4096
forward = np.array([np.cos(np.deg2rad(q[0])), -np.sin(np.deg2rad(q[0])), 0])
models = {}
nominal_mapping = None
for method in ['active_shoulder_elbow_mapping', 'same_pose_height_geometry']:
    armq = q.copy()
    if method == 'same_pose_height_geometry':
        armq[1:3] = pose['height_derived_q2_q3_deg']
    beta = -armq[1]-armq[2]
    cases=[]
    for gamma in [80.,90.,100.]:
        candidate = copy.deepcopy(mapping)
        cq = armq.copy()
        cq[3] = beta-gamma
        ref = cq[3]-count_delta
        candidate['joint_references']['4']['model_angle_deg'] = float(ref)
        candidate['model_mapping_verified'] = False
        anchors,tcp = state(cq)
        v = anchors[4]-anchors[3]
        actual_gamma = float(np.rad2deg(np.arctan2(v[2], v@forward)))
        assert abs(actual_gamma-gamma)<1e-8
        cases.append({'assumed_wrist_axis_gamma_deg':gamma,
                      'q_deg':cq.tolist(), 'candidate_id4_reference_deg_at_2046':float(ref),
                      'id4_in_model_limits':bool(np.all(np.deg2rad(cq[3])>=p.limits[3,0]) and np.deg2rad(cq[3])<=p.limits[3,1]),
                      'fk_gamma_deg':actual_gamma,'model_joint_anchors_mm':anchors.tolist(),
                      'model_tcp_mm':tcp.tolist()})
        if method == 'active_shoulder_elbow_mapping' and gamma == 90:
            nominal_mapping = candidate
    models[method]={'scenarios':cases,'shoulder_elbow_source':method}

assert nominal_mapping is not None
nominal_mapping['candidate_status']='offline_photo_upright_hypothesis'
nominal_mapping['not_for_motion']=True
nominal_mapping['execution_available']=False
nominal_mapping['candidate_evidence']={
    'pose':paths[1], 'assumed_gamma_deg':90,
    'basis':'qualitative_near_upright_photo_assumption_not_measured_angle',
    'active_shoulder_elbow_preserved':True,
    'hypothesis_sensitivity_angles_deg':[80,90,100],
    'sensitivity_is_not_a_measured_error_bound':True,
    'candidate_id4_reference_is_not_verified':True}

# Geometry replay from an archived target, not a live-target publication.
target=archived['input_target']
validate_target(target, target['published_at_unix_s'], 2.0, False,
                ROOT/'data/calibration/table_plane_pose.npz')
center,rot=candidate_transform(np.array(target['center_m']),target['square_yaw_deg'],base)
candidate_planner=CandidatePlanner(ROOT/paths[4],nominal_mapping)
samples=candidate_planner.plan(center,rot,np.array(target['size_m']))
assert len(samples)==41 and all(s['ik_ok'] for s in samples)
assert all(s['candidate_counts_in_single_turn'] for s in samples)
descend=[s for s in samples if s['phase']=='descend'][-1]
dq=np.rad2deg(descend['model_q_rad'])
_,dtcp=state(dq)
tcp_sensitivity=[]
for zero_error in [-10.,10.]:
    shifted=dq.copy();shifted[3]+=zero_error
    _,stcp=state(shifted)
    tcp_sensitivity.append({'assumed_wrist_zero_error_deg':zero_error,
                            'tcp_shift_at_model_grasp_mm':float(np.linalg.norm(stcp-dtcp))})
roundtrip=[]
for s in samples:
    for i,raw in enumerate(s['candidate_counts_unwrapped'],1):
        ref=nominal_mapping['joint_references'][str(i)]
        direction=nominal_mapping['joint_directions'][str(i)]['direction']
        restored=ref['model_angle_deg']+(raw-ref['position_count'])*360/4096*direction
        roundtrip.append(abs(restored-np.rad2deg(s['model_q_rad'][i-1])))
assert max(roundtrip)<1e-10
report={'schema_version':1,'kind':'so101_model_dimension_wrist_hypotheses',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'not_for_motion','execution_available':False,'hardware_writes_sent':0,
        'active_mapping_modified':False,'source_hashes':hashes,
        'fixed_model_dimensions_mm':{'shoulder_elbow':float(L2),'elbow_wrist':float(L3),
                                     'wrist_axes_in_plane':float(wrist_planar),'wrist_axes_lateral':float(wrist_lateral)},
        'reported_pose':pose['position_counts'],'models':models,
        'assumption_scope':'gamma90 is an unmeasured nominal assumption; gamma80/100 are illustrative sensitivity scenarios, not bounds or confidence intervals',
        'model_id4_limits_deg':np.rad2deg(p.limits[3]).tolist(),
        'grasp_zero_error_sensitivity':tcp_sensitivity,
        'offline_geometry_replay':{'source_archived_plan':paths[2],
            'source_archived_plan_sha256':hashes[paths[2]],
            'original_captured_at_unix_s':target['captured_at_unix_s'],
            'historical_schema_geometry_calibration_check':{
                'passed':True,'evaluation_time_original_published_at_unix_s':target['published_at_unix_s'],
                'current_calibration_hash_matches':True,'current_target_freshness_not_claimed':True},
            'current_live_target_validity_not_claimed':True,
            'model_ik_complete':True,'sample_count':len(samples),
            'sampled_penetrating_contacts':sum(len(s['penetrating_contacts']) for s in samples),
            'candidate_counts_in_single_turn':True,'max_count_angle_roundtrip_error_deg':max(roundtrip),
            'scope':'model target geometry replay only; IK does not validate wrist calibration; no current-pose approach or gripper-force/lift validation',
            'samples':samples}}

font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axs=plt.subplots(1,2,figsize=(11,5.6),gridspec_kw={'width_ratios':[1.3,1]})
cases=models['active_shoulder_elbow_mapping']['scenarios']
colors=['#25867f','#d88a12','#b44b63']
for case,color in zip(cases,colors):
    pts=np.array(case['model_joint_anchors_mm']);S=pts[1]
    # Align shoulder height for illustration only; never changes the base transform.
    prof=np.column_stack([(pts-S)@forward,pts[:,2]-S[2]+125])
    if case['assumed_wrist_axis_gamma_deg']==90:
        axs[0].plot(prof[1:4,0],prof[1:4,1],'-o',lw=6,color='#4c5968',zorder=1)
        for point,label in zip(prof[1:4],['肩轴','肘轴','腕轴']):
            offset=(-30,-16) if label=='肘轴' else (5,-16)
            axs[0].annotate(label,point,xytext=offset,textcoords='offset points',fontproperties=font)
    axs[0].plot(prof[3:5,0],prof[3:5,1],'-o',lw=3,color=color,
                label=f"W→R 连线假设 {case['assumed_wrist_axis_gamma_deg']:.0f}°")
axs[0].set_aspect('equal');axs[0].invert_xaxis();axs[0].grid(alpha=.2)
axs[0].set_xlabel('← 朝桌面工作区  /  朝肘部后方 →（mm）',fontproperties=font)
axs[0].set_ylabel('肩轴对齐 125 mm 后的示意高度（mm）',fontproperties=font)
axs[0].set_title('保持当前肩肘映射，仅比较腕部假设',fontproperties=font)
axs[0].legend(prop=font,loc='upper right',fontsize=9)
angles=[c['q_deg'][3] for c in cases]
axs[1].axhspan(-95,0,color='#e7f1eb',label='模型范围内的显示区间')
axs[1].axhline(-95,color='#b34343',ls='--',lw=1.5)
axs[1].bar([80,90,100],angles,color=colors,width=6,bottom=0)
for x,y in zip([80,90,100],angles):axs[1].text(x,y-2,f'{y:.2f}°',ha='center',va='top')
axs[1].set_ylim(-122,-70);axs[1].set_xticks([80,90,100]);axs[1].grid(axis='y',alpha=.2)
axs[1].set_xlabel('假设腕轴相对水平的角度 γ（°）',fontproperties=font)
axs[1].set_ylabel('对应模型腕关节角 q4（°）',fontproperties=font)
axs[1].set_title('90° 假设仍超出模型 −95° 下限',fontproperties=font)
axs[1].text(105,-93.8,'模型下限 −95°',ha='right',fontproperties=font,color='#b34343')
fig.suptitle('模型尺寸可复用；安装零位仍是假设',fontproperties=font,fontsize=16)
fig.text(.5,.01,'仅离线预览；80°/100°不是测得的误差界限。连线表示转轴关系，不代表外壳和线缆无碰撞。',
         ha='center',fontproperties=font,fontsize=9)
fig.tight_layout(rect=[0,.04,1,.94])
fig.savefig(OUT/'wrist_model_preview.png',dpi=150)
for name,data in [('wrist_candidate_mapping.json',nominal_mapping),('wrist_model_review.json',report)]:
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'model_dimensions_mm':report['fixed_model_dimensions_mm'],
                  'scenarios':[{'gamma':c['assumed_wrist_axis_gamma_deg'],'q4':c['q_deg'][3],
                                'ref4':c['candidate_id4_reference_deg_at_2046'],
                                'within_limits':c['id4_in_model_limits']} for c in cases],
                  'ik_samples':len(samples),'contact_count':report['offline_geometry_replay']['sampled_penetrating_contacts'],
                  'tcp_sensitivity':tcp_sensitivity},ensure_ascii=False))
