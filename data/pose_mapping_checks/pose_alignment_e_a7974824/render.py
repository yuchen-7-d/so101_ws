"""Render model-only pose hypotheses; never opens a camera or servo port."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/so101_pose_mesh_alignment/mpl')
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path('/home/yu/so101_ws')
OUT=Path(__file__).resolve().parent
source=ROOT/'data/pose_mapping_checks/wrist_model_e_ebbd4505/wrist_model_review.json'
report=json.loads(source.read_text())
modelpath=ROOT/'simulation/so101/scene_grasp.xml'
model=mujoco.MjModel.from_xml_path(str(modelpath))
model.vis.global_.offwidth=640
model.vis.global_.offheight=800
model.vis.headlight.ambient[:]=[.45,.45,.45]
model.vis.headlight.diffuse[:]=[.7,.7,.7]
# Hide the virtual box for pose comparison only; the approach checker keeps it.
box=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'a_box_geom')
model.geom_group[box]=5
data=mujoco.MjData(model)
names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
jointids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
qadr=model.jnt_qposadr[jointids]
camera=mujoco.MjvCamera()
camera.lookat[:]=[.055,0,.18]
camera.distance=.57
camera.azimuth=271.85
camera.elevation=-3
option=mujoco.MjvOption()
option.geomgroup[3]=0
option.geomgroup[5]=0
option.sitegroup[:]=0
cases=report['models']['active_shoulder_elbow_mapping']['scenarios']
images=[]
renderer=mujoco.Renderer(model,height=800,width=640)
try:
    for case in cases:
        data.qpos[qadr[:5]]=np.deg2rad(case['q_deg'])
        data.qpos[qadr[5]]=.05  # illustrative near-closed aperture; not count3951 calibration
        mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera,scene_option=option)
        images.append(renderer.render().copy())
finally:
    renderer.close()

font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(1,3,figsize=(12,6.2))
titles=['A：向纸箱侧前倾','B：近似竖直','C：向肘部侧后倾']
for ax,img,case,title in zip(axes,images,cases,titles):
    ax.imshow(img);ax.axis('off')
    label=f"{title}\nγ={case['assumed_wrist_axis_gamma_deg']:.0f}°，q4={case['q_deg'][3]:.1f}°"
    ax.set_title(label,fontproperties=font,fontsize=12)
    limit='模型腕角在范围内' if case['id4_in_model_limits'] else '模型腕角超出下限'
    ax.text(.5,-.035,limit,transform=ax.transAxes,ha='center',fontproperties=font,
            color='#247d61' if case['id4_in_model_limits'] else '#b04040')
fig.suptitle('同一组 E 姿态读数：仅改变腕部零位假设',fontproperties=font,fontsize=16)
fig.text(.5,.025,'固定相机作定性比较；未标定照片视角。夹爪开口仅作示意；B/C 超限姿态只作静态展示。',
         ha='center',fontproperties=font,fontsize=10)
fig.tight_layout(rect=[0,.06,1,.93])
fig.savefig(OUT/'pose_comparison.png',dpi=160)
metadata={'schema_version':1,'kind':'so101_static_mesh_pose_comparison',
          'source':str(source.relative_to(ROOT)),
          'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
          'source_model_sha256':hashlib.sha256((ROOT/'simulation/so101/so101_new_calib.xml').read_bytes()).hexdigest(),
          'camera':{'lookat':camera.lookat.tolist(),'distance':camera.distance,
                    'azimuth':camera.azimuth,'elevation':camera.elevation,'fitted_to_photo':False},
          'gripper_model_angle_rad':.05,'gripper_count_calibrated':False,
          'gamma_scenarios_deg':[80,90,100],'estimated_closest_from_user_photo':'B_qualitative_only',
          'photo_registration_performed':False,'model_mapping_verified':False,
          'physics_steps':0,'hardware_writes_sent':0,'execution_available':False,
          'scope':'model-only static rendering; no user photo editing, no pixel measurement, no calibration or motion acceptance'}
(OUT/'render_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print(OUT/'pose_comparison.png')
