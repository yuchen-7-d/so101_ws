"""Static model-route keyframes; no physical device access."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/so101_pose_f_route/mpl')
import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from search_route import build_case

out=Path(__file__).resolve().parent
review=json.loads((out/'route_review.json').read_text())
route=review['search']['route']
assert review['execution_available'] is False
assert route is not None
auditor,start,goal,context=build_case()
assert auditor.audit(start,start)['status']=='no_sampled_penetration'
model,data=auditor.model,auditor.data
model.vis.global_.offwidth=600;model.vis.global_.offheight=740
model.vis.headlight.ambient[:]=[.45,.45,.45]
camera=mujoco.MjvCamera();camera.lookat[:]=[.13,0,.19]
camera.distance=.75;camera.azimuth=271.9;camera.elevation=-10
option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
renderer=mujoco.Renderer(model,height=740,width=600)
images=[]
try:
    for q in route['waypoints_rad']:
        data.qpos[auditor.qadr]=q
        mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera,scene_option=option)
        images.append(renderer.render().copy())
finally:
    renderer.close()
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(1,len(images),figsize=(12,6.2))
titles=['F 起点（候选映射）','先调整 ID1/2/3/5，保持 ID4','再转 ID4，达到预抓取点']
for ax,img,title,q in zip(axes,images,titles,route['waypoints_rad']):
    ax.imshow(img);ax.axis('off');ax.set_title(title,fontproperties=font,fontsize=12)
    ax.text(.5,-.035,f'模型腕角 q4={np.rad2deg(q[3]):.1f}°',transform=ax.transAxes,
            ha='center',fontproperties=font,fontsize=10)
fig.suptitle('两段离线候选：先把臂部伸出，再转腕',fontproperties=font,fontsize=16)
fig.text(.5,.025,'仅模型采样检查；支撑大纸箱未建模。夹爪全程假设 q=0.6，开爪过渡未验证。禁止作为实机指令。',
         ha='center',fontproperties=font,fontsize=10)
fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(out/'route_keyframes.png',dpi=160)
print(out/'route_keyframes.png')
