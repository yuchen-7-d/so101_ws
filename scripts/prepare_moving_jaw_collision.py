from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET


def main():
    workspace = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(
        str(workspace / 'simulation/so101/scene_grasp.xml')
    )

    mesh_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_MESH, 'moving_jaw_so101_v1'
    )
    if mesh_id < 0:
        raise RuntimeError('找不到活动爪网格')

    v_start = model.mesh_vertadr[mesh_id]
    v_count = model.mesh_vertnum[mesh_id]
    f_start = model.mesh_faceadr[mesh_id]
    f_count = model.mesh_facenum[mesh_id]

    vertices = model.mesh_vert[v_start:v_start+v_count].copy()
    faces = model.mesh_face[f_start:f_start+f_count].copy()

    print(f'顶点:{vertices.shape},三角面:{faces.shape}')

    geom_id = mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_GEOM,'moving_jaw_visual'
    )
    if geom_id < 0:
        raise RuntimeError('找不到活动爪碰撞几何')

    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(
        rotation, model.geom_quat[geom_id]
    )
    rotation = rotation.reshape(3,3)

    vertices_body = (
        vertices @ rotation.T + model.geom_pos[geom_id]
    )

    print('最小坐标(mm):', np.round(vertices_body.min(axis=0) * 1000, 3))
    print('最大坐标(mm):', np.round(vertices_body.max(axis=0) * 1000, 3))

    y_edges = np.array([
        -0.083, -0.067, -0.062, -0.035, 0.011
    ])

    triangles = vertices_body[faces]

    cut_points = {}

    edge_start = triangles.reshape(-1, 3)
    edge_end = triangles[:, [1, 2, 0], :].reshape(-1, 3)

    for cut_y in y_edges[1:-1]:
        crossing = (
            (edge_start[:, 1] - cut_y)
            * (edge_end[:, 1] - cut_y)
        ) < 0

        a = edge_start[crossing]
        b = edge_end[crossing]

        t = (cut_y - a[:, 1]) / (b[:, 1] - a[:, 1])
        points = a + t[:, None] * (b - a)
        points = np.unique(np.round(points, 9),axis=0)
        cut_points[cut_y] = points

        print(
            f'切面Y={cut_y * 1000:.1f}mm,'
            f'交点记录数:{len(points)}'
        )

    y = vertices_body[:,1]

    segments = []

    for i in range(len(y_edges) - 1):
        y_min = y_edges[i]
        y_max = y_edges[i + 1]

        mask = (y >= y_min) & (y <= y_max)
        point_groups = [vertices_body[mask]]

        for boundary in (y_min, y_max):
            if boundary in cut_points:
                point_groups.append(cut_points[boundary])

        segment_points = np.unique(
            np.round(np.vstack(point_groups), 9),
            axis=0,
        )
        segments.append(segment_points)

        print(f'分段{i}:补点并去重后:{len(segment_points)}')

    collision_parts = []

    for i, segment_points in enumerate(segments):
        hull = ConvexHull(segment_points)
        hull_points = segment_points[hull.vertices]

        collision_parts.append(hull_points)

        print(
            f'凸包{i}:顶点数:{len(hull_points)},'
            f'体积:{hull.volume * 1000**3:.2f}mm³'
        )

    asset_root = ET.Element('mujocoinclude')

    for i, hull_points in enumerate(collision_parts):
        vertex_text = ' '.join(
            f'{value:.9f}' for value in hull_points.ravel()
        )

        ET.SubElement(
            asset_root,
            'mesh',
            name=f'moving_jaw_collision_part_{i}',
            vertex=vertex_text,
        )

    output_path = (
        workspace / 'simulation' / 'so101'
        / 'moving_jaw_collision_assets.xml'
    )

    ET.indent(asset_root, space='  ')
    ET.ElementTree(asset_root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True,
    )

    print(f'已保存碰撞网格资源:{output_path}')


if __name__ == '__main__':
    main()
