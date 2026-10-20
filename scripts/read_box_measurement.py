from pathlib import Path
import json
import numpy as np


def main():
    workspase = Path(__file__).resolve().parents[1]
    input_path = workspase / 'data' / 'box_top_measurement.json'

    json_text = input_path.read_text(encoding='utf-8')
    measurement = json.loads(json_text)

    if measurement is None:
        print('这份记录没有通过尺寸筛选的检测结果')
        return

    center_m = measurement['center_m']
    yaw_deg = measurement['square_yaw_deg']

    print(f'顶面中心:{center_m}')
    print(f'顶面方向:{yaw_deg:.2f}')

    x_table, y_table, z_table = center_m

    center_sim_demo = [x_table,-y_table,-z_table]
    yaw_sim_demo_deg = -yaw_deg

    print(f'仿真演示中心:{center_sim_demo}')
    print(f'仿真演示方向:{yaw_sim_demo_deg:.2f}')

    reference_path = workspase / 'data' / 'base_reference_initial.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))

    origin_table = np.asarray(reference['origin_table_m'])

    R_table_from_ref = np.column_stack([
        reference['x_axis_table'],
        reference['y_axis_table'],
        reference['z_axis_table'],
    ])

    center_ref = R_table_from_ref.T @ (
        np.asarray(center_m) - origin_table
    )

    print(f'临时底盘参考中心:{center_ref}')

    theta = np.deg2rad(yaw_deg)
    direction_table = np.array([
        np.cos(theta), np.sin(theta), 0.0,
    ])

    direction_ref = R_table_from_ref.T @ direction_table

    yaw_ref_deg = np.rad2deg(np.arctan2(
        direction_ref[1], direction_ref[0]
    ))
    yaw_ref_deg = (yaw_ref_deg + 90.0) % 180.0 - 90.0

    print(f'临时参考系纸盒边方向:{yaw_ref_deg:.2f}°')

    reference_origin_in_model_assumed = np.array([
        0.064635292, 0.0, 0.0,
    ])

    center_model_assumed = (
        center_ref + reference_origin_in_model_assumed
    )

    print(f'初始模型目标中心(m):{center_model_assumed}')
    print(f'初始模型纸盒边方向:{yaw_ref_deg:.2f}°')


if __name__ == '__main__':
    main()
