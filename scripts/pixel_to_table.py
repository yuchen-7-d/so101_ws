from pathlib import Path
import cv2
import numpy as np

def pixel_to_table(
        u, v, camera_matrix, dist_coeffs,
        rotation, camera_origin_table,
):
    pixel = np.array([[[u, v]]], dtype=np.float64)
    
    normalized = cv2.undistortPoints(
        pixel,
        camera_matrix,
        dist_coeffs,
    )

    xn, yn = normalized.reshape(2)
    ray_camera = np.array([xn, yn, 1.0])
    ray_table = rotation.T @ ray_camera

    if abs(ray_table[2]) <1e-9:
        print('射线与桌面接近平行，无法稳定求交')
        return

    scale = -camera_origin_table[2] / ray_table[2]

    if scale <= 0:
        print('交点位于相机后方，检查坐标变换')
        return

    point_table = camera_origin_table + scale * ray_table
    return point_table

def main():
    project_dir = Path(__file__).resolve().parents[1]
    pose_path = (
        project_dir / 'data' / 'calibration'
        / 'table_plane_pose.npz'
    )

    with np.load(pose_path, allow_pickle=False) as pose:
        camera_matrix = pose['camera_matrix']
        dist_coeffs = pose['dist_coeffs']
        rvec = pose['rvec']
        tvec = pose['tvec'].reshape(3)
        image_size = tuple(pose['image_size'].tolist())

    rotation, _ = cv2.Rodrigues(rvec)
    camera_origin_table = -rotation.T @ tvec

    u, v = image_size[0] / 2, image_size[1] / 2
    point_table = pixel_to_table(
        u, v, camera_matrix, dist_coeffs,
        rotation, camera_origin_table,
    )
    if point_table is not None:
        print(f'桌面交点:{point_table}')


if __name__ == '__main__':
    main()
