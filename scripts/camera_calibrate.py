from pathlib import Path
import numpy as np
import cv2


def main():
    pattern_size = (6,9)
    square_size_m = 0.027
    cols, rows = pattern_size

    image_dir = (
        Path(__file__).resolve().parents[1]
        / 'data' / 'calibration' / 'images'
    )
    image_paths = sorted(image_dir.glob('*.png'))

    object_template = np.zeros((cols*rows, 3), dtype=np.float32)
    object_template[:,:2] = (
        np.mgrid[0:cols,0:rows].T.reshape(-1,2)
    )
    object_template *= square_size_m
    object_points = []
    image_points = []
    image_size = None
    used_image_paths = []

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f'读取失败:{image_path.name}')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise RuntimeError('【照片尺寸不一致')

        found, corners = cv2.findChessboardCorners(gray,pattern_size)

        if not found:
            print(f'未找到完整棋盘:{image_path.name}')
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (5,5),
            (-1,-1),
            criteria,
        )

        object_points.append(object_template.copy())
        image_points.append(refined_corners)
        used_image_paths.append(image_path)

    if not image_points:
        raise RuntimeError('没有可用于标定的照片')

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        current_size,
        None,
        None,
    )

    per_view_errors = []

    for i in range(len(image_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs,
        )

        residual = (
            image_points[i].reshape(-1,2)
            - projected.reshape(-1,2)
        )
        error = float(
            np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
        )

        per_view_errors.append(
            (error, used_image_paths[i].name)
        )

    per_view_errors.sort(reverse=True)

    print(f'逐图误差最高3张 (有效{len(per_view_errors)}张)')
    for error, name in per_view_errors[:3]:
        print(f'{name}:{error:.4f}像素')

    print(
        f'照片:{len(image_paths)},'
        f'角点坐标形状:{object_template.shape}'
    )


if __name__ == '__main__':
    main()