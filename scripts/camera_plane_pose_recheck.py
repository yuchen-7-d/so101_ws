from pathlib import Path
import numpy as np
import cv2


def main():
    project_dir = Path(__file__).resolve().parents[1]
    calibration_path = (
        project_dir / 'data' / 'calibration'
        / 'camera_calibration.npz'
    )

    plane_pose_path = (
        calibration_path.parent / 'table_plane_pose_candidate.npz'
    )

    with np.load(calibration_path, allow_pickle=False) as calibration:
        camera_matrix = calibration['camera_matrix']
        dist_coeffs = calibration['dist_coeffs']
        image_size = tuple(
            calibration['image_size'].tolist()
        )

    pattern_size = (6,9)
    square_size_m = 0.027
    cols, rows = pattern_size

    object_points = np.zeros(
        (cols * rows, 3), dtype=np.float32
    )

    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    object_points *= square_size_m

    cap = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2)

    try:
        if not cap.isOpened():
            raise RuntimeError('摄像头打开失败')

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('摄像头取帧失败')

            current_size = (frame.shape[1], frame.shape[0])
            if current_size != image_size:
                raise RuntimeError('实际图片与标定尺寸不一致')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)

            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )

            display_frame = frame.copy()
            if found:

                refined_corners = cv2.cornerSubPix(
                    gray, corners, (5,5), (-1,-1), criteria
                )

                # 兼容 (N, 2) 和 (N, 1, 2) 两种返回形状。
                refined_corners = refined_corners.reshape(-1, 1, 2)

                # 仅用于当前固定视角的原点方向核对。
                if refined_corners[0, 0, 0] > refined_corners[-1, 0, 0]:
                    refined_corners = refined_corners[::-1].copy()

                pose_ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    refined_corners,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                cv2.drawChessboardCorners(
                    display_frame, pattern_size, refined_corners, found
                )

                if pose_ok:
                    cv2.drawFrameAxes(
                        display_frame, camera_matrix, dist_coeffs,
                        rvec, tvec, 0.05
                    )

            cv2.imshow('SO101 plane pose', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('p'), ord('s')):
                if not found or not pose_ok:
                    print('当前帧没有有效棋盘位姿')
                    continue

                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec, tvec, camera_matrix, dist_coeffs
                )

                measured = refined_corners

                residual = (
                    measured.reshape(-1, 2)
                    - projected.reshape(-1, 2)
                )

                pose_rms = float(
                    np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
                )

                x, y, z = tvec.ravel()
                print('当前rvec:', rvec.ravel())
                print('当前tvec:', tvec.ravel())
                print(
                    f'棋盘原点相机坐标(m):({x:.4f}, {y:.4f}, {z:.4f}),'
                    f'重投影RMS:{pose_rms:.4f}像素'
                )

                if key == ord('s'):
                    if plane_pose_path.exists():
                        print('参数位姿文件已存在，不覆盖')
                        continue

                    np.savez(
                        plane_pose_path,
                        rvec=rvec,
                        tvec=tvec,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        image_size=np.asanyarray(image_size),
                        pattern_size=np.asanyarray(pattern_size),
                        square_size_m=square_size_m,
                        pose_rms=pose_rms,
                        transform_direction='board_to_camera',
                    )
                    print(f'已保存桌面参考位姿:{plane_pose_path}')

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
