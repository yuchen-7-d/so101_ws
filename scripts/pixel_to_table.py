from pathlib import Path
import cv2
import numpy as np


def pixel_to_table(
        u, v, camera_matrix, dist_coeffs,
        rotation, camera_origin_table,plant_z=0,
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

    scale = (plant_z-camera_origin_table[2]) / ray_table[2]

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

    clicked_pixels = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point_table = pixel_to_table(
            x, y, camera_matrix, dist_coeffs,
            rotation, camera_origin_table,plant_z=0,
        )
        if point_table is not None:
            clicked_pixels.append((x,y))
            print(f'像素:({x},{y})桌面点:{point_table}')

    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

    try:
        if not cap.isOpened():
            raise RuntimeError('摄像头打开失败')

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

        window_name = 'SO101 TABLE CHECK'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                raise RuntimeError('摄像头取帧失败')

            if (frame.shape[1], frame.shape[0]) != image_size:
                raise RuntimeError('实际画面尺寸与标定尺寸不一致')

            display_frame = frame.copy()
            cv2.drawFrameAxes(
                display_frame, camera_matrix, dist_coeffs,
                rvec,tvec,0.10,
            )

            for number, piexl in enumerate(clicked_pixels, start=1):
                cv2.circle(display_frame, piexl, 3, (0,0,255),-1)
                cv2.putText(
                    display_frame,
                    str(number),
                    (piexl[0] + 5,piexl[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,0.255),
                    1,
                )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, (6,9))

                if not found:
                    print('未找到棋盘')
                    continue

                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                refined_corners = cv2.cornerSubPix(
                    gray,corners,(5,5),(-1,-1),criteria
                )

                pixels = refined_corners.reshape(-1,2)
                indices = (0,5,48)
                points = []

                for index in indices:
                    u, v = pixels[index]
                    point = pixel_to_table(
                        u, v, camera_matrix, dist_coeffs,
                        rotation, camera_origin_table,
                    )
                    if point is None:
                        break
                    points.append(point)

                if len(points) != 3:
                    print('桌面求交失败')
                    continue

                row_mm = np.linalg.norm(points[1]-points[0]) * 1000
                col_mm = np.linalg.norm(points[2]-points[0]) * 1000

                print(f'棋盘行方向跨度:{row_mm:.2f}毫米，参考135毫米')
                print(f'棋盘列方向跨度:{col_mm:.2f}毫米，参考216毫米')

    finally:
        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
    main()
