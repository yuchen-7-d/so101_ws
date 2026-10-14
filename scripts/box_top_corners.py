from pathlib import Path
from ultralytics import YOLO
from pixel_to_table import pixel_to_table
import cv2
import numpy as np


def main():
    workspace = Path(__file__).resolve().parents[1]
    image_path = workspace / 'data' / 'box_test.jpg'
    weights_path = (
        workspace / 'runs' / 'detect' / 'a_box_v4'
        / 'weights' / 'best.pt'
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError('图片读取失败')

    pose_path = (
        workspace / 'data' / 'calibration'
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

    if (image.shape[1],image.shape[0]) != image_size:
        raise RuntimeError('图片尺寸与标定尺寸不一致')

    print(f'标定尺寸:{image_size}相机位置:{camera_origin_table}')

    model = YOLO(weights_path)
    result = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device='cpu',
        verbose=False,
        save=False,
    )[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError('未检测到纸盒')

    index = int(boxes.conf.argmax().item())
    x1, y1, x2, y2 = boxes.xyxy[index].tolist()
    print(f'纸盒框:({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})')

    height, weight = image.shape[:2]
    padding = 8

    left = max(0, int(x1)-padding)
    top = max(0,int(y1)-padding)
    right = min(weight, int(x2)+padding)
    bottom = min(height, int(y2)+padding)

    roi = image[top:bottom, left:right]

    if roi.size == 0:
        raise RuntimeError('纸盒裁剪区域空')

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred,50,150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        close_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = roi.shape[0] * roi.shape[1] * 0.03

    areas = [cv2.contourArea(contour) for contour in contours]
    max_area = max(areas, default=0)
    print(f'轮廓总数:{len(contours)},面积门槛:{min_area:.1f},最大面积:{max_area:.1f}')

    large_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]
    print(f'较大轮廓数量:{len(large_contours)}')

    quadrilaterals = []

    for i, contour in enumerate(large_contours):
        hull = cv2.convexHull(contour)

        perimeter = cv2.arcLength(hull,True)
        approx = cv2.approxPolyDP(
            hull, 0.02 * perimeter, True,
        )

        print(f'候选{i}:顶点数={len(approx)}')

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quadrilaterals.append(approx)

    max_edges_error_mm = 3.0
    valid_quadrilaterals = []

    for index, quad in enumerate(quadrilaterals):
        candidate_points = []

        for u, v in quad.reshape(4,2) + (left,top):
            point= pixel_to_table(
                u, v, camera_matrix, dist_coeffs,
                rotation, camera_origin_table,plant_z=-0.074,
            )

            if point is None:
                break

            candidate_points.append(point)

        if len(candidate_points) != 4:
            print(f'四边形{index}求交失败')
            continue

        lengths_mm = []

        for i in range(4):
            j = (i + 1) % 4
            length = np.linalg.norm(
                candidate_points[j] - candidate_points[i]
            ) * 1000
            lengths_mm.append(length)

        score = max(abs(length - 40) for length in lengths_mm)
        print(f'四边形{index}:最大边长偏差:{score:.2f}')

        if score <= max_edges_error_mm:
            valid_quadrilaterals.append(quad)

    quadrilaterals = valid_quadrilaterals

    if len(quadrilaterals) != 1:

        debug_view = roi.copy()
        colors = ((0,0,255),(255,0,0))

        cv2.drawContours(
            debug_view, large_contours, -1, (0,255,0), 1
        )

        for index, qual in enumerate(quadrilaterals):
            color = colors[index % 2]

            cv2.drawContours(
                debug_view, [qual], -1, color, 1,
            )

        cv2.imshow(
            'candidate check',
            cv2.resize(debug_view, None, fx=3, fy=3)
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        raise RuntimeError(f'四边形候选数量:{len(quadrilaterals)}')

    corners_roi = quadrilaterals[0].astype('float32')

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )

    refined_roi = cv2.cornerSubPix(
        gray,
        corners_roi.copy(),
        (3,3),
        (-1,-1),
        criteria,
    ).reshape(4,2)

    corners_image = refined_roi + (left, top)

    points_table = []

    for i, (u,v) in enumerate(corners_image):
        point = pixel_to_table(
        u, v, camera_matrix, dist_coeffs,
        rotation, camera_origin_table,plant_z=-0.074,
        )

        if point is None:
            raise RuntimeError(f'角点{i}求交失败')

        points_table.append(point)

    points_table = np.asarray(points_table)
    center_table = points_table.mean(axis=0)
    print(f'顶面中心:{center_table}')

    for i in range(4):
        j = (i + 1) % 4

        length_mm = np.linalg.norm(
            points_table[j] - points_table[i]
        ) * 1000

        error_mm = length_mm - 40.0

        print(
            f'边{i}-{j}:{length_mm:.2f}mm,'
            f'偏差:{error_mm:+.2f}mm'
        )

    contour_view = roi.copy()
    cv2.drawContours(
        contour_view, large_contours, -1, (0,255,0), 1
    )

    if quadrilaterals:
        cv2.drawContours(
            contour_view, quadrilaterals, -1, (0,0,255), 1,
        )

    for u, v in refined_roi:
        point = (int(round(u)),int(round(v)))
        cv2.circle(contour_view, point, 2, (255,0,0), -1)

    display_roi = cv2.resize(contour_view, None, fx=3, fy=3)
    display_edges = cv2.resize(
        close_edges, None, fx=3, fy=3,
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imshow('box ROI', display_roi)
    cv2.imshow('box edges', display_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
