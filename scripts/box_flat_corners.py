from pathlib import Path
from ultralytics import YOLO
from pixel_to_table import pixel_to_table
import cv2
import numpy as np
import json


def match_rectangle_edges(lengths_mm):
    measured_mm = np.asarray(lengths_mm, dtype=float)

    patterns_mm = np.array([
        [40, 74, 40, 74],
        [74, 40, 74, 40],
    ], dtype=float)

    max_errors_mm = np.max(
        np.abs(patterns_mm - measured_mm),
        axis=1,
    )

    best_index = int(np.argmin(max_errors_mm))

    return patterns_mm[best_index], float(max_errors_mm[best_index])

def main():
    workspace = Path(__file__).resolve().parents[1]
    image_path = workspace / 'data' / 'box_samples' / 'box_20261019_063449_365756.jpg'
    weights_path = (
        workspace / 'runs' / 'detect' / 'a_box_v4'
        / 'weights' / 'best.pt'
    )
    top_height_m = 0.04

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


    _, top_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
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

    print(f'尺寸筛选前四边形数量:{len(quadrilaterals)}')

    raw_quadrilaterals = quadrilaterals.copy()

    candidate_error_limit_mm = 4.0
    max_edges_error_mm = 3.0
    valid_quadrilaterals = []

    for index, quad in enumerate(quadrilaterals):
        candidate_points = []

        for u, v in quad.reshape(4,2) + (left,top):
            point= pixel_to_table(
                u, v, camera_matrix, dist_coeffs,
                rotation, camera_origin_table,plant_z=-top_height_m
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

        _, score = match_rectangle_edges(lengths_mm)
        print(
            f'四边形{index}:边长:{np.round(lengths_mm, 2)},'
            f'最大边长偏差:{score:.2f}mm'
        )

        if score <= candidate_error_limit_mm:
            valid_quadrilaterals.append(quad)

    quadrilaterals = valid_quadrilaterals

    if len(quadrilaterals) != 1:

        debug_view = roi.copy()
        colors = ((0,0,255),(255,0,0))

        cv2.drawContours(
            debug_view, large_contours, -1, (0,255,0), 1
        )

        for index, qual in enumerate(raw_quadrilaterals):
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
        (2,2),
        (-1,-1),
        criteria,
    ).reshape(4,2)

    shift_px = np.linalg.norm(
        refined_roi - corners_roi.reshape(4,2),
        axis=1,
    )
    print(f'角点细化位移(px):{np.round(shift_px, 2)}')

    corners_image = refined_roi + (left, top)

    points_table = []

    for i, (u,v) in enumerate(corners_image):
        point = pixel_to_table(
        u, v, camera_matrix, dist_coeffs,
        rotation, camera_origin_table,plant_z=-top_height_m,
        )

        if point is None:
            raise RuntimeError(f'角点{i}求交失败')

        points_table.append(point)

    points_table = np.asarray(points_table)
    center_table = points_table.mean(axis=0)

    final_lengths_mm = []

    for i in range(4):
        j = (i + 1) % 4
        length_mm = np.linalg.norm(
            points_table[j] - points_table[i]
        ) * 1000
        final_lengths_mm.append(length_mm)

    expected_mm, final_score = match_rectangle_edges(final_lengths_mm)

    for i, length_mm in enumerate(final_lengths_mm):
        j = (i + 1) % 4
        error_mm = length_mm - expected_mm[i]

        print(
            f'边{i}-{j}:{length_mm:.2f}mm,'
            f'偏差:{error_mm:+.2f}mm'
        )

    measurement = None

    if final_score <= max_edges_error_mm:
        print(f'顶面中心:{center_table}')

        long_edge_index = int(np.argmax(expected_mm))
        j = (long_edge_index + 1) % 4

        edge_table = (
            points_table[j] - points_table[long_edge_index]
        )

        edge_yaw_deg = np.degrees(
            np.arctan2(edge_table[1], edge_table[0])
        )

        long_axis_yaw_deg = (edge_yaw_deg + 90.0) % 180.0 - 90.0

        print(f'顶面长轴方向:{long_axis_yaw_deg}')

        measurement = {
            'coordinate_system':'table',
            'center_m':center_table.tolist(),
            'long_axis_yaw_deg':float(long_axis_yaw_deg),
            'max_edge_error_mm':float(final_score)
        }

    else:
        print(f'细化后的尺寸未通过,最大偏差:{final_score:.2f}mm')

    output_path = workspace / 'data' / 'box_flat_measurement.json'

    json_text = json.dumps(measurement, indent=2)
    output_path.write_text(json_text + '\n', encoding='utf-8')

    print(f'测试结果已保存至:{output_path}')

    contour_view = roi.copy()
    cv2.drawContours(
        contour_view, large_contours, -1, (0,255,0), 1
    )

    if quadrilaterals:
        cv2.drawContours(
            contour_view, quadrilaterals, -1, (0,0,255), 1,
        )

    for corner_index, (u,v) in enumerate(refined_roi):
        point = (int(round(u)), int(round(v)))
        cv2.circle(contour_view, point, 2, (255,0,0), -1)

        cv2.putText(
            contour_view,
            str(corner_index),
            (point[0] - 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,(0,0,0),1,
        )

    if final_score <= max_edges_error_mm:
        center_pixel, _ = cv2.projectPoints(
            center_table.reshape(1,3),
            rvec,tvec,
            camera_matrix,dist_coeffs,
        )

        u_center, v_center = center_pixel.reshape(2)

        center_roi = (
            int(round(u_center - left)),
            int(round(v_center - top)),
        )

        cv2.circle(
            contour_view, center_roi, 3, (0,255,255), -1,
        )

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
