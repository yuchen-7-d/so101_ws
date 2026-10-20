"""Publish current box top measurements. This module has no robot-control path."""

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np


BOX_SIZE_M = [0.04, 0.04, 0.074]
TARGET_CLASS = "a_box"
MAX_EDGE_ERROR_MM = 3.0


class MeasurementError(ValueError):
    """A frame does not provide one unambiguous, valid known-size box top."""


def atomic_write_json(path, record):
    """Never expose partially written JSON or retain old coordinates on invalidation."""
    path = Path(path)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_calibration(path):
    """Read and hash exactly the same bytes; table Z is downward in this project."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        matrix = np.asarray(data["camera_matrix"], dtype=float)
        distortion = np.asarray(data["dist_coeffs"], dtype=float)
        rvec = np.asarray(data["rvec"], dtype=float).reshape(3)
        tvec = np.asarray(data["tvec"], dtype=float).reshape(3)
        image_size_raw = np.asarray(data["image_size"], dtype=float).reshape(2)
        transform = str(data["transform_direction"].item())
        rms = float(data["pose_rms"])
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise MeasurementError("相机内参无效")
    if (distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()
            or not np.isfinite(rvec).all() or not np.isfinite(tvec).all()
            or not np.isfinite(image_size_raw).all()
            or np.any(image_size_raw <= 0) or np.any(image_size_raw != np.floor(image_size_raw))
            or not np.isfinite(rms) or rms < 0 or transform != "board_to_camera"):
        raise MeasurementError("桌面标定参数或变换方向无效")
    image_size = tuple(int(value) for value in image_size_raw)
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "camera_matrix": matrix, "dist_coeffs": distortion,
        "rotation": rotation, "camera_origin_table": -rotation.T @ tvec,
        "image_size": image_size,
        "provenance": {"path": str(path), "sha256": sha256(raw).hexdigest(),
                       "image_size": list(image_size), "transform_direction": transform,
                       "pose_rms_px": rms},
    }


def diagnostic_numbers(values):
    numbers = np.asarray(values, dtype=float)
    result = numbers.astype(object)
    result[~np.isfinite(numbers)] = None
    return result.tolist()


def measure_top_corners(corners_image, calibration, diagnostics=None):
    """Project an ordered convex quadrilateral onto the known 74 mm top plane."""
    corners = np.asarray(corners_image, dtype=float)
    detail = {} if diagnostics is None else diagnostics
    detail.update(stage="corners", valid=False, corners_image_px=diagnostic_numbers(corners))
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise MeasurementError("顶面四角无效")
    width, height = calibration["image_size"]
    if (np.any(corners < 0) or np.any(corners[:, 0] >= width)
            or np.any(corners[:, 1] >= height)
            or not cv2.isContourConvex(corners.astype(np.float32).reshape(4, 1, 2))):
        raise MeasurementError("顶面四角超出画面或不构成凸四边形")
    detail["stage"] = "projection"
    normalized = cv2.undistortPoints(corners.reshape(4, 1, 2),
                                    calibration["camera_matrix"], calibration["dist_coeffs"]).reshape(4, 2)
    rays = np.column_stack((normalized, np.ones(4))) @ calibration["rotation"]
    origin = calibration["camera_origin_table"]
    if not np.isfinite(rays).all() or np.any(np.abs(rays[:, 2]) < 1e-9):
        raise MeasurementError("顶面投影射线与平面接近平行")
    scales = (-BOX_SIZE_M[2] - origin[2]) / rays[:, 2]
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise MeasurementError("顶面投影位于相机后方")
    points = origin + rays * scales[:, None]
    lengths_mm = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) * 1000
    max_error = float(np.max(np.abs(lengths_mm - 40.0)))
    diagonals_mm = np.linalg.norm(points[[2, 3]] - points[[0, 1]], axis=1) * 1000
    detail.update(stage="geometry", side_lengths_mm=diagnostic_numbers(lengths_mm),
                  side_errors_mm=diagnostic_numbers(lengths_mm-40),
                  diagonal_lengths_mm=diagnostic_numbers(diagonals_mm),
                  diagonal_errors_mm=diagnostic_numbers(diagonals_mm-np.sqrt(2)*40))
    if (not np.isfinite(points).all() or max_error > MAX_EDGE_ERROR_MM
            or np.max(np.abs(diagonals_mm - np.sqrt(2) * 40)) > np.sqrt(2) * MAX_EDGE_ERROR_MM):
        raise MeasurementError("顶面几何未通过40毫米方形尺寸检查")
    edge = points[1] - points[0]
    detail.update(stage="accepted", valid=True)
    yaw = float((np.degrees(np.arctan2(edge[1], edge[0])) + 45) % 90 - 45)
    return {"center_m": points.mean(axis=0).tolist(), "square_yaw_deg": yaw,
            "max_edge_error_mm": max_error}


def extract_topface(image, bbox, calibration, diagnostics=None):
    """Reuse the existing ROI/contour/subpixel procedure without files or GUI."""
    detail = {} if diagnostics is None else diagnostics
    detail["stage"] = "frame"
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8
            or (image.shape[1], image.shape[0]) != calibration["image_size"]):
        raise MeasurementError("帧尺寸/格式与640×480标定不一致")
    bbox = np.asarray(bbox, dtype=float)
    detail.update(stage="roi", bbox_xyxy=diagnostic_numbers(bbox))
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        raise MeasurementError("检测框无效")
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    left, top = max(0, int(x1)-8), max(0, int(y1)-8)
    right, bottom = min(width, int(x2)+8), min(height, int(y2)+8)
    if right-left < 6 or bottom-top < 6:
        raise MeasurementError("纸盒裁剪区域过小或为空")
    roi = image[top:bottom, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detail.update(stage="candidate_filter", roi_xyxy=[left, top, right, bottom],
                  contour_count=len(contours), large_contour_count=0, convex_quad_count=0,
                  geometry_reject_count=0, candidates=[], candidate_records_limit=12)
    candidates = []
    for contour_index, contour in enumerate(contours):
        if cv2.contourArea(contour) < roi.shape[0] * roi.shape[1] * .03:
            continue
        detail["large_contour_count"] += 1
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, .02 * cv2.arcLength(hull, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        detail["convex_quad_count"] += 1
        candidate_detail = {"contour_index": contour_index}
        if len(detail["candidates"]) < 12:
            detail["candidates"].append(candidate_detail)
        try:
            measure_top_corners(approx.reshape(4, 2) + (left, top), calibration, candidate_detail)
        except MeasurementError as error:
            candidate_detail["reason"] = str(error)
            detail["geometry_reject_count"] += 1
            continue
        candidates.append(approx)
        detail["selected_coarse"] = candidate_detail
    detail["candidate_count"] = len(candidates)
    if len(candidates) != 1:
        raise MeasurementError(f"合格顶面候选数量={len(candidates)}，需要唯一顶面")
    corners_roi = candidates[0].astype(np.float32)
    detail["stage"] = "refinement"
    if (np.any(corners_roi[:, 0, 0] < 3) or np.any(corners_roi[:, 0, 1] < 3)
            or np.any(corners_roi[:, 0, 0] >= roi.shape[1]-3)
            or np.any(corners_roi[:, 0, 1] >= roi.shape[0]-3)):
        raise MeasurementError("顶面角点太靠近裁剪边缘")
    refined = cv2.cornerSubPix(gray, corners_roi.copy(), (2, 2), (-1, -1),
                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01))
    # Revalidate the refined geometry; accepting the coarse candidate is insufficient.
    detail.update(stage="refined_geometry", corner_shift_px=diagnostic_numbers(
        np.linalg.norm(refined.reshape(4, 2)-corners_roi.reshape(4, 2), axis=1)), refined={})
    measurement = measure_top_corners(refined.reshape(4, 2) + (left, top), calibration, detail["refined"])
    detail["stage"] = "accepted"
    return measurement


def target_class_id(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    matches = [int(key) for key, name in items if name == TARGET_CLASS]
    if len(matches) != 1:
        raise MeasurementError("权重必须包含唯一的 a_box 类别")
    return matches[0]


def select_target_box(result, class_id, diagnostics=None):
    detail = {} if diagnostics is None else diagnostics
    detail.update(stage="detector", target_box_count=0)
    if result.boxes is None or len(result.boxes) == 0:
        raise MeasurementError("未检测到纸盒")
    boxes = result.boxes
    classes = boxes.cls.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    coords = boxes.xyxy.detach().cpu().numpy()
    if (not np.isfinite(classes).all() or not np.isfinite(confidence).all()
            or not np.isfinite(coords).all()):
        raise MeasurementError("检测结果包含非有限数值")
    indices = np.flatnonzero(classes == class_id)
    detail["target_box_count"] = len(indices)
    if len(indices) != 1:
        raise MeasurementError(f"a_box 检测数量={len(indices)}，需要唯一纸盒")
    index = int(indices[0])
    detail.update(bbox_xyxy=coords[index].tolist(), detection_confidence=float(confidence[index]))
    return coords[index].tolist(), float(confidence[index])


def load_detector(weights_path):
    # Keep pure geometry and JSON tests independent of torch/Ultralytics.
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def make_record(context, frame_id, captured_at, measurement=None, reason="ok"):
    record = {"schema_version": 1, "kind": "so101_box_grasp_target", **context,
              "frame_id": frame_id, "captured_at_unix_s": captured_at,
              "published_at_unix_s": time.time(), "valid": measurement is not None,
              "reason": reason, "coordinate_system": "table", "size_m": list(BOX_SIZE_M)}
    if measurement is not None:
        record.update(measurement)
    return record


def annotate_frame(frame, record):
    display = frame.copy()
    detail = record.get("diagnostics", {})
    bbox = detail.get("bbox_xyxy")
    if bbox is not None and np.asarray(bbox).shape == (4,) and all(value is not None for value in bbox):
        x1, y1, x2, y2 = (round(value) for value in bbox)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 1)
    shapes = [(f"C{item['contour_index']}", item) for item in detail.get("candidates", [])]
    if detail.get("refined"):
        shapes.append(("refined", detail["refined"]))
    for name, item in shapes:
        points = np.asarray(item.get("corners_image_px", []), dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        points = np.rint(points).astype(np.int32)
        color = (0, 210, 0) if item.get("valid") else (0, 80, 255)
        cv2.polylines(display, [points], True, color, 1)
        cv2.putText(display, name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        if name == "refined":
            for point in points:
                cv2.circle(display, tuple(point), 3, color, -1)
    label = "VALID top 40x40 mm / height 74 mm" if record["valid"] else "NO VALID TARGET"
    cv2.putText(display, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (0, 220, 0) if record["valid"] else (0, 0, 255), 1)
    cv2.putText(display, f"stage={detail.get('stage', 'unknown')}  s: save  q: stop",
                (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 100, 0), 1)
    return display


def save_vision_diagnostic(frame, annotated, record, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"frame_{record['frame_id']}_", dir=directory))
    for name, image in (("raw.png", frame), ("annotated.png", annotated)):
        if not cv2.imwrite(str(folder/name), image):
            raise OSError(f"诊断图片保存失败: {folder/name}")
    # Written last: a directory without this manifest is an incomplete snapshot.
    path = folder / "diagnostic.json"
    atomic_write_json(path, {"schema_version": 1, "kind": "so101_grasp_vision_diagnostic",
                            "saved_at_unix_s": time.time(), "raw_image": "raw.png",
                            "annotated_image": "annotated.png", "target_record": record})
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="识别74毫米高、40毫米方形顶面的纸盒并发布当前目标；不控制机械臂")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--image", type=Path, help="离线图片预览；记录明确标记为offline")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true", help="不显示相机窗口，Ctrl+C退出")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    output = args.output or workspace / "data" / "box_grasp_target.json"
    calibration_path = workspace / "data" / "calibration" / "table_plane_pose.npz"
    weights_path = workspace / "runs" / "detect" / "a_box_v4" / "weights" / "best.pt"
    protected_sources = [calibration_path, weights_path] + ([args.image] if args.image else [])
    for source_path in protected_sources:
        if (output.resolve() == source_path.resolve()
                or (output.exists() and source_path.exists() and os.path.samefile(output, source_path))):
            parser.error("--output 不能覆盖源图片、相机标定或YOLO权重（包含文件别名）")
    mode = "offline" if args.image else "live"
    context = {"source_mode": mode, "session_id": str(uuid4()),
               "source": {"image_path": str(args.image.resolve())} if args.image else {"camera_device": args.camera},
               "weights": {}, "calibration": {}}
    cap = None
    frame_id, captured_at, exit_reason = 0, None, "stopped"
    previous_sigterm = None
    try:
        atomic_write_json(output, make_record(context, 0, None, reason="starting"))
        calibration = load_calibration(calibration_path)
        context["calibration"] = calibration["provenance"]
        weights_hash = sha256(weights_path.read_bytes()).hexdigest()
        model = load_detector(weights_path)
        if sha256(weights_path.read_bytes()).hexdigest() != weights_hash:
            raise MeasurementError("加载期间模型权重发生变化")
        class_id = target_class_id(model.names)
        context["weights"] = {"path": str(weights_path), "sha256": weights_hash,
                              "class_id": class_id, "class_name": TARGET_CLASS}
        if args.image:
            source_path = args.image.resolve()
            before = source_path.stat()
            source_bytes = source_path.read_bytes()
            after = source_path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise MeasurementError("读取期间离线图片发生变化")
            frame = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise MeasurementError("离线图片读取失败")
            captured_at = before.st_mtime
            context["source"].update(image_mtime_unix_s=captured_at,
                                       image_sha256=sha256(source_bytes).hexdigest())
            frames = [(frame, captured_at)]
        else:
            def stop(signum, frame):
                raise KeyboardInterrupt
            previous_sigterm = signal.signal(signal.SIGTERM, stop)
            cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError("摄像头打开失败")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, calibration["image_size"][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration["image_size"][1])
            context["source"]["buffer_size_one_requested"] = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            context["source"]["timestamp_kind"] = "host_read_return_not_hardware_exposure"

            def captured_frames():
                while True:
                    ok, frame = cap.read()
                    # Host receipt time, not a hardware timestamp; precedes inference.
                    capture_time = time.time()
                    if not ok or frame is None:
                        raise RuntimeError("摄像头取帧失败")
                    yield frame, capture_time
            frames = captured_frames()
        previous_status = None
        last_print = 0.0
        for frame, captured_at in frames:
            frame_id += 1
            diagnostics = {"stage": "frame"}
            try:
                if (frame.shape[1], frame.shape[0]) != calibration["image_size"]:
                    raise MeasurementError("实际画面尺寸与标定尺寸不一致")
                result = model.predict(source=frame, conf=.25, imgsz=640, device="cpu",
                                       verbose=False, save=False)[0]
                bbox, confidence = select_target_box(result, class_id, diagnostics)
                measurement = extract_topface(frame, bbox, calibration, diagnostics)
                measurement.update(detection_confidence=confidence, bbox_xyxy=bbox)
                record = make_record(context, frame_id, captured_at, measurement)
            except MeasurementError as error:
                record = make_record(context, frame_id, captured_at, reason=str(error))
            record["diagnostics"] = diagnostics
            atomic_write_json(output, record)
            now = time.monotonic()
            status = (record["valid"], record["reason"])
            if status != previous_status or now-last_print >= 1:
                print(f"{mode} 帧{frame_id}: " +
                      (f"顶面中心(m)={np.round(record['center_m'], 4)}, 方向={record['square_yaw_deg']:.2f}°, "
                       f"边长最大误差={record['max_edge_error_mm']:.2f}mm" if record["valid"] else record["reason"]), flush=True)
                previous_status, last_print = status, now
                print(f"  stage={diagnostics['stage']} contours={diagnostics.get('contour_count', '-')} "
                      f"quads={diagnostics.get('convex_quad_count', '-')} "
                      f"accepted={diagnostics.get('candidate_count', '-')}", flush=True)
            if args.image:
                print(f"离线记录: {output}；源图时间={captured_at:.3f}，不是实时目标。", flush=True)
                return 0 if record["valid"] else 2
            if not args.headless:
                display = annotate_frame(frame, record)
                cv2.imshow("SO101 box target - q to stop", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    try:
                        path = save_vision_diagnostic(frame, display, record, workspace/"data"/"grasp_vision_debug")
                        print(f"同帧诊断已保存: {path}", flush=True)
                    except (OSError, cv2.error) as error:
                        print(f"诊断保存失败: {error}", file=sys.stderr, flush=True)
        return 0
    except KeyboardInterrupt:
        exit_reason = "interrupted"
        return 130
    except Exception as error:
        exit_reason = f"error: {error}"
        print(exit_reason, file=sys.stderr, flush=True)
        if mode == "offline":
            atomic_write_json(output, make_record(context, frame_id, captured_at, reason=exit_reason))
        return 1
    finally:
        try:
            if mode == "live":
                atomic_write_json(output, make_record(context, frame_id, captured_at, reason=exit_reason))
        finally:
            try:
                if cap is not None:
                    cap.release()
            finally:
                try:
                    if previous_sigterm is not None:
                        signal.signal(signal.SIGTERM, previous_sigterm)
                finally:
                    if not args.headless and mode == "live":
                        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
