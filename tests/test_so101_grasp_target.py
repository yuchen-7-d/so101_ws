"""Perception acceptance tests: fake detections/camera and synthetic geometry only."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import so101_grasp_target as target


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Boxes:
    def __init__(self, classes, coords=None):
        self.cls = Tensor(classes)
        self.conf = Tensor([.9] * len(classes))
        self.xyxy = Tensor(coords if coords is not None else [[290, 210, 350, 270]] * len(classes))

    def __len__(self):
        return len(self.cls.values)


class GraspTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calibration_path = self.root / "data" / "calibration" / "table_plane_pose.npz"
        self.calibration_path.parent.mkdir(parents=True)
        np.savez(self.calibration_path,
                 camera_matrix=np.array([[426., 0, 320], [0, 426., 240], [0, 0, 1]]),
                 dist_coeffs=np.zeros(5), rvec=np.zeros(3), tvec=np.array([0., 0., .5]),
                 image_size=np.array([640, 480]), transform_direction="board_to_camera", pose_rms=.2)
        self.calibration = target.load_calibration(self.calibration_path)
        weights = self.root / "runs" / "detect" / "a_box_v4" / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"fake weights for tests; never loaded")
        self.frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        self.corners = np.array([[300., 220.], [340., 220.], [340., 260.], [300., 260.]])
        self.measurement = target.measure_top_corners(self.corners, self.calibration)

    def test_synthetic_projection_is_known_top_center_and_rejects_false_squares(self):
        np.testing.assert_allclose(self.measurement["center_m"], [0, 0, -.074], atol=1e-12)
        self.assertAlmostEqual(self.measurement["square_yaw_deg"], 0)
        self.assertLess(self.measurement["max_edge_error_mm"], 1e-10)
        cases = (self.corners * [1.2, 1.0],
                 np.array([[300, 220], [340, 220], [360, 254.641], [320, 254.641]]),
                 self.corners[[0, 2, 1, 3]], np.full((4, 2), np.nan))
        for corners in cases:
            with self.subTest(corners=corners.tolist()):
                with self.assertRaises(target.MeasurementError):
                    target.measure_top_corners(corners, self.calibration)

    def test_calibration_rejects_wrong_transform_and_nonfinite_intrinsics(self):
        original = {"camera_matrix": self.calibration["camera_matrix"], "dist_coeffs": np.zeros(5),
                    "rvec": np.zeros(3), "tvec": np.array([0., 0., .5]), "image_size": [640, 480],
                    "transform_direction": "board_to_camera", "pose_rms": .2}
        for changed in ({"transform_direction": "camera_to_board"},
                        {"camera_matrix": np.full((3, 3), np.nan)}):
            with self.subTest(changed=changed):
                np.savez(self.calibration_path, **{**original, **changed})
                with self.assertRaises(target.MeasurementError):
                    target.load_calibration(self.calibration_path)

    def test_target_class_is_exact_and_multiple_boxes_are_ambiguous(self):
        self.assertEqual(target.target_class_id({0: "other", 1: "a_box"}), 1)
        with self.assertRaises(target.MeasurementError):
            target.target_class_id({0: "box"})
        result = SimpleNamespace(boxes=Boxes([0, 1]))
        bbox, confidence = target.select_target_box(result, 1)
        self.assertEqual(bbox, [290, 210, 350, 270])
        self.assertEqual(confidence, .9)
        for boxes in (Boxes([]), Boxes([0]), Boxes([1, 1]), Boxes([1], [[0, 0, np.nan, 2]])):
            with self.subTest(boxes=boxes.cls.values.tolist()):
                with self.assertRaises(target.MeasurementError):
                    target.select_target_box(SimpleNamespace(boxes=boxes), 1)

    def test_extraction_revalidates_refined_geometry_and_frame_size(self):
        # Expanded bbox origin is (282, 202), so the synthetic top starts at (18,18).
        contour = np.array([[[18, 18]], [[58, 18]], [[58, 58]], [[18, 58]]], dtype=np.int32)
        with patch.object(target.cv2, "findContours", return_value=([contour], None)), \
                patch.object(target.cv2, "cornerSubPix", side_effect=lambda gray, points, *args: points):
            measured = target.extract_topface(self.frame, [290, 210, 350, 270], self.calibration)
        np.testing.assert_allclose(measured["center_m"], [0, 0, -.074], atol=1e-12)
        with patch.object(target.cv2, "findContours", return_value=([contour], None)), \
                patch.object(target.cv2, "cornerSubPix", return_value=contour.astype(np.float32) * 1.5):
            with self.assertRaises(target.MeasurementError):
                target.extract_topface(self.frame, [290, 210, 350, 270], self.calibration)
        for frame, bbox in ((self.frame[:200], [290, 210, 350, 270]),
                            (self.frame, [0, 0, np.nan, 2]),
                            (self.frame, [700, 700, 710, 710])):
            with self.assertRaises(target.MeasurementError):
                target.extract_topface(frame, bbox, self.calibration)

    def test_no_unique_top_face_cannot_reuse_previous_coordinates(self):
        contour = np.array([[[18, 18]], [[58, 18]], [[58, 58]], [[18, 58]]], dtype=np.int32)
        for contours in ([], [contour, contour]):
            with patch.object(target.cv2, "findContours", return_value=(contours, None)):
                with self.assertRaises(target.MeasurementError):
                    target.extract_topface(self.frame, [290, 210, 350, 270], self.calibration)
        output = self.root / "out.json"
        context = {"source_mode": "live", "session_id": "test"}
        target.atomic_write_json(output, target.make_record(context, 1, 100., self.measurement))
        invalid = target.make_record(context, 2, 101., reason="lost detection")
        target.atomic_write_json(output, invalid)
        saved = json.loads(output.read_text())
        self.assertFalse(saved["valid"])
        self.assertNotIn("center_m", saved)
        self.assertNotIn("square_yaw_deg", saved)
        with self.assertRaises(ValueError):
            target.atomic_write_json(output, {"bad": float("nan")})
        self.assertEqual(json.loads(output.read_text()), invalid)
        self.assertEqual(list(self.root.glob(".out.json.*")), [])

    def test_offline_image_uses_source_mtime_and_never_opens_camera(self):
        source = self.root / "image.png"
        cv2.imwrite(str(source), self.frame)
        os.utime(source, (1234.5, 1234.5))
        output = self.root / "offline.json"
        model = SimpleNamespace(names={0: "a_box"}, predict=lambda **kw: [SimpleNamespace(boxes=Boxes([0]))])
        with patch.object(target, "load_detector", return_value=model), \
                patch.object(target, "extract_topface", return_value=dict(self.measurement)), \
                patch.object(target.cv2, "VideoCapture") as camera, \
                contextlib.redirect_stdout(io.StringIO()):
            result = target.main(["--workspace", str(self.root), "--image", str(source), "--output", str(output)])
        record = json.loads(output.read_text())
        self.assertEqual(result, 0)
        camera.assert_not_called()
        self.assertTrue(record["valid"])
        self.assertEqual(record["source_mode"], "offline")
        self.assertEqual(record["captured_at_unix_s"], 1234.5)
        self.assertEqual(record["source"]["image_mtime_unix_s"], 1234.5)
        self.assertEqual(record["calibration"]["sha256"], self.calibration["provenance"]["sha256"])
        self.assertEqual(record["weights"]["class_name"], "a_box")

    def test_live_invalidation_capture_time_and_exit_use_fake_camera(self):
        output = self.root / "live.json"
        target.atomic_write_json(output, {"valid": True, "center_m": [1, 2, 3]})
        now, records, calls = [0.], [], [0]
        real_write = target.atomic_write_json

        def write(path, record):
            records.append(json.loads(json.dumps(record)))
            return real_write(path, record)

        def read():
            now[0] += 100
            return True, self.frame.copy()

        def predict(**kwargs):
            now[0] += 10  # Inference is slower than capture; timestamp must remain capture time.
            calls[0] += 1
            return [SimpleNamespace(boxes=Boxes([0] if calls[0] == 1 else []))]

        camera = SimpleNamespace(isOpened=lambda: True, set=lambda *args: True,
                                 read=read, release=lambda: None)
        model = SimpleNamespace(names={0: "a_box"}, predict=predict)
        with patch.object(target, "load_detector", return_value=model), \
                patch.object(target, "extract_topface", return_value=dict(self.measurement)), \
                patch.object(target, "atomic_write_json", side_effect=write), \
                patch.object(target.time, "time", side_effect=lambda: now[0]), \
                patch.object(target.cv2, "VideoCapture", return_value=camera), \
                patch.object(target.cv2, "imshow"), patch.object(target.cv2, "destroyAllWindows"), \
                patch.object(target.cv2, "waitKey", side_effect=[-1, ord("q")]), \
                contextlib.redirect_stdout(io.StringIO()):
            result = target.main(["--workspace", str(self.root), "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual([record["valid"] for record in records], [False, True, False, False])
        accepted = records[1]
        self.assertEqual((accepted["captured_at_unix_s"], accepted["published_at_unix_s"]), (100., 110.))
        self.assertEqual(accepted["source_mode"], "live")
        self.assertEqual(accepted["frame_id"], 1)
        for record in (records[0], records[2], records[3]):
            self.assertNotIn("center_m", record)
        self.assertEqual(json.loads(output.read_text())["reason"], "stopped")

    def test_startup_error_invalidates_existing_live_target_without_camera(self):
        output = self.root / "old.json"
        target.atomic_write_json(output, {"valid": True, "center_m": [1, 2, 3]})
        with patch.object(target, "load_detector", side_effect=RuntimeError("bad model")), \
                patch.object(target.cv2, "VideoCapture") as camera, \
                contextlib.redirect_stderr(io.StringIO()):
            result = target.main(["--workspace", str(self.root), "--output", str(output), "--headless"])
        record = json.loads(output.read_text())
        self.assertEqual(result, 1)
        self.assertFalse(record["valid"])
        self.assertNotIn("center_m", record)
        self.assertIn("bad model", record["reason"])
        camera.assert_not_called()

    def test_offline_missing_image_and_no_detection_both_replace_old_target(self):
        output = self.root / "old.json"
        source = self.root / "image.png"
        model = SimpleNamespace(names={0: "a_box"}, predict=lambda **kw: [SimpleNamespace(boxes=Boxes([]))])
        for source_exists in (False, True):
            with self.subTest(source_exists=source_exists):
                if source_exists:
                    cv2.imwrite(str(source), self.frame)
                target.atomic_write_json(output, {"valid": True, "center_m": [1, 2, 3]})
                with patch.object(target, "load_detector", return_value=model), \
                        contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    result = target.main(["--workspace", str(self.root), "--image", str(source), "--output", str(output)])
                self.assertEqual(result, 2 if source_exists else 1)
                record = json.loads(output.read_text())
                self.assertFalse(record["valid"])
                self.assertNotIn("center_m", record)
                self.assertEqual(record["source_mode"], "offline")

    def test_output_cannot_overwrite_source_image_weights_calibration_or_alias(self):
        source = self.root / "image.png"
        cv2.imwrite(str(source), self.frame)
        alias = self.root / "image_alias.json"
        os.link(source, alias)
        weights = self.root / "runs" / "detect" / "a_box_v4" / "weights" / "best.pt"
        for output in (source, self.calibration_path, weights, alias):
            with self.subTest(output=str(output)):
                original = output.read_bytes()
                with patch.object(target, "load_detector") as detector, \
                        patch.object(target, "atomic_write_json") as write, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        target.main(["--workspace", str(self.root), "--image", str(source), "--output", str(output)])
                self.assertEqual(stopped.exception.code, 2)
                self.assertEqual(output.read_bytes(), original)
                detector.assert_not_called()
                write.assert_not_called()

    def test_failed_exit_invalidation_still_releases_camera_and_restores_signal(self):
        output = self.root / "live.json"
        camera = SimpleNamespace(isOpened=lambda: True, set=lambda *args: True,
                                 read=lambda: (True, self.frame.copy()), release=Mock())
        model = SimpleNamespace(names={0: "a_box"}, predict=lambda **kw: [SimpleNamespace(boxes=Boxes([0]))])
        real_write = target.atomic_write_json

        def write(path, record):
            if record["reason"] == "stopped":
                raise OSError("simulated output failure during cleanup")
            return real_write(path, record)

        with patch.object(target, "load_detector", return_value=model), \
                patch.object(target, "extract_topface", return_value=dict(self.measurement)), \
                patch.object(target, "atomic_write_json", side_effect=write), \
                patch.object(target.cv2, "VideoCapture", return_value=camera), \
                patch.object(target.cv2, "imshow"), \
                patch.object(target.cv2, "destroyAllWindows") as destroy, \
                patch.object(target.cv2, "waitKey", return_value=ord("q")), \
                patch.object(target.signal, "signal", wraps=target.signal.signal) as signal_handler, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "simulated output failure"):
                target.main(["--workspace", str(self.root), "--output", str(output)])
        camera.release.assert_called_once()
        destroy.assert_called_once()
        self.assertEqual(signal_handler.call_count, 2)

    def test_diagnostics_distinguish_coarse_and_refined_geometry_rejection(self):
        contour = np.array([[[18, 18]], [[58, 18]], [[58, 58]], [[18, 58]]], dtype=np.int32)
        for stage in ("candidate_filter", "refined_geometry"):
            detail = {}
            coarse = contour * 2 if stage == "candidate_filter" else contour
            with self.subTest(stage=stage), \
                    patch.object(target.cv2, "findContours", return_value=([coarse], None)), \
                    patch.object(target.cv2, "cornerSubPix", return_value=contour.astype(np.float32)*1.5):
                with self.assertRaises(target.MeasurementError):
                    target.extract_topface(self.frame, [290, 210, 350, 270], self.calibration, detail)
            self.assertEqual(detail["stage"], stage)
            self.assertEqual(detail["candidate_count"], 0 if stage == "candidate_filter" else 1)
            metrics = detail["candidates"][0] if stage == "candidate_filter" else detail["refined"]
            self.assertFalse(metrics["valid"])
            self.assertEqual(len(metrics["side_lengths_mm"]), 4)
            self.assertEqual(len(metrics["diagonal_lengths_mm"]), 2)
            if stage == "refined_geometry":
                self.assertTrue(detail["selected_coarse"]["valid"])
                self.assertTrue(any(value > 0 for value in detail["corner_shift_px"]))

    def test_diagnostics_are_bounded_and_nonfinite_geometry_remains_invalid(self):
        contour = np.array([[[18, 18]], [[58, 18]], [[58, 58]], [[18, 58]]], dtype=np.int32)
        detail = {}
        with patch.object(target.cv2, "findContours", return_value=([contour]*20, None)):
            with self.assertRaises(target.MeasurementError):
                target.extract_topface(self.frame, [290, 210, 350, 270], self.calibration, detail)
        self.assertEqual((detail["contour_count"], detail["candidate_count"]), (20, 20))
        self.assertEqual(len(detail["candidates"]), 12)
        detail = {}
        with self.assertRaises(target.MeasurementError):
            target.measure_top_corners(np.full((4, 2), np.nan), self.calibration, detail)
        self.assertFalse(detail["valid"])
        self.assertEqual(detail["corners_image_px"], [[None, None]]*4)
        json.dumps(detail, allow_nan=False)

    def test_s_snapshot_uses_displayed_frame_without_new_inference_or_target_write(self):
        output = self.root / "live.json"
        frames = [self.frame.copy(), self.frame.copy()]
        frames[0][0, 0], frames[1][0, 0] = [10, 20, 30], [30, 20, 10]
        camera = SimpleNamespace(isOpened=lambda: True, set=lambda *args: True,
                                 read=Mock(side_effect=[(True, frame) for frame in frames]), release=Mock())
        model = SimpleNamespace(names={0: "a_box"}, predict=Mock(return_value=[SimpleNamespace(boxes=Boxes([0]))]))
        real_save = target.save_vision_diagnostic

        def save(frame, display, record, directory):
            before = output.read_bytes()
            result = real_save(frame, display, record, directory)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(json.loads(before)["frame_id"], 1)
            return result

        with patch.object(target, "load_detector", return_value=model), \
                patch.object(target, "extract_topface", return_value=dict(self.measurement)), \
                patch.object(target, "save_vision_diagnostic", side_effect=save), \
                patch.object(target.cv2, "VideoCapture", return_value=camera), \
                patch.object(target.cv2, "imshow"), patch.object(target.cv2, "destroyAllWindows"), \
                patch.object(target.cv2, "waitKey", side_effect=[ord("s"), ord("q")]), \
                contextlib.redirect_stdout(io.StringIO()):
            result = target.main(["--workspace", str(self.root), "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual((camera.read.call_count, model.predict.call_count), (2, 2))
        metadata_paths = list((self.root/"data"/"grasp_vision_debug").glob("*/diagnostic.json"))
        self.assertEqual(len(metadata_paths), 1)
        path = metadata_paths[0]
        snapshot = json.loads(path.read_text())
        self.assertEqual(snapshot["kind"], "so101_grasp_vision_diagnostic")
        self.assertEqual(snapshot["target_record"]["frame_id"], 1)
        self.assertNotIn("valid", snapshot)
        np.testing.assert_array_equal(cv2.imread(str(path.parent/"raw.png")), frames[0])
        self.assertFalse(np.array_equal(cv2.imread(str(path.parent/"annotated.png")), frames[0]))

    def test_failed_snapshot_image_write_leaves_no_complete_manifest(self):
        record = target.make_record({"source_mode": "live", "session_id": "test"}, 12, 100., self.measurement)
        original = json.dumps(record, sort_keys=True)
        write_image = cv2.imwrite

        def fail_annotated(path, image):
            return False if Path(path).name == "annotated.png" else write_image(path, image)

        with patch.object(target.cv2, "imwrite", side_effect=fail_annotated):
            with self.assertRaises(OSError):
                target.save_vision_diagnostic(self.frame, self.frame, record, self.root/"debug")
        self.assertEqual(list((self.root/"debug").glob("*/diagnostic.json")), [])
        self.assertEqual(json.dumps(record, sort_keys=True), original)


if __name__ == "__main__":
    unittest.main()
