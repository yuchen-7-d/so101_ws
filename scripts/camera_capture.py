from pathlib import Path
from datetime import datetime
import cv2


def main():
    image_path = (
        Path(__file__).resolve().parents[1]
        /'data'
        /'box.jpg'
    )
    samples_dir = image_path.parent / 'box_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

    try:
        if not cap.isOpened():
            raise RuntimeError('摄像头打开失败')

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        first_frame = True

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                raise RuntimeError('摄像头取帧失败')

            if first_frame:
                print(f'首次帧尺寸:{frame.shape}')
                first_frame = False

            cv2.imshow('SO101 camera', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                saved = cv2.imwrite(str(image_path), frame)

                if not saved:
                    raise RuntimeError('图片保存失败')

            elif key == ord('d'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                sample_path = samples_dir / f'box_{timestamp}.jpg'
                saved = cv2.imwrite(str(sample_path), frame)

                if not saved:
                    raise RuntimeError('样本保存失败')

                print(f'样本已保存至:{sample_path}')

            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
