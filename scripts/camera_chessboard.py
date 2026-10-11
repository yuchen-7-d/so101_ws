import cv2
from pathlib import Path
from datetime import datetime


def main():


    pattern_size = (6,9)
    square_size_m = 0.027

    save_dir = ( 
        Path(__file__).resolve().parents[1]
        / 'data' / 'calibration' / 'images'
    )
    save_dir.mkdir(parents=True, exist_ok=True)

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(gray, pattern_size)

            display_frame = frame.copy()

            if found:
                cv2.drawChessboardCorners(
                    display_frame, pattern_size, corners, found
                )
            cv2.imshow('SO101 chessboard', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                if not found:
                    print('未找到完整棋盘')
                    continue

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                image_path = save_dir / f'chessboard_{timestamp}.png'

                saved = cv2.imwrite(str(image_path), frame)

                if not saved:
                    raise RuntimeError('标定照片保存失败')

                print(f'已保存:{image_path.name}')

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()