from pathlib import Path
from ultralytics import YOLO
import cv2


def main():
    workspace = Path(__file__).resolve().parents[1]
    image_path = workspace / 'data' / 'box_test.kpg'
    weights_path = (
        workspace / 'runs' / 'detect' / 'a_box_v2'
        / 'weights' / 'best.pt'
    )

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError('图片读取失败')

    model = YOLO(weights_path)
    result = model.predict(
        source=model,
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
    x1, x2, y1, y2 = boxes.xyxy[index].tolist()
    print(f'纸盒框:({x1:.1f},{x2:.1f},{y1:.1f},{y2:.1f})')


if __name__ == '__main__':
    main()
