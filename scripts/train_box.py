from pathlib import Path
from ultralytics import YOLO


def main():
    workspace = Path(__file__).resolve().parents[1]

    model_path = workspace / 'models' / 'yolo11n.pt'
    data_path = workspace / 'datasets' / 'a_box_v1' / 'data.yaml'

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=100,
        imgsz=640,
        batch=4,
        device='cpu',
        workers=0,
        amp=False,
        project=str(workspace / 'runs' / 'detect'),
        name='a_box_v2',
    )


if __name__ == '__main__':
    main()
