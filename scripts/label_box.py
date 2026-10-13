from pathlib import Path
import cv2


def main():
    samples_dir = (
        Path(__file__).resolve().parents[1]
        /'data'
        /'box_samples_v4'
    )

    image_files = sorted(samples_dir.glob('*.jpg'))

    if not image_files:
        raise RuntimeError('没有找到纸盒样本')

    pending_images = [
        path for path in image_files
        if not path.with_suffix('.txt').exists()
    ]

    if not pending_images:
        print('所有样本已标注')
        return

    for image_path in pending_images:

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f'图片读取失败:{image_path}')

        try:

            label_path = image_path.with_suffix('.txt')

            cv2.imshow('sample preview', image)
            print(f'{image_path.name}: a=框选纸盒,b=确认无纸盒,q=退出')
            key = cv2.waitKey(0) & 0xFF

            if key == ord('b'):
                label_path.write_text('',encoding='utf-8')
                print(f'背景标签已保存:{label_path.name}')
                continue
            elif key == ord('q'):
                return
            elif key != ord('a'):
                continue

            cv2.destroyWindow('sample preview')

            x, y, w, h = cv2.selectROI(
                'label box',
                image,
                showCrosshair=True,
                fromCenter=False,
            )

            if w <= 0 or h <= 0:
                print('已取消框选')
                continue

            image_height, image_width = image.shape[:2]
            class_id = 0

            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            box_width = w / image_width
            box_height = h / image_height

            label_line = (
                f'{class_id} {x_center:.6f} {y_center:.6f} '
                f'{box_width:.6f} {box_height:.6f}\n'
            )

            label_path.write_text(label_line, encoding='utf-8')
            print(f'{label_path.name}:{label_line.strip()}')

        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
