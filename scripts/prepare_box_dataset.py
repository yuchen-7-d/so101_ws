from pathlib import Path
import random
import shutil


def main():
    workspace = Path(__file__).resolve().parents[1]
    samples_dir = workspace / 'data' / 'box_samples'
    dataset_dir = workspace / 'datasets' / 'a_box_v1'

    if dataset_dir.exists():
        raise FileExistsError(f'目录已存在:{dataset_dir}')

    image_files = sorted(samples_dir.glob('*.jpg'))
    random.Random(42).shuffle(image_files)

    splits = {
        'train': image_files[4:],
        'val': image_files[:4],
    }

    for split, files in splits.items():
        image_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in files:
            label_path = image_path.with_suffix('.txt')

            shutil.copy2(image_path, image_dir)
            shutil.copy2(label_path, label_dir)

        print(f'{split}:{len(files)}')


if __name__ == '__main__':
    main()
