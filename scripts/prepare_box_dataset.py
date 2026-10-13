from pathlib import Path
import shutil


def main():
    workspace = Path(__file__).resolve().parents[1]
    train_samples_dir = workspace / 'data' / 'box_samples_v3'
    val_samples_dir = workspace / 'data' / 'box_samples_v3_val'
    extra_samples_dir = workspace / 'data' / 'box_samples_v4'
    dataset_dir = workspace / 'datasets' / 'a_box_v4'

    if dataset_dir.exists():
        raise FileExistsError(f'目录已存在:{dataset_dir}')

    splits = {
        'train': (
            sorted(train_samples_dir.glob('*.jpg'))
            + sorted(extra_samples_dir.glob('*.jpg'))
        ),
        'val': sorted(val_samples_dir.glob('*.jpg')),
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
