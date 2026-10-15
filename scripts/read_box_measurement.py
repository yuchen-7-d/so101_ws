from pathlib import Path
import json


def main():
    workspase = Path(__file__).resolve().parents[1]
    input_path = workspase / 'data' / 'box_top_measurement.json'

    json_text = input_path.read_text(encoding='utf-8')
    measurement = json.loads(json_text)

    if measurement is None:
        print('这份记录没有通过尺寸筛选的检测结果')
        return

    center_m = measurement['center_m']
    yaw_deg = measurement['square_yaw_deg']

    print(f'顶面中心:{center_m}')
    print(f'顶面方向:{yaw_deg:.2f}')


if __name__ == '__main__':
    main()
