#!/usr/bin/env python3
"""给候选规划器的旧姿态诊断绑定原映射；不访问串口、不运行规划。"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


PLANNER = 'scripts/so101_plan_grasp.py'
HISTORY = 'data/so101_initial_pose_mapping.json'
BACKUP = 'data/pose_mapping_checks/id2_mapping_upx7molo'
EXPECTED = {
    PLANNER: 'a03e536c0331df863a01b8c83d13df62b7c383128df9a3fe5785f06e669f17f0',
    'data/so101_joint_mapping.json': 'dd714b5934ce60d3c8162c64ce31ac72e76bf1c89ad139da81ea56fce92cdeb8',
    'data/so101_initial_pose.json': 'f91f9601f769ee23a489d7f50abfb00e872d22e82a373a1f8c3709db4aa44148',
    BACKUP + '/so101_joint_mapping.before.json': '716caa4a722dc165d870a8122efe68131a045e365e68dda7df05fc08cf24b150',
    BACKUP + '/migration.completed.json': '9ddcbf347cadde87b6394c534b19dfff03ee6671951e2022d78c4f529d10edf0',
}

HELPER = '''def load_historical_pose_mapping(workspace, pose_bytes):
    """原始姿态只用与它绑定的原映射解释；不得回退到当前映射。"""
    path = Path(workspace) / 'data/so101_initial_pose_mapping.json'
    source_bytes = path.read_bytes()
    record = json.loads(source_bytes)
    if (record.get('schema_version') != 1
            or record.get('kind') != 'so101_historical_pose_mapping'
            or record.get('scope') != 'historical_diagnostics_only'
            or record.get('execution_available') is not False):
        raise ValueError('历史姿态映射记录格式或用途不匹配')
    pose_hash = hashlib.sha256(pose_bytes).hexdigest()
    if record.get('pose_sha256') != pose_hash:
        raise ValueError('历史姿态文件已变化，不能沿用旧映射绑定')
    mapping = record['mapping']
    canonical = json.dumps(mapping, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    mapping_hash = hashlib.sha256(canonical).hexdigest()
    if record.get('mapping_canonical_sha256') != mapping_hash:
        raise ValueError('历史姿态映射内容与摘要不匹配')
    return mapping, {
        'source': str(path),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'pose_sha256': pose_hash,
        'mapping_canonical_sha256': mapping_hash,
        'scope': 'historical_diagnostics_only',
        'is_live_start': False,
    }


'''

REPLACEMENTS = [
    ('def mapping_diagnostics(mapping, pose, limits):',
     HELPER + 'def mapping_diagnostics(mapping, pose, limits):'),
    ("    pose = json.loads(pose_path.read_text())\n",
     "    pose_bytes = pose_path.read_bytes()\n"
     "    pose = json.loads(pose_bytes)\n"
     "    historical_mapping, historical_reference = load_historical_pose_mapping(workspace, pose_bytes)\n"),
    ('    diagnostics = mapping_diagnostics(mapping, pose, planner.limits)\n',
     '    diagnostics = mapping_diagnostics(historical_mapping, pose, planner.limits)\n'),
    ("            'historical_pose_diagnostics': diagnostics,\n",
     "            'historical_pose_diagnostics': diagnostics,\n"
     "            'historical_pose_mapping_reference': historical_reference,\n"),
    ("            'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()\n"
     "                      for p in (reference_path, mapping_path, pose_path)}}",
     "            'files': {**{str(p): hashlib.sha256(p.read_bytes()).hexdigest()\n"
     "                         for p in (reference_path, mapping_path, pose_path)},\n"
     "                      historical_reference['source']: historical_reference['source_sha256']}}"),
    ("        'data/so101_initial_pose.json', 'data/calibration/table_plane_pose.npz',\n",
     "        'data/so101_initial_pose.json', 'data/so101_initial_pose_mapping.json',\n"
     "        'data/calibration/table_plane_pose.npz',\n"),
    ("    for joint in record['historical_pose_diagnostics']:\n",
     "    print('历史姿态诊断采用与旧姿态绑定的原映射；它不是当前起点。')\n"
     "    for joint in record['historical_pose_diagnostics']:\n"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def patch_source(source):
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            raise ValueError('规划器结构不匹配，拒绝修改：' + old[:80])
        source = source.replace(old, new, 1)
    compile(source, PLANNER, 'exec')
    return source.encode('utf-8')


def original_source(patched):
    source = patched.decode('utf-8')
    for old, new in reversed(REPLACEMENTS):
        if source.count(new) != 1:
            raise ValueError('规划器版本不匹配，拒绝修改')
        source = source.replace(new, old, 1)
    if sha(source.encode('utf-8')) != EXPECTED[PLANNER]:
        raise ValueError('规划器不是已检查的原版或本次修正版')
    return source


def atomic_replace(path, content, mode=0o644):
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
            name = stream.name
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def prepare(workspace):
    sources = {}
    for relative, expected in EXPECTED.items():
        content = (workspace / relative).read_bytes()
        if relative != PLANNER and sha(content) != expected:
            raise ValueError('输入已变化，拒绝使用旧迁移依据：' + relative)
        sources[relative] = content
    before = sources[PLANNER]
    base_source = (before.decode('utf-8') if sha(before) == EXPECTED[PLANNER]
                   else original_source(before))
    patched = patch_source(base_source)
    old_mapping = json.loads(sources[BACKUP + '/so101_joint_mapping.before.json'])
    current_mapping = json.loads(sources['data/so101_joint_mapping.json'])
    expected_current = copy.deepcopy(old_mapping)
    expected_current['joint_references']['2']['position_count'] = 2005
    if (old_mapping['joint_references']['2']['position_count'] != 824
            or current_mapping != expected_current
            or current_mapping.get('model_mapping_verified') is not False):
        raise ValueError('映射变化并非已核对的 ID2 824→2005')
    # 核验关键计算函数完全没改，修改范围仅限历史来源、报告和输出文件保护。
    before_defs = {n.name: ast.dump(n) for n in ast.parse(base_source).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    after_defs = {n.name: ast.dump(n) for n in ast.parse(patched).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in before_defs.keys() - {'build_report', 'main'}:
        if before_defs[name] != after_defs[name]:
            raise ValueError('意外改动原计算函数：' + name)
    canonical = json.dumps(old_mapping, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    history = json_bytes({
        'schema_version': 1,
        'kind': 'so101_historical_pose_mapping',
        'scope': 'historical_diagnostics_only',
        'execution_available': False,
        'pose_path': 'data/so101_initial_pose.json',
        'pose_sha256': EXPECTED['data/so101_initial_pose.json'],
        'source_mapping_path': BACKUP + '/so101_joint_mapping.before.json',
        'source_mapping_sha256': EXPECTED[BACKUP + '/so101_joint_mapping.before.json'],
        'migration_record_path': BACKUP + '/migration.completed.json',
        'migration_record_sha256': EXPECTED[BACKUP + '/migration.completed.json'],
        'mapping_canonical_sha256': sha(canonical),
        'mapping': old_mapping,
        'note': '保存迁移前诊断的解释依据；不改写旧姿态，不证明历史或当前实机映射已标定。',
    })
    path = workspace / HISTORY
    if path.exists() and path.read_bytes() != history:
        raise ValueError('历史映射文件已存在且内容不同，拒绝覆盖：' + str(path))
    return sources, patched, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/yu/so101_ws'))
    parser.add_argument('--apply', action='store_true', help='备份后写入软件补丁；不启动 ROS 或控制器')
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    sources, patched, history = prepare(workspace)
    installed = (sources[PLANNER] == patched and (workspace / HISTORY).exists())
    if installed:
        print('状态：already_applied；文件已是本次修正版，没有重复修改。')
        return 0
    print('历史诊断：原始姿态 ID2=3692 配原参考824，保留原有 -111.75° 越界提示。')
    print('当前规划：继续使用 ID2 参考2005；模型角度、方向、关节限位及碰撞检查不变。')
    if not args.apply:
        print('状态：preview_only；加入 --apply 才写入这两份软件文件。')
        return 0
    # 先检查，再创建不可覆盖的备份；任何输入变化都不继续替换规划器。
    parent = workspace / 'data/pose_mapping_checks'
    backup = Path(tempfile.mkdtemp(prefix='history_mapping_', dir=parent))
    mode = stat.S_IMODE((workspace / PLANNER).stat().st_mode)
    atomic_replace(backup / 'so101_plan_grasp.before.py', sources[PLANNER], mode)
    atomic_replace(backup / 'so101_initial_pose_mapping.proposed.json', history)
    atomic_replace(backup / 'patch.prepared.json', json_bytes({
        'status': 'prepared', 'planner_before_sha256': sha(sources[PLANNER]),
        'planner_after_sha256': sha(patched), 'history_sha256': sha(history),
        'register_writes': 0, 'execution_available': False,
    }))
    for relative, original in sources.items():
        if (workspace / relative).read_bytes() != original:
            raise ValueError('准备期间输入发生变化；备份保留于 ' + str(backup))
    history_path = workspace / HISTORY
    if history_path.exists() and history_path.read_bytes() != history:
        raise ValueError('准备期间历史映射发生变化，拒绝覆盖')
    # 顺序有意如此：新数据先落盘，之后新规划器才能读取它。
    atomic_replace(history_path, history)
    atomic_replace(workspace / PLANNER, patched, mode)
    if ((workspace / PLANNER).read_bytes() != patched or history_path.read_bytes() != history):
        raise ValueError('写入后核验失败；请保留备份 ' + str(backup))
    record = {
        'status': 'historical_mapping_bound',
        'planner_sha256': sha(patched), 'history_sha256': sha(history),
        'history_pose_unchanged': (workspace / 'data/so101_initial_pose.json').read_bytes()
                                  == sources['data/so101_initial_pose.json'],
        'current_mapping_unchanged': (workspace / 'data/so101_joint_mapping.json').read_bytes()
                                     == sources['data/so101_joint_mapping.json'],
        'model_mapping_verified': False, 'execution_available': False,
        'register_writes': 0,
    }
    if not record['history_pose_unchanged'] or not record['current_mapping_unchanged']:
        raise ValueError('原姿态或当前映射被其他进程更改；请保留备份 ' + str(backup))
    atomic_replace(backup / 'patch.completed.json', json_bytes(record))
    print('备份目录：', backup)
    print('状态：historical_mapping_bound')
    print('规划器语法及计算函数保留检查通过。')
    print('旧姿态文件及当前映射保持原样；新增与旧姿态绑定的原映射副本。')
    print('未访问串口；寄存器写入=0；未启动 ROS、MuJoCo 或抓取。')
    print('完整记录：', backup / 'patch.completed.json')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'未完成：{type(exc).__name__}: {exc}')
        raise SystemExit(1)
