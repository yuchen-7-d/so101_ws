"""Update only ID2's software reference, using one audited device-change log.

Default: preview. --apply: back up and atomically replace the mapping JSON.
No serial, ROS, camera, MuJoCo or actuator access. No historical pose rewrite.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


ROOT = Path('/home/yu/so101_ws')
JOURNAL = ROOT/'data/real_control_runs/id2_offset_k0gboaq2.jsonl'
JOURNAL_SHA = '5b6dafa9f1371cd3d664622fb8646fe264129a53e2c8a21ae1a8cf4e33bf3889'
OLD_MAPPING_SHA = '716caa4a722dc165d870a8122efe68131a045e365e68dda7df05fc08cf24b150'
MAPPING = ROOT/'data/so101_joint_mapping.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def checked_proposal(journal_path):
    raw = journal_path.read_bytes()
    require(sha(raw) == JOURNAL_SHA, '日志不是已逐包复核的完整原始记录；停止')
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    require(rows[0]['stage'] == 'proposal_and_original_mapping' and rows[-1]['stage'] == 'final',
            '日志首尾不完整')
    proof, result = rows[0]['data'], rows[-1]['data']
    require(result['test'] == 'id2_fixed_offset_change'
            and result['status'] == 'offset_readback_verified_mapping_pending'
            and result['offset_readback_verified'] is True
            and result['release_confirmed'] is True and result['lock_confirmed'] is True
            and result['software_mapping_updated'] is False and result['journal_errors'] == [],
            '日志没有确认本次设备修改成功')
    require(result['cleanup']['readback'] == {'40': 0, '48': 0, '55': 1, '31': 3229},
            '偏移/扭矩/输出/锁定读回不符')
    old = proof['mapping_before_utf8'].encode('utf-8')
    require(sha(old) == OLD_MAPPING_SHA, '日志中的原映射与已复核原文件不同')
    original = json.loads(old)
    require(original['joint_references']['2']['position_count'] == 824
            and original['joint_directions']['2']['direction'] == 1
            and original['model_mapping_verified'] is False, '原映射条件不符')
    candidate = copy.deepcopy(original)
    candidate['joint_references']['2']['position_count'] = 2005
    new = encoded(candidate)
    observation = result['actual_position_after']
    angle = candidate['joint_references']['2']['model_angle_deg']+(observation-2005)*360/4096
    evidence = {
        'schema_version': 1, 'kind': 'id2_software_mapping_migration',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_journal': str(journal_path), 'source_journal_sha256': JOURNAL_SHA,
        'old_mapping_sha256': sha(old), 'new_mapping_sha256': sha(new),
        'only_changed_field': 'joint_references.2.position_count',
        'old_value': 824, 'new_value': 2005,
        'expected_id2_device_offset_raw': 3229, 'expected_id2_device_offset_signed': -1181,
        'device_configuration_source': 'saved audited journal; no live device check',
        'journal_position_counts': observation, 'journal_position_model_angle_deg': angle,
        'model_mapping_verified': False, 'power_cycle_persistence_verified': False,
        'execution_available': False, 'register_writes': 0,
        'historical_pose_files_modified': False,
        'old_plans_and_start_snapshots_reusable_with_new_mapping': False,
        'historical_pose_diagnostics_need_encoder_offset_epoch_handling': True,
    }
    return old, new, evidence


def sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_durable(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def update_mapping(mapping_path, journal_path, apply=False):
    old, new, evidence = checked_proposal(journal_path)
    require(not mapping_path.is_symlink() and mapping_path.is_file(), '映射路径不是普通文件，拒绝替换')
    current = mapping_path.read_bytes()
    evidence['mapping_path'] = str(mapping_path)
    if current == new:
        evidence['status'] = 'already_updated_no_write'
        return evidence
    require(current == old, '当前映射与备份不一致；不会覆盖其他修改')
    if not apply:
        evidence['status'] = 'preview_only'
        return evidence

    backup_root = mapping_path.parent/'pose_mapping_checks'
    backup_root.mkdir(parents=True, exist_ok=True)
    sync_dir(backup_root.parent)
    directory = Path(tempfile.mkdtemp(prefix='id2_mapping_', dir=backup_root))
    evidence['backup_directory'] = str(directory)
    print('备份目录：', directory, flush=True)
    create_durable(directory/'so101_joint_mapping.before.json', old)
    create_durable(directory/'so101_joint_mapping.proposed.json', new)
    create_durable(directory/'device_change_source.jsonl', journal_path.read_bytes())
    evidence['status'] = 'prepared_not_applied'
    create_durable(directory/'migration.prepared.json', encoded(evidence))
    sync_dir(directory)
    sync_dir(backup_root)
    require(sha((directory/'so101_joint_mapping.before.json').read_bytes()) == OLD_MAPPING_SHA
            and sha((directory/'device_change_source.jsonl').read_bytes()) == JOURNAL_SHA,
            '备份读回检查失败；映射未替换')

    fd, name = tempfile.mkstemp(prefix='.id2_mapping_', suffix='.tmp', dir=mapping_path.parent)
    pending = Path(name)
    replaced = False
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), stat.S_IMODE(mapping_path.stat().st_mode))
            stream.write(new)
            stream.flush()
            os.fsync(stream.fileno())
        require(not mapping_path.is_symlink() and mapping_path.read_bytes() == old,
                '准备期间映射发生变化；取消替换')
        require(sha(journal_path.read_bytes()) == JOURNAL_SHA, '准备期间日志发生变化；取消替换')
        os.replace(pending, mapping_path)
        replaced = True
        sync_dir(mapping_path.parent)
        require(mapping_path.read_bytes() == new, '替换后读回与预期不一致')
        evidence['status'] = 'mapping_updated'
        create_durable(directory/'migration.completed.json', encoded(evidence))
        sync_dir(directory)
    except BaseException:
        print('映射文件是否已执行原子替换：', replaced, flush=True)
        print('保留以上备份；本程序不自动回滚覆盖文件。', flush=True)
        raise
    finally:
        if pending.exists():
            pending.unlink()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, default=JOURNAL)
    parser.add_argument('--apply', action='store_true', help='备份后只更新软件 ID2 参考计数')
    args = parser.parse_args()
    try:
        result = update_mapping(MAPPING, args.journal.resolve(), args.apply)
        print('状态：', result['status'])
        print('ID2 软件参考计数：824 → 2005；参考角和方向保持不变。')
        print('model_mapping_verified：False')
        print('日志中的 ID2=957 对应模型角：', f"{result['journal_position_model_angle_deg']:.3f}°")
        print('映射摘要：', result['new_mapping_sha256'])
        if 'backup_directory' in result:
            print('完整记录：', Path(result['backup_directory'])/'migration.completed.json')
        print('未访问串口，寄存器写入=0；未修改历史姿态文件。')
        print('旧计划和旧起点快照不再配套；历史姿态诊断仍需适配计数编号变化。')
        return 0
    except Exception as error:
        print(f'更新未完成：{type(error).__name__}: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
