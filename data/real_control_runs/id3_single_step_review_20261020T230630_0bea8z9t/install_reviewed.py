"""Install reviewed diagnostic files and offline evidence; no serial imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path('/home/yu/so101_ws')
STAGE = Path('/tmp/so101_elbow_single_stage')
COMPARISON = Path('/tmp/so101_elbow_comparison_stage')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def replace_bytes(destination, data, mode):
    fd, name = tempfile.mkstemp(prefix='.elbow_install_', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)

plan = json.loads((STAGE/'installation_plan.json').read_text())
for name, expected in plan['existing_sha256'].items():
    actual = digest((ROOT/name).read_bytes())
    if actual != expected:
        raise SystemExit(f'Existing file changed; no installation: {name}')
prepared, originals = {}, {}
for name, expected in plan['candidate_sha256'].items():
    data = (STAGE/name).read_bytes()
    if digest(data) != expected:
        raise SystemExit(f'Candidate changed; no installation: {name}')
    destination = ROOT/name
    if name not in plan['existing_sha256'] and destination.exists():
        raise SystemExit(f'New destination already exists: {name}')
    prepared[name] = data
    originals[name] = ((destination.read_bytes(), destination.stat().st_mode & 0o777)
                       if destination.exists() else None)
for name, expected in plan['evidence_sha256'].items():
    path = Path(name)
    if digest(path.read_bytes()) != expected:
        raise SystemExit(f'Evidence changed; no installation: {path}')

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
archive = Path(tempfile.mkdtemp(prefix=f'id3_single_step_review_{stamp}',
                               dir=ROOT/'data/real_control_runs'))
for name, original in originals.items():
    if original is not None:
        saved = archive/'before'/name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(original[0])
for name, data in prepared.items():
    saved = archive/'reviewed'/name
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(data)
for source in ('comparison.json', 'analyze.py'):
    shutil.copy2(COMPARISON/source, archive/source)
for source in ('review.patch', 'test_results.txt', 'installation_plan.json', 'install_reviewed.py'):
    shutil.copy2(STAGE/source, archive/source)

changed = []
try:
    for name, data in prepared.items():
        mode = originals[name][1] if originals[name] else 0o644
        replace_bytes(ROOT/name, data, mode)
        changed.append(name)
    for name, expected in plan['candidate_sha256'].items():
        if digest((ROOT/name).read_bytes()) != expected:
            raise RuntimeError(f'Installed verification failed: {name}')
except BaseException:
    for name in reversed(changed):
        if originals[name] is None:
            (ROOT/name).unlink()
        else:
            replace_bytes(ROOT/name, *originals[name])
    raise

manifest = {
    **plan,
    'installed_at_utc': datetime.now(timezone.utc).isoformat(),
    'installed_sha256': {name: digest((ROOT/name).read_bytes()) for name in prepared},
    'offline_review': 'Independent review passed; mocked serial tests only.',
    'real_serial_opened': False,
    'real_motion_executed': False,
    'scope': 'Explicit single-step negative-eight diagnostic under reduced load; no grasp acceptance.',
}
(archive/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'archive': str(archive), 'installed_sha256': manifest['installed_sha256']},
                 ensure_ascii=False, indent=2))
