"""Install reviewed fixed-profile diagnostics; no serial imports or access."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

root = Path('/home/yu/so101_ws')
stage = Path('/tmp/so101_shoulder_relative_stage')
sha = lambda data: hashlib.sha256(data).hexdigest()
plan = json.loads((stage/'installation_plan.json').read_text())
for name, expected in plan['existing_sha256'].items():
    if expected is None:
        if (root/name).exists():
            raise SystemExit(f'Unexpected existing destination: {name}')
        continue
    if sha((root/name).read_bytes()) != expected:
        raise SystemExit(f'File changed; no installation: {name}')
prepared, originals = {}, {}
for name, expected in plan['candidate_sha256'].items():
    data = (stage/name).read_bytes()
    if sha(data) != expected:
        raise SystemExit(f'Candidate changed; no installation: {name}')
    destination = root/name
    if destination.exists() and name not in plan['existing_sha256']:
        raise SystemExit(f'Unexpected existing destination: {name}')
    prepared[name] = data
    originals[name] = ((destination.read_bytes(), destination.stat().st_mode & 0o777)
                       if destination.exists() else None)
for name, expected in plan['evidence_sha256'].items():
    if sha((stage/name).read_bytes()) != expected:
        raise SystemExit(f'Evidence changed; no installation: {name}')

def replace(destination, data, mode):
    fd, tmp = tempfile.mkstemp(prefix='.relative_probe_install_', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
archive = Path(tempfile.mkdtemp(prefix='shoulder_relative_review_'+stamp,
                               dir=root/'data/real_control_runs'))
for name, data in prepared.items():
    target = archive/'reviewed'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if originals[name] is not None:
        target = archive/'before'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(originals[name][0])
for name in [*plan['evidence_sha256'], 'installation_plan.json', 'install_reviewed.py']:
    shutil.copy2(stage/name, archive/name)
changed = []
try:
    for name, data in prepared.items():
        replace(root/name, data, originals[name][1] if originals[name] else 0o644)
        changed.append(name)
    for name, expected in plan['candidate_sha256'].items():
        if sha((root/name).read_bytes()) != expected:
            raise RuntimeError(f'Installed verification failed: {name}')
except BaseException:
    for name in reversed(changed):
        if originals[name] is None:
            (root/name).unlink()
        else:
            replace(root/name, *originals[name])
    raise
manifest = {**plan, 'installed_at_utc':datetime.now(timezone.utc).isoformat(),
            'real_serial_opened':False, 'real_motion_executed':False}
(archive/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'archive':str(archive),'installed_sha256':plan['candidate_sha256']},indent=2))
