"""Install handover display change; no serial access."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

root = Path('/home/yu/so101_ws')
stage = Path('/tmp/so101_elbow_handover_display_stage')
name = 'scripts/so101_elbow_probe.py'
expected = json.loads((stage/'baseline_sha256.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
for relative, value in expected.items():
    if sha((root/relative).read_bytes()) != value:
        raise SystemExit(f'File changed; no installation: {relative}')
destination = root/name
before = destination.read_bytes()
after = (stage/name).read_bytes()
candidate = (stage/'reviewed_sha256.txt').read_text().strip()
if sha(after) != candidate:
    raise SystemExit('Reviewed candidate changed; no installation.')
analysis = json.loads((stage/'run_analysis.json').read_text())
if sha(Path(analysis['source_path']).read_bytes()) != analysis['source_sha256']:
    raise SystemExit('Source run changed; no installation.')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
archive = Path(tempfile.mkdtemp(prefix='id3_handover_display_'+stamp,
                               dir=root/'data/real_control_runs'))
(archive/'so101_elbow_probe.before.py').write_bytes(before)
(archive/'so101_elbow_probe.reviewed.py').write_bytes(after)
for item in ('run_analysis.json','review.patch','test_results.txt','baseline_sha256.json','install_ui.py'):
    shutil.copy2(stage/item, archive/item)
fd, temp_name = tempfile.mkstemp(prefix='.elbow_display_', dir=destination.parent)
try:
    with os.fdopen(fd, 'wb') as output:
        output.write(after)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temp_name, destination.stat().st_mode & 0o777)
    os.replace(temp_name, destination)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
assert sha(destination.read_bytes()) == candidate
manifest = {
    'installed_at_utc': datetime.now(timezone.utc).isoformat(),
    'file': name, 'before_sha256': sha(before), 'after_sha256': candidate,
    'change': 'Handover feedback display and detailed baseline rejection message only.',
    'control_parameters_changed': False, 'real_serial_opened': False,
    'offline_tests_passed': 35,
}
(archive/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'archive':str(archive), **manifest}, ensure_ascii=False, indent=2))
