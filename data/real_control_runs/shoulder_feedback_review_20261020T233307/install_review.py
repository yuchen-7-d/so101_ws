"""Install reviewed diagnostic text and archive an offline historical analysis."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path('/home/yu/so101_ws')
STAGE = Path('/tmp/so101_probe_timeout_diagnostics_stage')
GEOMETRY = Path('/tmp/so101_shoulder_log_fk_ijyxou1a')
ARCHIVE = ROOT / 'data/real_control_runs/shoulder_feedback_review_20261020T233307'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((STAGE / 'baseline_sha256.json').read_text())
    for name, expected in baseline.items():
        path = ROOT / name
        actual = sha(path) if path.exists() else None
        if actual != expected:
            raise RuntimeError(f'Project changed since review: {name}')
    sources = {
        'viewer.py': (GEOMETRY / 'viewer.py', '956bdb782550cef526293e71e11ffc63bde2a168fbb0ee1ff0d968f436b55dcf'),
        'build_report.py': (GEOMETRY / 'build_report.py', 'b85170c9ead86f21bc5c2fa0e02bf3db51bff34eda0c248bef110bffadbe1b66'),
        'shoulder_feedback_fk.json': (GEOMETRY / 'shoulder_feedback_fk.json', '47e930a0f64e5b80eeb69fdd7003af0707eb45e823e6a8c59abf91ef487d9a4a'),
        'shoulder_feedback_comparison.png': (GEOMETRY / 'shoulder_feedback_comparison.png', 'da516547431ae602d9589ed29aa7976ade3731ebc1b37bce192f4bf29f5d4e6f'),
        'reviewed_so101_relative_probe.py': (STAGE / 'scripts/so101_relative_probe.py', '58583b0271eeab475f340a32e6bb766e1f23d619552e8f759e736e8a93e0b2f3'),
        'test_so101_probe_timeout_diagnostics.py': (STAGE / 'tests/test_so101_probe_timeout_diagnostics.py', '1e488e6ebc055cf37ee8d191d3692a7d9fe9ee5eba70eaa208aa26cf9d4ec39c'),
    }
    for name in ('review.patch', 'baseline_sha256.json', 'test_results.txt', 'baseline_protocol_comparison.json'):
        path = STAGE / name
        sources[name] = (path, sha(path))
    path = Path('/tmp/so101_shoulder_result_review/interpretation.json')
    sources[path.name] = (path, sha(path))
    for name, (path, expected) in sources.items():
        if sha(path) != expected:
            raise RuntimeError(f'Staged artifact changed: {name}')
    report = json.loads((GEOMETRY / 'shoulder_feedback_fk.json').read_text())
    for path, expected in report['source_sha256'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f'Offline input changed: {path}')
    ARCHIVE.mkdir(exist_ok=False)
    for name, (path, _) in sources.items():
        shutil.copy2(path, ARCHIVE / name)
    shutil.copy2(ROOT / 'scripts/so101_relative_probe.py', ARCHIVE / 'so101_relative_probe.before.py')
    shutil.copy2(__file__, ARCHIVE / 'install_review.py')
    # Install the new test first; the only changed production file is last.
    changes = [
        ('test_so101_probe_timeout_diagnostics.py', 'tests/test_so101_probe_timeout_diagnostics.py'),
        ('reviewed_so101_relative_probe.py', 'scripts/so101_relative_probe.py'),
    ]
    for source_name, target_name in changes:
        target = ROOT / target_name
        fd, temp = tempfile.mkstemp(prefix='.reviewed_', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write((ARCHIVE / source_name).read_bytes())
            os.chmod(temp, target.stat().st_mode & 0o777 if target.exists() else 0o644)
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        if sha(target) != sources[source_name][1]:
            raise RuntimeError(f'Installed hash mismatch: {target_name}')
    manifest = {
        'archive': str(ARCHIVE), 'hardware_access_by_installer': False,
        'changed_production_files': ['scripts/so101_relative_probe.py'],
        'scope': 'Timeout diagnostic text only; no control parameter, threshold, protocol or acceptance changes.',
        'tests_passed': 52,
        'artifacts_sha256': {name: expected for name, (_, expected) in sources.items()},
    }
    (ARCHIVE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
