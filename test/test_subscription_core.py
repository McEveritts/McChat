"""Keep database module initialization and its cached engines outside the test runner."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def test_subscription_core_cases_in_isolated_process():
    repo = Path(__file__).resolve().parents[1]
    scratch = Path.home() / '.tmp' / 'mcchat-subscriptions'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix='subscription-core-cases-') as tmp:
        env = dict(os.environ)
        home = Path(tmp) / 'home'
        data = Path(tmp) / 'data'
        (home / '.tmp' / 'mcchat-subscriptions').mkdir(parents=True)
        data.mkdir()
        env.update(
            HOME=str(home), TMPDIR=tmp, DATA_DIR=str(data),
            DATABASE_URL=f'sqlite:///{data / "test.db"}',
            ENABLE_DB_MIGRATIONS='false', USE_SLIM_DOCKER='true',
        )
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', 'test/subscription_core_cases.py'],
            cwd=repo, env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
