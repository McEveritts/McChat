from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import sys
import tempfile
import uuid
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from open_webui.subscriptions.executor import (
    SubprocessExecutor,
    _CONCURRENCY_SEMAPHORE,
    MAX_CONCURRENT_PROCESSES,
    MAX_STDIN_BYTES,
    MAX_STDOUT_BYTES,
    MAX_STDERR_BYTES,
    MAX_STREAM_LINE_BYTES,
    _LoopBoundSemaphore,
)


@pytest.mark.asyncio
async def test_paused_consumer_deadline_reaps_child_and_releases_permit():
    baseline = _CONCURRENCY_SEMAPHORE._value
    stream = SubprocessExecutor.execute_stream([
        sys.executable, '-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(30)'
    ], timeout=0.15)
    pid = int(await anext(stream))
    try:
        assert _CONCURRENCY_SEMAPHORE._value == baseline - 1
        # Do not pull another item or close the generator until after checking
        # independently enforced resource cleanup.
        await asyncio.sleep(0.65)
        assert not _is_pid_alive(pid)
        assert _CONCURRENCY_SEMAPHORE._value == baseline
        with pytest.raises(TimeoutError):
            await anext(stream)
    finally:
        await stream.aclose()
    assert _CONCURRENCY_SEMAPHORE._value == baseline


@pytest.mark.asyncio
async def test_admission_reuses_uncontended_permits():
    limiter = _LoopBoundSemaphore(2)
    await limiter.acquire()
    await limiter.acquire()
    pending = asyncio.create_task(limiter.acquire())
    try:
        await asyncio.sleep(0)
        assert not pending.done()
        limiter.release()
        await asyncio.wait_for(pending, timeout=1)
        assert limiter.locked()
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        limiter.release()
        limiter.release()


@pytest.fixture(scope='session', autouse=True)
def setup_safe_tmpdir():
    """Ensure safe TMPDIR under ~/.tmp/mcchat-subscriptions for executor tests."""
    base_tmp = os.path.expanduser('~/.tmp/mcchat-subscriptions')
    session_tmp = os.path.join(base_tmp, f'test_executor_{uuid.uuid4().hex[:8]}')
    os.makedirs(session_tmp, exist_ok=True)
    old_tmpdir = os.environ.get('TMPDIR')
    os.environ['TMPDIR'] = session_tmp
    yield session_tmp
    if old_tmpdir is not None:
        os.environ['TMPDIR'] = old_tmpdir
    else:
        os.environ.pop('TMPDIR', None)
    shutil.rmtree(session_tmp, ignore_errors=True)


@pytest.fixture(autouse=True)
def track_and_cleanup_processes(monkeypatch):
    """Track all subprocesses spawned during tests and guarantee aggressive cleanup

    even if assertion errors occur.
    """
    spawned_procs = []
    spawned_pgids = []

    orig_exec = asyncio.create_subprocess_exec

    async def tracking_exec(*args, **kwargs):
        proc = await orig_exec(*args, **kwargs)
        spawned_procs.append(proc)
        if proc.pid:
            spawned_pgids.append(proc.pid)
        return proc

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', tracking_exec)

    yield spawned_procs

    # Teardown: ensure every spawned group and proc is terminated
    for pgid in spawned_pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:
            pass

    for proc in spawned_procs:
        try:
            if proc.returncode is None:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:
            pass


def _is_pid_alive(pid: int) -> bool:
    """Check if a specific process PID is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _is_pgid_alive(pgid: int) -> bool:
    """Check if any process in process group PGID is still alive."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# ==============================================================================
# 1. Blocked stdin timeout & cancel with PID cleaned
# ==============================================================================

@pytest.mark.asyncio
async def test_blocked_stdin_stream_timeout_cleaned(track_and_cleanup_processes):
    """Child ignores stdin and sleeps. execute_stream must time out promptly and kill child."""
    child_code = "import time; time.sleep(60)"
    cmd = [sys.executable, "-c", child_code]
    large_input = "A" * (1024 * 1024)  # 1 MB input

    stream = SubprocessExecutor.execute_stream(cmd, input_data=large_input, timeout=0.2)
    start_time = asyncio.get_running_loop().time()

    with pytest.raises(TimeoutError):
        await anext(stream)

    elapsed = asyncio.get_running_loop().time() - start_time
    assert elapsed < 1.0, f"Expected prompt timeout within ~0.2-0.5s, took {elapsed}s"

    await asyncio.sleep(0.05)
    assert len(track_and_cleanup_processes) >= 1
    proc = track_and_cleanup_processes[0]
    assert not _is_pgid_alive(proc.pid), "Process group still alive after stream stdin timeout!"


@pytest.mark.asyncio
async def test_blocked_stdin_stream_cancel_cleaned(track_and_cleanup_processes):
    """External task cancellation during blocked stdin write cleans up child group."""
    child_code = "import time; time.sleep(60)"
    cmd = [sys.executable, "-c", child_code]
    large_input = "B" * (1024 * 1024)

    stream = SubprocessExecutor.execute_stream(cmd, input_data=large_input, timeout=10.0)

    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.08)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.05)
    assert len(track_and_cleanup_processes) >= 1
    proc = track_and_cleanup_processes[0]
    assert not _is_pgid_alive(proc.pid), "Process group still alive after stream stdin cancellation!"


@pytest.mark.asyncio
async def test_blocked_stdin_command_timeout_cleaned(track_and_cleanup_processes):
    """execute_command with blocked stdin times out and kills process group."""
    child_code = "import time; time.sleep(60)"
    cmd = [sys.executable, "-c", child_code]
    large_input = "C" * (1024 * 1024)

    start_time = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await SubprocessExecutor.execute_command(cmd, input_data=large_input, timeout=0.2)

    elapsed = asyncio.get_running_loop().time() - start_time
    assert elapsed < 1.0, f"Expected command timeout within ~0.2-0.5s, took {elapsed}s"

    await asyncio.sleep(0.05)
    assert len(track_and_cleanup_processes) >= 1
    proc = track_and_cleanup_processes[0]
    assert not _is_pgid_alive(proc.pid), "Process group still alive after command stdin timeout!"


# ==============================================================================
# 2. Stderr flood success bounded retained (no pipe deadlock)
# ==============================================================================

@pytest.mark.asyncio
async def test_stderr_flood_command_success_bounded():
    """Child emits 1.5MB stderr then stdout pong. Stderr must drain continuously to avoid pipe deadlock."""
    child_code = (
        "import sys\n"
        "sys.stderr.write('E' * 1572864)\n"  # 1.5 MB
        "sys.stderr.flush()\n"
        "sys.stdout.write('pong\\n')\n"
        "sys.stdout.flush()\n"
    )
    cmd = [sys.executable, "-c", child_code]

    start_time = asyncio.get_running_loop().time()
    code, out, err = await SubprocessExecutor.execute_command(cmd, timeout=3.0)
    elapsed = asyncio.get_running_loop().time() - start_time

    assert code == 0
    assert out.strip() == "pong"
    assert len(err.encode('utf-8')) <= MAX_STDERR_BYTES
    assert elapsed < 2.0, f"Command hung on stderr pipe buffer, took {elapsed}s"


@pytest.mark.asyncio
async def test_stderr_flood_stream_success():
    """Streaming command drains massive stderr without hanging."""
    child_code = (
        "import sys, time\n"
        "sys.stderr.write('S' * 1048576)\n"
        "sys.stderr.flush()\n"
        "sys.stdout.write('line1\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('S' * 524288)\n"
        "sys.stderr.flush()\n"
        "sys.stdout.write('line2\\n')\n"
        "sys.stdout.flush()\n"
    )
    cmd = [sys.executable, "-c", child_code]

    lines = []
    async for line in SubprocessExecutor.execute_stream(cmd, timeout=3.0):
        lines.append(line.strip())

    assert lines == ["line1", "line2"]


# ==============================================================================
# 3. 128KiB frame success and >1MiB failure
# ==============================================================================

@pytest.mark.asyncio
async def test_stream_frame_size_128kib_success():
    """128KiB frame succeeds because StreamReader limit is set to 1MiB (not default 64KiB)."""
    child_code = (
        "import sys\n"
        "sys.stdout.write('F' * 131072 + '\\n')\n"  # 128 KiB
        "sys.stdout.flush()\n"
    )
    cmd = [sys.executable, "-c", child_code]

    frames = []
    async for frame in SubprocessExecutor.execute_stream(cmd, timeout=2.0):
        frames.append(frame)

    assert len(frames) == 1
    assert len(frames[0]) == 131073  # 128 KiB + '\n'


@pytest.mark.asyncio
async def test_stream_frame_size_over_1mib_failure():
    """Frame exceeding 1MiB limit fails deterministically with ValueError and kills process."""
    child_code = (
        "import sys\n"
        "sys.stdout.write('X' * (1024 * 1024 + 1024) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    cmd = [sys.executable, "-c", child_code]

    with pytest.raises(ValueError) as exc_info:
        async for _ in SubprocessExecutor.execute_stream(cmd, timeout=2.0):
            pass

    assert "exceeded" in str(exc_info.value).lower()


# ==============================================================================
# 4. Output cap enforcement (command and stream)
# ==============================================================================

@pytest.mark.asyncio
async def test_total_output_cap_command():
    """Non-streaming command exceeding MAX_STDOUT_BYTES (4MB) raises ValueError."""
    child_code = (
        "import sys\n"
        "sys.stdout.write('O' * (5 * 1024 * 1024))\n"  # 5 MB
        "sys.stdout.flush()\n"
    )
    cmd = [sys.executable, "-c", child_code]

    with pytest.raises(ValueError) as exc_info:
        await SubprocessExecutor.execute_command(cmd, timeout=3.0)

    assert "stdout exceeded limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_total_output_cap_stream():
    """Streaming command with frames below 1MiB but total exceeding 4MB raises ValueError."""
    # 50 lines of 100KiB = 5 MB total
    child_code = (
        "import sys\n"
        "chunk = 'M' * (100 * 1024) + '\\n'\n"
        "for _ in range(50):\n"
        "    sys.stdout.write(chunk)\n"
        "    sys.stdout.flush()\n"
    )
    cmd = [sys.executable, "-c", child_code]

    with pytest.raises(ValueError) as exc_info:
        async for _ in SubprocessExecutor.execute_stream(cmd, timeout=3.0):
            pass

    assert "total output exceeded limit" in str(exc_info.value).lower()


# ==============================================================================
# 5. Nonzero exit handling
# ==============================================================================

@pytest.mark.asyncio
async def test_nonzero_exit_command():
    """execute_command returns exit code without raising, allowing adapters to handle errors."""
    child_code = (
        "import sys\n"
        "sys.stderr.write('provider error details\\n')\n"
        "sys.exit(42)\n"
    )
    cmd = [sys.executable, "-c", child_code]

    code, out, err = await SubprocessExecutor.execute_command(cmd, timeout=2.0)
    assert code == 42
    assert "provider error details" in err


@pytest.mark.asyncio
async def test_nonzero_exit_stream():
    """execute_stream raises RuntimeError on non-zero exit code with sanitized error."""
    child_code = (
        "import sys\n"
        "sys.stderr.write('stream failure /home/betheenvy/secret.key\\n')\n"
        "sys.exit(7)\n"
    )
    cmd = [sys.executable, "-c", child_code]

    with pytest.raises(RuntimeError) as exc_info:
        async for _ in SubprocessExecutor.execute_stream(cmd, timeout=2.0):
            pass

    msg = str(exc_info.value)
    assert "Subscription CLI failed with exit code 7" in msg
    assert "/home/betheenvy" not in msg  # Home path must be redacted


# ==============================================================================
# 6. Grandchildren cleanup (detached descendant, leader exit, timeout, cancel)
# ==============================================================================

@pytest.mark.asyncio
async def test_detached_descendant_successful_command_cleaned():
    """Leader spawns detached background process and exits 0 immediately.

    Grandchild must be terminated and reaped on command completion.
    """
    child_code = (
        "import subprocess, sys, os\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "sys.stdout.write(f'{os.getpid()}:{p.pid}\\n')\n"
        "sys.stdout.flush()\n"
        "sys.exit(0)\n"
    )
    cmd = [sys.executable, "-c", child_code]

    code, out, err = await SubprocessExecutor.execute_command(cmd, timeout=2.0)
    assert code == 0
    leader_pid, grandchild_pid = map(int, out.strip().split(":"))

    await asyncio.sleep(0.1)
    assert not _is_pid_alive(leader_pid), "Leader process remained alive!"
    assert not _is_pid_alive(grandchild_pid), "Detached grandchild remained alive after successful exit 0!"


@pytest.mark.asyncio
async def test_descendant_timeout_cleaned():
    """Leader spawns grandchild and both hang. Timeout must clean both up."""
    child_code = (
        "import subprocess, sys, os, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "sys.stdout.write(f'{os.getpid()}:{p.pid}\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    cmd = [sys.executable, "-c", child_code]

    # Use stream so we can read the PIDs before timing out
    stream = SubprocessExecutor.execute_stream(cmd, timeout=0.3)
    pids_line = await anext(stream)
    leader_pid, grandchild_pid = map(int, pids_line.strip().split(":"))

    with pytest.raises(TimeoutError):
        await anext(stream)

    await asyncio.sleep(0.1)
    assert not _is_pid_alive(leader_pid), "Leader alive after timeout!"
    assert not _is_pid_alive(grandchild_pid), "Grandchild alive after timeout!"


@pytest.mark.asyncio
async def test_descendant_generator_exit_cleaned():
    """Consumer breaking out of stream (GeneratorExit) kills leader and grandchild."""
    child_code = (
        "import subprocess, sys, os, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "sys.stdout.write(f'{os.getpid()}:{p.pid}\\n')\n"
        "sys.stdout.flush()\n"
        "while True:\n"
        "    sys.stdout.write('tick\\n')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.05)\n"
    )
    cmd = [sys.executable, "-c", child_code]

    leader_pid = None
    grandchild_pid = None

    async for line in SubprocessExecutor.execute_stream(cmd, timeout=5.0):
        if ":" in line:
            leader_pid, grandchild_pid = map(int, line.strip().split(":"))
        else:
            # Break after reading one tick
            break

    assert leader_pid is not None and grandchild_pid is not None
    await asyncio.sleep(0.1)
    assert not _is_pid_alive(leader_pid), "Leader alive after GeneratorExit!"
    assert not _is_pid_alive(grandchild_pid), "Grandchild alive after GeneratorExit!"


# ==============================================================================
# 7. Semaphore queue deadline & release after failure
# ==============================================================================

@pytest.mark.asyncio
async def test_semaphore_queue_deadline_and_release(monkeypatch):
    """Queued admission has bounded timeout and slots are released on failure."""
    # Force single concurrency slot
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr('open_webui.subscriptions.executor._CONCURRENCY_SEMAPHORE', sem)

    # Acquire the slot with a background sleep
    cmd_sleep = [sys.executable, "-c", "import time; time.sleep(0.8)"]
    task1 = asyncio.create_task(SubprocessExecutor.execute_command(cmd_sleep, timeout=2.0))
    await asyncio.sleep(0.05)

    # Attempt second command with small timeout (0.15s) while slot is held
    cmd_quick = [sys.executable, "-c", "print('fast')"]
    start_time = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError) as exc_info:
        await SubprocessExecutor.execute_command(cmd_quick, timeout=0.15)

    elapsed = asyncio.get_running_loop().time() - start_time
    assert elapsed < 0.6, f"Semaphore wait took too long ({elapsed}s)"
    assert "concurrency slot" in str(exc_info.value).lower()

    # Cancel task1 and verify slot is immediately released and usable
    task1.cancel()
    try:
        await task1
    except (asyncio.CancelledError, Exception):
        pass

    # Now run command should succeed immediately
    code, out, _ = await SubprocessExecutor.execute_command(cmd_quick, timeout=1.0)
    assert code == 0
    assert out.strip() == "fast"


# ==============================================================================
# 8. No raw prompt or stderr log leakage
# ==============================================================================

@pytest.mark.asyncio
async def test_no_raw_prompt_or_stderr_logged(caplog):
    """Logs must never contain input prompt text or raw process stderr."""
    secret_prompt = "SUPER_SECRET_USER_PROMPT_987654321"
    secret_err = "CRITICAL_INTERNAL_DIAGNOSTIC_ERR_123456"

    child_code = (
        "import sys\n"
        f"sys.stderr.write('{secret_err}\\n')\n"
        "sys.exit(1)\n"
    )
    cmd = [sys.executable, "-c", child_code]

    with caplog.at_level(logging.DEBUG):
        try:
            async for _ in SubprocessExecutor.execute_stream(cmd, input_data=secret_prompt, timeout=2.0):
                pass
        except RuntimeError:
            pass

    log_text = caplog.text
    assert secret_prompt not in log_text, "Secret prompt leaked into execution logs!"
    assert secret_err not in log_text, "Raw process stderr leaked into execution logs!"
