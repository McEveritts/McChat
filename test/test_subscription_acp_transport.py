from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from open_webui.subscriptions.acp_client import (
    AcpClient,
    AcpError,
    AcpProcessError,
    AcpRequestError,
    DEFAULT_CLIENT_INFO,
    DEFAULT_CLIENT_CAPABILITIES,
    MAX_PENDING_REQUESTS,
)
from open_webui.subscriptions.antigravity_profile import (
    ANTIGRAVITY_AUTH_BROWSER_MARKER,
    ANTIGRAVITY_AUTH_STDOUT_PREFIX,
)
from open_webui.subscriptions.executor import (
    _CONCURRENCY_SEMAPHORE,
    MAX_STDERR_BYTES,
    MAX_STDIN_BYTES,
    MAX_STDOUT_BYTES,
    MAX_STREAM_LINE_BYTES,
)

log = logging.getLogger(__name__)


@pytest.fixture
def tmp_path():
    """Keep transport fixtures private without changing the runner environment."""
    base = Path.home() / '.tmp' / 'mcchat-subscriptions'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix='acp-transport-') as directory:
        yield Path(directory)


@pytest.mark.asyncio
async def test_replay_genuine_nativeinitialize_fixture(tmp_path):
    """Replay genuine native initialize request/response from audit-7 fixture

    verifying protocolVersion 2, truthful McChat client identity, and agentCapabilities.
    """
    replay_path = Path(__file__).parent / 'fixtures/subscriptions/antigravity-initialize-1.4.5.json'
    fixture = json.loads(replay_path.read_text())
    expected_response = fixture['response']

    resp_json = json.dumps(expected_response)

    script = f"""
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
assert req['method'] == 'initialize'
assert req['params']['protocolVersion'] == 2
assert req['params']['clientInfo']['name'] == 'mcchat'
resp = json.loads({json.dumps(resp_json)})
resp['id'] = req['id']
print(json.dumps(resp), flush=True)
sys.stdin.readline()
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        result = await client.initialize(timeout=5.0)

        assert result['protocolVersion'] == 2
        assert result['agentInfo']['name'] == 'antigravity-acp'
        assert result['agentInfo']['version'] == '1.4.5'
        assert result['agentCapabilities']['auth']['logout'] == {}
        assert client.proc is not None
        assert client.admitted is True
    finally:
        await client.close()
        assert client.admitted is False


@pytest.mark.asyncio
async def test_queued_admission_deadline_bounds_waiting(tmp_path):
    """When admission semaphore has 0 permits, spawn must obey the deadline and not hang."""
    saturated_sem = asyncio.Semaphore(0)
    client = AcpClient(timeout=0.1)
    with patch('open_webui.subscriptions.acp_client._CONCURRENCY_SEMAPHORE', saturated_sem):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await client.spawn([sys.executable, '-c', 'pass'], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        elapsed = time.monotonic() - start
        assert elapsed < 0.25, f'Admission wait exceeded deadline: took {elapsed:.3f}s'
        assert client.admitted is False

    # Also test spawn(timeout=0.1) override on client with larger overall timeout
    client2 = AcpClient(timeout=180)
    with patch('open_webui.subscriptions.acp_client._CONCURRENCY_SEMAPHORE', saturated_sem):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await client2.spawn(
                [sys.executable, '-c', 'pass'],
                env={'PATH': os.environ['PATH']},
                cwd=str(tmp_path),
                timeout=0.1,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 0.25, f'Spawn timeout override exceeded: took {elapsed:.3f}s'
        assert client2.admitted is False


@pytest.mark.asyncio
async def test_blocked_stdin_times_out_within_request_timeout(tmp_path):
    """When an ACP child stops reading stdin, request timeout must bound stdin drain and not hang."""
    script = 'import time; time.sleep(60)'
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        start = time.monotonic()
        # 600KB payload will fill OS pipe buffer and block in stdin.drain()
        with pytest.raises(TimeoutError):
            await client.request('session/prompt', {'text': 'x' * 600000}, timeout=0.1)
        elapsed = time.monotonic() - start
        assert elapsed < 0.25, f'Blocked stdin was not bounded by request timeout: took {elapsed:.3f}s'
    finally:
        await client.close()
        assert client.admitted is False


@pytest.mark.asyncio
async def test_long_frames_128kib_success_and_over_1mib_failure(tmp_path):
    """Test stdout frame limits: 128 KiB frame succeeds, while >1 MiB frame fails with ValueError."""
    script = """
import sys, json
while True:
    line = sys.stdin.readline()
    if not line:
        break
    req = json.loads(line)
    method = req.get('method')
    req_id = req.get('id')
    if method == 'ping_128k':
        resp = {'jsonrpc': '2.0', 'id': req_id, 'result': {'data': 'a' * 131072}}
        print(json.dumps(resp), flush=True)
    elif method == 'ping_huge':
        # Send frame exceeding 1 MiB (1024 * 1024 + 100 bytes)
        resp = {'jsonrpc': '2.0', 'id': req_id, 'result': {'data': 'b' * (1024 * 1024 + 100)}}
        print(json.dumps(resp), flush=True)
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))

        # 1. 128 KiB frame must succeed (default 64 KiB buffer would have failed)
        res_128k = await client.request('ping_128k', timeout=5.0)
        assert len(res_128k['data']) == 131072

        # 2. Frame exceeding 1 MiB must fail with ValueError
        with pytest.raises((ValueError, AcpProcessError)) as excinfo:
            await client.request('ping_huge', timeout=5.0)
        assert 'Stream frame size exceeded limit' in str(excinfo.value)
    finally:
        await client.close()
        assert client.admitted is False


@pytest.mark.asyncio
async def test_stderr_flood_newline_free_does_not_deadlock(tmp_path):
    """Child writing 4MB of newline-free stderr diagnostics must not deadlock client or stop drain."""
    script = """
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
# Write 4MB without any newline
sys.stderr.write('x' * 4000000)
sys.stderr.flush()
print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'protocolVersion': 2}}), flush=True)
sys.stdin.read()
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        res = await client.initialize(timeout=5.0)
        assert res['protocolVersion'] == 2

        # Check stderr drain is still running and bounded
        assert client.stderr_task is not None
        assert not client.stderr_task.done(), 'Stderr task terminated unexpectedly'
        assert client.stderr_bytes_captured <= MAX_STDERR_BYTES
    finally:
        await client.close()
        assert client.admitted is False


@pytest.mark.asyncio
async def test_total_output_cap_enforced(tmp_path):
    """Cumulative stdout exceeding MAX_STDOUT_BYTES (4 MiB) fails with ValueError."""
    # Child continuously sends 64 KiB notifications until stopped
    script = """
import sys, json, time
line = sys.stdin.readline()
req = json.loads(line)
for i in range(100):
    chunk = {'jsonrpc': '2.0', 'method': 'session/update', 'params': {'chunk': 'x' * 65536}}
    print(json.dumps(chunk), flush=True)
sys.stdin.read()
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        # Trigger child
        with pytest.raises((ValueError, AcpProcessError)) as excinfo:
            await client.request('start', timeout=5.0)
        assert 'total output exceeded limit' in str(excinfo.value).lower()
    finally:
        await client.close()
        assert client.admitted is False


@pytest.mark.asyncio
async def test_descendant_cleanup_on_leader_exit_0(tmp_path):
    """When leader process exits 0 after spawning a grandchild, client.close() kills grandchild."""
    pid_file = tmp_path / 'grandchild.pid'
    script = f"""
import subprocess, sys, pathlib
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path({json.dumps(str(pid_file))}).write_text(str(child.pid))
"""
    client = AcpClient(timeout=30)
    child_pid = None
    try:
        await client.spawn([sys.executable, '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))

        # Wait for leader to write pid file
        for _ in range(100):
            if pid_file.exists() and client.proc.returncode is not None:
                break
            await asyncio.sleep(0.02)

        child_pid = int(pid_file.read_text().strip())

        # Verify grandchild is currently alive
        os.kill(child_pid, 0)

        # Close client; must kill entire process group
        await client.close()

        # Grandchild must now be dead (short poll for kernel delivery)
        dead = False
        for _ in range(50):
            try:
                os.kill(child_pid, 0)
                await asyncio.sleep(0.02)
            except ProcessLookupError:
                dead = True
                break
        assert dead is True, 'Grandchild process survived client close'
        assert client.admitted is False
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await client.close()


@pytest.mark.asyncio
async def test_cancellations_at_spawn_request_close(tmp_path):
    """Verify clean resource and permit cleanup under cancellation at spawn, request, and close."""
    # 1. Cancellation during request
    script = """
import sys, json, time
line = sys.stdin.readline()
time.sleep(10)
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        task = asyncio.create_task(client.request('slow_op'))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(client.pending_requests) == 0
    finally:
        await client.close()
        assert client.admitted is False

    # 2. Cancellation during close
    client2 = AcpClient(timeout=30)
    await client2.spawn([sys.executable, '-c', 'import time; time.sleep(60)'], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
    close_task = asyncio.create_task(client2.close())
    await asyncio.sleep(0.01)  # Yield so close() starts execution
    close_task.cancel()
    try:
        await close_task
    except asyncio.CancelledError:
        pass
    assert client2.admitted is False


@pytest.mark.asyncio
async def test_no_residual_tasks_or_resource_warnings(tmp_path):
    """Ensure no unjoined helper tasks or pending futures remain after client close."""
    script = """
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'protocolVersion': 2}}), flush=True)
"""
    client = AcpClient(timeout=30)
    await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
    await client.initialize(timeout=5.0)

    stdout_task = client.stdout_task
    stderr_task = client.stderr_task

    await client.close()

    assert stdout_task.done()
    assert stderr_task.done()
    assert client.is_closed is True
    assert len(client.pending_requests) == 0
    assert client.admitted is False


@pytest.mark.asyncio
async def test_callbacks_sync_and_awaitable_with_backpressure_and_error_propagation(tmp_path):
    """Session update callbacks handle sync and awaitable callables, backpressure, and exception propagation."""
    # Part A: Awaitable callback with backpressure
    events = []

    async def async_callback(update):
        await asyncio.sleep(0.05)
        events.append(update.get('text'))

    script = """
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
for word in ['hello', 'world']:
    print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {'text': word}}), flush=True)
print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'stopReason': 'end_turn'}}), flush=True)
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn(
            [sys.executable, '-u', '-c', script],
            env={'PATH': os.environ['PATH']},
            cwd=str(tmp_path),
            on_session_update=async_callback,
        )
        res = await client.prompt('sess_1', 'say hi', timeout=5.0)
        assert res['stopReason'] == 'end_turn'
        assert events == ['hello', 'world']
    finally:
        await client.close()

    # Part B: Callback exception must propagate and fail pending RPCs, not be swallowed
    def exploding_callback(update):
        raise RuntimeError('Queue consumer failed')

    client2 = AcpClient(timeout=30)
    try:
        await client2.spawn(
            [sys.executable, '-u', '-c', script],
            env={'PATH': os.environ['PATH']},
            cwd=str(tmp_path),
            on_session_update=exploding_callback,
        )
        with pytest.raises(RuntimeError) as excinfo:
            await client2.prompt('sess_2', 'say hi', timeout=5.0)
        assert 'Queue consumer failed' in str(excinfo.value)
    finally:
        await client2.close()


@pytest.mark.asyncio
async def test_auth_marker_parsing_bounded_and_never_logs_raw_url(tmp_path):
    """Google auth URL marker parsed from stderr without logging raw URLs, and sanitized properly."""
    secret_url = 'https://accounts.google.com/o/oauth2/v2/auth?client_id=secret123&state=xyz'
    captured_urls = []

    async def auth_url_handler(url: str):
        captured_urls.append(url)

    script = f"""
import sys, json
marker = {json.dumps(ANTIGRAVITY_AUTH_BROWSER_MARKER)}
url = {json.dumps(secret_url)}
sys.stderr.write(marker + json.dumps(url) + '\\n')
sys.stderr.flush()
line = sys.stdin.readline()
req = json.loads(line)
print(json.dumps({{'jsonrpc': '2.0', 'id': req['id'], 'result': {{'protocolVersion': 2}}}}), flush=True)
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn(
            [sys.executable, '-u', '-c', script],
            env={'PATH': os.environ['PATH']},
            cwd=str(tmp_path),
            on_authorization_url=auth_url_handler,
        )
        await client.initialize(timeout=5.0)

        assert captured_urls == [secret_url]

        # Verify sanitized stderr does NOT contain raw secret URL
        clean_err = client._get_sanitized_stderr()
        assert secret_url not in clean_err
        assert '[REDACTED_AUTH_URL]' in clean_err or '[REDACTED_AUTH_MARKER]' in clean_err
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unsupported_callback_refusal_jsonrpc(tmp_path):
    """Agent-to-client request is safely rejected with JSON-RPC error -32601."""
    script = """
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
# Send callback request to client requiring response
print(json.dumps({'jsonrpc': '2.0', 'id': 99, 'method': 'fs/readFile', 'params': {'path': '/etc/passwd'}}), flush=True)
resp_line = sys.stdin.readline()
resp = json.loads(resp_line)
assert resp['id'] == 99
assert resp['error']['code'] == -32601
# Now respond to original initialize
print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'protocolVersion': 2}}), flush=True)
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        res = await client.initialize(timeout=5.0)
        assert res['protocolVersion'] == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_max_stdin_bytes_limit(tmp_path):
    """Attempting to send payload exceeding MAX_STDIN_BYTES (2 MiB) raises ValueError."""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-c', 'import time; time.sleep(60)'], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        huge_text = 'x' * (MAX_STDIN_BYTES + 1024)
        with pytest.raises(ValueError) as excinfo:
            await client.request('test', {'payload': huge_text})
        assert 'exceeds limit' in str(excinfo.value)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_protocol_version_negotiation_failure(tmp_path):
    """Negotiation of protocolVersion != 2 raises AcpProcessError."""
    script = """
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'protocolVersion': 1}}), flush=True)
"""
    client = AcpClient(timeout=30)
    try:
        await client.spawn([sys.executable, '-u', '-c', script], env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
        with pytest.raises(AcpProcessError) as excinfo:
            await client.initialize(timeout=5.0)
        assert 'unsupported protocolVersion' in str(excinfo.value)
    finally:
        await client.close()

@pytest.mark.asyncio
async def test_rpc_deadline_includes_write_and_response_wait():
    """A slow write must not grant a fresh response timeout afterward."""
    client = AcpClient(timeout=5)
    async def slow_send(*args, **kwargs):
        await asyncio.sleep(0.08)
    start = time.monotonic()
    try:
        with patch.object(client, '_send_raw', new=slow_send):
            with pytest.raises(TimeoutError):
                await client.request('slow-roundtrip', timeout=0.12)
        assert time.monotonic() - start < 0.18
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_concurrent_close_finishes_after_caller_cancelled(tmp_path):
    """Disconnect and auth cancellation must both wait until the same child is reaped."""
    import open_webui.subscriptions.acp_client as transport
    client = AcpClient(timeout=5)
    await client.spawn([sys.executable, '-c', 'import time; time.sleep(30)'],
                       env={'PATH': os.environ['PATH']}, cwd=str(tmp_path))
    entered = asyncio.Event()
    release = asyncio.Event()
    original_cleanup = transport._cleanup_process
    async def slow_cleanup(*args, **kwargs):
        entered.set()
        await release.wait()
        await original_cleanup(*args, **kwargs)
    with patch.object(transport, '_cleanup_process', new=slow_cleanup):
        first = asyncio.create_task(client.close())
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        try:
            assert not second.done()
            first.cancel()
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
    assert first.cancelled()
    assert client.proc.returncode is not None
    assert not client.admitted
