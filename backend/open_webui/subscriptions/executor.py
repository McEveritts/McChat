from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import AsyncGenerator

from open_webui.subscriptions.security import sanitize_error_message

log = logging.getLogger(__name__)


def _parse_bounded_int(env_var: str, default: int, min_val: int, max_val: int) -> int:
    raw = os.getenv(env_var)
    if not raw:
        return default
    try:
        val = int(raw.strip())
        return max(min_val, min(val, max_val))
    except (ValueError, TypeError):
        return default


# Concurrency ceiling and memory/output bounds with valid environment bounds
MAX_CONCURRENT_PROCESSES = _parse_bounded_int('MAX_CONCURRENT_SUBSCRIPTIONS', 4, 1, 64)
MAX_STDIN_BYTES = _parse_bounded_int('MAX_SUBSCRIPTION_STDIN_BYTES', 2 * 1024 * 1024, 1024, 32 * 1024 * 1024)      # 2 MB limit for prompt input
MAX_STDOUT_BYTES = _parse_bounded_int('MAX_SUBSCRIPTION_STDOUT_BYTES', 4 * 1024 * 1024, 1024, 64 * 1024 * 1024)   # 4 MB ceiling for non-streaming output & total stream
MAX_STDERR_BYTES = _parse_bounded_int('MAX_SUBSCRIPTION_STDERR_BYTES', 512 * 1024, 1024, 16 * 1024 * 1024)        # 512 KB ceiling for stderr capture
MAX_STREAM_LINE_BYTES = _parse_bounded_int('MAX_SUBSCRIPTION_STREAM_FRAME_BYTES', 1024 * 1024, 1024, 16 * 1024 * 1024)  # 1 MB per streamed line


class _LoopBoundSemaphore:
    """An asyncio.Semaphore-compatible manager that provides a semaphore bound to

    the currently running event loop. This ensures that cross-test event loop
    lifecycles in pytest or server restarts do not reuse or leak dead-loop semaphores.
    """

    def __init__(self, value: int):
        self._default_value = value
        self._semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    def get_semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Semaphore(self._default_value)

        # Prune closed loops
        dead_loops = [l for l in self._semaphores if l.is_closed()]
        for l in dead_loops:
            self._semaphores.pop(l, None)

        sem = self._semaphores.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self._default_value)
            self._semaphores[loop] = sem
        return sem

    @property
    def _value(self) -> int:
        return self.get_semaphore()._value

    async def acquire(self) -> bool:
        return await self.get_semaphore().acquire()

    def release(self) -> None:
        self.get_semaphore().release()

    def locked(self) -> bool:
        return self.get_semaphore().locked()

    async def __aenter__(self):
        await self.acquire()
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


_CONCURRENCY_SEMAPHORE = _LoopBoundSemaphore(MAX_CONCURRENT_PROCESSES)


def _terminate_pgid_sync(pgid: int) -> None:
    """Send SIGTERM to process group."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    except Exception as exc:
        log.debug('Failed to send SIGTERM to process group %s: %s', pgid, exc)


def _kill_pgid_sync(pgid: int) -> None:
    """Send SIGKILL to process group, terminating all descendants."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    except Exception as exc:
        log.debug('Failed to send SIGKILL to process group %s: %s', pgid, exc)


async def _cleanup_process(proc: asyncio.subprocess.Process, pgid: int, grace_period: float = 0.3) -> None:
    """Safely terminate the process group and reap child process to prevent zombie leaks.

    Ensures descendants that ignore SIGTERM receive SIGKILL even if the group leader has exited.
    Explicitly handles and suppresses asyncio.CancelledError during cleanup waits to ensure
    termination guarantees cannot be interrupted by caller cancellation.
    """
    if proc.stdin and not proc.stdin.is_closing():
        try:
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.debug('Error closing stdin in cleanup: %s', exc)

    # 1. Send SIGTERM to process group
    _terminate_pgid_sync(pgid)

    # 2. Wait bounded grace period for leader to exit gracefully
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_period)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception as exc:
            log.debug('Error awaiting proc exit in grace period: %s', exc)

    # 3. Unconditionally send SIGKILL to the process group to reap any lingering descendants
    _kill_pgid_sync(pgid)

    # 4. Final bounded wait to reap leader process
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception as exc:
            log.debug('Error awaiting proc reap: %s', exc)

    # 5. Ensure open transports and pipes are closed to prevent loop teardown warnings
    if proc._transport and not proc._transport.is_closing():
        try:
            proc._transport.close()
        except Exception as exc:
            log.debug('Error closing subprocess transport: %s', exc)


class SubprocessExecutor:
    """Bounded, secure async subprocess runner with process group isolation,

    resource ceilings, concurrent stream draining, and prompt/error sanitization.
    """

    @classmethod
    async def execute_command(
        cls,
        cmd: list[str],
        input_data: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 180,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a non-streaming command within bounded memory, time, and concurrency budgets."""
        if not cmd:
            raise ValueError('Command list cannot be empty')

        safe_bin = os.path.basename(cmd[0]) if cmd else 'unknown'
        log.debug('Executing subscription command: %s', safe_bin)

        if input_data is not None and len(input_data.encode('utf-8')) > MAX_STDIN_BYTES:
            raise ValueError(f'Input prompt size exceeds maximum allowed limit ({MAX_STDIN_BYTES} bytes)')

        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout)

        sem = _CONCURRENCY_SEMAPHORE
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f'Command timed out waiting for concurrency slot after {timeout} seconds')

        acquired = False
        proc = None
        pgid = None
        all_tasks: list[asyncio.Task] = []

        try:
            try:
                acquired = await asyncio.wait_for(sem.acquire(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f'Command timed out waiting for concurrency slot after {timeout} seconds')

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f'Command timed out before process spawn after {timeout} seconds')

            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=cwd,
                    limit=MAX_STREAM_LINE_BYTES,
                    start_new_session=True,
                ),
                timeout=remaining,
            )
            pgid = proc.pid

            stdout_chunks: list[bytes] = []
            stderr_chunks: list[bytes] = []
            stdout_len = 0
            stderr_retained_len = 0

            async def _write_stdin():
                if input_data is None or proc.stdin is None:
                    return
                try:
                    proc.stdin.write(input_data.encode('utf-8'))
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
                    pass
                except Exception as exc:
                    log.debug('Error writing stdin: %s', exc)
                finally:
                    try:
                        if proc.stdin and not proc.stdin.is_closing():
                            proc.stdin.close()
                        try:
                            await asyncio.wait_for(proc.stdin.wait_closed(), timeout=0.2)
                        except (asyncio.TimeoutError, asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                            pass
                    except Exception as exc:
                        log.debug('Error closing stdin: %s', exc)

            async def _read_stdout():
                nonlocal stdout_len
                if proc.stdout is None:
                    return
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    stdout_chunks.append(chunk)
                    stdout_len += len(chunk)
                    if stdout_len > MAX_STDOUT_BYTES:
                        raise ValueError(f'Process stdout exceeded limit of {MAX_STDOUT_BYTES} bytes')

            async def _read_stderr():
                nonlocal stderr_retained_len
                if proc.stderr is None:
                    return
                while True:
                    chunk = await proc.stderr.read(65536)
                    if not chunk:
                        break
                    if stderr_retained_len < MAX_STDERR_BYTES:
                        take = min(len(chunk), MAX_STDERR_BYTES - stderr_retained_len)
                        stderr_chunks.append(chunk[:take])
                        stderr_retained_len += take

            async def _watch_leader_exit():
                """If leader process exits while descendants keep stdout/stderr open,

                terminate the process group so reader tasks reach EOF promptly.
                """
                while True:
                    await asyncio.sleep(0.05)
                    if proc.returncode is not None:
                        break
                    try:
                        os.kill(pgid, 0)
                    except ProcessLookupError:
                        break
                await asyncio.sleep(0.05)
                _kill_pgid_sync(pgid)

            watch_task = asyncio.create_task(_watch_leader_exit())
            stdin_task = asyncio.create_task(_write_stdin())
            stdout_task = asyncio.create_task(_read_stdout())
            stderr_task = asyncio.create_task(_read_stderr())
            wait_task = asyncio.create_task(proc.wait())

            primary_tasks = [stdout_task, stderr_task, wait_task]
            all_tasks = [stdin_task, stdout_task, stderr_task, wait_task, watch_task]

            remaining = max(0.01, deadline - loop.time())
            done, pending = await asyncio.wait(all_tasks, timeout=remaining, return_when=asyncio.FIRST_EXCEPTION)

            if not done and pending:
                raise asyncio.TimeoutError()

            for t in done:
                if t.exception() is not None:
                    raise t.exception()

            primary_pending = [t for t in primary_tasks if not t.done()]
            if primary_pending:
                remaining = max(0.01, deadline - loop.time())
                done2, pending2 = await asyncio.wait(primary_pending, timeout=remaining, return_when=asyncio.ALL_COMPLETED)
                for t in done2:
                    if t.exception() is not None:
                        raise t.exception()
                if pending2:
                    raise asyncio.TimeoutError()

            returncode = proc.returncode if proc.returncode is not None else 0
            stdout_str = b''.join(stdout_chunks).decode('utf-8', errors='replace')
            stderr_str = b''.join(stderr_chunks).decode('utf-8', errors='replace')
            return returncode, stdout_str, stderr_str

        except TimeoutError:
            log.warning('Command timed out after %s seconds: %s', timeout, safe_bin)
            raise
        except asyncio.CancelledError:
            log.info('Subprocess execution cancelled: %s', safe_bin)
            raise
        except Exception:
            raise
        finally:
            if proc is not None:
                curr_pgid = pgid or proc.pid
                await _cleanup_process(proc, curr_pgid)

            for t in all_tasks:
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception) as exc:
                        log.debug('Helper task join during cleanup: %s', exc)

            if acquired:
                sem.release()

    @classmethod
    async def execute_stream(
        cls,
        cmd: list[str],
        input_data: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 180,
        cwd: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream output line-by-line with process-group cleanup, bounded memory, and error propagation."""
        if not cmd:
            raise ValueError('Command list cannot be empty')

        safe_bin = os.path.basename(cmd[0]) if cmd else 'unknown'
        log.debug('Executing subscription stream: %s', safe_bin)

        if input_data is not None and len(input_data.encode('utf-8')) > MAX_STDIN_BYTES:
            raise ValueError(f'Input prompt size exceeds maximum allowed limit ({MAX_STDIN_BYTES} bytes)')

        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout)

        sem = _CONCURRENCY_SEMAPHORE
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f'Command stream timed out waiting for concurrency slot after {timeout} seconds')

        acquired = False
        proc = None
        pgid = None
        stdin_task = None
        stderr_task = None
        watch_task = None
        deadline_task = None
        cleanup_task = None

        async def _close_stream_resources():
            nonlocal acquired
            try:
                if proc is not None:
                    await _cleanup_process(proc, pgid or proc.pid)
            finally:
                for task in (stdin_task, stderr_task, watch_task):
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                if acquired:
                    sem.release()
                    acquired = False

        async def _ensure_stream_closed():
            nonlocal cleanup_task
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(_close_stream_resources())
            cancelled = False
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup_task.result()
            if cancelled:
                raise asyncio.CancelledError

        async def _expire_stream():
            await asyncio.sleep(max(0, deadline - loop.time()))
            await _ensure_stream_closed()

        try:
            try:
                acquired = await asyncio.wait_for(sem.acquire(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f'Command stream timed out waiting for concurrency slot after {timeout} seconds')

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f'Command stream timed out before process spawn after {timeout} seconds')

            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=cwd,
                    limit=MAX_STREAM_LINE_BYTES,
                    start_new_session=True,
                ),
                timeout=remaining,
            )
            pgid = proc.pid

            stderr_chunks: list[bytes] = []
            stderr_retained_len = 0

            async def _feed_stdin():
                if input_data is None or proc.stdin is None:
                    return
                try:
                    proc.stdin.write(input_data.encode('utf-8'))
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
                    pass
                except Exception as exc:
                    log.debug('Error writing stream stdin: %s', exc)
                finally:
                    try:
                        if proc.stdin and not proc.stdin.is_closing():
                            proc.stdin.close()
                        try:
                            await asyncio.wait_for(proc.stdin.wait_closed(), timeout=0.2)
                        except (asyncio.TimeoutError, asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                            pass
                    except Exception as exc:
                        log.debug('Error closing stream stdin: %s', exc)

            async def _drain_stderr():
                nonlocal stderr_retained_len
                if proc.stderr is None:
                    return
                try:
                    while True:
                        chunk = await proc.stderr.read(65536)
                        if not chunk:
                            break
                        if stderr_retained_len < MAX_STDERR_BYTES:
                            take = min(len(chunk), MAX_STDERR_BYTES - stderr_retained_len)
                            stderr_chunks.append(chunk[:take])
                            stderr_retained_len += take
                except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    log.debug('Error draining stream stderr: %s', exc)

            async def _watch_leader_exit_stream():
                while True:
                    await asyncio.sleep(0.05)
                    if proc.returncode is not None:
                        break
                    try:
                        os.kill(pgid, 0)
                    except ProcessLookupError:
                        break
                await asyncio.sleep(0.05)
                _kill_pgid_sync(pgid)

            stdin_task = asyncio.create_task(_feed_stdin())
            stderr_task = asyncio.create_task(_drain_stderr())
            watch_task = asyncio.create_task(_watch_leader_exit_stream())
            # A yield suspends this generator while the HTTP consumer writes.
            # Resource deadlines must remain active without another __anext__.
            deadline_task = asyncio.create_task(_expire_stream())

            total_output_bytes = 0
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f'Command stream timed out after {timeout} seconds')

                try:
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=remaining,
                    )
                except ValueError as ve:
                    raise ValueError(f'Stream frame size exceeded limit of {MAX_STREAM_LINE_BYTES} bytes') from ve

                if loop.time() >= deadline:
                    raise TimeoutError(f'Command stream timed out after {timeout} seconds')
                if not line_bytes:
                    break

                if len(line_bytes) > MAX_STREAM_LINE_BYTES:
                    raise ValueError(f'Stream frame size exceeded limit of {MAX_STREAM_LINE_BYTES} bytes')

                total_output_bytes += len(line_bytes)
                if total_output_bytes > MAX_STDOUT_BYTES:
                    raise ValueError(f'Stream total output exceeded limit of {MAX_STDOUT_BYTES} bytes')

                yield line_bytes.decode('utf-8', errors='replace')

            # Wait for process exit within remaining deadline
            remaining = max(0.01, deadline - loop.time())
            try:
                await asyncio.wait_for(proc.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f'Command stream timed out waiting for process exit after {timeout} seconds')

            # Wait briefly for stderr drain to finish
            if stderr_task and not stderr_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stderr_task), timeout=min(0.5, remaining))
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as exc:
                    log.debug('Error waiting for stderr task in stream: %s', exc)

            if proc.returncode != 0 and proc.returncode is not None:
                raw_err = b''.join(stderr_chunks).decode('utf-8', errors='replace').strip()
                sanitized_err = sanitize_error_message(raw_err or f'exit code {proc.returncode}')
                log.warning('Stream process exited with non-zero code %d', proc.returncode)
                raise RuntimeError(f'Subscription CLI failed with exit code {proc.returncode}: {sanitized_err}')

        except TimeoutError:
            log.warning('Command stream timed out after %s seconds: %s', timeout, safe_bin)
            raise
        except (asyncio.CancelledError, GeneratorExit):
            log.info('Command stream cancelled or consumer closed: %s', safe_bin)
            raise
        except Exception:
            raise
        finally:
            try:
                if deadline_task is not None and not deadline_task.done():
                    deadline_task.cancel()
                    try:
                        await deadline_task
                    except (asyncio.CancelledError, Exception):
                        pass
            finally:
                await _ensure_stream_closed()
