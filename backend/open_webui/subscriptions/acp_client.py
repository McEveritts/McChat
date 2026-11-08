from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import signal
import subprocess
import time
from typing import Any, Callable

from open_webui.subscriptions.antigravity_profile import (
    ANTIGRAVITY_AUTH_BROWSER_MARKER,
    ANTIGRAVITY_AUTH_STDOUT_PREFIX,
    MAX_BROWSER_HELPER_LINE_LENGTH,
)
from open_webui.subscriptions.executor import (
    _CONCURRENCY_SEMAPHORE,
    MAX_STDERR_BYTES,
    MAX_STDIN_BYTES,
    MAX_STDOUT_BYTES,
    MAX_STREAM_LINE_BYTES,
    _cleanup_process,
    _kill_pgid_sync,
    _terminate_pgid_sync,
)
from open_webui.subscriptions.security import sanitize_error_message

log = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = {
    "name": "mcchat",
    "title": "McChat",
    "version": "0.13.0",
}

DEFAULT_CLIENT_CAPABILITIES = {
    "fs": {
        "readTextFile": False,
        "writeTextFile": False,
    },
    "terminal": False,
}

MAX_PENDING_REQUESTS = 64
DEFAULT_OVERALL_TIMEOUT = 180.0


class AcpError(Exception):
    """Base error for Agent Client Protocol communication."""
    pass


class AcpRequestError(AcpError):
    """JSON-RPC error returned by ACP agent."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"ACP Error [{code}]: {message}")
        self.code = code
        self.error_message = message
        self.data = data


class AcpProcessError(AcpError):
    """Error representing process failure, protocol mismatch, or premature termination."""

    def __init__(self, message: str, exit_code: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class AcpClient:
    """Async bidirectional JSON-RPC 2.0 client for official Agent Client Protocol runtimes.

    Enforces strict protocol contracts:
    - ProtocolVersion 2 negotiation
    - Single monotonic overall deadline bounding all operations (admission, spawn, writes, waits, dispatch)
    - Truthful default McChat clientInfo identity
    - Tool disablement via session metadata (_meta.antigravity.enabledTools: [])
    - JSON-RPC 2.0 method not found / capability refusal for unsupported agent callbacks
    - Full process group containment (SIGTERM -> grace period -> SIGKILL) even on leader exit 0
    - Chunk-based stderr draining with bounded URL marker parsing and memory bounds
    - stdout line limit (1 MiB) and total stdout limit (4 MiB)
    - Stdin payload bound (2 MiB) with timed drain
    - Bounded concurrent pending requests
    - Sync and awaitable session update callbacks with exception propagation
    - Admission concurrency ceiling integration with McChat limiter
    """

    def __init__(self, timeout: float = DEFAULT_OVERALL_TIMEOUT) -> None:
        self.overall_timeout: float = float(timeout)
        self._created_at: float = time.monotonic()
        self.deadline: float = self._created_at + self.overall_timeout

        self.proc: asyncio.subprocess.Process | None = None
        self.pgid: int | None = None
        self.pending_requests: dict[int | str, asyncio.Future[Any]] = {}
        self.next_request_id: int = 1
        self.on_authorization_url: Callable[[str], Any] | None = None
        self.on_session_update: Callable[[dict[str, Any]], Any] | None = None

        self.stderr_chunks: list[bytes] = []
        self.stderr_bytes_captured: int = 0
        self.total_stdout_bytes: int = 0

        self.stdout_task: asyncio.Task[None] | None = None
        self.stderr_task: asyncio.Task[None] | None = None
        self.is_closed: bool = False
        self.admitted: bool = False
        self._write_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def stderr_lines(self) -> list[str]:
        raw = b"".join(self.stderr_chunks).decode("utf-8", errors="replace")
        return raw.splitlines()

    def _remaining_timeout(self, local_timeout: float | None = None) -> float:
        now = time.monotonic()
        remaining = self.deadline - now
        if remaining <= 0:
            raise TimeoutError(f"ACP client deadline exceeded ({self.overall_timeout:.1f}s)")
        if local_timeout is not None:
            return min(float(local_timeout), remaining)
        return remaining

    def _get_sanitized_stderr(self) -> str:
        """Return sanitized, redacted captured stderr text."""
        raw = b"".join(self.stderr_chunks).decode("utf-8", errors="replace")
        clean = re.sub(r'https://accounts\.google\.com\S*', '[REDACTED_AUTH_URL]', raw)
        clean = re.sub(r'__T3_ANTIGRAVITY_AUTH_URL__\S*', '[REDACTED_AUTH_MARKER]', clean)
        return sanitize_error_message(clean)

    def _fail_pending_requests(self, exc: Exception) -> None:
        """Fail all pending requests with given exception."""
        for req_id, fut in list(self.pending_requests.items()):
            if not fut.done():
                fut.set_exception(exc)
        self.pending_requests.clear()

    def _fail_pending_requests_on_eof(self) -> None:
        """If stdout closed while requests were pending, fail them coherently."""
        exit_code = self.proc.returncode if self.proc else None
        sanitized_stderr = self._get_sanitized_stderr()
        for req_id, fut in list(self.pending_requests.items()):
            if not fut.done():
                fut.set_exception(
                    AcpProcessError(
                        "ACP server process closed connection before responding.",
                        exit_code=exit_code,
                        stderr=sanitized_stderr,
                    )
                )
        self.pending_requests.clear()

    async def spawn(
        self,
        command: list[str],
        env: dict[str, str],
        cwd: str,
        on_authorization_url: Callable[[str], Any] | None = None,
        on_session_update: Callable[[dict[str, Any]], Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Spawn the ACP server subprocess in an isolated process group with shared concurrency limiting."""
        if self.is_closed:
            raise AcpProcessError("Cannot spawn: ACP client is closed.")

        rem = self._remaining_timeout(timeout)

        # 1. Acquire admission semaphore bounded by deadline
        try:
            await asyncio.wait_for(_CONCURRENCY_SEMAPHORE.acquire(), timeout=rem)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(f"ACP client timed out waiting for concurrency slot after {rem:.2f}s") from exc
        self.admitted = True

        self.on_authorization_url = on_authorization_url
        self.on_session_update = on_session_update

        try:
            rem = self._remaining_timeout(timeout)
            self.proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=cwd,
                    limit=MAX_STREAM_LINE_BYTES,  # 1 MiB line buffer instead of default 64 KiB
                    start_new_session=True,  # Isolate process group
                ),
                timeout=rem,
            )
            self.pgid = self.proc.pid
        except BaseException:
            if self.proc is not None:
                curr_pgid = self.pgid or self.proc.pid
                _kill_pgid_sync(curr_pgid)
                if self.proc.stdin and not self.proc.stdin.is_closing():
                    try:
                        self.proc.stdin.close()
                    except Exception:
                        pass
            if self.admitted:
                _CONCURRENCY_SEMAPHORE.release()
                self.admitted = False
            raise

        self.stdout_task = asyncio.create_task(self._drain_stdout())
        self.stderr_task = asyncio.create_task(self._drain_stderr())

    async def _check_auth_marker(self, text: str) -> None:
        """Inspect line/buffer for Google authorization markers."""
        if not text:
            return

        raw_url: str | None = None
        if ANTIGRAVITY_AUTH_BROWSER_MARKER in text:
            marker_idx = text.find(ANTIGRAVITY_AUTH_BROWSER_MARKER)
            payload = text[marker_idx + len(ANTIGRAVITY_AUTH_BROWSER_MARKER):].strip()
            try:
                val = json.loads(payload)
                if isinstance(val, str) and val.startswith("https://accounts.google.com"):
                    raw_url = val
            except Exception as e:
                log.debug("Failed to parse browser marker auth URL: %s", type(e).__name__)
        elif text.startswith(ANTIGRAVITY_AUTH_STDOUT_PREFIX):
            cand = text[len(ANTIGRAVITY_AUTH_STDOUT_PREFIX):].strip()
            if cand.startswith("https://accounts.google.com"):
                raw_url = cand

        if raw_url and self.on_authorization_url:
            try:
                res = self.on_authorization_url(raw_url)
                if inspect.isawaitable(res):
                    rem = self._remaining_timeout()
                    await asyncio.wait_for(res, timeout=rem)
            except Exception as e:
                log.error("Exception in on_authorization_url callback: %s", type(e).__name__)
                self._fail_pending_requests(e)
                raise

    async def _drain_stderr(self) -> None:
        """Read and buffer stderr up to MAX_STDERR_BYTES in chunks, intercepting Google authorization URLs

        without blocking even on oversized or newline-free output.
        """
        if not self.proc or not self.proc.stderr:
            return

        marker_buf = ""
        MAX_SCAN_BUF = MAX_BROWSER_HELPER_LINE_LENGTH * 2

        try:
            while not self.is_closed:
                try:
                    chunk = await self.proc.stderr.read(65536)
                except (asyncio.CancelledError, GeneratorExit):
                    break
                except Exception as e:
                    log.debug("Exception reading ACP stderr chunk: %s", type(e).__name__)
                    break

                if not chunk:
                    break

                # Bounded diagnostics retention
                if self.stderr_bytes_captured < MAX_STDERR_BYTES:
                    take = min(len(chunk), MAX_STDERR_BYTES - self.stderr_bytes_captured)
                    self.stderr_chunks.append(chunk[:take])
                    self.stderr_bytes_captured += take

                # Check for URL markers
                text = chunk.decode("utf-8", errors="replace")
                marker_buf += text

                while "\n" in marker_buf:
                    line, marker_buf = marker_buf.split("\n", 1)
                    line = line.rstrip("\r")
                    await self._check_auth_marker(line)

                if len(marker_buf) > MAX_SCAN_BUF:
                    await self._check_auth_marker(marker_buf)
                    marker_buf = marker_buf[-256:]

            if marker_buf:
                await self._check_auth_marker(marker_buf.strip())

        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception as e:
            log.debug("Exception in ACP stderr drain: %s", type(e).__name__)

    async def _drain_stdout(self) -> None:
        """Parse incoming JSON-RPC 2.0 lines and dispatch requests, notifications, and responses."""
        if not self.proc or not self.proc.stdout:
            return

        try:
            while not self.is_closed:
                try:
                    line_bytes = await self.proc.stdout.readline()
                except ValueError:
                    err = ValueError(f"Stream frame size exceeded limit of {MAX_STREAM_LINE_BYTES} bytes")
                    self._fail_pending_requests(err)
                    break
                except (asyncio.CancelledError, GeneratorExit):
                    break
                except Exception as e:
                    log.debug("Exception reading ACP stdout: %s", type(e).__name__)
                    break

                if not line_bytes:
                    break

                if len(line_bytes) > MAX_STREAM_LINE_BYTES:
                    err = ValueError(f"Stream frame size exceeded limit of {MAX_STREAM_LINE_BYTES} bytes")
                    self._fail_pending_requests(err)
                    break

                self.total_stdout_bytes += len(line_bytes)
                if self.total_stdout_bytes > MAX_STDOUT_BYTES:
                    err = ValueError(f"Stream total output exceeded limit of {MAX_STDOUT_BYTES} bytes")
                    self._fail_pending_requests(err)
                    break

                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Check for stdout Google sign-in announcement
                if line.startswith(ANTIGRAVITY_AUTH_STDOUT_PREFIX):
                    cand = line[len(ANTIGRAVITY_AUTH_STDOUT_PREFIX):].strip()
                    if cand.startswith("https://accounts.google.com") and self.on_authorization_url:
                        try:
                            res = self.on_authorization_url(cand)
                            if inspect.isawaitable(res):
                                rem = self._remaining_timeout()
                                await asyncio.wait_for(res, timeout=rem)
                        except Exception as e:
                            log.error("Exception in on_authorization_url: %s", e)
                            self._fail_pending_requests(e)
                            raise
                    continue

                try:
                    payload = json.loads(line)
                except Exception:
                    # Ignore non-JSON diagnostics
                    continue

                if not isinstance(payload, dict):
                    continue

                # 1. Handle JSON-RPC Response to our outgoing request
                if "id" in payload and ("result" in payload or "error" in payload):
                    req_id = payload.get("id")
                    fut = self.pending_requests.get(req_id)
                    if fut and not fut.done():
                        if "error" in payload:
                            err_info = payload["error"] or {}
                            fut.set_exception(
                                AcpRequestError(
                                    code=err_info.get("code", -32603),
                                    message=err_info.get("message", "Unknown error"),
                                    data=err_info.get("data"),
                                )
                            )
                        else:
                            fut.set_result(payload.get("result", {}))
                    continue

                # 2. Handle JSON-RPC Notification from agent (no id)
                if "method" in payload and "id" not in payload:
                    method = payload.get("method")
                    params = payload.get("params") or {}
                    if method == "session/update" and self.on_session_update:
                        try:
                            res = self.on_session_update(params)
                            if inspect.isawaitable(res):
                                rem = self._remaining_timeout()
                                await asyncio.wait_for(res, timeout=rem)
                        except Exception as e:
                            log.error("Exception in on_session_update: %s", type(e).__name__)
                            self._fail_pending_requests(e)
                            raise
                    continue

                # 3. Handle JSON-RPC Request from agent to client (requires response)
                if "method" in payload and "id" in payload:
                    req_id = payload.get("id")
                    method = payload.get("method", "")
                    log.info("ACP agent requested an unsupported callback; rejecting safely")
                    error_resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": "Method unsupported by client capabilities.",
                        },
                    }
                    try:
                        await self._send_raw(error_resp)
                    except Exception as e:
                        log.debug("Failed to send capability refusal: %s", type(e).__name__)
                    continue

        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception as e:
            log.debug("Exception in ACP stdout drain: %s", type(e).__name__)
            self._fail_pending_requests(e)
        finally:
            self._fail_pending_requests_on_eof()

    async def _send_raw(self, payload: dict[str, Any], timeout: float | None = None) -> None:
        """Serialize and write a JSON-RPC message to the server's standard input within remaining deadline."""
        if not self.proc or not self.proc.stdin or self.proc.returncode is not None or self.is_closed:
            raise AcpProcessError("Cannot send payload: ACP process is not running.")

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")

        if len(data) > MAX_STDIN_BYTES:
            raise ValueError(f"ACP payload size ({len(data)} bytes) exceeds limit ({MAX_STDIN_BYTES} bytes)")

        rem = self._remaining_timeout(timeout)
        try:
            await asyncio.wait_for(self._write_lock.acquire(), timeout=rem)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(f"Timed out acquiring write lock after {rem:.2f}s") from exc

        try:
            self.proc.stdin.write(data)
            rem = self._remaining_timeout(timeout)
            await asyncio.wait_for(self.proc.stdin.drain(), timeout=rem)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(f"Timed out writing to ACP stdin after {rem:.2f}s") from exc
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise AcpProcessError("ACP stdin pipe closed.") from exc
        finally:
            self._write_lock.release()

    async def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
        """Send a JSON-RPC 2.0 request and await the corresponding response within remaining deadline."""
        if self.is_closed:
            raise AcpProcessError("ACP client is closed.")

        if len(self.pending_requests) >= MAX_PENDING_REQUESTS:
            raise AcpRequestError(-32603, f"Maximum pending ACP requests exceeded ({MAX_PENDING_REQUESTS})")

        effective_timeout = self._remaining_timeout(timeout)

        loop = asyncio.get_running_loop()
        req_id = self.next_request_id
        self.next_request_id += 1

        fut: asyncio.Future[Any] = loop.create_future()
        self.pending_requests[req_id] = fut

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params if params is not None else {},
        }

        try:
            # One local RPC budget includes sending and waiting for its reply.
            async with asyncio.timeout(effective_timeout):
                await self._send_raw(msg, timeout=effective_timeout)
                return await fut
        except (asyncio.TimeoutError, TimeoutError) as e:
            fut.cancel()
            raise TimeoutError(f"ACP request {method!r} timed out after {timeout}s.") from e
        finally:
            self.pending_requests.pop(req_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> None:
        """Send a JSON-RPC 2.0 notification without expecting a response."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params if params is not None else {},
        }
        await self._send_raw(msg, timeout=timeout)

    async def cancel_session(self, session_id: str, timeout: float | None = 5.0) -> None:
        """Send session/cancel notification to abort in-flight prompt."""
        try:
            await self.notify("session/cancel", {"sessionId": session_id}, timeout=timeout)
        except Exception as e:
            log.debug("Failed to send session/cancel: %s", e)

    async def initialize(
        self,
        protocol_version: int = 2,
        client_info: dict[str, str] | None = None,
        client_capabilities: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Perform initial ACP handshake and validate protocolVersion 2 negotiation."""
        payload = {
            "protocolVersion": protocol_version,
            "clientInfo": client_info or DEFAULT_CLIENT_INFO,
            "clientCapabilities": client_capabilities or DEFAULT_CLIENT_CAPABILITIES,
        }
        result = await self.request("initialize", payload, timeout=timeout)
        if not isinstance(result, dict):
            raise AcpProcessError("ACP server returned non-dictionary initialize response")
        negotiated = result.get("protocolVersion")
        if type(negotiated) is not int or negotiated != 2:
            raise AcpProcessError(
                f"ACP server negotiated unsupported protocolVersion {negotiated!r} (expected 2)"
            )
        return result

    async def authenticate(self, method_id: str = "oauth-personal", timeout: float = 30.0) -> dict[str, Any]:
        """Eagerly authenticate with the ACP server."""
        return await self.request("authenticate", {"methodId": method_id}, timeout=timeout)

    async def create_session(
        self,
        cwd: str,
        enabled_tools: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Create a new conversational ACP session with tools disabled."""
        payload = {
            "cwd": cwd,
            "mcpServers": mcp_servers if mcp_servers is not None else [],
            "_meta": {
                "antigravity": {
                    "enabledTools": enabled_tools if enabled_tools is not None else [],
                }
            },
        }
        return await self.request("session/new", payload, timeout=timeout)

    async def set_config_option(
        self,
        session_id: str,
        config_id: str,
        value: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Configure session options such as model or thought_level."""
        payload = {
            "sessionId": session_id,
            "configId": config_id,
            "value": value,
        }
        return await self.request("session/set_config_option", payload, timeout=timeout)

    async def prompt(
        self,
        session_id: str,
        prompt_text: str,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        """Send prompt to ACP session and await terminal PromptResponse."""
        payload = {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": prompt_text}],
        }
        return await self.request("session/prompt", payload, timeout=timeout)

    async def logout(self, timeout: float = 30.0) -> dict[str, Any]:
        """Send logout request to clear provider credentials."""
        return await self.request("logout", {}, timeout=timeout)

    async def close(self) -> None:
        """All callers wait for one cleanup, even if the first caller is cancelled."""
        if self._close_task is None:
            self.is_closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        # Retrieve exceptions and finish cleanup before propagating cancellation.
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close_resources(self) -> None:
        proc = self.proc
        pgid = self.pgid or (proc.pid if proc else None)

        try:
            # 1. Fail any in-flight pending request futures
            sanitized_stderr = self._get_sanitized_stderr()
            for fut in list(self.pending_requests.values()):
                if not fut.done():
                    fut.set_exception(AcpProcessError("ACP client is closed.", stderr=sanitized_stderr))
            self.pending_requests.clear()

            # 2. Cancel and join helper tasks
            for task in (self.stdout_task, self.stderr_task):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            self.stdout_task = None
            self.stderr_task = None

            # 3. Process group cleanup
            if proc is not None and pgid is not None:
                await _cleanup_process(proc, pgid, grace_period=0.2)
                # Ensure kernel delivers SIGKILL to lingering descendants
                await asyncio.sleep(0.05)
            elif pgid is not None:
                _kill_pgid_sync(pgid)
                await asyncio.sleep(0.05)

            # Ensure open transports and pipes are closed to prevent unclosed resource warnings
            if proc is not None:
                for pipe in (proc.stdout, proc.stderr):
                    if pipe and hasattr(pipe, "_transport") and pipe._transport:
                        try:
                            pipe._transport.close()
                        except Exception:
                            pass
        except BaseException as exc:
            if pgid is not None:
                _kill_pgid_sync(pgid)
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            # 4. Release admission permit unconditionally on every exit path
            if self.admitted:
                _CONCURRENCY_SEMAPHORE.release()
                self.admitted = False

    async def __aenter__(self) -> AcpClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
