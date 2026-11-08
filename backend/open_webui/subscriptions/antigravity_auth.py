from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from open_webui.subscriptions.acp_client import AcpClient, AcpRequestError
from open_webui.subscriptions.antigravity_profile import (
    AntigravityProfileInfo,
    atomic_write_private,
    build_antigravity_environment,
    get_default_profile_dir,
    parse_antigravity_authorization_url,
    prepare_antigravity_profile,
    profile_identity,
    validate_antigravity_callback_url,
    validate_owned_profile,
)
from open_webui.subscriptions.security import validate_binary_path

log = logging.getLogger(__name__)

AUTH_TIMEOUT_SECONDS = 300


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforce fail-closed loopback callback relay by rejecting all HTTP redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"HTTP redirects are blocked during loopback OAuth callback relay (status {code}).")


@dataclass
class AuthFlowState:
    flow_id: str
    owner_user_id: str
    phase: str  # 'idle', 'starting', 'waiting', 'verifying', 'succeeded', 'failed', 'cancelled'
    profile_identity: str = ""
    binary_path: str = ""
    config_dir: str = ""
    authorization_url: str | None = None
    redirect_uri: str | None = None
    state: str | None = None
    message: str | None = None
    expires_at: float = field(default_factory=lambda: time.time() + AUTH_TIMEOUT_SECONDS)
    callback_sent: bool = False
    cancelled: bool = False
    client: AcpClient | None = None
    task: asyncio.Task[None] | None = None


def _resolve_actual_binary(binary_path: str, config_dir: str = "") -> str:
    """Resolve actual binary through manager's Gemini adapter without gemini CLI fallback."""
    if binary_path and str(binary_path).strip():
        return validate_binary_path(str(binary_path).strip(), "gemini")

    try:
        from open_webui.subscriptions.manager import subscription_manager

        adapter = subscription_manager.get_adapter("gemini")
        if adapter:
            from open_webui.subscriptions.config import ProviderConfigModel

            cfg_model = ProviderConfigModel(binary_path="", config_dir=config_dir)
            installed, res = adapter.is_installed(cfg_model)
            if installed:
                return validate_binary_path(res, "gemini")
            raise RuntimeError(f"Google Antigravity ACP runtime not found: {res}")
    except ImportError:
        pass

    raise RuntimeError("Google Antigravity ACP runtime is not installed or available.")


class AntigravityAuthController:
    """Manages explicit owner-initiated Google sign-in flows for Antigravity ACP.

    Enforces:
    - Single-owner authorization with exact profile scoping
    - Ephemeral flow IDs with 300s bounded lifetimes
    - Standalone expiry enforcement and atomic cancellation/reaping
    - In-memory authorization URL capture with strict loopback/state validation
    - SSRF-protected, redirect-disabled callback forwarding to local listener
    - Clean cancellation, session teardown, and native logout RPC
    - Bounded atomic storage for profile-scoped verified metadata
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._transition_lock = asyncio.Lock()
        self._active_flow: AuthFlowState | None = None

    @property
    def is_authenticated(self) -> bool:
        if self._active_flow and self._active_flow.profile_identity:
            st = self.get_account_state(
                binary_path=self._active_flow.binary_path,
                config_dir=self._active_flow.config_dir,
            )
            return bool(st.get("authenticated", False))
        return False

    @property
    def discovered_models(self) -> list[dict[str, Any]]:
        if self._active_flow and self._active_flow.profile_identity:
            st = self.get_account_state(
                binary_path=self._active_flow.binary_path,
                config_dir=self._active_flow.config_dir,
            )
            return list(st.get("models", []))
        return []

    def get_account_state(self, *, binary_path: str = "", config_dir: str = "") -> dict[str, Any]:
        """Return profile-scoped public verified metadata without inspecting credential files.

        Always read fresh from bounded, typed app-owned account_state.json.
        Unknown state: authenticated=False, models=[], version=None, epoch=0.
        """
        pid = profile_identity(binary_path=binary_path, config_dir=config_dir)
        try:
            prof_dir = validate_owned_profile(config_dir)
            state_file = prof_dir / "antigravity-acp" / "account_state.json"
            if state_file.is_symlink() or not state_file.is_file():
                raise ValueError("No verified public account metadata.")
        except (OSError, ValueError):
            return {"authenticated": False, "models": [], "version": None, "epoch": 0}

        try:
            # Reject oversized files to prevent memory exhaustion / DoS
            if state_file.stat().st_size > 131072:
                return {"authenticated": False, "models": [], "version": None, "epoch": 0}
            with state_file.open("rb") as stream:
                raw = stream.read(131073)
            if len(raw) > 131072:
                raise ValueError("Public metadata size exceeded.")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {"authenticated": False, "models": [], "version": None, "epoch": 0}

            # Reject foreign profile metadata
            if data.get("profile_identity") != pid:
                return {"authenticated": False, "models": [], "version": None, "epoch": 0}

            # Reject strings pretending booleans: must be strict bool
            auth_val = data.get("authenticated")
            if not isinstance(auth_val, bool):
                return {"authenticated": False, "models": [], "version": None, "epoch": 0}

            # Validate bounded models list
            models_raw = data.get("models")
            if not isinstance(models_raw, list) or len(models_raw) > 500:
                return {"authenticated": False, "models": [], "version": None, "epoch": 0}

            models_clean: list[dict[str, Any]] = []
            for m in models_raw:
                if not isinstance(m, dict):
                    return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                m_id = m.get("id")
                m_name = m.get("name", m_id)
                if not isinstance(m_id, str) or not m_id or len(m_id) > 256 or re.search(r"[\x00-\x1f\x7f]", m_id):
                    return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                if not isinstance(m_name, str) or len(m_name) > 256 or re.search(r"[\x00-\x1f\x7f]", m_name):
                    return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                clean_model: dict[str, Any] = {"id": m_id, "name": m_name}

                # Optional reasoning_efforts: list[{value: str, name: str}]
                if "reasoning_efforts" in m:
                    efforts_raw = m.get("reasoning_efforts")
                    if not isinstance(efforts_raw, list) or len(efforts_raw) > 64:
                        return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                    clean_efforts = []
                    for eff in efforts_raw:
                        if not isinstance(eff, dict):
                            return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                        ev = eff.get("value")
                        en = eff.get("name", ev)
                        if not isinstance(ev, str) or not ev or len(ev) > 64 or re.search(r"[\x00-\x1f\x7f]", ev):
                            return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                        if not isinstance(en, str) or len(en) > 64 or re.search(r"[\x00-\x1f\x7f]", en):
                            return {"authenticated": False, "models": [], "version": None, "epoch": 0}
                        clean_efforts.append({"value": ev, "name": en})
                    clean_model["reasoning_efforts"] = clean_efforts
                models_clean.append(clean_model)

            version_val = data.get("version")
            if version_val is not None:
                if not isinstance(version_val, str) or len(version_val) > 64 or re.search(r"[\x00-\x1f\x7f]", version_val):
                    version_val = None

            epoch_val = data.get("epoch", 0)
            if type(epoch_val) is not int or not 0 <= epoch_val < (1 << 63):
                raise ValueError("Invalid public metadata revision.")

            res = {
                "authenticated": auth_val,
                "models": models_clean,
                "version": version_val,
                "epoch": epoch_val,
            }
            return dict(res)
        except Exception as e:
            log.debug("Notice reading profile account state: %s", type(e).__name__)

        return {
            "authenticated": False,
            "models": [],
            "version": None,
            "epoch": 0,
        }

    @contextmanager
    def _profile_operation_lock(self, config_dir: str):
        """Prevent concurrent native sign-in/logout mutations by different workers."""
        profile = validate_owned_profile(config_dir)
        lock_path = profile / ".auth_flow.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(descriptor)

    @contextmanager
    def _account_write_lock(self, config_dir: str):
        """Serialize public metadata CAS across processes without blocking the event loop."""
        profile = validate_owned_profile(config_dir)
        acp_dir = profile / "antigravity-acp"
        acp_dir.mkdir(mode=0o700, exist_ok=True)
        lock_path = acp_dir / ".account_state.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield acp_dir / "account_state.json"
        finally:
            os.close(descriptor)

    def set_authenticated_models(
        self,
        models: list[dict[str, Any]],
        account: str | None = None,
        *,
        binary_path: str = "",
        config_dir: str = "",
        version: str | None = None,
        expected_epoch: str | int | None = None,
        expected_profile_identity: str | None = None,
        current_model_id: str | None = None,
        rotate_epoch: bool = False,
    ) -> bool:
        """Merge native metadata within the captured authentication generation.

        Catalog changes keep the generation stable. Initial verification, explicit
        sign-in and logout advance it so stale work cannot restore revoked access.
        """
        # Bound and validate native metadata before entering the short file transaction.
        def safe_text(value: Any, limit: int) -> bool:
            return isinstance(value, str) and 0 < len(value) <= limit and not re.search(r"[\x00-\x1f\x7f]", value)

        if not isinstance(models, list) or len(models) > 500:
            return False
        clean_models = []
        for model in models:
            if not isinstance(model, dict):
                return False
            mid, name = model.get("id"), model.get("name", model.get("id"))
            if not safe_text(mid, 256) or not safe_text(name, 256):
                return False
            clean = {"id": mid, "name": name}
            if "reasoning_efforts" in model:
                efforts = model["reasoning_efforts"]
                if not isinstance(efforts, list) or len(efforts) > 64:
                    return False
                clean["reasoning_efforts"] = []
                for effort in efforts:
                    if not isinstance(effort, dict):
                        return False
                    value, label = effort.get("value"), effort.get("name", effort.get("value"))
                    if not safe_text(value, 64) or not safe_text(label, 64):
                        return False
                    clean["reasoning_efforts"].append({"value": value, "name": label})
            clean_models.append(clean)
        if version is not None and not safe_text(version, 64):
            return False
        if current_model_id is not None and not safe_text(current_model_id, 256):
            return False
        try:
            with self._account_write_lock(config_dir) as state_file:
                verified_identity = profile_identity(binary_path=binary_path, config_dir=config_dir)
                if expected_profile_identity is not None and verified_identity != expected_profile_identity:
                    return False
                current = self.get_account_state(binary_path=binary_path, config_dir=config_dir)
                epoch = current["epoch"]
                if expected_epoch is not None and str(epoch) != str(expected_epoch):
                    return False
                if current_model_id is not None and current["authenticated"] and not rotate_epoch:
                    # Merge against fresh metadata while holding the file lock.
                    # Missing thinking options are authoritative for the selected
                    # model; only other models retain their own verified choices.
                    known = {model["id"]: model for model in current["models"]}
                    for model in clean_models:
                        old = known.get(model["id"], {})
                        if (model["id"] != current_model_id and "reasoning_efforts" not in model
                                and "reasoning_efforts" in old):
                            model["reasoning_efforts"] = old["reasoning_efforts"]
                next_epoch = epoch + 1 if rotate_epoch or not current["authenticated"] else epoch
                state = {
                    "profile_identity": verified_identity,
                    "authenticated": True, "models": clean_models,
                    "version": version, "epoch": next_epoch,
                }
                payload = (json.dumps(state) + "\n").encode()
                if len(payload) > 131072:
                    return False
                atomic_write_private(state_file, payload)
            return True
        except (OSError, ValueError):
            # Lock contention, unsafe profile or disk error requires explicit revalidation.
            return False

    def clear_authenticated_state(self, *, binary_path: str = "", config_dir: str = "") -> bool:
        """Write a revocation tombstone under the same cross-process CAS lock."""
        try:
            with self._account_write_lock(config_dir) as state_file:
                current = self.get_account_state(binary_path=binary_path, config_dir=config_dir)
                state = {
                    "profile_identity": profile_identity(binary_path=binary_path, config_dir=config_dir),
                    "authenticated": False, "models": [], "version": None,
                    "epoch": current["epoch"] + 1,
                }
                atomic_write_private(state_file, (json.dumps(state) + "\n").encode())
            return True
        except (OSError, ValueError):
            return False

    def get_flow_snapshot(
        self,
        owner_user_id: str,
        *,
        binary_path: str = "",
        config_dir: str = "",
    ) -> dict[str, Any]:
        """Return non-sensitive flow status visible to the user, strictly scoped to profile."""
        state = self.get_account_state(binary_path=binary_path, config_dir=config_dir)
        flow = self._active_flow

        expected_pid = (
            profile_identity(binary_path=binary_path, config_dir=config_dir)
            if (binary_path or config_dir)
            else None
        )

        if not flow or flow.owner_user_id != owner_user_id or (expected_pid and flow.profile_identity != expected_pid):
            return {
                "phase": "idle",
                "flow_id": None,
                "authorization_url": None,
                "message": None,
                "authenticated": state["authenticated"],
                "models": state["models"],
                "version": state.get("version"),
                "epoch": state.get("epoch", 0),
            }

        # Check expiration
        if time.time() >= flow.expires_at and flow.phase in ("starting", "waiting", "verifying"):
            flow.cancelled = True
            flow.phase = "failed"
            flow.message = "Google sign-in attempt timed out. Please start sign-in again."
            if flow.task and not flow.task.done():
                flow.task.cancel()
            if flow.client:
                asyncio.create_task(flow.client.close())

        return {
            "phase": flow.phase,
            "flow_id": flow.flow_id,
            "authorization_url": flow.authorization_url if flow.phase == "waiting" else None,
            "message": flow.message,
            "authenticated": state["authenticated"],
            "models": state["models"],
            "version": state.get("version"),
            "epoch": state.get("epoch", 0),
        }

    async def start_sign_in(
        self,
        owner_user_id: str,
        binary_path: str,
        config_dir: str,
    ) -> dict[str, Any]:
        """Start a new Google OAuth sign-in flow.

        Serialized with _transition_lock to eliminate the replacement race (L8-02).
        """
        resolved_binary = _resolve_actual_binary(binary_path, config_dir)
        prof_id = profile_identity(binary_path=resolved_binary, config_dir=config_dir)

        async with self._transition_lock:
            # Terminate and join any existing in-progress flow safely before creating replacement
            old_flow: AuthFlowState | None = None
            async with self._lock:
                if self._active_flow and self._active_flow.phase in ("starting", "waiting", "verifying"):
                    old_flow = self._active_flow
                    old_flow.cancelled = True
                    old_flow.phase = "cancelled"
                    old_flow.message = "Cancelled by newer sign-in attempt."
                    if old_flow.task and not old_flow.task.done():
                        old_flow.task.cancel()

            if old_flow:
                if old_flow.client:
                    try:
                        await old_flow.client.close()
                    except Exception:
                        pass
                if old_flow.task:
                    try:
                        await asyncio.gather(old_flow.task, return_exceptions=True)
                    except Exception:
                        pass

            flow_id = str(uuid.uuid4())
            client = AcpClient(timeout=AUTH_TIMEOUT_SECONDS)
            flow = AuthFlowState(
                flow_id=flow_id,
                owner_user_id=owner_user_id,
                profile_identity=prof_id,
                binary_path=resolved_binary,
                config_dir=config_dir,
                phase="starting",
                message="Starting Google sign-in...",
                client=client,
            )

            async with self._lock:
                self._active_flow = flow

            # Launch background runner with overall 300s budget
            flow.task = asyncio.create_task(
                self._run_sign_in(flow, client, resolved_binary, config_dir)
            )

            return self.get_flow_snapshot(owner_user_id, binary_path=resolved_binary, config_dir=config_dir)

    async def _run_sign_in(self, flow: AuthFlowState, client: AcpClient, binary_path: str, config_dir: str) -> None:
        """Background coroutine that connects to ACP and handles the OAuth lifecycle."""
        scratch_dir = tempfile.mkdtemp(prefix="mcchat_acp_auth_")

        # Capture pre-verification epoch for CAS fencing
        pre_epoch = self.get_account_state(binary_path=binary_path, config_dir=config_dir).get("epoch", 0)

        def handle_auth_url(raw_url: str):
            if flow.cancelled or flow.phase in ("failed", "cancelled") or time.time() >= flow.expires_at:
                return
            parsed = parse_antigravity_authorization_url(raw_url)
            if parsed:
                flow.authorization_url = parsed["authorization_url"]
                flow.redirect_uri = parsed["redirect_uri"]
                flow.state = parsed["state"]
                flow.phase = "waiting"
                flow.message = "Open the Google sign-in link. If you are remote, paste the redirect URL here."

        try:
            async with asyncio.timeout(AUTH_TIMEOUT_SECONDS):
                profile_info = prepare_antigravity_profile(config_dir, binary_path)
                with self._profile_operation_lock(config_dir):
                    env = build_antigravity_environment(profile_info)

                    exe = validate_binary_path(binary_path, "gemini")
                    args = ["--uid="] if exe.endswith(".par") else ["acp"] if os.path.basename(exe).lower() in ("agy", "antigravity") else []
                    cmd = [exe] + args

                    await client.spawn(
                        cmd,
                        env=env,
                        cwd=scratch_dir,
                        on_authorization_url=handle_auth_url,
                    )

                    init_res = await client.initialize(
                        protocol_version=2,
                        client_info={"name": "mcchat", "title": "McChat", "version": "0.13.0"},
                        timeout=45.0,
                    )
                    agent_info = init_res.get("agentInfo", {})
                    actual_version = agent_info.get("version") or None

                    await client.authenticate(method_id="oauth-personal", timeout=240.0)

                    if flow.cancelled or flow.phase in ("failed", "cancelled") or time.time() >= flow.expires_at:
                        return

                    flow.phase = "verifying"
                    flow.message = "Checking Antigravity account access and available models..."

                    session_res = await client.create_session(cwd=scratch_dir, timeout=30.0)
                    config_options = session_res.get("configOptions", [])
                    discovered: list[dict[str, Any]] = []

                    try:
                        from open_webui.subscriptions.acp_catalog import extract_models

                        discovered = extract_models(config_options)
                    except ImportError:
                        for opt in config_options:
                            if opt.get("id") == "model":
                                for choice in opt.get("options", []):
                                    val = choice.get("value") or choice.get("id")
                                    name = choice.get("name", val)
                                    if val:
                                        discovered.append({"id": val, "name": name})

                    if flow.cancelled or flow.phase in ("failed", "cancelled") or time.time() >= flow.expires_at:
                        return

                    # Record verified models with captured expected_epoch CAS fence
                    recorded = self.set_authenticated_models(
                        discovered,
                        binary_path=binary_path,
                        config_dir=config_dir,
                        version=actual_version,
                        expected_epoch=pre_epoch,
                        expected_profile_identity=flow.profile_identity,
                        rotate_epoch=True,
                    )

                    if not recorded:
                        flow.phase = "cancelled"
                        flow.message = "Google sign-in was superseded or invalidated."
                        return

                    flow.phase = "succeeded"
                    flow.authorization_url = None
                    flow.message = f"Signed in with Google. {len(discovered)} model(s) available."

        except (TimeoutError, asyncio.TimeoutError):
            flow.phase = "failed"
            flow.message = "Google sign-in attempt timed out. Please start sign-in again."
        except asyncio.CancelledError:
            flow.phase = "cancelled"
            flow.message = "Google sign-in was cancelled."
        except AcpRequestError as e:
            flow.phase = "failed"
            # Generic diagnostic classification: NEVER leak OAuth URL/state/code or raw tokens (SEC8-06)
            raw_err = str(getattr(e, "error_message", "") or e)
            if "SUBSCRIPTION_REQUIRED" in raw_err:
                flow.message = "Google requires an eligible Antigravity subscription for this account."
            elif re.search(r"access_denied|denied access|cancelled", raw_err, re.I):
                flow.message = "Google sign-in was not approved. Start sign-in again."
            elif re.search(r"auth|token|credential|oauth|login|permission", raw_err, re.I):
                flow.message = "Google authentication failed. Please try signing in again."
            else:
                flow.message = "Google Antigravity service request failed."
        except Exception:
            flow.phase = "failed"
            flow.message = "Google sign-in encountered an unexpected error. Please try again."
        finally:
            try:
                await client.close()
            finally:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    async def complete_sign_in(
        self,
        owner_user_id: str,
        flow_id: str,
        callback_url: str,
    ) -> dict[str, Any]:
        """Relay the Google OAuth return URL to the active ACP loopback listener."""
        flow: AuthFlowState | None = None
        async with self._lock:
            if not self._active_flow or self._active_flow.flow_id != flow_id or self._active_flow.owner_user_id != owner_user_id:
                raise ValueError("No active sign-in flow matching this identifier.")

            flow = self._active_flow
            if flow.phase != "waiting" or not flow.redirect_uri or not flow.state:
                raise ValueError("Sign-in is not currently waiting for a callback URL.")

            if flow.callback_sent:
                raise ValueError("Callback URL was already submitted. Waiting for verification.")

            validated_url = validate_antigravity_callback_url(
                pending_redirect_uri=flow.redirect_uri,
                pending_state=flow.state,
                callback_url=callback_url,
            )

            flow.callback_sent = True
            flow.phase = "verifying"
            flow.message = "Submitting sign-in response to Antigravity..."

        loop = asyncio.get_running_loop()

        def _forward():
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler, NoRedirectHandler())
            req = urllib.request.Request(
                validated_url,
                headers={"User-Agent": "McChat-Antigravity-Auth-Relay/1.0"},
            )
            with opener.open(req, timeout=10) as resp:
                return resp.status

        try:
            await loop.run_in_executor(None, _forward)
        except Exception as e:
            log.debug("Callback relay completed or listener closed on consumption: %s", type(e).__name__)

        return self.get_flow_snapshot(owner_user_id, binary_path=flow.binary_path, config_dir=flow.config_dir)

    async def cancel_sign_in(self, owner_user_id: str, flow_id: str) -> dict[str, Any]:
        """Cancel an in-progress sign-in flow."""
        async with self._transition_lock:
            flow: AuthFlowState | None = None
            async with self._lock:
                flow = self._active_flow
                if not flow or flow.flow_id != flow_id or flow.owner_user_id != owner_user_id:
                    raise ValueError("No active sign-in flow matching this identifier.")

                flow.cancelled = True
                flow.phase = "cancelled"
                flow.message = "Google sign-in was cancelled by user."
                if flow.task and not flow.task.done():
                    flow.task.cancel()

            if flow.client:
                try:
                    await flow.client.close()
                except Exception:
                    pass
            if flow.task:
                try:
                    await asyncio.gather(flow.task, return_exceptions=True)
                except Exception:
                    pass

            return self.get_flow_snapshot(owner_user_id, binary_path=flow.binary_path, config_dir=flow.config_dir)

    async def sign_out(self, owner_user_id: str, binary_path: str, config_dir: str) -> dict[str, Any]:
        """Perform native ACP sign-out and clear isolated profile credentials.

        Serialized with _transition_lock.
        Reports failure if native logout could not be performed (dirty profile / missing capability / RPC failure).
        No manual token deletion (SEC8-05).
        """
        resolved_binary = _resolve_actual_binary(binary_path, config_dir)

        async with self._transition_lock:
            old_flow: AuthFlowState | None = None
            async with self._lock:
                if self._active_flow:
                    old_flow = self._active_flow
                    old_flow.cancelled = True
                    old_flow.phase = "cancelled"
                    old_flow.message = "Cancelled by sign-out."
                    if old_flow.task and not old_flow.task.done():
                        old_flow.task.cancel()

            if old_flow:
                if old_flow.client:
                    try:
                        await old_flow.client.close()
                    except Exception:
                        pass
                if old_flow.task:
                    try:
                        await asyncio.gather(old_flow.task, return_exceptions=True)
                    except Exception:
                        pass

            # Clear local public verification honestly
            verification_cleared = self.clear_authenticated_state(binary_path=resolved_binary, config_dir=config_dir)

            # Attempt native ACP logout RPC if runtime is executable
            native_logout_ok = False
            try:
                profile_info = prepare_antigravity_profile(config_dir, resolved_binary)
                with self._profile_operation_lock(config_dir):
                    env = build_antigravity_environment(profile_info)
                    exe = validate_binary_path(resolved_binary, "gemini")
                    args = ["--uid="] if exe.endswith(".par") else ["acp"] if os.path.basename(exe).lower() in ("agy", "antigravity") else []
                    cmd = [exe] + args

                    client = AcpClient(timeout=30.0)
                    with tempfile.TemporaryDirectory(prefix="mcchat_acp_logout_") as tmp_cwd:
                        await client.spawn(cmd, env=env, cwd=tmp_cwd)
                        try:
                            init_res = await client.initialize(
                                protocol_version=2,
                                client_info={"name": "mcchat", "title": "McChat", "version": "0.13.0"},
                                timeout=15.0,
                            )
                            caps = init_res.get("agentCapabilities", {}).get("auth", {})
                            # In official 1.4.5, agentCapabilities.auth.logout is {} (presence valid)
                            if "logout" in caps:
                                await client.logout(timeout=15.0)
                                native_logout_ok = True
                        finally:
                            await client.close()
            except Exception as e:
                log.debug("Native logout attempt completed with notice: %s", type(e).__name__)

            # Fence verifications which started while the native logout RPC was in flight.
            verification_cleared = self.clear_authenticated_state(
                binary_path=resolved_binary, config_dir=config_dir
            ) and verification_cleared
            if native_logout_ok and verification_cleared:
                return {
                    "phase": "idle",
                    "flow_id": None,
                    "success": True,
                    "message": "Signed out from Google Antigravity.",
                    "authenticated": False,
                    "models": [],
                }
            else:
                return {
                    "phase": "failed",
                    "flow_id": None,
                    "success": False,
                    "message": "Sign-out could not be fully verified. Login state unknown / revalidation required.",
                    "authenticated": False,
                    "models": [],
                }


antigravity_auth_controller = AntigravityAuthController()


def get_account_state(*, binary_path: str = "", config_dir: str = "") -> dict[str, Any]:
    return antigravity_auth_controller.get_account_state(binary_path=binary_path, config_dir=config_dir)


def set_authenticated_models(
    models: list[dict[str, Any]],
    *,
    binary_path: str = "",
    config_dir: str = "",
    version: str | None = None,
    expected_epoch: str | int | None = None,
    expected_profile_identity: str | None = None,
    current_model_id: str | None = None,
    rotate_epoch: bool = False,
) -> bool:
    return antigravity_auth_controller.set_authenticated_models(
        models,
        binary_path=binary_path,
        config_dir=config_dir,
        version=version,
        expected_epoch=expected_epoch,
        expected_profile_identity=expected_profile_identity,
        current_model_id=current_model_id,
        rotate_epoch=rotate_epoch,
    )


def clear_authenticated_state(*, binary_path: str = "", config_dir: str = "") -> bool:
    return antigravity_auth_controller.clear_authenticated_state(binary_path=binary_path, config_dir=config_dir)


def get_flow_snapshot(owner_user_id: str, *, binary_path: str = "", config_dir: str = "") -> dict[str, Any]:
    return antigravity_auth_controller.get_flow_snapshot(
        owner_user_id,
        binary_path=binary_path,
        config_dir=config_dir,
    )
