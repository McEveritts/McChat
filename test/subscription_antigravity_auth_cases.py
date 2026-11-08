from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure backend directory is in sys.path without loading DB modules at import time
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from open_webui.subscriptions.antigravity_profile import (
    APP_OWNERSHIP_MARKER,
    AntigravityProfileInfo,
    build_antigravity_environment,
    get_default_profile_dir,
    parse_antigravity_authorization_url,
    prepare_antigravity_profile,
    profile_identity,
    validate_antigravity_callback_url,
)
from open_webui.subscriptions.antigravity_auth import (
    AntigravityAuthController,
    AuthFlowState,
    NoRedirectHandler,
    get_account_state,
    set_authenticated_models,
    clear_authenticated_state,
    get_flow_snapshot,
)


class FakeAcpClient:
    """Mock ACP client simulating official 1.4.5 protocol responses."""

    def __init__(self, timeout: float = 300.0) -> None:
        self.timeout = timeout
        self.spawn_called = False
        self.spawn_cmd: list[str] = []
        self.spawn_env: dict[str, str] = {}
        self.spawn_cwd: str = ""
        self.on_authorization_url = None
        self.initialize_called = False
        self.client_info = None
        self.authenticate_called = False
        self.logout_called = 0
        self.create_session_called = False
        self.closed = False
        self.caps = {"auth": {"logout": {}}}  # official empty logout capabilities
        self.config_options = [
            {
                "id": "model",
                "type": "select",
                "currentValue": "gemini-2.5-flash",
                "options": [
                    {"value": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
                    {"value": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
                ],
            }
        ]

    async def spawn(self, cmd, env=None, cwd="", on_authorization_url=None, **kwargs):
        self.spawn_called = True
        self.spawn_cmd = cmd
        self.spawn_env = env or {}
        self.spawn_cwd = cwd
        self.on_authorization_url = on_authorization_url

    async def initialize(self, protocol_version=2, client_info=None, timeout=45.0):
        self.initialize_called = True
        self.client_info = client_info
        return {
            "protocolVersion": protocol_version,
            "agentInfo": {"name": "antigravity-acp", "version": "1.4.5"},
            "agentCapabilities": self.caps,
        }

    async def authenticate(self, method_id="oauth-personal", timeout=240.0):
        self.authenticate_called = True
        return {}

    async def create_session(self, cwd="", timeout=30.0):
        self.create_session_called = True
        return {
            "sessionId": "test-session-123",
            "configOptions": self.config_options,
        }

    async def logout(self, timeout=15.0):
        self.logout_called += 1
        return {}

    async def close(self):
        self.closed = True


# =========================================================================
# 1. Dirty/Foreign Profiles & Settings Preservation Tests
# =========================================================================


def test_dirty_foreign_profiles_rejected(tmp_path):
    """Reject existing foreign or dirty directories before mkdir/chmod/write."""
    foreign_dir = tmp_path / "foreign-profile"
    foreign_dir.mkdir()
    (foreign_dir / "unrelated_file.txt").write_text("user data")

    # Directory is non-empty and lacks McChat ownership marker
    with pytest.raises(ValueError, match="missing McChat ownership marker"):
        prepare_antigravity_profile(foreign_dir)

    # Global or T3 directory adoption rejected
    home = Path.home().resolve()
    with pytest.raises(ValueError, match="Cannot adopt global"):
        prepare_antigravity_profile(home / ".gemini")

    with pytest.raises(ValueError, match="Cannot adopt global"):
        prepare_antigravity_profile(home / ".t3")


def test_unsafe_mcp_and_hooks_rejected(tmp_path):
    """Reject profiles containing unsafe native MCP, hooks, skills, or symlinks."""
    owned_dir = tmp_path / "owned-profile"
    owned_dir.mkdir()
    (owned_dir / APP_OWNERSHIP_MARKER).write_text('{"app": "mcchat"}')

    # Unsafe MCP config
    (owned_dir / "config").mkdir()
    (owned_dir / "config" / "mcp_config.json").write_text('{"mcpServers": {}}')
    with pytest.raises(ValueError, match="unsafe native MCP"):
        prepare_antigravity_profile(owned_dir)

    (owned_dir / "config" / "mcp_config.json").unlink()

    # Unsafe hooks
    (owned_dir / "config" / "hooks.json").write_text('{"hooks": []}')
    with pytest.raises(ValueError, match="unsafe native MCP"):
        prepare_antigravity_profile(owned_dir)

    (owned_dir / "config" / "hooks.json").unlink()

    # Unsafe symlink
    target_file = tmp_path / "target.txt"
    target_file.write_text("hello")
    symlink_path = owned_dir / "link.txt"
    symlink_path.symlink_to(target_file)
    with pytest.raises(ValueError, match="Symlinks detected inside profile"):
        prepare_antigravity_profile(owned_dir)


def test_legitimate_profile_preserves_existing_settings_and_creates_backup(tmp_path):
    """Maintain nonsensitive app config while preserving existing unrelated settings & backups."""
    profile_dir = tmp_path / "valid-profile"
    profile_dir.mkdir()
    (profile_dir / APP_OWNERSHIP_MARKER).write_text('{"app": "mcchat"}')

    acp_dir = profile_dir / "antigravity-acp"
    acp_dir.mkdir()
    settings_file = acp_dir / "settings.json"
    initial_settings = {"custom_feature": True, "editor": "vim", "auth": {"type": "legacy-auth"}}
    settings_file.write_text(json.dumps(initial_settings))

    with patch(
        "open_webui.subscriptions.antigravity_profile.verify_browser_suppression",
        return_value="mock_browser",
    ):
        prof_info = prepare_antigravity_profile(profile_dir)

    # Check backup was created
    backup_file = acp_dir / "settings.json.mcchat_backup"
    assert backup_file.is_file()
    assert json.loads(backup_file.read_text()) == initial_settings

    # Check settings updated to oauth-personal while preserving other fields
    updated = json.loads(settings_file.read_text())
    assert updated["custom_feature"] is True
    assert updated["editor"] == "vim"
    assert updated["auth"]["type"] == "oauth-personal"

    # Subsequent call without changes should NOT rewrite settings
    mtime_before = settings_file.stat().st_mtime_ns
    with patch(
        "open_webui.subscriptions.antigravity_profile.verify_browser_suppression",
        return_value="mock_browser",
    ):
        prepare_antigravity_profile(profile_dir)
    mtime_after = settings_file.stat().st_mtime_ns
    assert mtime_before == mtime_after


# =========================================================================
# 2. Ambient Secrets Stripping Tests
# =========================================================================


def test_ambient_server_secrets_stripped(tmp_path):
    """Sanitize environment by allowing only SAFE_ENV_KEYS and stripping credentials."""
    prof_info = AntigravityProfileInfo(
        gemini_home=tmp_path / "home",
        acp_dir=tmp_path / "home" / "antigravity-acp",
        tmp_dir=tmp_path / "home" / "antigravity-acp" / "tmp",
        settings_path=tmp_path / "home" / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path="/mock/harness",
    )

    dirty_env = {
        "HOME": "/home/user",
        "PATH": "/usr/bin:/bin",
        "WEBUI_SECRET_KEY": "super-secret-webui-key",
        "DATABASE_URL": "postgresql://user:pass@localhost/db",
        "OPENAI_API_KEY": "sk-openai-12345",
        "ANTHROPIC_API_KEY": "sk-ant-12345",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-key",
        "GOOGLE_CLOUD_ACCESS_TOKEN": "ya29.secret",
        "GOOGLE_GENAI_USE_GCA": "1",
        "GEMINI_API_KEY": "ai-studio-key",
    }

    clean_env = build_antigravity_environment(prof_info, base_env=dirty_env)

    # Ensure none of the secrets leaked
    forbidden_keys = [
        "WEBUI_SECRET_KEY",
        "DATABASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_CLOUD_ACCESS_TOKEN",
        "GOOGLE_GENAI_USE_GCA",
        "GEMINI_API_KEY",
    ]
    for key in forbidden_keys:
        assert key not in clean_env

    # Ensure authoritative profile variables are set
    assert clean_env["GEMINI_HOME"] == str(prof_info.gemini_home)
    assert clean_env["AGY_ACP_FORCE_FILE_STORAGE"] == "1"
    assert clean_env["BROWSER"] == "echo %s"
    assert clean_env["PYTHONUNBUFFERED"] == "1"
    assert clean_env["ELECTRON_RUN_AS_NODE"] == "1"
    assert clean_env["HOME"] == "/home/user"


# =========================================================================
# 3. Callback URL Validation & SSRF Guard Tests
# =========================================================================


def test_callback_url_validation_and_ssrf_guard():
    """Validate loopback, port, single state/code, userinfo, fragment, and opaque encoding."""
    pending_redirect = "http://127.0.0.1:45678/"
    pending_state = "unique-state-token-123"

    # Valid loopback callback preserving opaque percent-encoded characters
    valid_callback = (
        "http://127.0.0.1:45678/?state=unique-state-token-123&code=4%2F0AX4X%26injected%3D1"
    )
    result = validate_antigravity_callback_url(
        pending_redirect_uri=pending_redirect,
        pending_state=pending_state,
        callback_url=valid_callback,
    )
    assert result.startswith("http://127.0.0.1:45678/?")
    assert "state=unique-state-token-123" in result
    assert "4%2F0AX4X%26injected%3D1" in result

    # SSRF / non-loopback rejected
    with pytest.raises(ValueError, match="127.0.0.1"):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://192.168.1.1:45678/?state=unique-state-token-123&code=abc"
        )

    # Port mismatch rejected
    with pytest.raises(ValueError, match="port does not match"):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://127.0.0.1:8080/?state=unique-state-token-123&code=abc"
        )

    # Path mismatch rejected
    with pytest.raises(ValueError, match="does not match the active sign-in endpoint"):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://127.0.0.1:45678/other?state=unique-state-token-123&code=abc"
        )

    # Userinfo rejected
    with pytest.raises(ValueError):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://user:pass@127.0.0.1:45678/?state=unique-state-token-123&code=abc"
        )

    # Fragment rejected
    with pytest.raises(ValueError):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://127.0.0.1:45678/?state=unique-state-token-123&code=abc#fragment"
        )

    # State mismatch rejected
    with pytest.raises(ValueError, match="Sign-in state mismatch"):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://127.0.0.1:45678/?state=wrong-state&code=abc"
        )

    # Duplicate state parameter rejected
    with pytest.raises(ValueError, match="Sign-in state mismatch"):
        validate_antigravity_callback_url(
            pending_redirect, pending_state, "http://127.0.0.1:45678/?state=unique-state-token-123&state=tamper&code=abc"
        )


def test_callback_relay_blocks_redirects():
    """Fail closed on HTTP redirects during loopback callback relay."""
    handler = NoRedirectHandler()
    with pytest.raises(ValueError, match="HTTP redirects are blocked"):
        handler.redirect_request(MagicMock(), None, 302, "Found", {}, "http://example.com")


# =========================================================================
# 4. Native Logout Capabilities & Empty Dict Handling Tests
# =========================================================================


@pytest.mark.asyncio
async def test_native_empty_logout_capability(tmp_path):
    """Recognize official 1.4.5 agentCapabilities.auth.logout={} presence as valid logout."""
    controller = AntigravityAuthController()
    fake_client = FakeAcpClient()
    fake_client.caps = {"auth": {"logout": {}}}  # official empty object

    fake_profile = AntigravityProfileInfo(
        gemini_home=tmp_path,
        acp_dir=tmp_path / "antigravity-acp",
        tmp_dir=tmp_path / "tmp",
        settings_path=tmp_path / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path=None,
    )
    fake_profile.acp_dir.mkdir(parents=True, exist_ok=True)
    (fake_profile.gemini_home / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')
    (tmp_path / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')
    token_file = fake_profile.acp_dir / "acp_token.json"
    token_file.write_text('{"token": "fake-token"}')

    with patch("open_webui.subscriptions.antigravity_auth.AcpClient", return_value=fake_client), \
         patch("open_webui.subscriptions.antigravity_auth.prepare_antigravity_profile", return_value=fake_profile), \
         patch("open_webui.subscriptions.antigravity_auth.build_antigravity_environment", return_value={}), \
         patch("open_webui.subscriptions.antigravity_auth.validate_binary_path", return_value="/bin/sh"):

        res = await controller.sign_out("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))

    # Verify truthful logout response and RPC invocation
    assert fake_client.logout_called == 1
    assert fake_client.client_info == {"name": "mcchat", "title": "McChat", "version": "0.13.0"}
    assert res["success"] is True
    assert res["phase"] == "idle"
    assert res["authenticated"] is False
    assert res["message"] == "Signed out from Google Antigravity."


# =========================================================================
# 5. Expiry & Cancellation Race Tests
# =========================================================================


@pytest.mark.asyncio
async def test_expired_auth_flow_cannot_mark_success(tmp_path):
    """Cancel and fence expired sign-in flows, preventing later success writes."""
    controller = AntigravityAuthController()
    began = asyncio.Event()
    release = asyncio.Event()

    class SlowAcpClient(FakeAcpClient):
        async def authenticate(self, **kwargs):
            began.set()
            await release.wait()
            return {}

    fake_profile = AntigravityProfileInfo(
        gemini_home=tmp_path,
        acp_dir=tmp_path / "antigravity-acp",
        tmp_dir=tmp_path / "tmp",
        settings_path=tmp_path / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path=None,
    )
    fake_profile.acp_dir.mkdir(parents=True, exist_ok=True)
    (fake_profile.gemini_home / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')

    with patch("open_webui.subscriptions.antigravity_auth.AcpClient", SlowAcpClient), \
         patch("open_webui.subscriptions.antigravity_auth.prepare_antigravity_profile", return_value=fake_profile), \
         patch("open_webui.subscriptions.antigravity_auth.build_antigravity_environment", return_value={}), \
         patch("open_webui.subscriptions.antigravity_auth.validate_binary_path", return_value="/bin/sh"):

        await controller.start_sign_in("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        await asyncio.wait_for(began.wait(), timeout=1.0)

        flow = controller._active_flow
        assert flow is not None
        # Simulate expiry
        flow.expires_at = time.time() - 1

        snapshot = controller.get_flow_snapshot("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        assert snapshot["phase"] == "failed"
        assert flow.cancelled is True

        # Now release the slow auth call
        release.set()
        await asyncio.gather(flow.task, return_exceptions=True)

    # Expired flow must NOT authenticate the instance!
    account_state = controller.get_account_state(binary_path="/bin/sh", config_dir=str(tmp_path))
    assert account_state["authenticated"] is False
    assert flow.phase in ("failed", "cancelled")


@pytest.mark.asyncio
async def test_starting_new_flow_cancels_and_reaps_old_flow(tmp_path):
    """Starting a replacement sign-in flow cancels and closes the prior active flow."""
    controller = AntigravityAuthController()
    first_client = FakeAcpClient()
    second_client = FakeAcpClient()
    clients = [first_client, second_client]

    fake_profile = AntigravityProfileInfo(
        gemini_home=tmp_path,
        acp_dir=tmp_path / "antigravity-acp",
        tmp_dir=tmp_path / "tmp",
        settings_path=tmp_path / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path=None,
    )
    fake_profile.acp_dir.mkdir(parents=True, exist_ok=True)
    (fake_profile.gemini_home / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')

    with patch("open_webui.subscriptions.antigravity_auth.AcpClient", side_effect=clients), \
         patch("open_webui.subscriptions.antigravity_auth.prepare_antigravity_profile", return_value=fake_profile), \
         patch("open_webui.subscriptions.antigravity_auth.build_antigravity_environment", return_value={}), \
         patch("open_webui.subscriptions.antigravity_auth.validate_binary_path", return_value="/bin/sh"):

        snap1 = await controller.start_sign_in("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        first_flow = controller._active_flow
        assert first_flow is not None

        # Start replacement
        snap2 = await controller.start_sign_in("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        assert first_flow.cancelled is True
        assert first_client.closed is True
        assert snap2["flow_id"] != snap1["flow_id"]


# =========================================================================
# 6. Profile Scoping & Restart State Tests
# =========================================================================


def test_profile_scoping_and_restart_state(tmp_path):
    """Verified models and status are scoped to exact profile and survive controller restarts."""
    controller1 = AntigravityAuthController()

    prof_a = tmp_path / "profile-a"
    prof_b = tmp_path / "profile-b"
    prof_a.mkdir()
    prof_b.mkdir()
    (prof_a / "antigravity-acp").mkdir()
    (prof_b / "antigravity-acp").mkdir()

    for profile in (prof_a, prof_b):
        (profile / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')

    bin_a = "/bin/sh"
    bin_b = "/bin/bash"

    models_a = [{"id": "model-a1", "name": "Model A1"}, {"id": "model-a2", "name": "Model A2"}]
    controller1.set_authenticated_models(models_a, binary_path=bin_a, config_dir=str(prof_a), version="1.4.5")

    # Profile A should be authenticated
    state_a = controller1.get_account_state(binary_path=bin_a, config_dir=str(prof_a))
    assert state_a["authenticated"] is True
    assert len(state_a["models"]) == 2

    # Profile B with different config_dir or binary must NOT be authenticated!
    state_b = controller1.get_account_state(binary_path=bin_b, config_dir=str(prof_b))
    assert state_b["authenticated"] is False
    assert state_b["models"] == []

    # Simulate worker restart by creating a new controller instance
    controller2 = AntigravityAuthController()
    restored_a = controller2.get_account_state(binary_path=bin_a, config_dir=str(prof_a))
    assert restored_a["authenticated"] is True
    assert restored_a["models"] == models_a

    # Clear state on Profile A
    controller2.clear_authenticated_state(binary_path=bin_a, config_dir=str(prof_a))
    cleared_a = controller2.get_account_state(binary_path=bin_a, config_dir=str(prof_a))
    assert cleared_a["authenticated"] is False


# =========================================================================
# 7. UI Auth-Only Refresh Handlers Test (Node VM)
# =========================================================================


def test_ui_auth_only_refresh_handlers_via_node():
    """Verify in Node that auth-only transitions refresh models/status without config writes."""
    node_script = """
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const src = fs.readFileSync('src/lib/components/admin/Settings/Connections.svelte', 'utf8');
const ra = src.indexOf('let refreshingSubscriptionsAuth = false;');
const a = src.indexOf('const updateSubscriptionsHandler =');
const b = src.indexOf('const updateConnectionsHandler =', a);
assert.ok(ra >= 0 && a > ra && b > a);

const config = {
  ENABLE_SUBSCRIPTIONS: true,
  SUBSCRIPTIONS_OWNER_ID: 'owner',
  current_user_id: 'owner',
  SUBSCRIPTIONS_PROVIDERS: { gemini: { enabled: true } }
};
const calls = { settingsWrites: 0, modelRefresh: 0, statusRefresh: 0 };
const context = vm.createContext({
  subscriptionsConfig: config,
  initialSubscriptionsSnapshot: JSON.stringify(config),
  savingSubscriptions: false,
  toast: { error: () => {}, success: () => {} },
  $i18n: { t: v => v },
  localStorage: { token: 'test-token' },
  updateSubscriptionsConfig: async () => { calls.settingsWrites++; return config; },
  models: { set: async () => {} },
  getModels: async () => { calls.modelRefresh++; return []; },
  getSubscriptionProviders: async () => { calls.statusRefresh++; return []; },
  getSubscriptionsConfig: async () => config
});

// Execute auth-only refresh handler
const authHandler = new vm.Script(src.slice(ra, a) + '; refreshSubscriptionsAuthHandler').runInContext(context);
await authHandler();
assert.equal(calls.settingsWrites, 0, 'auth refresh must not trigger settings writes');
assert.equal(calls.modelRefresh, 1, 'models must refresh on auth update');
assert.equal(calls.statusRefresh, 1, 'providers status must refresh on auth update');

// Execute updateSubscriptionsHandler with unchanged config (should be no-op)
const updateHandler = new vm.Script(src.slice(a, b) + '; updateSubscriptionsHandler').runInContext(context);
await updateHandler();
assert.equal(calls.settingsWrites, 0, 'unchanged config must not trigger settings save');

// Execute updateSubscriptionsHandler after actual configuration change
config.SUBSCRIPTIONS_PROVIDERS.gemini.enabled = false;
await updateHandler();
assert.equal(calls.settingsWrites, 1, 'actual config change must trigger save');
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# =========================================================================
# 8. Route-Level Owner Denial & Isolated Dependency Tests
# =========================================================================


def _get_isolated_router():
    """Import router with isolated auth dependencies without loading DB or ChromaDB."""
    import types
    if "open_webui.utils.auth" not in sys.modules:
        auth_mock = types.ModuleType("open_webui.utils.auth")
        auth_mock.get_admin_user = lambda: None
        auth_mock.get_verified_user = lambda: None
        sys.modules["open_webui.utils.auth"] = auth_mock

    from open_webui.routers.subscriptions import (
        _require_owner,
        start_provider_signin,
        get_provider_auth_status,
        complete_provider_signin,
        cancel_provider_signin,
        logout_provider,
        ProviderAuthCallbackForm,
        ProviderAuthCancelForm,
    )
    return {
        "_require_owner": _require_owner,
        "start_provider_signin": start_provider_signin,
        "get_provider_auth_status": get_provider_auth_status,
        "complete_provider_signin": complete_provider_signin,
        "cancel_provider_signin": cancel_provider_signin,
        "logout_provider": logout_provider,
        "ProviderAuthCallbackForm": ProviderAuthCallbackForm,
        "ProviderAuthCancelForm": ProviderAuthCancelForm,
    }


@pytest.mark.asyncio
async def test_route_owner_denial():
    """Enforce strict owner-only access across all subscription auth route endpoints."""
    routes = _get_isolated_router()
    _require_owner = routes["_require_owner"]
    from fastapi import HTTPException

    owner_user = SimpleNamespace(id="owner_1", role="admin")
    attacker_user = SimpleNamespace(id="attacker_2", role="admin")

    with patch("open_webui.subscriptions.manager.subscription_manager.get_config_dict", new_callable=AsyncMock) as mock_cfg:
        # Case 1: Unclaimed ownership
        mock_cfg.return_value = {"SUBSCRIPTIONS_OWNER_ID": ""}
        with pytest.raises(HTTPException) as exc_info:
            await _require_owner(owner_user)
        assert exc_info.value.status_code == 403
        assert "claimed before managing authentication" in exc_info.value.detail

        # Case 2: Claimed ownership, non-owner caller
        mock_cfg.return_value = {"SUBSCRIPTIONS_OWNER_ID": "owner_1"}
        with pytest.raises(HTTPException) as exc_info:
            await _require_owner(attacker_user)
        assert exc_info.value.status_code == 403
        assert "Only the designated subscription owner" in exc_info.value.detail

        # Case 3: Claimed ownership, verified owner caller
        owner_id = await _require_owner(owner_user)
        assert owner_id == "owner_1"


@pytest.mark.asyncio
async def test_route_endpoints_reject_non_owner():
    """Verify route functions reject non-owner requests with 403."""
    routes = _get_isolated_router()
    from fastapi import HTTPException

    attacker_user = SimpleNamespace(id="attacker_2", role="admin")
    with patch("open_webui.subscriptions.manager.subscription_manager.get_config_dict", new_callable=AsyncMock) as mock_cfg:
        mock_cfg.return_value = {"SUBSCRIPTIONS_OWNER_ID": "owner_1"}

        with pytest.raises(HTTPException) as exc:
            await routes["start_provider_signin"]("gemini", user=attacker_user)
        assert exc.value.status_code == 403

        with pytest.raises(HTTPException) as exc:
            await routes["get_provider_auth_status"]("gemini", user=attacker_user)
        assert exc.value.status_code == 403

        form = routes["ProviderAuthCallbackForm"](flow_id="f1", callback_url="http://127.0.0.1:45678/?state=s&code=c")
        with pytest.raises(HTTPException) as exc:
            await routes["complete_provider_signin"](MagicMock(), "gemini", form_data=form, user=attacker_user)
        assert exc.value.status_code == 403

        cancel_form = routes["ProviderAuthCancelForm"](flow_id="f1")
        with pytest.raises(HTTPException) as exc:
            await routes["cancel_provider_signin"]("gemini", form_data=cancel_form, user=attacker_user)
        assert exc.value.status_code == 403

        with pytest.raises(HTTPException) as exc:
            await routes["logout_provider"](MagicMock(), "gemini", user=attacker_user)
        assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_route_signin_resolves_actual_binary():
    """Verify start_provider_signin resolves actual binary via manager without gemini CLI fallback."""
    routes = _get_isolated_router()
    owner_user = SimpleNamespace(id="owner_1", role="admin")

    mock_cfg = SimpleNamespace(binary_path="", config_dir="")
    fake_adapter = MagicMock()
    fake_adapter.is_installed.return_value = (True, "/mock/detected/agy_acp_server.par")

    with patch("open_webui.subscriptions.manager.subscription_manager.get_config_dict", new_callable=AsyncMock, return_value={"SUBSCRIPTIONS_OWNER_ID": "owner_1"}), \
         patch("open_webui.subscriptions.manager.subscription_manager.get_runtime_config", new_callable=AsyncMock, return_value=(True, "owner_1", 180, {"gemini": mock_cfg})), \
         patch("open_webui.subscriptions.manager.subscription_manager.get_adapter", return_value=fake_adapter), \
         patch("open_webui.subscriptions.antigravity_auth.antigravity_auth_controller.start_sign_in", new_callable=AsyncMock) as mock_start:

        mock_start.return_value = {"phase": "starting"}
        res = await routes["start_provider_signin"]("gemini", user=owner_user)
        assert res["phase"] == "starting"
        mock_start.assert_called_once_with(
            owner_user_id="owner_1",
            binary_path="/mock/detected/agy_acp_server.par",
            config_dir="",
        )


@pytest.mark.asyncio
async def test_route_cache_invalidation_on_verified_changes():
    """Verify callback success and logout invalidate shared model caches and publish events."""
    routes = _get_isolated_router()
    owner_user = SimpleNamespace(id="owner_1", role="admin")
    mock_request = MagicMock()
    mock_request.app.state.redis = None
    mock_request.app.state.BASE_MODELS = ["old-model"]
    mock_request.app.state.MODELS = {"old-model": {}}

    form = routes["ProviderAuthCallbackForm"](flow_id="f1", callback_url="http://127.0.0.1:45678/?state=s&code=c")

    with patch("open_webui.subscriptions.manager.subscription_manager.get_config_dict", new_callable=AsyncMock, return_value={"SUBSCRIPTIONS_OWNER_ID": "owner_1"}), \
         patch("open_webui.subscriptions.manager.subscription_manager.get_runtime_config", new_callable=AsyncMock, return_value=(True, "owner_1", 180, {})), \
         patch("open_webui.subscriptions.manager.subscription_manager.get_adapter", return_value=None), \
         patch("open_webui.subscriptions.antigravity_auth.antigravity_auth_controller.complete_sign_in", new_callable=AsyncMock, return_value={"phase": "succeeded", "authenticated": True}), \
         patch("open_webui.subscriptions.antigravity_auth.antigravity_auth_controller.sign_out", new_callable=AsyncMock, return_value={"phase": "idle", "authenticated": False}), \
         patch("open_webui.routers.subscriptions._invalidate_models_cache", new_callable=AsyncMock) as mock_inval:

        await routes["complete_provider_signin"](mock_request, "gemini", form_data=form, user=owner_user)
        assert mock_inval.call_count == 1

        await routes["logout_provider"](mock_request, "gemini", user=owner_user)
        assert mock_inval.call_count == 2


@pytest.mark.asyncio
async def test_config_update_asgi_route_is_unique_and_requires_admin():
    """Exercise actual route matching so private helpers cannot shadow admin updates."""
    _get_isolated_router()
    from collections import Counter
    from fastapi import FastAPI, HTTPException
    from httpx import ASGITransport, AsyncClient
    from open_webui.routers import subscriptions as module
    app = FastAPI()
    app.include_router(module.router, prefix='/subscriptions')
    registered = Counter((route.path, method) for route in app.routes for method in getattr(route, 'methods', []))
    assert all(count == 1 for count in registered.values())
    owner = SimpleNamespace(id='owner', role='admin')
    async def require_admin():
        raise HTTPException(401, 'Authentication required')
    app.dependency_overrides[module.get_admin_user] = require_admin
    body = {'ENABLE_SUBSCRIPTIONS': False, 'SUBSCRIPTIONS_OWNER_ID': 'owner',
            'SUBSCRIPTIONS_PROVIDERS': {}}
    with patch.object(module.subscription_manager, 'update_config', new=AsyncMock(return_value={'ENABLE_SUBSCRIPTIONS': False})) as update, \
         patch.object(module, '_invalidate_models_cache', new=AsyncMock()) as invalidate:
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://mcchat.invalid') as client:
            denied = await client.post('/subscriptions/config/update?user=untrusted', json=body)
            assert denied.status_code == 401
            update.assert_not_awaited()
            invalidate.assert_not_awaited()
            app.dependency_overrides[module.get_admin_user] = lambda: owner
            accepted = await client.post('/subscriptions/config/update', json=body)
            assert accepted.status_code == 200, accepted.text
            assert accepted.json()['current_user_id'] == 'owner'
            update.assert_awaited_once()
            invalidate.assert_awaited_once()


# =========================================================================
# 9. Round 9 Regressions: Concurrency, CAS, Symlinks, Bad Settings, Logout
# =========================================================================


@pytest.mark.asyncio
async def test_gated_concurrency_replacement_race(tmp_path):
    """Verify serialized lifecycle transition lock closes the replacement race (L8-02)."""
    controller = AntigravityAuthController()
    gate = asyncio.Event()

    class ControlledAcpClient(FakeAcpClient):
        async def spawn(self, *args, **kwargs):
            await gate.wait()
            return await super().spawn(*args, **kwargs)

    fake_profile = AntigravityProfileInfo(
        gemini_home=tmp_path,
        acp_dir=tmp_path / "antigravity-acp",
        tmp_dir=tmp_path / "tmp",
        settings_path=tmp_path / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path=None,
    )

    with patch("open_webui.subscriptions.antigravity_auth.AcpClient", side_effect=lambda **kw: ControlledAcpClient(**kw)), \
         patch("open_webui.subscriptions.antigravity_auth.prepare_antigravity_profile", return_value=fake_profile), \
         patch("open_webui.subscriptions.antigravity_auth.build_antigravity_environment", return_value={}), \
         patch("open_webui.subscriptions.antigravity_auth.validate_binary_path", return_value="/bin/sh"):

        flow1 = await controller.start_sign_in("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        f1_id = flow1["flow_id"]
        assert controller._active_flow is not None
        assert controller._active_flow.flow_id == f1_id

        # Concurrently start replacement flow
        gate.set()
        flow2 = await controller.start_sign_in("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        f2_id = flow2["flow_id"]

        assert f2_id != f1_id
        assert controller._active_flow.flow_id == f2_id
        snap = controller.get_flow_snapshot("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        assert snap["flow_id"] == f2_id


def test_cross_controller_freshness_and_cas_fencing(tmp_path):
    """Verify separate controller instances observe logout/new catalog and CAS fences stale verification (L8-03, SEC8-03)."""
    prof = tmp_path / "cas-profile"
    prof.mkdir()
    (prof / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat","version":1}\n')
    (prof / "antigravity-acp").mkdir()

    bin_path = "/bin/sh"
    controller_a = AntigravityAuthController()
    controller_b = AntigravityAuthController()

    kwargs = {"binary_path": bin_path, "config_dir": str(prof)}

    # Controller A records initial authenticated models
    models_v1 = [{"id": "model-1", "name": "Model 1"}]
    ok1 = controller_a.set_authenticated_models(models_v1, **kwargs, version="1.5.0")
    assert ok1 is True

    # Controller B reads fresh state from disk
    state_b1 = controller_b.get_account_state(**kwargs)
    assert state_b1["authenticated"] is True
    assert state_b1["models"] == models_v1
    assert state_b1["version"] == "1.5.0"
    epoch_v1 = state_b1["epoch"]

    # Controller A clears authentication (logout writes revocation tombstone with new epoch)
    controller_a.clear_authenticated_state(**kwargs)

    # Controller B immediately observes revocation on fresh read
    state_b2 = controller_b.get_account_state(**kwargs)
    assert state_b2["authenticated"] is False
    assert state_b2["models"] == []
    assert state_b2["version"] is None
    assert state_b2["epoch"] > epoch_v1

    # Stale in-flight verification with old expected_epoch fails CAS and does not overwrite tombstone
    stale_models = [{"id": "stale-model", "name": "Stale Model"}]
    cas_res = controller_b.set_authenticated_models(stale_models, **kwargs, expected_epoch=epoch_v1)
    assert cas_res is False

    # State remains revoked
    state_b3 = controller_b.get_account_state(**kwargs)
    assert state_b3["authenticated"] is False
    assert state_b3["models"] == []


def test_badsettings_preserved_byte_exact_and_auth_null_handled(tmp_path):
    """Preserve malformed settings byte-exact, reject non-JSON objects, handle auth:null/string, and protect prior backups (PC8-05)."""
    # Case 1: Malformed JSON settings.json must be preserved byte-exact and fail closed
    prof1 = tmp_path / "bad-json-prof"
    prof1.mkdir()
    (prof1 / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat","version":1}\n')
    (prof1 / "antigravity-acp").mkdir()
    malformed_content = b"not valid json\ncustom-private-setting=preserve-me\n"
    (prof1 / "antigravity-acp" / "settings.json").write_bytes(malformed_content)

    with patch("open_webui.subscriptions.antigravity_profile.verify_browser_suppression", return_value="mock"):
        with pytest.raises(ValueError, match="Antigravity settings must contain a JSON object"):
            prepare_antigravity_profile(prof1)

    assert (prof1 / "antigravity-acp" / "settings.json").read_bytes() == malformed_content
    assert not (prof1 / "antigravity-acp" / "settings.json.mcchat_backup").exists()

    # Case 2: JSON array instead of object
    prof2 = tmp_path / "array-json-prof"
    prof2.mkdir()
    (prof2 / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat","version":1}\n')
    (prof2 / "antigravity-acp").mkdir()
    array_content = b'["item1", "item2"]\n'
    (prof2 / "antigravity-acp" / "settings.json").write_bytes(array_content)

    with patch("open_webui.subscriptions.antigravity_profile.verify_browser_suppression", return_value="mock"):
        with pytest.raises(ValueError, match="Antigravity settings must contain a JSON object"):
            prepare_antigravity_profile(prof2)
    assert (prof2 / "antigravity-acp" / "settings.json").read_bytes() == array_content

    # Case 3: auth is null, custom keys preserved, backup created
    prof3 = tmp_path / "auth-null-prof"
    prof3.mkdir()
    (prof3 / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat","version":1}\n')
    (prof3 / "antigravity-acp").mkdir()
    null_auth_content = b'{"editor": "vim", "auth": null}\n'
    (prof3 / "antigravity-acp" / "settings.json").write_bytes(null_auth_content)

    with patch("open_webui.subscriptions.antigravity_profile.verify_browser_suppression", return_value="mock"):
        prepare_antigravity_profile(prof3)

    backup_file = prof3 / "antigravity-acp" / "settings.json.mcchat_backup"
    assert backup_file.is_file()
    assert backup_file.read_bytes() == null_auth_content

    updated = json.loads((prof3 / "antigravity-acp" / "settings.json").read_text())
    assert updated["editor"] == "vim"
    assert updated["auth"]["type"] == "oauth-personal"

    # Case 4: Subsequent run does not overwrite prior backup
    (prof3 / "antigravity-acp" / "settings.json").write_bytes(b'{"editor": "vim", "auth": "legacy-auth"}\n')
    with patch("open_webui.subscriptions.antigravity_profile.verify_browser_suppression", return_value="mock"):
        prepare_antigravity_profile(prof3)

    assert backup_file.read_bytes() == null_auth_content
    sec_backups = list((prof3 / "antigravity-acp").glob("settings.json.mcchat_backup.*"))
    assert len(sec_backups) >= 1


def test_backupsymlink_target_outside_profile_rejected(tmp_path):
    """Reject nested symlinked backup files without reading or overwriting outside targets (SEC8-02)."""
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_text("synthetic sentinel content")

    prof = tmp_path / "symlink-prof"
    prof.mkdir()
    (prof / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat","version":1}\n')
    acp = prof / "antigravity-acp"
    acp.mkdir()
    (acp / "settings.json").write_text('{"auth": {"type": "gemini-api-key"}}')

    # settings.json.mcchat_backup is a symlink to outside file
    (acp / "settings.json.mcchat_backup").symlink_to(outside)

    with patch("open_webui.subscriptions.antigravity_profile.verify_browser_suppression", return_value="mock"):
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            prepare_antigravity_profile(prof)

    # Outside file must remain untouched
    assert outside.read_text() == "synthetic sentinel content"


def test_malformed_account_metadata_rejected(tmp_path):
    """Reject foreign profile metadata, strings pretending booleans, and oversized state data (SEC8-03)."""
    prof = tmp_path / "meta-prof"
    prof.mkdir()
    (prof / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat","version":1}\n')
    acp = prof / "antigravity-acp"
    acp.mkdir()
    state_file = acp / "account_state.json"

    controller = AntigravityAuthController()
    kwargs = {"binary_path": "/bin/sh", "config_dir": str(prof)}
    pid = profile_identity(**kwargs)

    # Case 1: authenticated is a string "true" instead of bool True
    state_file.write_text(json.dumps({
        "profile_identity": pid,
        "authenticated": "true",
        "models": [{"id": "m1", "name": "M1"}],
        "version": "1.0",
        "epoch": 1,
    }))
    assert controller.get_account_state(**kwargs)["authenticated"] is False

    # Case 2: foreign profile identity
    state_file.write_text(json.dumps({
        "profile_identity": "antigravity:foreign_id",
        "authenticated": True,
        "models": [{"id": "m1", "name": "M1"}],
        "version": "1.0",
        "epoch": 1,
    }))
    assert controller.get_account_state(**kwargs)["authenticated"] is False

    # Case 3: model id contains control characters
    state_file.write_text(json.dumps({
        "profile_identity": pid,
        "authenticated": True,
        "models": [{"id": "m1\x00bad", "name": "M1"}],
        "version": "1.0",
        "epoch": 1,
    }))
    assert controller.get_account_state(**kwargs)["authenticated"] is False

    # Case 4: oversized metadata (>128KB)
    state_file.write_text(" " * 140000)
    assert controller.get_account_state(**kwargs)["authenticated"] is False


@pytest.mark.asyncio
async def test_native_failure_logout_reports_failure_honestly(tmp_path):
    """Sign-out reports failure if native logout could not be performed; no manual token deletion (SEC8-05)."""
    controller = AntigravityAuthController()

    fake_profile = AntigravityProfileInfo(
        gemini_home=tmp_path,
        acp_dir=tmp_path / "antigravity-acp",
        tmp_dir=tmp_path / "tmp",
        settings_path=tmp_path / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path=None,
    )
    fake_profile.acp_dir.mkdir(parents=True, exist_ok=True)
    (fake_profile.gemini_home / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')
    token_file = fake_profile.acp_dir / "acp_token.json"
    token_file.write_text('{"token": "fake-token"}')

    # Simulate dirty profile causing prepare_antigravity_profile to fail
    with patch("open_webui.subscriptions.antigravity_auth.prepare_antigravity_profile", side_effect=ValueError("Profile contains unsafe native MCP")), \
         patch("open_webui.subscriptions.antigravity_auth.validate_binary_path", return_value="/bin/sh"):
        res = await controller.sign_out("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))

    # Verify failure reported honestly
    assert res["success"] is False
    assert res["phase"] == "failed"
    assert "revalidation required" in res["message"] or "Login state unknown" in res["message"]
    # Token file must NOT be manually deleted
    assert token_file.exists()
    # Local public verification must still be cleared
    assert res["authenticated"] is False
    snap = controller.get_account_state(binary_path="/bin/sh", config_dir=str(tmp_path))
    assert snap["authenticated"] is False


@pytest.mark.asyncio
async def test_synthetic_diagnostic_redaction_in_auth_errors(tmp_path):
    """Auth failures redact sensitive OAuth URLs, codes, states, and bearer tokens (SEC8-06)."""
    controller = AntigravityAuthController()

    sensitive_leak = (
        "Upstream OAuth failure: https://accounts.google.com/o/oauth2/v2/auth?state=LEAKED_STATE&code=LEAKED_CODE "
        "Authorization: Bearer ya29.LEAKED_SECRET_TOKEN file=/home/betheenvy/secret/token.json"
    )

    class LeakingClient(FakeAcpClient):
        async def authenticate(self, **kwargs):
            raise AcpRequestError(1001, sensitive_leak)

    fake_profile = AntigravityProfileInfo(
        gemini_home=tmp_path,
        acp_dir=tmp_path / "antigravity-acp",
        tmp_dir=tmp_path / "tmp",
        settings_path=tmp_path / "antigravity-acp" / "settings.json",
        browser_command="echo %s",
        harness_path=None,
    )

    with patch("open_webui.subscriptions.antigravity_auth.AcpClient", return_value=LeakingClient()), \
         patch("open_webui.subscriptions.antigravity_auth.prepare_antigravity_profile", return_value=fake_profile), \
         patch("open_webui.subscriptions.antigravity_auth.build_antigravity_environment", return_value={}), \
         patch("open_webui.subscriptions.antigravity_auth.validate_binary_path", return_value="/bin/sh"):

        await controller.start_sign_in("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        if controller._active_flow and controller._active_flow.task:
            await asyncio.gather(controller._active_flow.task, return_exceptions=True)

        snap = controller.get_flow_snapshot("owner_1", binary_path="/bin/sh", config_dir=str(tmp_path))
        assert snap["phase"] == "failed"
        msg = snap["message"] or ""

        # None of the sensitive markers must leak into the user message
        assert "LEAKED_STATE" not in msg
        assert "LEAKED_CODE" not in msg
        assert "LEAKED_SECRET_TOKEN" not in msg
        assert "https://accounts.google.com" not in msg
        assert "/home/betheenvy/secret" not in msg


@pytest.mark.asyncio
async def test_unsupported_provider_auth_polling_returns_400_or_404():
    """Unsupported provider polling returns 400 or 404, never authenticated:true (SEC8-08)."""
    from fastapi import HTTPException

    routes = _get_isolated_router()
    owner_user = SimpleNamespace(id="owner_1", role="admin")

    with patch("open_webui.subscriptions.manager.subscription_manager.get_config_dict", new_callable=AsyncMock) as mock_cfg:
        mock_cfg.return_value = {
            "ENABLE_SUBSCRIPTIONS": True,
            "SUBSCRIPTIONS_OWNER_ID": "owner_1",
            "SUBSCRIPTIONS_PROVIDERS": {},
        }

        # Claude auth polling -> 400 Bad Request
        with pytest.raises(HTTPException) as exc1:
            await routes["get_provider_auth_status"]("claude", user=owner_user)
        assert exc1.value.status_code == 400

        # Codex auth polling -> 400 Bad Request
        with pytest.raises(HTTPException) as exc2:
            await routes["get_provider_auth_status"]("codex", user=owner_user)
        assert exc2.value.status_code == 400

        # Unknown provider auth polling -> 404 Not Found
        with pytest.raises(HTTPException) as exc3:
            await routes["get_provider_auth_status"]("does-not-exist", user=owner_user)
        assert exc3.value.status_code == 404


def _concurrent_metadata_claim(profile, barrier, results):
    controller = AntigravityAuthController()
    barrier.wait(timeout=5)
    results.put(controller.set_authenticated_models(
        [{"id": "verified", "name": "Verified"}],
        binary_path="/bin/sh", config_dir=profile, expected_epoch=0,
    ))


def test_metadata_compare_and_set_is_atomic_across_processes(tmp_path):
    """Two workers verifying revision 0 cannot both commit a successful claim."""
    import multiprocessing

    profile = tmp_path / "owned"
    prepare_antigravity_profile(profile)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    workers = [ctx.Process(target=_concurrent_metadata_claim, args=(str(profile), barrier, results)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        outcomes = [results.get(timeout=10) for _ in workers]
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
        assert sorted(outcomes) == [False, True]
        controller = AntigravityAuthController()
        kwargs = {"binary_path": "/bin/sh", "config_dir": str(profile)}
        revision = controller.get_account_state(**kwargs)["epoch"]
        assert controller.clear_authenticated_state(**kwargs)
        assert not AntigravityAuthController().set_authenticated_models(
            [{"id": "stale", "name": "Stale"}], **kwargs, expected_epoch=revision,
        )
        assert controller.get_account_state(**kwargs)["authenticated"] is False
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)
        results.close()
        results.join_thread()


def test_metadata_never_adopts_unmarked_or_symlinked_profile(tmp_path):
    """A matching fingerprint cannot authorize foreign or symlinked public state."""
    controller = AntigravityAuthController()
    foreign = tmp_path / "foreign"
    acp = foreign / "antigravity-acp"
    acp.mkdir(parents=True)
    kwargs = {"binary_path": "/bin/sh", "config_dir": str(foreign)}
    state_file = acp / "account_state.json"
    payload = json.dumps({
        "profile_identity": profile_identity(**kwargs), "authenticated": True,
        "models": [{"id": "foreign", "name": "Foreign"}], "version": None, "epoch": 1,
    })
    state_file.write_text(payload)
    assert not controller.get_account_state(**kwargs)["authenticated"]
    assert not controller.set_authenticated_models([], **kwargs)
    assert not controller.clear_authenticated_state(**kwargs)
    with pytest.raises(ValueError, match="ownership marker"):
        prepare_antigravity_profile(foreign)
    assert state_file.read_text() == payload

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / APP_OWNERSHIP_MARKER).write_text('{"app":"mcchat"}')
    (owned / "antigravity-acp").symlink_to(acp, target_is_directory=True)
    symlinked = {"binary_path": "/bin/sh", "config_dir": str(owned)}
    assert not controller.get_account_state(**symlinked)["authenticated"]
    assert not controller.set_authenticated_models([], **symlinked)
    assert not controller.clear_authenticated_state(**symlinked)
    assert state_file.read_text() == payload


def test_native_reasoning_metadata_roundtrip_and_size_bound(tmp_path):
    profile = tmp_path / "owned"
    prepare_antigravity_profile(profile)
    controller = AntigravityAuthController()
    kwargs = {"binary_path": "/bin/sh", "config_dir": str(profile)}
    models = [{"id": "native", "name": "Native", "reasoning_efforts": [
        {"value": "minimal", "name": "Minimal"}, {"value": "extra_high", "name": "Extra high"},
    ]}]
    assert controller.set_authenticated_models(models, **kwargs, version="native-version")
    assert AntigravityAuthController().get_account_state(**kwargs)["models"] == models
    before = (profile / "antigravity-acp" / "account_state.json").read_bytes()
    assert not controller.set_authenticated_models([{"id": 42, "name": "Invalid"}], **kwargs)
    assert not controller.set_authenticated_models([{"id": "bad\x00id", "name": "Invalid"}], **kwargs)
    assert not controller.set_authenticated_models([
        {"id": str(index) + "x" * 250, "name": "x" * 256} for index in range(500)
    ], **kwargs)
    assert (profile / "antigravity-acp" / "account_state.json").read_bytes() == before


@pytest.mark.asyncio
async def test_native_logout_does_not_race_another_workers_signin(tmp_path):
    profile = tmp_path / "owned"
    prepare_antigravity_profile(profile)
    controller = AntigravityAuthController()
    other_worker = AntigravityAuthController()
    fake_client = FakeAcpClient()
    kwargs = {"binary_path": "/bin/sh", "config_dir": str(profile)}
    assert controller.set_authenticated_models([], **kwargs)
    with other_worker._profile_operation_lock(str(profile)), \
         patch("open_webui.subscriptions.antigravity_auth._resolve_actual_binary", return_value="/bin/sh"), \
         patch("open_webui.subscriptions.antigravity_auth.AcpClient", return_value=fake_client):
        result = await controller.sign_out("owner", **kwargs)
    assert result["success"] is False
    assert fake_client.logout_called == 0
    assert not controller.get_account_state(**kwargs)["authenticated"]


def test_signout_ui_reports_native_failure_and_refreshes_models():
    """Execute the actual UI handler against both native logout outcomes."""
    source = REPO_ROOT / "src/lib/components/admin/Settings/Connections/SubscriptionConnection.svelte"
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const match = source.match(/const handleSignOut = async \(\) => \{([\s\S]*?)\n\t\};/);
assert.ok(match);
const body = match[1].replace(/catch \(err: any\)/g, 'catch (err)');
(async () => {
  for (const success of [false, true]) {
    const calls = { success: 0, error: 0, refresh: 0 };
    const context = {
      signingOut: false, authFlow: {}, returnUrl: 'sensitive-return-url',
      stopPolling() {}, localStorage: {token:'synthetic'}, provider: {id:'gemini'},
      providerSignOut: async () => ({success, message:'Revalidation required'}),
      toast: {success() { calls.success++; }, error() { calls.error++; }},
      $i18n: {t: value => value}, notifyAuthChanged() { calls.refresh++; },
    };
    vm.createContext(context);
    await vm.runInContext('(async () => {' + body + '})()', context);
    assert.equal(calls.success, success ? 1 : 0);
    assert.equal(calls.error, success ? 0 : 1);
    assert.equal(calls.refresh, 1);
    assert.equal(context.returnUrl, '');
    assert.equal(context.signingOut, false);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(["node", "-e", script, str(source)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_catalog_updates_keep_auth_generation_and_signin_logout_fence_old_work(tmp_path):
    binary = tmp_path / 'agy_acp_server'
    binary.write_text('# synthetic identity only\n'); binary.chmod(0o700)
    profile = tmp_path / 'owned'; prepare_antigravity_profile(profile, binary)
    controller = AntigravityAuthController()
    kwargs = {'binary_path': str(binary), 'config_dir': str(profile)}
    identity = profile_identity(**kwargs)
    models = [{'id': 'native', 'name': 'Native'}]
    assert controller.set_authenticated_models(models, **kwargs, expected_epoch=0, expected_profile_identity=identity)
    epoch = controller.get_account_state(**kwargs)['epoch']
    assert controller.set_authenticated_models(models, **kwargs, expected_epoch=epoch,
        expected_profile_identity=identity, current_model_id='native')
    assert controller.get_account_state(**kwargs)['epoch'] == epoch
    assert controller.set_authenticated_models(models, **kwargs, expected_epoch=epoch,
        expected_profile_identity=identity, rotate_epoch=True)
    assert controller.get_account_state(**kwargs)['epoch'] == epoch + 1
    assert not controller.set_authenticated_models(models, **kwargs, expected_epoch=epoch,
        expected_profile_identity=identity, current_model_id='native')
    assert controller.clear_authenticated_state(**kwargs)
    assert not controller.set_authenticated_models(models, **kwargs, expected_epoch=epoch + 1,
        expected_profile_identity=identity, current_model_id='native')
    assert controller.get_account_state(**kwargs)['authenticated'] is False


@pytest.mark.asyncio
async def test_auth_cleanup_cancellation_still_removes_private_scratch(tmp_path):
    from open_webui.subscriptions.acp_client import AcpClient
    binary = tmp_path / 'agy_acp_server'
    binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o700)
    profile = tmp_path / 'owned'; prepare_antigravity_profile(profile, binary)
    controller = AntigravityAuthController(); client = AcpClient()
    client.spawn = AsyncMock(); client.initialize = AsyncMock(return_value={'protocolVersion':2,'agentInfo':{'version':'synthetic'}})
    client.authenticate = AsyncMock(return_value={})
    client.create_session = AsyncMock(return_value={'sessionId':'session','configOptions':[{'id':'model','currentValue':'native','options':[{'value':'native','name':'Native'}]}]})
    began = asyncio.Event(); release = asyncio.Event(); close = client._close_resources
    async def gated_close():
        began.set(); await release.wait(); await close()
    client._close_resources = gated_close
    flow = AuthFlowState(flow_id='flow',owner_user_id='owner',phase='starting',
        profile_identity=profile_identity(binary_path=str(binary),config_dir=str(profile)))
    task = asyncio.create_task(controller._run_sign_in(flow,client,str(binary),str(profile)))
    await asyncio.wait_for(began.wait(), 2)
    scratch = Path(client.spawn.call_args.kwargs['cwd'])
    assert flow.phase == 'succeeded'
    task.cancel(); release.set()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert client._close_task.done()
    assert not scratch.exists()


@pytest.mark.asyncio
async def test_runtime_replacement_during_signin_cannot_certify_new_binary(tmp_path):
    binary = tmp_path / 'agy_acp_server'
    binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o700)
    profile = tmp_path / 'owned'; prepare_antigravity_profile(profile, binary)
    controller = AntigravityAuthController(); client = FakeAcpClient()
    original_authenticate = client.authenticate
    async def replaced(**kwargs):
        binary.write_text('#!/bin/sh\n# replacement\nexit 0\n')
        return await original_authenticate(**kwargs)
    client.authenticate = replaced
    flow = AuthFlowState(flow_id='flow',owner_user_id='owner',phase='starting',
        profile_identity=profile_identity(binary_path=str(binary),config_dir=str(profile)))
    await controller._run_sign_in(flow,client,str(binary),str(profile))
    assert flow.phase == 'cancelled'
    assert client.closed
    assert not controller.get_account_state(binary_path=str(binary),config_dir=str(profile))['authenticated']


@pytest.mark.asyncio
async def test_successful_provider_test_invalidates_model_catalog_caches():
    _get_isolated_router()
    from open_webui.routers import subscriptions as router
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    owner = SimpleNamespace(id='owner',role='admin')
    form = router.ProviderTestRequest()
    for success in (True, False):
        with patch.object(router.subscription_manager,'test_provider',new=AsyncMock(return_value={'success':success})), \
             patch.object(router,'_invalidate_models_cache',new=AsyncMock()) as invalidate:
            result = await router.test_subscription_provider(request,'gemini',form,owner)
        assert result['success'] is success
        assert invalidate.await_count == int(success)


@pytest.mark.asyncio
async def test_catalog_refresh_preserves_active_models_and_notifies_owner_only():
    import types
    from open_webui.subscriptions.cache import invalidate_subscription_models_cache
    model_module = types.ModuleType('open_webui.utils.models')
    model_module.get_all_models = SimpleNamespace(cache=SimpleNamespace(clear=AsyncMock()))
    model_module.BASE_MODELS_CACHE_KEY = 'synthetic-base-cache'
    event_module = types.ModuleType('open_webui.events')
    event_module.EVENTS = SimpleNamespace(MODEL_PROVIDER_CONFIG_UPDATED='provider-updated')
    event_module.publish_event = AsyncMock()
    socket_module = types.ModuleType('open_webui.socket.main')
    socket_module.emit_to_users = AsyncMock()
    redis = SimpleNamespace(delete=AsyncMock())
    active = {'subscription:gemini:native': {'id':'subscription:gemini:native'}}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=redis,MODELS=active,BASE_MODELS=['old'])))
    owner = SimpleNamespace(id='owner',role='admin')
    with patch.dict(sys.modules, {'open_webui.utils.models':model_module, 'open_webui.events':event_module,
                                'open_webui.socket.main':socket_module}):
        await invalidate_subscription_models_cache(request, owner, preserve_active_models=True)
    assert request.app.state.MODELS is active and active
    assert request.app.state.BASE_MODELS == []
    redis.delete.assert_awaited_once_with('synthetic-base-cache')
    socket_module.emit_to_users.assert_awaited_once_with(
        'events', {'data': {'type':'models:refresh','data':{}}}, ['owner'])
