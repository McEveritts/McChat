from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

ANTIGRAVITY_AUTH_STDOUT_PREFIX = "Open the following link to authenticate the ACP server: "
ANTIGRAVITY_AUTH_BROWSER_MARKER = "__T3_ANTIGRAVITY_AUTH_URL__"
ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE = "Sign in to Antigravity in Settings before you continue."

MAX_AUTHORIZATION_URL_LENGTH = 16_384
MAX_BROWSER_HELPER_LINE_LENGTH = len(ANTIGRAVITY_AUTH_BROWSER_MARKER) + MAX_AUTHORIZATION_URL_LENGTH + 64

REMOVED_ENV_KEYS = {
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GCLOUD_PROJECT",
    "CLOUDSDK_CORE_PROJECT",
    "AGY_ACP_CCPA_PROJECT",
    "AGY_ACP_ENABLE_OAUTH",
    "GEMINI_HOME",
    "AGY_ACP_FORCE_FILE_STORAGE",
    "ANTIGRAVITY_HARNESS_PATH",
    "BROWSER",
    "PYTHONUNBUFFERED",
    "ELECTRON_RUN_AS_NODE",
}

BROWSER_HELPER_SOURCE = (
    "import sys, json; "
    "url = sys.argv[1] if len(sys.argv) > 1 else ''; "
    "sys.stderr.write('__T3_ANTIGRAVITY_AUTH_URL__' + json.dumps(url) + chr(10)); "
    "sys.stderr.flush(); "
    "sys.exit(0)"
)
BROWSER_PREFLIGHT_URL = "https://example.invalid/t3-antigravity-browser-preflight"


APP_OWNERSHIP_MARKER = ".mcchat_profile_owned"


@dataclass(frozen=True)
class AntigravityProfileInfo:
    gemini_home: Path
    acp_dir: Path
    tmp_dir: Path
    settings_path: Path
    browser_command: str
    harness_path: str | None


def get_default_profile_dir() -> Path:
    """Return dedicated private profile directory for McChat Antigravity."""
    return Path.home() / ".gemini-mcchat"


def profile_identity(*, binary_path: str, config_dir: str = "") -> str:
    """Bind public verification to the configured profile and runtime file identity."""
    binary = Path(os.path.expanduser(binary_path)).resolve() if binary_path.strip() else None
    profile = Path(os.path.expanduser(config_dir)).resolve() if config_dir.strip() else get_default_profile_dir().resolve()
    fingerprint = []
    resolved_h = resolve_harness_path(binary) if binary else None
    harness_path = Path(resolved_h) if resolved_h else None
    for path in (binary, harness_path):
        if path is None:
            continue
        try:
            info = path.stat()
            content_digest = ""
            try:
                with open(path, "rb") as f:
                    chunk = f.read(65536)
                    content_digest = hashlib.sha256(chunk).hexdigest()
            except OSError:
                pass
            fingerprint.append((str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, content_digest))
        except OSError:
            fingerprint.append((str(path), None))
    payload = json.dumps([str(profile), str(binary) if binary else "", fingerprint], sort_keys=True)
    return "antigravity:" + hashlib.sha256(payload.encode()).hexdigest()


def validate_owned_profile(profile_dir: Path | str | None, *, allow_new: bool = False) -> Path:
    """Validate profile control paths without reading native credentials or modifying files."""
    path = Path(os.path.expanduser(str(profile_dir))) if profile_dir else get_default_profile_dir()
    if path.is_symlink():
        raise ValueError("Symlinks are not permitted for an Antigravity profile.")
    path = path.resolve()
    home = Path.home().resolve()
    forbidden = [home / name for name in (".t3", ".gemini", ".codex", ".claude", ".config/gemini")]
    if path in (home, Path("/"), home / ".config") or any(path == root or path.is_relative_to(root) for root in forbidden):
        raise ValueError("Cannot adopt global, user home, or T3 profile directory.")
    if not path.exists():
        if allow_new:
            return path
        raise ValueError("An app-owned Antigravity profile is required.")
    if not path.is_dir():
        raise ValueError("Antigravity profile is not a directory.")
    marker = path / APP_OWNERSHIP_MARKER
    if not marker.exists():
        if allow_new and not any(path.iterdir()):
            return path
        raise ValueError("Profile is missing McChat ownership marker.")
    if marker.is_symlink():
        raise ValueError("Symlinks detected inside profile directory.")
    if not marker.is_file() or marker.stat().st_size > 1024:
        raise ValueError("Invalid McChat profile ownership marker.")
    try:
        ownership = json.loads(marker.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid McChat profile ownership marker.") from exc
    if not isinstance(ownership, dict) or ownership.get("app") != "mcchat":
        raise ValueError("Invalid McChat profile ownership marker.")

    # Only control/configuration directories are traversed. Native extraction
    # files under antigravity-acp/tmp are private runtime artifacts.
    roots = [path, path / "antigravity-acp"]
    config = path / "config"
    if config.is_symlink():
        raise ValueError("Symlinks detected inside profile directory.")
    if config.exists():
        roots.extend([config, *[p for p in config.rglob("*") if p.is_dir() and not p.is_symlink()]])
    for root in roots:
        if root.is_symlink():
            raise ValueError("Symlinks detected inside profile directory.")
        if root.is_dir():
            for entry in root.iterdir():
                if entry.is_symlink():
                    raise ValueError("Symlinks detected inside profile directory.")
    tmp_dir = path / "antigravity-acp" / "tmp"
    if tmp_dir.is_symlink():
        raise ValueError("Symlinks detected inside profile directory.")

    acp = path / "antigravity-acp"
    unsafe = [
        path / "mcp_config.json", path / "hooks.json", path / "skills",
        config / "mcp_config.json", config / "hooks.json", config / "skills",
        acp / "mcp_config.json", acp / "hooks.json", acp / "skills",
    ]
    if any(item.exists() for item in unsafe):
        raise ValueError("Profile contains unsafe native MCP, hook, or skill configuration.")
    return path


def atomic_write_private(path: Path, data: bytes) -> None:
    """Replace app-owned data using a private unique temporary file."""
    if path.is_symlink():
        raise ValueError(f"Cannot write to symlink: {path}")
    parent = path.parent
    if parent.is_symlink():
        raise ValueError(f"Parent directory is a symlink: {parent}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", dir=parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
        if path.is_symlink():
            raise ValueError(f"Target changed to symlink: {path}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def resolve_harness_path(binary_path: str | Path) -> str | None:
    """Locate the official localharness_external executable corresponding to the ACP binary."""
    if not binary_path:
        return None
    try:
        bin_path = Path(binary_path).resolve()
    except Exception:
        return None

    # Check sibling in same directory (preserves Whatbox launcher and colocated harness)
    sibling = bin_path.parent / "localharness_external"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling.resolve())

    # Check Whatbox standard libexec directory
    libexec_harness = Path.home() / ".local" / "libexec" / "t3-antigravity" / "localharness_external"
    if libexec_harness.is_file() and os.access(libexec_harness, os.X_OK):
        try:
            home_local = (Path.home() / ".local").resolve()
            if bin_path.is_relative_to(home_local):
                return str(libexec_harness.resolve())
        except Exception:
            pass

    return None


def verify_browser_suppression(helper_executable: str) -> str:
    """Verify that the browser suppression helper works as expected without launching host browsers."""
    browser_cmd = f"{shlex.quote(helper_executable)} -c {shlex.quote(BROWSER_HELPER_SOURCE)} %s"
    if ":" in helper_executable or "\n" in helper_executable or "\r" in helper_executable:
        raise RuntimeError("Helper executable path contains invalid characters for BROWSER environment variable.")

    try:
        proc = subprocess.run(
            [helper_executable, "-c", BROWSER_HELPER_SOURCE, BROWSER_PREFLIGHT_URL],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        raise RuntimeError(f"Antigravity browser suppression verification failed to execute: {e}") from e

    expected_stderr = f'{ANTIGRAVITY_AUTH_BROWSER_MARKER}"{BROWSER_PREFLIGHT_URL}"\n'
    if proc.returncode != 0 or proc.stdout != "" or proc.stderr != expected_stderr:
        raise RuntimeError(
            f"Antigravity browser suppression could not be verified (code={proc.returncode}, stderr={proc.stderr!r})."
        )

    return browser_cmd


def prepare_antigravity_profile(
    profile_dir: Path | str | None = None,
    binary_path: str | Path = "",
) -> AntigravityProfileInfo:
    """Prepare a dedicated private Antigravity profile with isolated settings and browser suppression."""
    gemini_home = validate_owned_profile(profile_dir, allow_new=True)
    acp_dir = gemini_home / "antigravity-acp"
    tmp_dir = acp_dir / "tmp"
    settings_path = acp_dir / "settings.json"
    original = None
    existing = {}

    if settings_path.exists():
        if settings_path.is_symlink():
            raise ValueError("Antigravity settings cannot be a symlink.")
        if settings_path.stat().st_size > 65536:
            raise ValueError("Antigravity settings exceed the supported size.")
        original = settings_path.read_bytes()
        try:
            existing = json.loads(original)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Antigravity settings must contain a JSON object; the original was preserved.") from exc
        if not isinstance(existing, dict):
            raise ValueError("Antigravity settings must contain a JSON object; the original was preserved.")

    for directory in (gemini_home, acp_dir, tmp_dir):
        if directory.is_symlink():
            raise ValueError(f"Symlinks are not permitted for profile directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            directory.chmod(0o700)
        except OSError:
            pass

    marker = gemini_home / APP_OWNERSHIP_MARKER
    if not marker.exists():
        atomic_write_private(marker, b'{"app":"mcchat","version":1}\n')

    auth = existing.get("auth") if isinstance(existing, dict) else None
    if not isinstance(auth, dict) or auth.get("type") != "oauth-personal":
        if original is not None:
            backup = acp_dir / "settings.json.mcchat_backup"
            if backup.exists() or backup.is_symlink():
                import time
                backup = acp_dir / f"settings.json.mcchat_backup.{time.time_ns()}"
            if backup.is_symlink():
                raise ValueError(f"Symlinks are not permitted for backup file: {backup}")
            atomic_write_private(backup, original)
        existing["auth"] = {**(auth if isinstance(auth, dict) else {}), "type": "oauth-personal"}
        atomic_write_private(settings_path, (json.dumps(existing, indent=2) + "\n").encode())

    # Verify browser suppression command using current Python interpreter
    browser_command = verify_browser_suppression(sys.executable)

    # Locate harness if binary given
    harness_path = resolve_harness_path(binary_path) if binary_path else None

    return AntigravityProfileInfo(
        gemini_home=gemini_home,
        acp_dir=acp_dir,
        tmp_dir=tmp_dir,
        settings_path=settings_path,
        browser_command=browser_command,
        harness_path=harness_path,
    )


def build_antigravity_environment(
    profile_info: AntigravityProfileInfo,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct sanitized execution environment for Antigravity ACP runtime."""
    from open_webui.subscriptions.adapters.common import SAFE_ENV_KEYS
    source = base_env if base_env is not None else os.environ
    env = {key: source[key] for key in SAFE_ENV_KEYS if key in source}

    # Remove all ambient API keys and cloud credentials
    for key in list(env.keys()):
        if key.upper() in REMOVED_ENV_KEYS:
            env.pop(key, None)

    # Set authoritative Antigravity environment
    env["GEMINI_HOME"] = str(profile_info.gemini_home)
    env["AGY_ACP_FORCE_FILE_STORAGE"] = "1"
    env["BROWSER"] = profile_info.browser_command
    env["PYTHONUNBUFFERED"] = "1"
    env["ELECTRON_RUN_AS_NODE"] = "1"
    env["TMPDIR"] = str(profile_info.tmp_dir)

    # Preserve host HOME so launcher can access Whatbox ~/.local wrappers
    if "HOME" not in env:
        env["HOME"] = str(Path.home())

    if profile_info.harness_path:
        env["ANTIGRAVITY_HARNESS_PATH"] = profile_info.harness_path

    return env


def parse_antigravity_authorization_url(raw_url: str) -> dict[str, Any] | None:
    """Parse and validate Google OAuth authorization URL emitted by Antigravity."""
    if not raw_url or len(raw_url) > MAX_AUTHORIZATION_URL_LENGTH or re.search(r"\s", raw_url):
        return None

    try:
        url = urlparse(raw_url)
    except Exception:
        return None

    if url.scheme != "https" or url.netloc != "accounts.google.com" or url.path != "/o/oauth2/v2/auth":
        return None
    if url.username or url.password or url.fragment:
        return None

    from urllib.parse import parse_qs
    params = parse_qs(url.query, keep_blank_values=False)

    if len(params.get("response_type", [])) != 1 or params["response_type"][0] != "code":
        return None

    states = params.get("state", [])
    if len(states) != 1:
        return None
    state = states[0]
    if not state or len(state) > 512 or re.search(r"\s", state):
        return None

    redirect_uris = params.get("redirect_uri", [])
    if len(redirect_uris) != 1:
        return None
    redirect_uri = redirect_uris[0]

    # Validate loopback binding: http://127.0.0.1:<port>/ with port >= 1024
    match = re.match(r"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/$", redirect_uri)
    if not match:
        return None
    port = int(match.group(1))
    if port < 1024 or port > 65535:
        return None

    return {
        "authorization_url": raw_url,
        "redirect_uri": redirect_uri,
        "state": state,
        "port": port,
    }


def validate_antigravity_callback_url(
    pending_redirect_uri: str,
    pending_state: str,
    callback_url: str,
) -> str:
    """Validate incoming OAuth callback URL and guard against SSRF or state tampering."""
    if not callback_url or len(callback_url) > MAX_AUTHORIZATION_URL_LENGTH or re.search(r"\s", callback_url):
        raise ValueError("Invalid callback URL length or format.")

    try:
        url = urlparse(callback_url)
    except Exception as e:
        raise ValueError(f"Malformed callback URL: {e}") from e

    if url.scheme != "http" or url.hostname != "127.0.0.1":
        raise ValueError("Callback URL must connect to 127.0.0.1 loopback.")

    try:
        expected_url = urlparse(pending_redirect_uri)
    except Exception as e:
        raise ValueError(f"Malformed pending redirect URI: {e}") from e

    if url.username or url.password or url.fragment or url.params or url.path != expected_url.path:
        raise ValueError("Callback URL does not match the active sign-in endpoint.")
    if url.port != expected_url.port:
        raise ValueError("Callback URL port does not match active sign-in listener.")

    from urllib.parse import parse_qs
    params = parse_qs(url.query, keep_blank_values=False)

    states = params.get("state", [])
    if len(states) != 1 or states[0] != pending_state:
        raise ValueError("Sign-in state mismatch. The callback does not match this active attempt.")

    codes = params.get("code", [])
    if len(codes) != 1 or not codes[0].strip():
        # Check if error returned by Google
        errors = params.get("error", [])
        if errors:
            raise ValueError(f"Google sign-in returned error: {errors[0]}")
        raise ValueError("Missing authorization code in callback URL.")

    # Reconstruct clean loopback GET target to prevent request smuggling
    code = codes[0].strip()
    from urllib.parse import urlencode
    return f"http://127.0.0.1:{url.port}/?" + urlencode({'state': pending_state, 'code': code})
