from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import HTTPException, status

from open_webui.subscriptions.acp_catalog import (
    _get_field,
    extract_models,
    find_thinking_option,
    flatten_option_choices,
)
from open_webui.subscriptions.acp_client import AcpClient, AcpError, AcpProcessError, AcpRequestError
from open_webui.subscriptions.adapters.common import (
    ERROR_CLI_API_KEY_NOT_SUPPORTED,
    ERROR_CLI_AUTH_REQUIRED,
    ERROR_CLI_EXECUTION_ERROR,
    ERROR_CLI_INVALID_RESPONSE,
    ERROR_CLI_NOT_INSTALLED,
    ERROR_CLI_PREMATURE_EOF,
    ERROR_CLI_STREAM_ERROR,
    ERROR_CLI_UNSUPPORTED_VERSION,
    build_sanitized_env,
    create_request_scratch_dir,
    format_structured_prompt,
    validate_subscription_chat_controls,
)
from open_webui.subscriptions.antigravity_auth import antigravity_auth_controller
from open_webui.subscriptions.antigravity_profile import (
    ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE,
    AntigravityProfileInfo,
    build_antigravity_environment,
    prepare_antigravity_profile,
    profile_identity,
    resolve_harness_path,
)
from open_webui.subscriptions.base import BaseSubscriptionAdapter
from open_webui.subscriptions.config import ModelDefinition, ProviderConfigModel
from open_webui.subscriptions.security import sanitize_error_message, validate_binary_path, validate_config_dir

log = logging.getLogger(__name__)

# Standard launcher path on Whatbox
OPERATOR_LAUNCHER_PATH = "/home/betheenvy/.local/libexec/t3-antigravity/agy_acp_server.par"

# Truthful native client identity and capabilities
CLIENT_INFO = {
    "name": "mcchat",
    "title": "McChat",
    "version": "0.13.0",
}
CLIENT_CAPABILITIES = {
    "fs": {
        "readTextFile": False,
        "writeTextFile": False,
    },
    "terminal": False,
}

# Strict bounds for output and queues (4 MiB ceiling)
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
STREAM_QUEUE_MAXSIZE = 100


def validate_acp_model_id(model_id: str, max_length: int = 256) -> str:
    """Validate an ACP model identifier as an opaque, bounded string.

    Allows advertised model strings (including dotted or complex formats)
    while enforcing bounded length and rejecting control characters.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model identifier must be a non-empty string.")
    cleaned = model_id.strip()
    if len(cleaned) > max_length:
        raise ValueError(f"Model identifier exceeds maximum length of {max_length} characters.")
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        raise ValueError("Model identifier contains invalid control characters.")
    return cleaned


def parse_gemini_usage(raw_usage: Any) -> dict[str, Any] | None:
    """Parse usage statistics from official native ACP PromptResponse usage.

    Recognizes official schema fields:
      inputTokens, outputTokens, cachedReadTokens, cachedWriteTokens, thoughtTokens, totalTokens
    Maps metrics faithfully to OpenAI-compatible format, preserving upstream totals
    and omitting unavailable breakdowns without fabricating zeros.
    """
    if not isinstance(raw_usage, dict) or not raw_usage:
        return None

    # Official ACP camelCase field names with snake_case and legacy fallback
    input_tokens = raw_usage.get("inputTokens")
    if input_tokens is None:
        input_tokens = raw_usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = raw_usage.get("promptTokens")
    if input_tokens is None:
        input_tokens = raw_usage.get("prompt_tokens")

    output_tokens = raw_usage.get("outputTokens")
    if output_tokens is None:
        output_tokens = raw_usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = raw_usage.get("completionTokens")
    if output_tokens is None:
        output_tokens = raw_usage.get("completion_tokens")

    total_tokens = raw_usage.get("totalTokens")
    if total_tokens is None:
        total_tokens = raw_usage.get("total_tokens")

    cached_read = raw_usage.get("cachedReadTokens")
    if cached_read is None:
        cached_read = raw_usage.get("cached_read_tokens")
    if cached_read is None:
        cached_read = raw_usage.get("cachedTokens")
    if cached_read is None:
        cached_read = raw_usage.get("cached_tokens")

    cached_write = raw_usage.get("cachedWriteTokens")
    if cached_write is None:
        cached_write = raw_usage.get("cached_write_tokens")

    thought_tokens = raw_usage.get("thoughtTokens")
    if thought_tokens is None:
        thought_tokens = raw_usage.get("thought_tokens")

    # If no token metrics are present, return None
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    usage: dict[str, Any] = {}

    if input_tokens is not None:
        usage["prompt_tokens"] = int(input_tokens)
    if output_tokens is not None:
        usage["completion_tokens"] = int(output_tokens)

    if total_tokens is not None:
        usage["total_tokens"] = int(total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        usage["total_tokens"] = int(input_tokens) + int(output_tokens)

    # Provide reasoning token details if present
    if thought_tokens is not None and int(thought_tokens) > 0:
        usage["completion_tokens_details"] = {"reasoning_tokens": int(thought_tokens)}

    # Provide cache details if present
    prompt_details: dict[str, int] = {}
    if cached_read is not None and int(cached_read) > 0:
        usage["cached_tokens"] = int(cached_read)
        prompt_details["cached_tokens"] = int(cached_read)
    if cached_write is not None and int(cached_write) > 0:
        prompt_details["cached_write_tokens"] = int(cached_write)

    if prompt_details:
        usage["prompt_tokens_details"] = prompt_details

    return usage


def classify_acp_error(exc: Exception | str) -> str:
    """Classify an error from the Antigravity ACP runtime into a safe, bounded public error message.

    Strictly satisfies SEC8-06: never echoes raw upstream exception text, AcpRequestError.error_message,
    authorization URLs, tokens, or diagnostics that could leak OAuth URL/code/state, Bearer tokens,
    API keys, or local filesystem paths.
    """
    if isinstance(exc, str):
        msg = exc
        code = None
    elif isinstance(exc, AcpRequestError):
        msg = exc.error_message or ""
        code = exc.code
    elif isinstance(exc, AcpProcessError):
        msg = str(exc)
        code = None
    elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "Antigravity request timed out."
    elif isinstance(exc, asyncio.CancelledError):
        return "Antigravity operation was cancelled."
    else:
        msg = str(exc)
        code = None

    msg_upper = msg.upper()

    # Authentication / sign-in required
    if (
        code == -32000
        or "SUBSCRIPTION_REQUIRED" in msg_upper
        or "AUTH_REQUIRED" in msg_upper
        or "UNAUTHENTICATED" in msg_upper
        or "SIGN_IN_REQUIRED" in msg_upper
        or "LOGIN_REQUIRED" in msg_upper
        or "OAUTH" in msg_upper
    ):
        return ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE

    # API key errors
    if "API_KEY" in msg_upper or "API KEY" in msg_upper or "INVALID_API_KEY" in msg_upper:
        return "Invalid or unauthorized API key."

    # Privacy / Terms acceptance
    if "PRIVACY" in msg_upper or "TERMS" in msg_upper or "POLICY" in msg_upper:
        return "Privacy policy or terms acceptance required."

    # Permission / access denied
    if "PERMISSION_DENIED" in msg_upper or "ACCESS_DENIED" in msg_upper or "FORBIDDEN" in msg_upper:
        return "Antigravity subscription access denied or permission insufficient."

    # Quota / rate limit
    if "QUOTA_EXCEEDED" in msg_upper or "RESOURCE_EXHAUSTED" in msg_upper or "RATE_LIMIT" in msg_upper:
        return "Antigravity request rate limit or quota exceeded."

    if "MISSING STOPREASON" in msg_upper:
        return "Antigravity response missing stopReason (CLI_INVALID_RESPONSE)."
    if "CLI_INVALID_RESPONSE" in msg_upper:
        return "Invalid Antigravity response (CLI_INVALID_RESPONSE)."

    # Model / option unavailable
    if "NOT AVAILABLE IN ANTIGRAVITY SESSION" in msg_upper:
        return "Requested model is not available in Antigravity session."
    if "MODEL_NOT_FOUND" in msg_upper or "MODEL_UNAVAILABLE" in msg_upper:
        return "Requested model is not available in Antigravity session."

    if "INVALID_ARGUMENT" in msg_upper or "BAD_REQUEST" in msg_upper:
        return "Invalid request parameters sent to Antigravity runtime."

    if "PROCESS_EXITED" in msg_upper or "SPAWN" in msg_upper or "FAILED TO START" in msg_upper:
        return "Antigravity runtime process terminated unexpectedly."

    if ("OUTPUT" in msg_upper and ("EXCEED" in msg_upper or "LIMIT" in msg_upper)) or "MAX_OUTPUT" in msg_upper:
        return "Output limit exceeded (4MB)"

    if "MODEL_CHANGED" in msg_upper:
        return "Model was replaced by runtime during generation (MODEL_CHANGED)."

    if "REASONING EFFORT" in msg_upper:
        return "Requested reasoning effort is not available for model."

    return "Antigravity runtime execution failed."


class GeminiAdapter(BaseSubscriptionAdapter):
    """Adapter for Google Gemini subscription access via Google official Antigravity ACP.

    Enforces strict protocol contracts:
    - Official operator runtime launcher: `agy_acp_server.par`
    - ACP protocolVersion 2 with truthful McChat clientInfo and empty client tool capabilities
    - Strict empty native tool allowlist via `enabled_tools: []` on session creation
    - Explicit Google OAuth sign-in flow owned by official Antigravity harness
    - Dedicated private GEMINI_HOME profile directory with mode 0700
    - Complete browser suppression helper verified at runtime preflight
    - Rejection of agent tool/fs/terminal callbacks
    - Dynamic account-entitled model discovery from advertised session configOptions
    - Monotonic overall request timeout applied via AcpClient constructor
    - Full process group containment with bounded stream queues and backpressure
    """

    provider_id = "gemini"
    default_name = "Google Gemini (Antigravity ACP Subscription)"
    execution_available = True

    def is_installed(self, config: ProviderConfigModel | None = None) -> tuple[bool, str]:
        """Detect whether the official Antigravity ACP runtime or configured binary is available."""
        if config and config.binary_path and config.binary_path.strip():
            raw_path = config.binary_path.strip()
            try:
                validated = validate_binary_path(raw_path, "gemini")
            except ValueError as e:
                return False, str(e)

            if os.path.sep in validated or os.path.isabs(validated):
                resolved = os.path.realpath(os.path.expanduser(validated))
                if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
                    return True, resolved
                return False, f"Configured binary does not exist or is not executable: {validated}"
            else:
                found = shutil.which(validated)
                if found and os.access(found, os.X_OK):
                    try:
                        return True, validate_binary_path(found, "gemini")
                    except ValueError as e:
                        return False, str(e)
                return False, f"Configured binary '{validated}' not found in PATH or not executable."

        # Check operator-configured launcher path on Whatbox
        if os.path.isfile(OPERATOR_LAUNCHER_PATH) and os.access(OPERATOR_LAUNCHER_PATH, os.X_OK):
            return True, OPERATOR_LAUNCHER_PATH

        # Fall back to standard executable lookups in PATH (only official Antigravity/agy names)
        for cand in ("agy_acp_server.par", "agy_acp_server", "agy", "antigravity"):
            found = shutil.which(cand)
            if found and os.access(found, os.X_OK):
                try:
                    return True, validate_binary_path(found, "gemini")
                except ValueError:
                    continue

        return False, "Google Antigravity ACP server runtime not found."

    def _get_scoped_auth_state(self, config: ProviderConfigModel | None) -> tuple[bool, list[dict[str, str]], str | None]:
        """Query profile-scoped account state from auth controller."""
        installed, binary = self.is_installed(config)
        binary_path = binary if installed else ""
        config_dir = config.config_dir if config and config.config_dir else ""

        try:
            state = antigravity_auth_controller.get_account_state(binary_path=binary_path, config_dir=config_dir)
            return bool(state.get("authenticated", False)), list(state.get("models", [])), state.get("version")
        except Exception:
            return False, [], None

    async def check_auth_status(self, config: ProviderConfigModel | None = None) -> dict[str, Any]:
        """Inspect installation and auth status.

        Crucially: Background status probes NEVER initiate login, session creation,
        hook loading, or MCP execution. Account status remains 'unknown' until an
        explicit owner action (Sign-in or Test Connection) verifies it.
        """
        installed, binary_or_err = self.is_installed(config)
        if not installed:
            return {
                "installed": False,
                "authenticated": False,
                "execution_available": False,
                "status": "not_installed",
                "version": None,
                "message": binary_or_err,
                "reason": binary_or_err,
            }

        is_authenticated, models, version = self._get_scoped_auth_state(config)

        if is_authenticated:
            models_count = len(models)
            return {
                "installed": True,
                "authenticated": True,
                "execution_available": True,
                "status": "authenticated",
                "version": version,
                "message": f"Connected to Google Antigravity ({models_count} models discovered).",
                "reason": "",
            }

        return {
            "installed": True,
            "authenticated": False,
            "execution_available": True,
            "status": "unknown",
            "version": version,
            "message": "Account unverified. Click 'Sign in with Google' in Connections to link your subscription.",
            "reason": "Account unverified. Explicit sign-in or test required.",
        }

    def get_supported_models(self, config: ProviderConfigModel | None = None) -> list[dict[str, Any]]:
        """Return supported model definitions strictly for verified accounts.

        Unverified accounts return an empty list; optional configured restrictions
        intersect the verified catalog. Never injects fabricated fallback models.
        """
        is_authenticated, verified_models, _ = self._get_scoped_auth_state(config)
        if not is_authenticated or not verified_models:
            return []

        # If administrator configured explicit model restrictions, intersect with real catalog
        if config and config.models:
            allowed_ids = {m.id for m in config.models}
            return [m for m in verified_models if m.get("id") in allowed_ids]

        return list(verified_models)

    def _build_spawn_command_and_env(
        self,
        config: ProviderConfigModel | None,
        scratch_dir: str,
    ) -> tuple[list[str], dict[str, str], AntigravityProfileInfo]:
        """Prepare isolated execution environment, binary command, and profile info."""
        installed, binary_path = self.is_installed(config)
        if not installed:
            raise RuntimeError(f"{ERROR_CLI_NOT_INSTALLED}: {binary_path}")

        profile_dir = config.config_dir if config and config.config_dir else ""
        profile_info = prepare_antigravity_profile(profile_dir, binary_path=binary_path)
        env = build_antigravity_environment(profile_info)

        # Determine arguments: .par requires --uid= on Linux; agy CLI requires acp
        exe_basename = os.path.basename(binary_path).lower()
        if exe_basename.endswith(".par"):
            args = ["--uid="]
        elif exe_basename in ("agy", "antigravity"):
            args = ["acp"]
        else:
            args = []

        cmd = [binary_path] + args
        return cmd, env, profile_info

    async def _configure_session(
        self,
        client: AcpClient,
        session_id: str,
        initial_config_options: list[dict[str, Any]],
        requested_model: str | None,
        requested_effort: str | None,
        timeout: float = 10.0,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Configure model and reasoning effort options on an active ACP session.

        Enforces:
        - Exact validation of requested model against advertised configOptions
        - Immediate rejection of unavailable model before prompt execution
        - Refreshed configOptions after model selection before configuring effort
        - Exact lookup of thinking option by category='thought_level' (id='thinking_level')
        - Verification that requested reasoning effort matches offered choice values
        - Immediate rejection of unsupported effort before prompt execution without echoing raw input
        """
        available_models = extract_models(initial_config_options)
        available_model_ids = {m["id"]: m.get("name", m["id"]) for m in available_models}

        # Locate model option and current selected value
        model_opt = next(
            (o for o in initial_config_options if o.get("id") == "model" or o.get("category") == "model"),
            None,
        )
        current_model = None
        if model_opt:
            current_model = model_opt.get("currentValue") or model_opt.get("current_value")

        current_options = initial_config_options
        selected_model: str

        if requested_model and requested_model not in ("", "default", "auto"):
            if requested_model not in available_model_ids:
                raise RuntimeError(
                    f"{ERROR_CLI_EXECUTION_ERROR}: Requested model {requested_model!r} is not available in Antigravity session."
                )

            if requested_model != current_model:
                model_opt_id = model_opt.get("id", "model") if model_opt else "model"
                set_res = await client.set_config_option(
                    session_id, model_opt_id, requested_model, timeout=timeout
                )
                if isinstance(set_res, dict) and "configOptions" in set_res:
                    current_options = set_res["configOptions"]
            selected_model = requested_model
        else:
            # Use default advertised model from session
            selected_model = current_model or (available_models[0]["id"] if available_models else "default")

        # Configure reasoning effort (thinking level) if requested
        if requested_effort is not None:
            if not isinstance(requested_effort, str):
                raise RuntimeError(f"{ERROR_CLI_EXECUTION_ERROR}: Reasoning effort must be a string.")
            cleaned_effort = requested_effort.strip()
            if not cleaned_effort:
                raise RuntimeError(f"{ERROR_CLI_EXECUTION_ERROR}: Reasoning effort must be non-empty.")
            if len(cleaned_effort) > 64:
                raise RuntimeError(f"{ERROR_CLI_EXECUTION_ERROR}: Reasoning effort exceeds maximum allowed length.")
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned_effort):
                raise RuntimeError(f"{ERROR_CLI_EXECUTION_ERROR}: Reasoning effort contains invalid control characters.")

            thinking_opt = find_thinking_option(current_options)
            if not thinking_opt:
                raise RuntimeError(
                    f"{ERROR_CLI_EXECUTION_ERROR}: Thinking level configuration is not supported for model {selected_model!r}."
                )

            choices = flatten_option_choices(thinking_opt)
            choice_map = {c["value"].lower(): c["value"] for c in choices if c.get("value")}
            effort_key = cleaned_effort.lower()

            if effort_key not in choice_map:
                avail_str = ", ".join(sorted(choice_map.keys()))
                raise RuntimeError(
                    f"{ERROR_CLI_EXECUTION_ERROR}: Requested reasoning effort is not available for model {selected_model!r}. Available: [{avail_str}]"
                )

            target_val = choice_map[effort_key]
            thinking_opt_id = thinking_opt.get("id") or "thinking_level"
            set_effort_res = await client.set_config_option(
                session_id, thinking_opt_id, target_val, timeout=timeout
            )
            if isinstance(set_effort_res, dict) and "configOptions" in set_effort_res:
                current_options = set_effort_res["configOptions"]

        confirmed_model = next((option.get("currentValue") for option in current_options
            if isinstance(option, dict) and (option.get("id") == "model" or option.get("category") == "model")), None)
        if confirmed_model != selected_model:
            raise RuntimeError("MODEL_CHANGED: Native model selection was not confirmed.")
        if requested_effort is not None:
            confirmed_thinking = find_thinking_option(current_options)
            if not confirmed_thinking or confirmed_thinking.get("currentValue") != target_val:
                raise RuntimeError("Requested reasoning effort was not confirmed by the runtime.")
        return selected_model, current_options

    def _refresh_catalog(self, options, state, *, binary_path, config_dir, version):
        models = extract_models(options)
        if not models:
            return False
        selected_model = next((option.get('currentValue') for option in options
            if isinstance(option, dict) and (option.get('id') == 'model' or option.get('category') == 'model')), None)
        if selected_model not in {model['id'] for model in models}:
            return False
        recorded = antigravity_auth_controller.set_authenticated_models(
            models, binary_path=binary_path, config_dir=config_dir,
            version=version, expected_epoch=state['epoch'],
            expected_profile_identity=state['profile_identity'],
            current_model_id=selected_model,
        )
        if recorded:
            if not state.get('authenticated', False):
                state['epoch'] += 1
            state['authenticated'] = True
        return recorded

    async def test_connection(
        self,
        config: ProviderConfigModel,
        prompt: str = "Hello, reply with pong and nothing else.",
        model_id: str | None = None,
        timeout: int = 45,
    ) -> dict[str, Any]:
        """Run an isolated test prompt via Antigravity ACP to verify account entitlement and models."""
        installed, binary_path = self.is_installed(config)
        if not installed:
            return {
                "success": False,
                "elapsed_ms": 0,
                "message": f"Antigravity runtime not installed: {binary_path}",
            }

        start_time = time.monotonic()
        client = AcpClient(timeout=float(timeout))
        auth_url_detected = False

        def on_auth_url(_url: str):
            nonlocal auth_url_detected
            auth_url_detected = True
            raise RuntimeError("CLI_AUTH_REQUIRED: Sign in through Admin Connections.")

        scratch_ctx = create_request_scratch_dir(prefix="gemini_test_")
        scratch_dir = scratch_ctx.__enter__()

        try:
            cmd, env, profile_info = self._build_spawn_command_and_env(config, scratch_dir)

            verification_state = {
                **antigravity_auth_controller.get_account_state(
                    binary_path=binary_path, config_dir=str(profile_info.gemini_home)),
                'profile_identity': profile_identity(
                    binary_path=binary_path, config_dir=str(profile_info.gemini_home)),
            }

            await client.spawn(
                cmd,
                env=env,
                cwd=scratch_dir,
                on_authorization_url=on_auth_url,
            )

            # 1. Initialize with McChat identity and zero client tools
            init_res = await client.initialize(
                protocol_version=2,
                client_info=CLIENT_INFO,
                client_capabilities=CLIENT_CAPABILITIES,
                timeout=20.0,
            )
            init_version = (
                (init_res.get("agentInfo") or {}).get("version")
                if isinstance(init_res, dict)
                else None
            )

            # 2. Authenticate
            try:
                await client.authenticate(method_id="oauth-personal", timeout=20.0)
            except (AcpRequestError, AcpProcessError, TimeoutError) as auth_err:
                elapsed = int((time.monotonic() - start_time) * 1000)
                if auth_url_detected:
                    return {
                        "success": False,
                        "elapsed_ms": elapsed,
                        "message": "Google sign-in required. Use 'Sign in with Google' in Admin Connections.",
                    }
                classified = classify_acp_error(auth_err)
                return {
                    "success": False,
                    "elapsed_ms": elapsed,
                    "message": f"Authentication required: {classified}",
                }

            if auth_url_detected:
                elapsed = int((time.monotonic() - start_time) * 1000)
                return {
                    "success": False,
                    "elapsed_ms": elapsed,
                    "message": "Google sign-in required. Use 'Sign in with Google' in Admin Connections.",
                }

            # 3. Create Session with empty native tools
            session_res = await client.create_session(
                cwd=scratch_dir,
                enabled_tools=[],
                timeout=20.0,
            )
            session_id = session_res.get("sessionId")
            if not session_id:
                elapsed = int((time.monotonic() - start_time) * 1000)
                return {
                    "success": False,
                    "elapsed_ms": elapsed,
                    "message": "Failed to create Antigravity session (no sessionId returned).",
                }

            raw_options = session_res.get("configOptions", [])
            test_model = model_id or (config.models[0].id if config.models else None)
            selected_model, configured_options = await self._configure_session(
                client, session_id, raw_options, test_model, requested_effort=None, timeout=10.0
            )
            # 5. Send test prompt and capture assistant-only text
            received_text: list[str] = []

            def on_update(params: dict[str, Any]):
                if not isinstance(params, dict) or params.get("sessionId") != session_id:
                    return
                upd = params.get("update") or {}
                if upd.get("sessionUpdate") == "agent_message_chunk":
                    chunk_text = (upd.get("content") or {}).get("text", "")
                    if chunk_text:
                        received_text.append(chunk_text)

            client.on_session_update = on_update

            prompt_res = await client.prompt(session_id, prompt, timeout=float(timeout))
            elapsed = int((time.monotonic() - start_time) * 1000)

            stop_reason = prompt_res.get("stopReason")
            if stop_reason not in ("end_turn", "max_tokens"):
                return {
                    "success": False,
                    "elapsed_ms": elapsed,
                    "message": "CLI_INVALID_RESPONSE: Unexpected test termination.",
                }

            full_response = "".join(received_text).strip()
            if not full_response:
                return {
                    "success": False,
                    "elapsed_ms": elapsed,
                    "message": "Connection test failed: No assistant text received from Antigravity.",
                }

            if not self._refresh_catalog(configured_options, verification_state,
                binary_path=binary_path, config_dir=str(profile_info.gemini_home), version=init_version):
                raise RuntimeError("CLI_AUTH_REQUIRED: Verification was invalidated; sign in again.")

            return {
                "success": True,
                "elapsed_ms": elapsed,
                "message": f"Successfully connected to Google Antigravity in {elapsed}ms.",
                "response": full_response,
            }

        except Exception as e:
            elapsed = int((time.monotonic() - start_time) * 1000)
            classified = classify_acp_error(e)
            return {
                "success": False,
                "elapsed_ms": elapsed,
                "message": f"Connection test failed: {classified}",
            }
        finally:
            try:
                await client.close()
            finally:
                scratch_ctx.__exit__(None, None, None)

    async def chat_completion(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> dict[str, Any]:
        """Execute non-streaming chat completion through isolated Antigravity ACP session."""
        validate_subscription_chat_controls(request_data)
        validated_model = validate_acp_model_id(model_id)

        messages = request_data.get("messages", [])
        prompt_text = format_structured_prompt(messages)

        client = AcpClient(timeout=float(timeout))
        auth_url_detected = False

        def on_auth_url(_url: str):
            nonlocal auth_url_detected
            auth_url_detected = True
            raise RuntimeError("CLI_AUTH_REQUIRED: Sign in through Admin Connections.")

        received_text: list[str] = []
        received_bytes = 0
        session_id_holder: list[str] = []
        active_model_state: dict[str, Any] = {
            "model": validated_model,
            "chunks_received": False,
            "prompt_started": False,
            "epoch": 0,
            "error": None,
        }

        requested_effort = request_data.get("reasoning_effort")
        if not isinstance(requested_effort, str):
            requested_effort = None

        scratch_ctx = create_request_scratch_dir(prefix="gemini_chat_")
        scratch_dir = scratch_ctx.__enter__()

        try:
            cmd, env, profile_info = self._build_spawn_command_and_env(config, scratch_dir)
            binary_path = cmd[0]
            config_dir = str(profile_info.gemini_home)

            init_version: str | None = None
            account_state = antigravity_auth_controller.get_account_state(
                binary_path=binary_path, config_dir=config_dir)
            active_model_state.update(epoch=account_state["epoch"], authenticated=account_state["authenticated"])
            active_model_state["profile_identity"] = profile_identity(
                binary_path=binary_path, config_dir=config_dir)

            def on_update(params: dict[str, Any]):
                nonlocal received_bytes
                if not isinstance(params, dict) or not session_id_holder:
                    return
                # Strictly ignore updates for foreign or missing sessionId
                msg_sid = params.get("sessionId")
                if not msg_sid or msg_sid != session_id_holder[0]:
                    return

                upd = params.get("update") or {}
                session_update_type = upd.get("sessionUpdate")

                if session_update_type == "config_option_update":
                    if not active_model_state["prompt_started"]:
                        return
                    options = upd.get("configOptions")
                    if not isinstance(options, list):
                        raise RuntimeError("CLI_INVALID_RESPONSE: Invalid configuration update.")
                    self._refresh_catalog(options, active_model_state,
                        binary_path=binary_path, config_dir=config_dir, version=init_version)
                    model_option = next((option for option in options if isinstance(option, dict)
                        and (option.get("category") == "model" or option.get("id") == "model")), None)
                    if model_option and model_option.get("currentValue") != active_model_state["model"]:
                        raise RuntimeError("MODEL_CHANGED: Model changed; select the available model and retry.")
                    if requested_effort:
                        thinking = find_thinking_option(options)
                        if (not thinking or not isinstance(thinking.get("currentValue"), str)
                                or thinking["currentValue"].lower() != requested_effort.strip().lower()):
                            raise RuntimeError("Requested reasoning effort changed during generation.")
                    return

                # Strictly filter agent message text, never echo user input or diagnostics
                if session_update_type == "agent_message_chunk":
                    chunk = (upd.get("content") or {}).get("text", "")
                    if chunk:
                        chunk_bytes = len(chunk.encode("utf-8"))
                        if received_bytes + chunk_bytes > MAX_OUTPUT_BYTES:
                            raise RuntimeError("Antigravity response exceeded maximum output limit of 4MB.")
                        received_bytes += chunk_bytes
                        received_text.append(chunk)
                        active_model_state["chunks_received"] = True

            await client.spawn(
                cmd,
                env=env,
                cwd=scratch_dir,
                on_authorization_url=on_auth_url,
                on_session_update=on_update,
            )

            init_res = await client.initialize(
                protocol_version=2,
                client_info=CLIENT_INFO,
                client_capabilities=CLIENT_CAPABILITIES,
                timeout=25.0,
            )
            if isinstance(init_res, dict):
                init_version = (init_res.get("agentInfo") or {}).get("version")

            try:
                await client.authenticate(method_id="oauth-personal", timeout=25.0)
            except AcpRequestError as e:
                classified = classify_acp_error(e)
                if classified == ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE:
                    raise RuntimeError(f"{ERROR_CLI_AUTH_REQUIRED}: {ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE}") from e
                raise RuntimeError(f"{ERROR_CLI_EXECUTION_ERROR}: {classified}") from e

            if auth_url_detected:
                raise RuntimeError(f"{ERROR_CLI_AUTH_REQUIRED}: {ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE}")

            session_res = await client.create_session(
                cwd=scratch_dir,
                enabled_tools=[],
                timeout=25.0,
            )
            session_id = session_res.get("sessionId")
            if not session_id:
                raise RuntimeError(f"{ERROR_CLI_INVALID_RESPONSE}: No sessionId returned from session/new.")

            session_id_holder.append(session_id)

            # Configure model and reasoning effort from session configOptions
            raw_options = session_res.get("configOptions", [])

            selected_model, configured_options = await self._configure_session(
                client=client,
                session_id=session_id,
                initial_config_options=raw_options,
                requested_model=validated_model,
                requested_effort=requested_effort,
                timeout=10.0,
            )
            active_model_state["model"] = selected_model
            self._refresh_catalog(configured_options, active_model_state,
                binary_path=binary_path, config_dir=config_dir, version=init_version)

            # Native fallback must not be silently attributed to the originally requested model.
            active_model_state["prompt_started"] = True
            prompt_res = await client.prompt(session_id, prompt_text, timeout=float(timeout))

            if active_model_state["error"]:
                raise active_model_state["error"]

            stop_reason = prompt_res.get("stopReason")
            if not stop_reason:
                raise RuntimeError(f"{ERROR_CLI_INVALID_RESPONSE}: Antigravity prompt response missing stopReason.")

            if stop_reason == "end_turn":
                finish_reason = "stop"
            elif stop_reason == "max_tokens":
                finish_reason = "length"
            else:
                raise RuntimeError(f"{ERROR_CLI_INVALID_RESPONSE}: Unexpected prompt termination.")

            full_text = "".join(received_text)
            if not full_text.strip() and not received_text:
                raise RuntimeError(f"{ERROR_CLI_PREMATURE_EOF}: No assistant output received from Antigravity.")

            # Compute usage strictly from per-turn PromptResponse usage
            parsed_usage = parse_gemini_usage(prompt_res.get("usage"))

            final_reported_model = active_model_state["model"]
            res_payload = {
                "id": f"chatcmpl-gemini-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"subscription:gemini:{final_reported_model}",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_text,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
            }

            if parsed_usage:
                res_payload["usage"] = parsed_usage

            return res_payload

        except HTTPException:
            raise
        except Exception as e:
            classified = classify_acp_error(e)
            code = ERROR_CLI_AUTH_REQUIRED if "CLI_AUTH_REQUIRED" in str(e) else ERROR_CLI_EXECUTION_ERROR
            raise RuntimeError(f"{code}: {classified}") from e
        finally:
            try:
                await client.close()
            finally:
                scratch_ctx.__exit__(None, None, None)

    async def chat_completion_stream(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion through isolated Antigravity ACP session."""
        try:
            validate_subscription_chat_controls(request_data)
            validated_model = validate_acp_model_id(model_id)
        except Exception as e:
            classified = classify_acp_error(e)
            yield f'data: {{"error": {{"message": "{classified}"}}, "finish_reason": null}}\n\n'
            return

        messages = request_data.get("messages", [])
        prompt_text = format_structured_prompt(messages)

        scratch_ctx = create_request_scratch_dir(prefix="gemini_stream_")
        scratch_dir = scratch_ctx.__enter__()
        client = AcpClient(timeout=float(timeout))
        update_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
        scratch_closed = False

        def close_scratch():
            nonlocal scratch_closed
            if not scratch_closed:
                scratch_closed = True
                scratch_ctx.__exit__(None, None, None)

        session_id_holder: list[str] = []
        total_output_bytes = 0

        active_model_state: dict[str, Any] = {
            "model": validated_model,
            "chunks_sent": False,
            "prompt_started": False,
            "epoch": 0,
        }
        active_model_name: list[str] = [f"subscription:gemini:{validated_model}"]
        created_ts = int(time.time())
        chatcmpl_id = f"chatcmpl-gemini-{uuid.uuid4().hex[:12]}"

        reasoning_effort = request_data.get("reasoning_effort")
        if not isinstance(reasoning_effort, str):
            reasoning_effort = None

        async def on_auth_url(_url: str):
            await update_queue.put(("auth_required", ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE))

        async def on_update(params: dict[str, Any]):
            nonlocal total_output_bytes
            if not isinstance(params, dict) or not session_id_holder:
                return
            # Strictly ignore updates for foreign or missing sessionId
            msg_sid = params.get("sessionId")
            if not msg_sid or msg_sid != session_id_holder[0]:
                return

            upd = params.get("update") or {}
            update_type = upd.get("sessionUpdate")

            if update_type == "config_option_update":
                if not active_model_state["prompt_started"]:
                    return
                options = upd.get("configOptions")
                if not isinstance(options, list):
                    raise RuntimeError("CLI_INVALID_RESPONSE: Invalid configuration update.")
                self._refresh_catalog(options, active_model_state,
                    binary_path=binary_path, config_dir=config_dir, version=init_version)
                model_option = next((option for option in options if isinstance(option, dict)
                    and (option.get("category") == "model" or option.get("id") == "model")), None)
                if model_option and model_option.get("currentValue") != active_model_state["model"]:
                    raise RuntimeError("MODEL_CHANGED: Model changed; select the available model and retry.")
                if reasoning_effort:
                    thinking = find_thinking_option(options)
                    if (not thinking or not isinstance(thinking.get("currentValue"), str)
                            or thinking["currentValue"].lower() != reasoning_effort.strip().lower()):
                        raise RuntimeError("Requested reasoning effort changed during generation.")
                return

            # Only stream assistant text; strictly suppress user echo and diagnostics
            if update_type == "agent_message_chunk":
                chunk = (upd.get("content") or {}).get("text", "")
                if chunk:
                    chunk_len = len(chunk.encode("utf-8"))
                    if total_output_bytes + chunk_len > MAX_OUTPUT_BYTES:
                        await update_queue.put(("error", "Output limit exceeded (4MB)"))
                        return
                    total_output_bytes += chunk_len
                    active_model_state["chunks_sent"] = True
                    await update_queue.put(("chunk", chunk))
            elif update_type == "agent_thought_chunk":
                thought = (upd.get("content") or {}).get("text", "")
                if thought:
                    thought_len = len(thought.encode("utf-8"))
                    if total_output_bytes + thought_len > MAX_OUTPUT_BYTES:
                        await update_queue.put(("error", "Output limit exceeded (4MB)"))
                        return
                    total_output_bytes += thought_len
                    active_model_state["chunks_sent"] = True
                    await update_queue.put(("thought", thought))

        init_version: str | None = None
        binary_path = ""
        config_dir = ""

        async def _run_stream() -> None:
            nonlocal init_version, binary_path, config_dir
            try:
                cmd, env, profile_info = self._build_spawn_command_and_env(config, scratch_dir)
                binary_path = cmd[0]
                config_dir = str(profile_info.gemini_home)
                account_state = antigravity_auth_controller.get_account_state(
                    binary_path=binary_path, config_dir=config_dir)
                active_model_state.update(epoch=account_state["epoch"], authenticated=account_state["authenticated"])
                active_model_state["profile_identity"] = profile_identity(
                    binary_path=binary_path, config_dir=config_dir)
                await client.spawn(
                    cmd,
                    env=env,
                    cwd=scratch_dir,
                    on_authorization_url=on_auth_url,
                    on_session_update=on_update,
                )

                init_res = await client.initialize(
                    protocol_version=2,
                    client_info=CLIENT_INFO,
                    client_capabilities=CLIENT_CAPABILITIES,
                    timeout=25.0,
                )
                if isinstance(init_res, dict):
                    init_version = (init_res.get("agentInfo") or {}).get("version")

                try:
                    await client.authenticate(method_id="oauth-personal", timeout=25.0)
                except AcpRequestError as e:
                    classified = classify_acp_error(e)
                    if classified == ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE:
                        await update_queue.put(
                            ("error", f"{ERROR_CLI_AUTH_REQUIRED}: {ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE}")
                        )
                    else:
                        await update_queue.put(("error", f"{ERROR_CLI_STREAM_ERROR}: {classified}"))
                    return

                session_res = await client.create_session(
                    cwd=scratch_dir,
                    enabled_tools=[],
                    timeout=25.0,
                )
                sid = session_res.get("sessionId")
                if not sid:
                    await update_queue.put(("error", "No sessionId returned from Antigravity session/new."))
                    return

                session_id_holder.append(sid)

                # Configure model and reasoning effort from session configOptions
                raw_options = session_res.get("configOptions", [])

                selected_model, configured_options = await self._configure_session(
                    client=client,
                    session_id=sid,
                    initial_config_options=raw_options,
                    requested_model=validated_model,
                    requested_effort=reasoning_effort,
                    timeout=10.0,
                )
                active_model_state["model"] = selected_model
                active_model_name[0] = f"subscription:gemini:{selected_model}"
                self._refresh_catalog(configured_options, active_model_state,
                    binary_path=binary_path, config_dir=config_dir, version=init_version)

                active_model_state["prompt_started"] = True
                prompt_res = await client.prompt(sid, prompt_text, timeout=float(timeout))
                stop_reason = prompt_res.get("stopReason")
                if not stop_reason:
                    await update_queue.put(
                        ("error", f"{ERROR_CLI_INVALID_RESPONSE}: Antigravity prompt response missing stopReason.")
                    )
                    return

                if stop_reason == "end_turn":
                    finish_reason = "stop"
                elif stop_reason == "max_tokens":
                    finish_reason = "length"
                else:
                    await update_queue.put(
                        ("error", f"{ERROR_CLI_INVALID_RESPONSE}: Unexpected prompt termination.")
                    )
                    return

                turn_usage = prompt_res.get("usage")
                await update_queue.put(("done", {"usage": turn_usage, "finish_reason": finish_reason}))

            except asyncio.CancelledError:
                if session_id_holder:
                    await client.cancel_session(session_id_holder[0])
                raise
            except AcpRequestError as e:
                classified = classify_acp_error(e)
                if classified == ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE:
                    await update_queue.put(
                        ("error", f"{ERROR_CLI_AUTH_REQUIRED}: {ANTIGRAVITY_SIGN_IN_REQUIRED_MESSAGE}")
                    )
                else:
                    await update_queue.put(("error", f"{ERROR_CLI_STREAM_ERROR}: {classified}"))
            except Exception as e:
                classified = classify_acp_error(e)
                await update_queue.put(("error", f"{ERROR_CLI_STREAM_ERROR}: {classified}"))

        async def _produce_stream():
            try:
                # Include configuration and every queue write in one budget.
                # A paused consumer must not retain the native process/permit.
                async with asyncio.timeout(max(0, client.deadline - time.monotonic())):
                    await _run_stream()
            except TimeoutError:
                # Make room for a terminal failure even if backpressure filled
                # the queue. Buffered partial output cannot signal success.
                while not update_queue.empty():
                    update_queue.get_nowait()
                update_queue.put_nowait(("error", f"{ERROR_CLI_STREAM_ERROR}: Antigravity request timed out."))
            finally:
                try:
                    await client.close()
                finally:
                    close_scratch()

        stream_task = asyncio.create_task(_produce_stream())
        terminal_success = False
        final_finish_reason = "stop"
        captured_usage: dict[str, Any] | None = None

        try:
            while True:
                try:
                    event_type, event_data = await asyncio.wait_for(update_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if stream_task.done():
                        # Drain remaining items in queue before exiting
                        while not update_queue.empty():
                            event_type, event_data = update_queue.get_nowait()
                            if event_type == "chunk":
                                yield f'data: {json.dumps({"id": chatcmpl_id, "object": "chat.completion.chunk", "created": created_ts, "model": active_model_name[0], "choices": [{"index": 0, "delta": {"content": event_data}, "finish_reason": None}]})}\n\n'
                            elif event_type == "done":
                                terminal_success = True
                                if isinstance(event_data, dict):
                                    captured_usage = event_data.get("usage")
                                    final_finish_reason = event_data.get("finish_reason", "stop")
                            elif event_type in ("error", "auth_required"):
                                yield f'data: {json.dumps({"error": {"message": event_data}, "finish_reason": None})}\n\n'
                                return
                        break
                    continue

                if event_type == "chunk":
                    chunk_payload = {
                        "id": chatcmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": active_model_name[0],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event_data},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"

                elif event_type == "thought":
                    chunk_payload = {
                        "id": chatcmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": active_model_name[0],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": event_data},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"

                elif event_type == "done":
                    terminal_success = True
                    if isinstance(event_data, dict):
                        captured_usage = event_data.get("usage")
                        final_finish_reason = event_data.get("finish_reason", "stop")
                    break

                elif event_type in ("error", "auth_required"):
                    yield f'data: {json.dumps({"error": {"message": event_data}, "finish_reason": None})}\n\n'
                    return

            # Strict terminal check: Emit [DONE] ONLY if prompt completed with terminal success
            if terminal_success:
                parsed_u = parse_gemini_usage(captured_usage)
                final_chunk: dict[str, Any] = {
                    "id": chatcmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": active_model_name[0],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": final_finish_reason,
                        }
                    ],
                }
                if parsed_u:
                    final_chunk["usage"] = parsed_u

                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield f'data: {json.dumps({"error": {"message": ERROR_CLI_PREMATURE_EOF}, "finish_reason": None})}\n\n'

        except (asyncio.CancelledError, GeneratorExit):
            if session_id_holder:
                try:
                    await client.cancel_session(session_id_holder[0])
                except Exception:
                    pass
            stream_task.cancel()
            raise
        finally:
            if not stream_task.done():
                stream_task.cancel()
            cancelled_during_cleanup = False
            try:
                try:
                    await asyncio.shield(stream_task)
                except asyncio.CancelledError:
                    # Child cancellation during aclose is expected. Caller
                    # cancellation must propagate after resources are closed.
                    cancelled_during_cleanup = bool(asyncio.current_task().cancelling())
                except Exception:
                    pass
            finally:
                try:
                    await client.close()
                finally:
                    close_scratch()
            if cancelled_during_cleanup:
                raise asyncio.CancelledError
