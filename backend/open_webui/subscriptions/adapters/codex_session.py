from __future__ import annotations

import json
import asyncio
import weakref
import logging
import os
import shutil
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import HTTPException

from open_webui.subscriptions.adapters.common import (
    ERROR_CLI_API_KEY_NOT_SUPPORTED,
    ERROR_CLI_AUTH_REQUIRED,
    ERROR_CLI_EXECUTION_ERROR,
    ERROR_CLI_INVALID_RESPONSE,
    ERROR_CLI_NOT_INSTALLED,
    ERROR_CLI_PREMATURE_EOF,
    ERROR_CLI_STREAM_ERROR,
    ERROR_CLI_UNSUPPORTED_VERSION,
    ERROR_UNSUPPORTED_PARAMETER,
    build_sanitized_env,
    create_request_scratch_dir,
    format_structured_prompt,
    validate_model_identifier,
    validate_subscription_chat_controls,
)
from open_webui.subscriptions.base import BaseSubscriptionAdapter
from open_webui.subscriptions.config import ProviderConfigModel
from open_webui.subscriptions.executor import SubprocessExecutor
from open_webui.subscriptions.security import sanitize_error_message, validate_binary_path, validate_config_dir

log = logging.getLogger(__name__)

# Verified official bundled models from Codex CLI 0.166.0 (excluding internal/hidden models)
# Includes documented 'default' CLI alias mapped to omitting -m
SUPPORTED_CODEX_MODELS: list[dict[str, Any]] = [
    {'id': 'default', 'name': 'Codex Default (CLI Recommended)'},
    {'id': 'gpt-6.1-sol', 'name': 'GPT-6.1-Sol (Subscription)'},
    {'id': 'gpt-6-astra', 'name': 'GPT-6-Astra (Subscription)'},
    {'id': 'gpt-6.1-terra', 'name': 'GPT-6.1 Terra (Subscription)'},
    {'id': 'gpt-6-sol', 'name': 'GPT-6-Sol (Subscription)'},
    {'id': 'gpt-6-luna', 'name': 'GPT-6-Luna (Subscription)'},
    {'id': 'gpt-5.6-sol', 'name': 'GPT-5.6-Sol (Subscription)'},
    {'id': 'gpt-5.6-terra', 'name': 'GPT-5.6-Terra (Subscription)'},
    {'id': 'gpt-5.6-luna', 'name': 'GPT-5.6-Luna (Subscription)'},
    {'id': 'gpt-5.5', 'name': 'GPT-5.5 (Subscription)'},
]
SUPPORTED_MODEL_IDS = {m['id'] for m in SUPPORTED_CODEX_MODELS}

VALID_REASONING_EFFORTS = frozenset({'low', 'medium', 'high', 'xhigh', 'max', 'ultra'})


def parse_codex_usage(raw_usage: Any) -> dict[str, Any] | None:
    """Parse Codex usage metrics into standard OpenAI usage dict.

    Omits usage if unknown instead of fabricating zero token counts.
    """
    if not isinstance(raw_usage, dict):
        return None

    def metric(*keys):
        for key in keys:
            value = raw_usage.get(key)
            if type(value) is int and value >= 0:
                return value
        return None

    usage: dict[str, Any] = {}
    for name, aliases in (
        ('prompt_tokens', ('input_tokens', 'prompt_tokens')),
        ('completion_tokens', ('output_tokens', 'completion_tokens')),
        ('total_tokens', ('total_tokens',)),
    ):
        value = metric(*aliases)
        if value is not None:
            usage[name] = value
    if 'total_tokens' not in usage and 'prompt_tokens' in usage and 'completion_tokens' in usage:
        usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']

    cached = metric('cached_input_tokens', 'cache_read_input_tokens')
    if cached is not None:
        usage['prompt_tokens_details'] = {'cached_tokens': int(cached)}
        usage['cached_input_tokens'] = int(cached)

    reasoning = metric('reasoning_output_tokens')
    if reasoning is not None:
        usage['completion_tokens_details'] = {'reasoning_tokens': int(reasoning)}
        usage['reasoning_output_tokens'] = int(reasoning)

    return usage or None


class CodexSessionAdapter(BaseSubscriptionAdapter):
    """Production adapter for OpenAI Codex CLI (`codex`).

    Enforces fail-closed isolation and official non-interactive CLI contracts:
    - Current official Codex CLI 0.166.0+ non-interactive execution mode (`codex exec`)
    - Documented 'default' alias mapped to omitting `-m`, plus verified current model catalog
    - Preflight capability verification for required security flags
    - Strict allowlisted environment; sanitizedenv excludes ambient API/cloud/server credentials
    - CODEX_HOME explicitly preserved only from validated config.config_dir after sanitization
    - Zero auth.json inspection or credential copy/replay; queries official `codex login status`
    - Differentiates ChatGPT subscription OAuth from API key / personal access token authentication
    - Never leaks or logs account emails, tokens, or private paths
    - Prompts streamed via stdin (`-`); no argument leakage
    - Enforced host isolation:
        * `--ignore-user-config` ('Do not load $CODEX_HOME/config.toml; auth still uses CODEX_HOME')
        * `--ignore-rules`
        * `--ephemeral` (no disk session persistence)
        * `--skip-git-repo-check`
        * `--sandbox read-only`
        * `-c project_doc_max_bytes=0` (neutralizes auto-reading host repo docs)
        * `--enable skip_host_skill_discovery`
        * Multi-feature neutralization via repeatable `--disable`:
          shell_tool, hooks, plugins, apps, browser_use, computer_use,
          image_generation, multi_agent, view_image, skill_search, code_mode_host
        * Private mode 0700 per-request scratch working directory (`-C <scratch_cwd>`)
    - Structured JSONL event stream (`--json`); aggregates only `agent_message` text,
      strictly ignoring tool executions, command progress, or web searches
    - Strict terminal success verification (`turn.completed`) and exit code 0; fails closed on error
    - Rejection of unsupported chat controls (temperature, top_p, response_format, max_tokens, etc.)
    """

    provider_id = 'codex'
    default_name = 'OpenAI Codex (ChatGPT Subscription)'
    execution_available: bool = True

    def __init__(self) -> None:
        super().__init__(self.provider_id, self.default_name)
        self._model_catalogs = {}
        self._catalog_locks = weakref.WeakKeyDictionary()

    def is_installed(self, config: ProviderConfigModel) -> tuple[bool, str]:
        """Check if the codex CLI binary is installed and executable on the host."""
        try:
            binary = validate_binary_path(config.binary_path, 'codex')
        except ValueError as e:
            return False, str(e)

        resolved = shutil.which(binary)
        if resolved and os.access(resolved, os.X_OK):
            return True, resolved

        return False, f'Codex CLI executable "{binary}" was not found in PATH or is not executable.'

    def _get_env(self, config: ProviderConfigModel) -> dict[str, str]:
        """Construct sanitized environment with preserved, validated CODEX_HOME.

        Existing build_sanitized_env purges dangerous prefixes including CODEX_*.
        We reintroduce ONLY the validated, operator-provided CODEX_HOME directory
        without exposing ambient API keys or cloud credentials.
        """
        env = build_sanitized_env()

        if config.config_dir:
            cleaned_dir = validate_config_dir(config.config_dir)
            if cleaned_dir:
                env['CODEX_HOME'] = cleaned_dir

        return env

    async def _verify_capabilities(
        self, binary: str, env: dict[str, str], scratch_cwd: str
    ) -> tuple[bool, str, str | None]:
        """Preflight check: verify CLI execution, version, and required security flags."""
        # 1. Version check
        version: str | None = None
        try:
            code, stdout, _ = await SubprocessExecutor.execute_command(
                [binary, '--version'],
                env=env,
                timeout=10,
                cwd=scratch_cwd,
            )
            if code == 0 and stdout:
                version = stdout.strip()
            else:
                return False, f'Codex CLI version check returned non-zero code {code}.', None
        except Exception as e:
            return False, f'Failed to run Codex CLI version check: {sanitize_error_message(str(e))}', None

        # 2. Capabilities preflight via exec --help
        try:
            code, stdout, _ = await SubprocessExecutor.execute_command(
                [binary, 'exec', '--help'],
                env=env,
                timeout=10,
                cwd=scratch_cwd,
            )
            if code != 0 or not stdout:
                return False, 'Codex CLI failed exec --help capabilities check.', version

            required_flags = (
                '--ignore-user-config',
                '--sandbox',
                '--ephemeral',
                '--json',
                '--disable',
            )
            missing = [flag for flag in required_flags if flag not in stdout]
            if missing:
                return (
                    False,
                    f'Codex CLI lacks required security flags: {", ".join(missing)}.',
                    version,
                )
        except Exception as e:
            return False, f'Failed to run Codex CLI capabilities check: {sanitize_error_message(str(e))}', version

        return True, 'Capabilities verified.', version

    async def check_auth_status(self, config: ProviderConfigModel) -> dict[str, Any]:
        """Check Codex authentication status using official CLI login status command.

        Never reads, extracts, or inspects auth.json or OAuth tokens directly.
        Never exposes raw email addresses or token values in logs or status responses.
        Differentiates between ChatGPT subscription auth and API key auth.
        """
        installed, binary_or_err = self.is_installed(config)
        if not installed:
            return {
                'installed': False,
                'authenticated': False,
                'status': 'not_installed',
                'version': None,
                'message': binary_or_err,
                'execution_available': False,
                'reason': 'Codex CLI executable not found.',
            }

        env = self._get_env(config)

        with create_request_scratch_dir(prefix='codex_preflight_') as scratch_cwd:
            # 1. Preflight capabilities and version
            cap_ok, cap_msg, version = await self._verify_capabilities(binary_or_err, env, scratch_cwd)
            if not cap_ok:
                return {
                    'installed': True,
                    'authenticated': False,
                    'status': 'unsupported_cli',
                    'version': version,
                    'message': f'Unsupported Codex CLI: {cap_msg}',
                    'execution_available': False,
                    'reason': cap_msg,
                }

            # 2. Query official authentication status via 'codex login status'
            try:
                code, stdout, stderr = await SubprocessExecutor.execute_command(
                    [binary_or_err, 'login', 'status'],
                    env=env,
                    timeout=15,
                    cwd=scratch_cwd,
                )
            except Exception as e:
                return {
                    'installed': True,
                    'authenticated': False,
                    'status': 'error',
                    'version': version,
                    'message': 'Failed to execute Codex login status check.',
                    'execution_available': False,
                    'reason': f'Execution failed: {sanitize_error_message(str(e))}',
                }

        combined = f'{stdout or ""}\n{stderr or ""}'.strip()
        out_lower = combined.lower()

        # Check for unauthenticated condition
        if code != 0 or 'not logged in' in out_lower or 'unauthenticated' in out_lower:
            return {
                'installed': True,
                'authenticated': False,
                'status': 'unauthenticated',
                'version': version,
                'message': 'Codex CLI is not logged in. Please run "codex login" on the server terminal.',
                'execution_available': False,
                'reason': 'Codex CLI reported unauthenticated status.',
            }

        # Check for API key / access token vs subscription
        if (
            'api key' in out_lower
            or 'using access token' in out_lower
            or 'personal access token' in out_lower
            or 'workload identity' in out_lower
        ):
            return {
                'installed': True,
                'authenticated': False,
                'status': 'api_key_detected',
                'version': version,
                'message': (
                    'Codex CLI is configured with an API key or access token instead of a '
                    'ChatGPT subscription account. Please run "codex login" to connect your subscription.'
                ),
                'execution_available': False,
                'reason': f'Unsupported auth detected; subscription adapter requires ChatGPT account auth.',
            }

        # Confirmed ChatGPT login (sanitized: no email addresses or tokens logged)
        if 'chatgpt' in out_lower:
            return {
                'installed': True,
                'authenticated': True,
                'status': 'authenticated',
                'version': version,
                'message': f'Codex CLI is authenticated via ChatGPT subscription ({version or "ready"}).',
                'execution_available': True,
                'reason': None,
            }

        return {
            'installed': True,
            'authenticated': False,
            'status': 'unauthenticated',
            'version': version,
            'message': 'Codex CLI is not authenticated with a recognized subscription plan.',
            'execution_available': False,
            'reason': 'Unrecognized auth status response.',
        }

    def _catalog_key(self, config: ProviderConfigModel):
        installed, binary = self.is_installed(config)
        if not installed:
            return None
        try:
            info = os.stat(binary)
            return (os.path.realpath(binary), info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, config.config_dir)
        except OSError:
            return None

    def get_supported_models(self, config: ProviderConfigModel | None = None) -> list[dict[str, Any]]:
        """Return configured overrides or the last bounded native bundled catalog."""
        cfg = config or ProviderConfigModel()
        if cfg.models:
            configured = []
            for model in cfg.models:
                try:
                    configured.append({'id': validate_model_identifier(model.id), 'name': model.name or model.id})
                except ValueError:
                    log.debug('Ignoring an invalid configured Codex model identifier.')
            return configured
        cached = self._model_catalogs.get(self._catalog_key(cfg))
        if cached and cached[0] > time.monotonic():
            return [dict(model) for model in cached[1]]
        # Explicit compatibility fallback for offline discovery failures only.
        return [dict(model) for model in SUPPORTED_CODEX_MODELS]

    async def discover_models(self, config: ProviderConfigModel | None = None) -> list[dict[str, Any]]:
        """Cache offline native discovery by runtime identity, with bounded single-flight IO."""
        cfg = config or ProviderConfigModel()
        if cfg.models:
            return self.get_supported_models(cfg)
        key = self._catalog_key(cfg)
        if key is None:
            return []
        loop = asyncio.get_running_loop()
        lock = self._catalog_locks.setdefault(loop, asyncio.Lock())
        try:
            async with asyncio.timeout(12):
                async with lock:
                    cached = self._model_catalogs.get(key)
                    if cached and cached[0] > time.monotonic():
                        return [dict(model) for model in cached[1]]
                    models = [dict(model) for model in SUPPORTED_CODEX_MODELS]
                    cache_seconds = 30
                    try:
                        with create_request_scratch_dir(prefix='codex_models_') as scratch_cwd:
                            code, stdout, _ = await SubprocessExecutor.execute_command(
                                [key[0], 'debug', 'models', '--bundled'], env=self._get_env(cfg),
                                timeout=10, cwd=scratch_cwd)
                        if code == 0:
                            data = json.loads(stdout)
                            raw_models = data.get('models') if isinstance(data, dict) else None
                            if not isinstance(raw_models, list) or len(raw_models) > 500:
                                raise ValueError('Invalid bundled model catalog.')
                            discovered = [{'id': 'default', 'name': 'Codex Default (CLI Recommended)'}]
                            seen = {'default'}
                            for model in raw_models:
                                if not isinstance(model, dict) or model.get('visibility') == 'hide':
                                    continue
                                slug = validate_model_identifier(model.get('slug', ''))
                                if slug not in seen:
                                    name = model.get('display_name')
                                    discovered.append({'id': slug, 'name': name[:256] if isinstance(name, str) else slug})
                                    seen.add(slug)
                            models, cache_seconds = discovered, 300
                    except Exception as exc:
                        log.debug('Codex bundled catalog unavailable: %s', type(exc).__name__)
                    if key != self._catalog_key(cfg):
                        return self.get_supported_models(cfg)
                    self._model_catalogs[key] = (time.monotonic() + cache_seconds, models)
                    while len(self._model_catalogs) > 16:
                        self._model_catalogs.pop(next(iter(self._model_catalogs)))
                    return [dict(model) for model in models]
        except TimeoutError:
            return self.get_supported_models(cfg)

    def _validate_model_id(self, model_id: str, config: ProviderConfigModel | None = None) -> str:
        """Validate requested model against official aliases and validated overrides."""
        cleaned = validate_model_identifier(model_id)
        supported = {m['id'] for m in self.get_supported_models(config)}
        if cleaned not in supported:
            raise ValueError(f'Requested model "{cleaned}" is not supported by Codex subscription.')
        return cleaned

    def _extract_reasoning_effort(self, request_data: dict[str, Any]) -> str | None:
        """Honor explicit nested controls and reject unsupported native values."""
        sources = [request_data.get('params'), request_data.get('options'), request_data]
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in ('reasoning_effort', 'reasoning'):
                value = source.get(key)
                if value is None:
                    continue
                if not isinstance(value, str) or value.strip().lower() not in VALID_REASONING_EFFORTS:
                    raise HTTPException(400, f'Unsupported Codex reasoning effort (code: {ERROR_UNSUPPORTED_PARAMETER}).')
                return value.strip().lower()
        return None

    def _build_command_args(
        self,
        binary: str,
        model_id: str,
        scratch_cwd: str,
        reasoning_effort: str | None = None,
    ) -> list[str]:
        """Construct fail-closed, tool-disabled invocation arguments.

        - Disables user config to isolate from ambient settings: --ignore-user-config
        - Disables policy rules: --ignore-rules
        - Disables session persistence: --ephemeral
        - Disables git repo requirement: --skip-git-repo-check
        - Formats output as structured JSONL: --json
        - Enforces read-only sandbox: --sandbox read-only
        - Neutralizes repo doc reading: -c project_doc_max_bytes=0
        - Neutralizes host skills: --enable skip_host_skill_discovery
        - Neutralizes tools/plugins/apps: repeatable --disable FEATURE
        - Enforces isolated scratch working directory: -C <scratch_cwd>
        - Documented default model alias 'default' maps to omitting -m
        - Prompts fed strictly via stdin: '-'
        """
        cmd = [
            binary,
            'exec',
            '--ignore-user-config',
            '--ignore-rules',
            '--ephemeral',
            '--skip-git-repo-check',
            '--json',
            '--sandbox',
            'read-only',
            '-c',
            'project_doc_max_bytes=0',
            '--enable',
            'skip_host_skill_discovery',
            '--disable',
            'shell_tool',
            '--disable',
            'hooks',
            '--disable',
            'plugins',
            '--disable',
            'apps',
            '--disable',
            'browser_use',
            '--disable',
            'computer_use',
            '--disable',
            'image_generation',
            '--disable',
            'multi_agent',
            '--disable',
            'view_image',
            '--disable',
            'skill_search',
            '--disable',
            'code_mode_host',
            '-C',
            scratch_cwd,
        ]

        if reasoning_effort and reasoning_effort in VALID_REASONING_EFFORTS:
            cmd.extend(['-c', f'model_reasoning_effort="{reasoning_effort}"'])

        # Documented CLI default alias 'default' mapped to omitting -m
        if model_id and model_id != 'default':
            cmd.extend(['-m', model_id])

        cmd.append('-')
        return cmd

    async def chat_completion(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> dict[str, Any]:
        """Execute non-streaming chat completion with strict event verification."""
        # 1. Validate chat parameters against unsupported options
        validate_subscription_chat_controls(request_data)

        # 2. Execution-time subscription authentication check
        auth_status = await self.check_auth_status(config)
        if not auth_status.get('installed'):
            raise RuntimeError(auth_status.get('message', 'Codex CLI is not installed.'))
        if not auth_status.get('authenticated') or not auth_status.get('execution_available'):
            reason = auth_status.get('message', 'Codex CLI subscription is not available.')
            raise RuntimeError(f'Codex subscription execution error: {reason} (code: {ERROR_CLI_AUTH_REQUIRED})')

        installed, binary = self.is_installed(config)
        if not installed:
            raise RuntimeError(binary)

        await self.discover_models(config)
        validated_model = self._validate_model_id(model_id, config)
        messages = request_data.get('messages', [])
        prompt = format_structured_prompt(messages)
        env = self._get_env(config)
        reasoning_effort = self._extract_reasoning_effort(request_data)

        # 3. Execute subprocess in private mode 0700 scratch directory
        with create_request_scratch_dir(prefix='codex_chat_') as scratch_cwd:
            cmd = self._build_command_args(
                binary, validated_model, scratch_cwd, reasoning_effort=reasoning_effort
            )
            code, stdout, stderr = await SubprocessExecutor.execute_command(
                cmd,
                input_data=prompt,
                env=env,
                timeout=timeout,
                cwd=scratch_cwd,
            )

        if code != 0:
            log.warning('Codex CLI execution failed with exit code %s', code)
            raise RuntimeError(f'Codex subscription execution failed (code: {ERROR_CLI_EXECUTION_ERROR}).')

        # 4. Parse strict JSONL events: aggregate ONLY agent text, verify terminal turn.completed
        agent_text_parts: list[str] = []
        terminal_turn_seen = False
        terminal_success = False
        turn_usage: dict[str, Any] | None = None
        last_agent_msg: str | None = None

        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue
            except Exception:
                continue

            event_type = event.get('type')
            if event_type == 'item.completed':
                item = event.get('item', {})
                if isinstance(item, dict):
                    item_type = item.get('type') or item.get('kind')
                    # Aggregate ONLY agent text; ignore command_execution, tool_call, web_search, etc.
                    if item_type == 'agent_message':
                        text = item.get('text') or item.get('content') or item.get('message') or ''
                        if text:
                            agent_text_parts.append(text)
            elif event_type == 'agent_message':
                text = event.get('text') or event.get('content') or event.get('message') or ''
                if text:
                    agent_text_parts.append(text)
            elif event_type == 'turn.completed':
                terminal_turn_seen = True
                terminal_success = True
                if 'usage' in event:
                    turn_usage = parse_codex_usage(event['usage'])
                if 'last_agent_message' in event and event['last_agent_message']:
                    last_agent_msg = str(event['last_agent_message'])
            elif event_type in ('turn.failed', 'error'):
                raise RuntimeError(
                    f'Codex reported execution failure (code: {ERROR_CLI_EXECUTION_ERROR}).'
                )

        # 5. Strict terminal success check
        if not (terminal_turn_seen and terminal_success):
            raise RuntimeError(
                f'Codex subscription execution failed: missing terminal turn.completed event '
                f'(code: {ERROR_CLI_INVALID_RESPONSE}).'
            )

        content = ''.join(agent_text_parts) if agent_text_parts else (last_agent_msg or '')
        if not content:
            raise RuntimeError(
                f'Codex subscription response missing result content (code: {ERROR_CLI_INVALID_RESPONSE}).'
            )

        return self.build_openai_completion_dict(
            content=content,
            model_id=validated_model,
            usage=turn_usage,
        )

    async def chat_completion_stream(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion with strict EOF and terminal success verification."""
        # 1. Validate chat parameters against unsupported options
        try:
            validate_subscription_chat_controls(request_data)
        except Exception as e:
            err_chunk = {
                'error': {
                    'message': f'Invalid chat controls: {sanitize_error_message(str(e))}',
                    'type': 'invalid_request_error',
                }
            }
            yield f'data: {json.dumps(err_chunk)}\n\n'
            return

        # 2. Execution-time subscription authentication check
        auth_status = await self.check_auth_status(config)
        if (
            not auth_status.get('installed')
            or not auth_status.get('authenticated')
            or not auth_status.get('execution_available')
        ):
            reason = auth_status.get('message', 'Codex CLI subscription is not available.')
            err_chunk = {
                'error': {
                    'message': f'Codex subscription unavailable: {reason}',
                    'type': 'subscription_error',
                    'code': ERROR_CLI_AUTH_REQUIRED,
                }
            }
            yield f'data: {json.dumps(err_chunk)}\n\n'
            return

        installed, binary = self.is_installed(config)
        if not installed:
            err_chunk = {
                'error': {
                    'message': binary,
                    'type': 'subscription_error',
                    'code': ERROR_CLI_NOT_INSTALLED,
                }
            }
            yield f'data: {json.dumps(err_chunk)}\n\n'
            return

        await self.discover_models(config)
        validated_model = self._validate_model_id(model_id, config)
        messages = request_data.get('messages', [])
        prompt = format_structured_prompt(messages)
        env = self._get_env(config)
        try:
            reasoning_effort = self._extract_reasoning_effort(request_data)
        except HTTPException as exc:
            yield f'data: {json.dumps({"error": {"message": exc.detail, "code": ERROR_UNSUPPORTED_PARAMETER, "type": "invalid_request_error"}})}\n\n'
            return

        cid = f'chatcmpl-{uuid.uuid4().hex[:12]}'
        now = int(time.time())
        full_model = f'subscription:{self.provider_id}:{validated_model}'
        last_usage: dict[str, Any] | None = None

        terminal_turn_seen = False
        terminal_success = False
        stream_gen = None
        emitted_any_content = False

        with create_request_scratch_dir(prefix='codex_stream_') as scratch_cwd:
            cmd = self._build_command_args(
                binary, validated_model, scratch_cwd, reasoning_effort=reasoning_effort
            )
            try:
                stream_gen = SubprocessExecutor.execute_stream(
                    cmd,
                    input_data=prompt,
                    env=env,
                    timeout=timeout,
                    cwd=scratch_cwd,
                )

                async for raw_line in stream_gen:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError('Stream line is not a JSON object.')
                    except Exception:
                        raise RuntimeError(f'Malformed JSON event in stream (code: {ERROR_CLI_STREAM_ERROR}).')

                    event_type = event.get('type')
                    chunk_text = ''

                    if event_type == 'item.completed':
                        item = event.get('item', {})
                        if isinstance(item, dict):
                            item_type = item.get('type') or item.get('kind')
                            if item_type == 'agent_message':
                                chunk_text = item.get('text') or item.get('content') or item.get('message') or ''
                    elif event_type in ('agent_message_content_delta', 'content_block_delta'):
                        delta = event.get('delta', {})
                        if isinstance(delta, dict):
                            chunk_text = delta.get('text') or delta.get('content') or ''
                    elif event_type == 'agent_message':
                        chunk_text = event.get('text') or event.get('content') or event.get('message') or ''
                    elif event_type == 'turn.completed':
                        terminal_turn_seen = True
                        terminal_success = True
                        if 'usage' in event:
                            parsed_u = parse_codex_usage(event['usage'])
                            if parsed_u:
                                last_usage = parsed_u
                        if not emitted_any_content and event.get('last_agent_message'):
                            chunk_text = str(event['last_agent_message'])
                    elif event_type in ('turn.failed', 'error'):
                        raise RuntimeError(
                            f'Codex stream reported error (code: {ERROR_CLI_EXECUTION_ERROR}).'
                        )

                    if chunk_text:
                        emitted_any_content = True
                        sse_chunk = {
                            'id': cid,
                            'object': 'chat.completion.chunk',
                            'created': now,
                            'model': full_model,
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': {'content': chunk_text},
                                    'finish_reason': None,
                                }
                            ],
                        }
                        yield f'data: {json.dumps(sse_chunk)}\n\n'

                # Verify terminal turn.completed before emitting finish_reason: stop and [DONE]
                if not (terminal_turn_seen and terminal_success):
                    raise RuntimeError(
                        f'Codex stream finished prematurely without terminal success result '
                        f'(code: {ERROR_CLI_PREMATURE_EOF}).'
                    )

                # Successful completion: emit finish_reason: stop with accurate usage
                final_choice: dict[str, Any] = {
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop',
                }
                final_chunk: dict[str, Any] = {
                    'id': cid,
                    'object': 'chat.completion.chunk',
                    'created': now,
                    'model': full_model,
                    'choices': [final_choice],
                }
                if last_usage is not None:
                    final_chunk['usage'] = last_usage

                yield f'data: {json.dumps(final_chunk)}\n\n'
                yield 'data: [DONE]\n\n'

            except Exception as e:
                log.warning('Codex stream failed: %s', type(e).__name__)
                err_chunk = {
                    'error': {
                        'message': f'Codex subscription streaming failed (code: {ERROR_CLI_STREAM_ERROR}).',
                        'type': 'subscription_error',
                        'code': ERROR_CLI_STREAM_ERROR,
                    }
                }
                yield f'data: {json.dumps(err_chunk)}\n\n'
                # Strictly do NOT emit finish_reason: stop or [DONE] on failure

            finally:
                if stream_gen is not None and hasattr(stream_gen, 'aclose'):
                    try:
                        await stream_gen.aclose()
                    except Exception:
                        pass

    async def test_connection(
        self,
        config: ProviderConfigModel,
        prompt: str = 'Hello, reply with "pong" and nothing else.',
        model_id: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Run an isolated test prompt through Codex subscription adapter."""
        auth_status = await self.check_auth_status(config)
        if not auth_status.get('installed'):
            return {'success': False, 'message': auth_status.get('message', 'Codex CLI is not installed.')}
        if not auth_status.get('authenticated') or not auth_status.get('execution_available'):
            return {'success': False, 'message': auth_status.get('message', 'Codex CLI is not authenticated.')}

        selected_model = model_id
        if not selected_model:
            supported = await self.discover_models(config)
            selected_model = ('default' if any(model['id'] == 'default' for model in supported)
                else supported[0]['id'] if supported else 'default')
        start_time = time.time()

        try:
            res = await self.chat_completion(
                {'messages': [{'role': 'user', 'content': prompt}]},
                model_id=selected_model,
                config=config,
                timeout=timeout,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            content = res.get('choices', [{}])[0].get('message', {}).get('content', '')
            return {
                'success': True,
                'elapsed_ms': elapsed_ms,
                'model': selected_model,
                'response': content,
                'message': f'Successfully connected to Codex subscription in {elapsed_ms}ms.',
            }
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                'success': False,
                'elapsed_ms': elapsed_ms,
                'model': selected_model,
                'message': sanitize_error_message(str(e)),
            }
