from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import HTTPException, status

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
    parse_usage_dict,
    validate_model_identifier,
    validate_subscription_chat_controls,
)
from open_webui.subscriptions.base import BaseSubscriptionAdapter
from open_webui.subscriptions.config import ProviderConfigModel
from open_webui.subscriptions.executor import SubprocessExecutor
from open_webui.subscriptions.security import sanitize_error_message, validate_binary_path, validate_config_dir

log = logging.getLogger(__name__)

# Maintained official Claude Code CLI aliases (sonnet, opus, haiku)
# Retired models (claude-3-5, claude-3-7) are strictly excluded from defaults.
SUPPORTED_CLAUDE_MODELS = [
    {'id': 'sonnet', 'name': 'Claude Sonnet (Subscription)'},
    {'id': 'opus', 'name': 'Claude Opus (Subscription)'},
    {'id': 'haiku', 'name': 'Claude Haiku (Subscription)'},
]
SUPPORTED_MODEL_IDS = {m['id'] for m in SUPPORTED_CLAUDE_MODELS}
SUPPORTED_CLAUDE_EFFORT_LEVELS = frozenset({'low', 'medium', 'high', 'xhigh', 'max'})


class ClaudeCodeAdapter(BaseSubscriptionAdapter):
    """Adapter for Anthropic Claude Code CLI (`claude`).

    Enforces fail-closed isolation and strict official contracts:
    - Maintained model aliases: sonnet, opus, haiku (custom overrides strictly validated)
    - Query official `claude auth status` (emits JSON by default; no unverified --json flag)
    - Require subscription OAuth (`authMethod == "claude.ai"`); fail closed on API keys or malformed JSON
    - Preflight `--help` capability verification for required security flags
    - Strict allowlisted environment (PATH, HOME, LANG, TMPDIR); zero inheritance of secret/cloud/bearer vars
    - Official `--safe-mode` combined with `--settings '{"disableAllHooks": true}'` to neutralize policy hooks
    - Explicit empty tools (`--tools ""` and `--disallowedTools "*"`) and empty MCP configuration
    - Private mode 0700 per-request scratch cwd (no global cwd or import-time mkdir)
    - Prompts streamed via stdin (non-interactive print mode `-p`)
    - Strict JSON output parsing for nonstream; strict terminal success and EOF checks for streams
    - Accurate token counts with prompt cache metrics, omitting usage if unknown (no fabricated zeros)
    - Rejection of unsupported chat controls (temperature, top_p, response_format, max_tokens, etc.)
    """

    provider_id = 'claude'
    default_name = 'Claude Code (Anthropic Subscription)'

    def __init__(self) -> None:
        super().__init__(self.provider_id, self.default_name)

    def is_installed(self, config: ProviderConfigModel) -> tuple[bool, str]:
        try:
            binary = validate_binary_path(config.binary_path, 'claude')
        except ValueError as e:
            return False, str(e)

        resolved = shutil.which(binary)
        if resolved and os.access(resolved, os.X_OK):
            return True, resolved

        return False, f'Claude CLI executable "{binary}" was not found in PATH or is not executable.'

    def _get_env(self, config: ProviderConfigModel) -> dict[str, str]:
        extra: dict[str, str] = {
            'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS': '1',
            'DISABLE_TELEMETRY': '1',
            'DISABLE_AUTOUPDATER': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        }
        if config.config_dir:
            cleaned_dir = validate_config_dir(config.config_dir)
            if cleaned_dir:
                extra['CLAUDE_CONFIG_DIR'] = cleaned_dir

        return build_sanitized_env(extra)

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
                return False, f'CLI version check returned non-zero code {code}.', None
        except Exception as e:
            return False, f'Failed to run Claude CLI version check: {sanitize_error_message(str(e))}', None

        # 2. Capabilities preflight via --help
        try:
            code, stdout, _ = await SubprocessExecutor.execute_command(
                [binary, '--help'],
                env=env,
                timeout=10,
                cwd=scratch_cwd,
            )
            if code != 0 or not stdout:
                return False, 'Claude CLI failed --help capabilities check.', version

            required_flags = ('--safe-mode', '--settings', '--tools', '--disallowedTools')
            missing = [flag for flag in required_flags if flag not in stdout]
            if missing:
                return (
                    False,
                    f'Claude CLI lacks required security flags: {", ".join(missing)}.',
                    version,
                )

            self._has_effort_capability = ('--effort' in stdout)
        except Exception as e:
            return False, f'Failed to run Claude CLI capabilities preflight: {sanitize_error_message(str(e))}', version

        return True, 'Capabilities verified.', version

    async def _verify_effort_capability(
        self, binary: str, env: dict[str, str], scratch_cwd: str
    ) -> bool:
        """Verify whether the installed Claude CLI binary supports --effort via --help."""
        if getattr(self, '_has_effort_capability', None) is not None:
            return self._has_effort_capability
        try:
            code, stdout, _ = await SubprocessExecutor.execute_command(
                [binary, '--help'],
                env=env,
                timeout=10,
                cwd=scratch_cwd,
            )
            if code == 0 and stdout:
                has_cap = '--effort' in stdout
                self._has_effort_capability = has_cap
                return has_cap
        except Exception:
            pass
        return False

    async def check_auth_status(self, config: ProviderConfigModel) -> dict[str, Any]:
        """Check real authentication status using official CLI auth status command."""
        installed, binary_or_err = self.is_installed(config)
        if not installed:
            return {
                'installed': False,
                'authenticated': False,
                'status': 'not_installed',
                'version': None,
                'message': binary_or_err,
                'execution_available': False,
                'reason': 'Claude CLI executable not found.',
            }

        env = self._get_env(config)

        with create_request_scratch_dir(prefix='claude_preflight_') as scratch_cwd:
            # 1. Preflight capabilities and version
            cap_ok, cap_msg, version = await self._verify_capabilities(binary_or_err, env, scratch_cwd)
            if not cap_ok:
                return {
                    'installed': True,
                    'authenticated': False,
                    'status': 'unsupported_cli',
                    'version': version,
                    'message': f'Unsupported Claude CLI: {cap_msg}',
                    'execution_available': False,
                    'reason': cap_msg,
                }

            # 2. Query official auth status (emits JSON by default; DO NOT append --json)
            try:
                code, stdout, stderr = await SubprocessExecutor.execute_command(
                    [binary_or_err, 'auth', 'status'],
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
                    'message': 'Failed to query Claude auth status.',
                    'execution_available': False,
                    'reason': f'Execution failed: {sanitize_error_message(str(e))}',
                }

        if code != 0:
            return {
                'installed': True,
                'authenticated': False,
                'status': 'unauthenticated',
                'version': version,
                'message': 'Claude CLI is not authenticated. Please run "claude auth login" on the server terminal.',
                'execution_available': False,
                'reason': 'CLI returned non-zero exit code on auth status.',
            }

        # 3. Parse strict official JSON; fail closed without text substring guessing
        try:
            status_data = json.loads(stdout.strip())
            if not isinstance(status_data, dict):
                raise ValueError('Auth status output is not a JSON object.')
        except Exception:
            return {
                'installed': True,
                'authenticated': False,
                'status': 'error',
                'version': version,
                'message': 'Failed to parse Claude CLI authentication status (malformed response).',
                'execution_available': False,
                'reason': 'Malformed JSON from claude auth status.',
            }

        logged_in = bool(status_data.get('loggedIn'))
        auth_method = status_data.get('authMethod')

        if not logged_in:
            return {
                'installed': True,
                'authenticated': False,
                'status': 'unauthenticated',
                'version': version,
                'message': 'Claude CLI is not logged in. Please run "claude auth login" on the server terminal.',
                'execution_available': False,
                'reason': 'Claude CLI reported loggedIn=false.',
            }

        if auth_method == 'claude.ai':
            return {
                'installed': True,
                'authenticated': True,
                'status': 'authenticated',
                'version': version,
                'message': f'Claude Code is authenticated via Anthropic subscription ({version or "ready"}).',
                'execution_available': True,
                'reason': None,
            }
        elif auth_method in ('api_key', 'console', 'bedrock', 'vertex'):
            return {
                'installed': True,
                'authenticated': False,
                'status': 'api_key_detected',
                'version': version,
                'message': (
                    'Claude CLI is currently configured with an API key or cloud credentials instead '
                    'of an Anthropic subscription plan. Please run "claude auth login" to connect your subscription.'
                ),
                'execution_available': False,
                'reason': f'Unsupported auth method "{auth_method}"; subscription requires claude.ai OAuth.',
            }
        else:
            return {
                'installed': True,
                'authenticated': False,
                'status': 'unauthenticated',
                'version': version,
                'message': 'Claude CLI is not authenticated with a recognized subscription plan.',
                'execution_available': False,
                'reason': f'Unrecognized auth method "{auth_method}".',
            }

    def get_supported_models(self, config: ProviderConfigModel | None = None) -> list[dict[str, Any]]:
        """Return supported model catalog."""
        if config and config.models:
            valid_models: list[dict[str, Any]] = []
            for m in config.models:
                try:
                    cleaned_id = validate_model_identifier(m.id)
                    valid_models.append({'id': cleaned_id, 'name': m.name or cleaned_id})
                except ValueError:
                    log.warning('Ignoring configured Claude model with invalid ID: %s', m.id)
            if valid_models:
                return valid_models
        return SUPPORTED_CLAUDE_MODELS

    def _validate_model_id(self, model_id: str, config: ProviderConfigModel | None = None) -> str:
        """Validate requested model against official aliases and validated overrides."""
        cleaned = validate_model_identifier(model_id)
        supported = {m['id'] for m in self.get_supported_models(config)}
        if cleaned not in supported and cleaned not in SUPPORTED_MODEL_IDS:
            raise ValueError(f'Requested model "{cleaned}" is not supported by Claude subscription.')
        return cleaned

    def _extract_reasoning_effort(self, request_data: dict[str, Any]) -> str | None:
        """Extract requested reasoning effort from request_data, honoring per-chat/nested precedence."""
        for sub in ('params', 'options'):
            nested = request_data.get(sub)
            if isinstance(nested, dict):
                for key in ('reasoning_effort', 'reasoning'):
                    val = nested.get(key)
                    if val is not None:
                        return str(val).strip().lower()

        for key in ('reasoning_effort', 'reasoning'):
            val = request_data.get(key)
            if val is not None:
                return str(val).strip().lower()

        return None

    def _build_command_args(
        self,
        binary: str,
        model_id: str,
        streaming: bool,
        effort: str | None = None,
    ) -> list[str]:
        """Construct fail-closed, tool-disabled invocation arguments.

        - Disables policy hooks via --settings '{"disableAllHooks": true}'
        - Neutralizes all tools via --tools "" and --disallowedTools "*"
        - Neutralizes MCP servers via --strict-mcp-config and --mcp-config "{}"
        - Avoids --bare (preserves subscription OAuth)
        - Forwards validated --effort when requested
        """
        cmd = [
            binary,
            '-p',  # Non-interactive print mode (prompt fed via stdin)
            '--safe-mode',  # Safe mode
            '--settings', '{"disableAllHooks": true}',  # Documented hook neutralization
            '--tools', '',  # Disable all built-in tools (Bash, Edit, Read)
            '--disallowedTools', '*',  # Deny external and MCP tools
            '--strict-mcp-config',  # Prevent loading project/local MCP configs
            '--mcp-config', '{}',  # Explicit empty MCP config
            '--no-session-persistence',  # Do not record persistent session
        ]
        if streaming:
            cmd.extend(['--output-format', 'stream-json', '--verbose', '--include-partial-messages'])
        else:
            cmd.extend(['--output-format', 'json'])

        if model_id:
            cmd.extend(['--model', model_id])

        if effort:
            cmd.extend(['--effort', effort])

        return cmd

    async def chat_completion(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> dict[str, Any]:
        """Execute non-streaming chat completion with strict schema verification."""
        # 1. Validate chat parameters against unsupported options
        validate_subscription_chat_controls(request_data)

        # 2. Execution-time subscription authentication check
        auth_status = await self.check_auth_status(config)
        if not auth_status.get('installed'):
            raise RuntimeError(auth_status.get('message', 'Claude CLI is not installed.'))
        if not auth_status.get('authenticated') or not auth_status.get('execution_available'):
            reason = auth_status.get('message', 'Claude CLI subscription is not available.')
            raise RuntimeError(f'Claude subscription execution error: {reason} (code: {ERROR_CLI_AUTH_REQUIRED})')

        installed, binary = self.is_installed(config)
        if not installed:
            raise RuntimeError(binary)

        validated_model = self._validate_model_id(model_id, config)
        messages = request_data.get('messages', [])
        prompt = format_structured_prompt(messages)
        env = self._get_env(config)

        effort = self._extract_reasoning_effort(request_data)
        if effort is not None:
            if effort not in SUPPORTED_CLAUDE_EFFORT_LEVELS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f'Claude subscription does not support reasoning_effort "{effort}". '
                        f'Supported options: {", ".join(sorted(SUPPORTED_CLAUDE_EFFORT_LEVELS))} '
                        f'(code: {ERROR_UNSUPPORTED_PARAMETER}).'
                    ),
                )

        # 3. Execute subprocess in private mode 0700 scratch directory
        with create_request_scratch_dir(prefix='claude_chat_') as scratch_cwd:
            if effort is not None:
                effort_cap_ok = await self._verify_effort_capability(binary, env, scratch_cwd)
                if not effort_cap_ok:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'Installed Claude CLI does not support --effort (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                    )
            cmd = self._build_command_args(binary, validated_model, streaming=False, effort=effort)
            code, stdout, stderr = await SubprocessExecutor.execute_command(
                cmd,
                input_data=prompt,
                env=env,
                timeout=timeout,
                cwd=scratch_cwd,
            )

        if code != 0:
            log.warning('Claude CLI execution failed with exit code %s', code)
            raise RuntimeError(f'Claude subscription execution failed (code: {ERROR_CLI_EXECUTION_ERROR}).')

        # 4. Parse strict JSON nonstream output
        try:
            data = json.loads(stdout.strip())
            if not isinstance(data, dict):
                raise ValueError('Parsed output is not a JSON object.')
        except Exception:
            raise RuntimeError(
                f'Claude subscription execution failed: malformed JSON response (code: {ERROR_CLI_INVALID_RESPONSE}).'
            )

        # 5. Require success result/type/subtype; reject is_error or task failure markers
        if data.get('is_error') or data.get('type') == 'error':
            raise RuntimeError(f'Claude subscription task error reported (code: {ERROR_CLI_EXECUTION_ERROR}).')

        if data.get('type') != 'result' or data.get('subtype') != 'success':
            raise RuntimeError(
                f'Claude subscription execution did not return a successful result (code: {ERROR_CLI_EXECUTION_ERROR}).'
            )

        content = data.get('result')
        if content is None:
            raise RuntimeError(
                f'Claude subscription response missing result content (code: {ERROR_CLI_INVALID_RESPONSE}).'
            )

        # 6. Parse accurate token counts including cache metrics; omit if unknown
        usage = parse_usage_dict(data.get('usage'))

        return self.build_openai_completion_dict(
            content=str(content),
            model_id=validated_model,
            usage=usage,
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
        if not auth_status.get('installed') or not auth_status.get('authenticated') or not auth_status.get('execution_available'):
            reason = auth_status.get('message', 'Claude CLI subscription is not available.')
            err_chunk = {
                'error': {
                    'message': f'Claude subscription unavailable: {reason}',
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

        validated_model = self._validate_model_id(model_id, config)
        messages = request_data.get('messages', [])
        prompt = format_structured_prompt(messages)
        env = self._get_env(config)

        effort = self._extract_reasoning_effort(request_data)
        if effort is not None:
            if effort not in SUPPORTED_CLAUDE_EFFORT_LEVELS:
                err_chunk = {
                    'error': {
                        'message': (
                            f'Claude subscription does not support reasoning_effort "{effort}". '
                            f'Supported options: {", ".join(sorted(SUPPORTED_CLAUDE_EFFORT_LEVELS))} '
                            f'(code: {ERROR_UNSUPPORTED_PARAMETER}).'
                        ),
                        'type': 'invalid_request_error',
                        'code': ERROR_UNSUPPORTED_PARAMETER,
                    }
                }
                yield f'data: {json.dumps(err_chunk)}\n\n'
                return

        cid = f'chatcmpl-{uuid.uuid4().hex[:12]}'
        now = int(time.time())
        full_model = f'subscription:{self.provider_id}:{validated_model}'
        last_usage: dict[str, Any] | None = None

        terminal_result_seen = False
        terminal_success = False
        stream_gen = None

        with create_request_scratch_dir(prefix='claude_stream_') as scratch_cwd:
            if effort is not None:
                effort_cap_ok = await self._verify_effort_capability(binary, env, scratch_cwd)
                if not effort_cap_ok:
                    err_chunk = {
                        'error': {
                            'message': f'Installed Claude CLI does not support --effort (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                            'type': 'invalid_request_error',
                            'code': ERROR_UNSUPPORTED_PARAMETER,
                        }
                    }
                    yield f'data: {json.dumps(err_chunk)}\n\n'
                    return

            cmd = self._build_command_args(binary, validated_model, streaming=True, effort=effort)
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

                    if event_type == 'stream_event':
                        inner = event.get('event', {})
                        inner_type = inner.get('type')
                        if inner_type == 'content_block_delta':
                            delta = inner.get('delta', {})
                            if delta.get('type') == 'text_delta' or 'text' in delta:
                                chunk_text = delta.get('text', '')
                        elif inner_type == 'message_delta':
                            if 'usage' in inner:
                                parsed_u = parse_usage_dict(inner['usage'])
                                if parsed_u:
                                    last_usage = parsed_u

                    elif event_type == 'content_block_delta':
                        delta = event.get('delta', {})
                        if delta.get('type') == 'text_delta' or 'text' in delta:
                            chunk_text = delta.get('text', '')

                    elif event_type == 'result':
                        terminal_result_seen = True
                        if 'usage' in event:
                            parsed_u = parse_usage_dict(event['usage'])
                            if parsed_u:
                                last_usage = parsed_u

                        if event.get('is_error') or event.get('subtype') != 'success':
                            terminal_success = False
                            raise RuntimeError(
                                f'Claude stream result reported failure (code: {ERROR_CLI_EXECUTION_ERROR}).'
                            )
                        else:
                            terminal_success = True

                    elif event_type == 'error':
                        raise RuntimeError(f'Claude stream reported error event (code: {ERROR_CLI_EXECUTION_ERROR}).')

                    if chunk_text:
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

                # Verify terminal success before emitting finish_reason: stop and [DONE]
                if not (terminal_result_seen and terminal_success):
                    raise RuntimeError(
                        f'Claude stream finished prematurely without terminal success result (code: {ERROR_CLI_PREMATURE_EOF}).'
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
                log.warning('Claude stream failed (%s).', type(e).__name__)
                err_chunk = {
                    'error': {
                        'message': f'Claude subscription streaming failed (code: {ERROR_CLI_STREAM_ERROR}).',
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
        """Run an isolated test prompt through Claude subscription adapter."""
        auth_status = await self.check_auth_status(config)
        if not auth_status.get('installed'):
            return {'success': False, 'message': auth_status.get('message', 'Claude CLI is not installed.')}
        if not auth_status.get('authenticated') or not auth_status.get('execution_available'):
            return {'success': False, 'message': auth_status.get('message', 'Claude CLI is not authenticated.')}

        selected_model = model_id or (config.models[0].id if config.models else 'haiku')
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
                'message': f'Successfully connected to Claude subscription in {elapsed_ms}ms.',
            }
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                'success': False,
                'elapsed_ms': elapsed_ms,
                'model': selected_model,
                'message': sanitize_error_message(str(e)),
            }
