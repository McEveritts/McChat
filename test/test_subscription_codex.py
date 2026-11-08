from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
import unittest
import unittest.mock
from typing import Any

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import HTTPException

from open_webui.subscriptions.adapters.codex_session import (
    CodexSessionAdapter,
    SUPPORTED_CODEX_MODELS,
    SUPPORTED_MODEL_IDS,
    VALID_REASONING_EFFORTS,
    parse_codex_usage,
)
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
from open_webui.subscriptions.config import ProviderConfigModel, ModelDefinition


class TestCodexSessionAdapter(unittest.IsolatedAsyncioTestCase):
    """Rigorous contract, security, and protocol tests for CodexSessionAdapter."""

    def test_partial_usage_omits_unknown_metrics_and_preserves_explicit_zero(self):
        self.assertEqual(parse_codex_usage({'input_tokens': 0}), {'prompt_tokens': 0})
        self.assertEqual(parse_codex_usage({'output_tokens': 4}), {'completion_tokens': 4})
        self.assertEqual(parse_codex_usage({'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0}), {
            'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            'prompt_tokens_details': {'cached_tokens': 0}, 'cached_input_tokens': 0})
        self.assertEqual(parse_codex_usage({'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 7})['total_tokens'], 7)
        self.assertIsNone(parse_codex_usage({'input_tokens': 'unknown', 'output_tokens': -1}))

    async def test_status_requires_explicit_chatgpt_authentication(self):
        adapter = CodexSessionAdapter()
        for output in ('Logged in using an API key', 'Logged in', 'Logged in using an access token'):
            with self.subTest(output=output):
                with unittest.mock.patch.object(adapter, 'is_installed', return_value=(True, '/bin/codex')):
                    with unittest.mock.patch.object(adapter, '_verify_capabilities', new=unittest.mock.AsyncMock(return_value=(True, '', '0.166.0'))):
                        with unittest.mock.patch('open_webui.subscriptions.adapters.codex_session.SubprocessExecutor.execute_command', new=unittest.mock.AsyncMock(return_value=(0, output, ''))):
                            result = await adapter.check_auth_status(ProviderConfigModel())
                self.assertFalse(result['authenticated'])
                self.assertFalse(result['execution_available'])

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory(prefix='codex_test_')
        self.test_dir = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_fake_executable(self, name: str, script_content: str) -> str:
        """Create an executable bash script in the isolated test directory."""
        path = os.path.join(self.test_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('#!/bin/bash\n')
            f.write(script_content)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return path

    # -------------------------------------------------------------------------
    # 1. Environment Sanitization & CODEX_HOME
    # -------------------------------------------------------------------------
    def test_environment_sanitization_preserves_only_validated_codex_home(self):
        """Verify CODEX_HOME is preserved only when configured, and ambient keys are purged."""
        adapter = CodexSessionAdapter()
        test_env = {
            'PATH': '/usr/bin:/bin',
            'HOME': '/home/testuser',
            'LANG': 'en_US.UTF-8',
            'TMPDIR': '/tmp',
            'OPENAI_API_KEY': 'sk-proj-secret-12345',
            'CODEX_API_KEY': 'codex-secret-key',
            'CODEX_HOME': '/untrusted/ambient/codex',
            'AWS_SECRET_ACCESS_KEY': 'aws-secret-xyz',
            'WEBUI_SECRET_KEY': 'webui-secret-key-123',
            'DATABASE_URL': 'sqlite:///dangerous.db',
        }
        with unittest.mock.patch.dict(os.environ, test_env, clear=True):
            # Without configured config_dir: CODEX_HOME must be purged
            cfg_no_dir = ProviderConfigModel()
            env_no_dir = adapter._get_env(cfg_no_dir)
            self.assertEqual(env_no_dir.get('PATH'), '/usr/bin:/bin')
            self.assertEqual(env_no_dir.get('HOME'), '/home/testuser')
            self.assertNotIn('CODEX_HOME', env_no_dir)
            self.assertNotIn('OPENAI_API_KEY', env_no_dir)
            self.assertNotIn('CODEX_API_KEY', env_no_dir)
            self.assertNotIn('AWS_SECRET_ACCESS_KEY', env_no_dir)

            # With validated configured config_dir: ONLY validated CODEX_HOME is set
            safe_dir = os.path.join(self.test_dir, 'safe_codex_home')
            os.makedirs(safe_dir, exist_ok=True)
            cfg_with_dir = ProviderConfigModel(config_dir=safe_dir)
            env_with_dir = adapter._get_env(cfg_with_dir)
            self.assertEqual(os.path.realpath(env_with_dir.get('CODEX_HOME')), os.path.realpath(safe_dir))
            self.assertNotIn('OPENAI_API_KEY', env_with_dir)
            self.assertNotIn('CODEX_API_KEY', env_with_dir)

    # -------------------------------------------------------------------------
    # 2. Binary Installation & Verification
    # -------------------------------------------------------------------------
    def test_is_installed_validation(self):
        """Verify is_installed returns True for executable, False for missing or invalid."""
        adapter = CodexSessionAdapter()

        fake_binary = self._create_fake_executable('codex_bin', 'exit 0\n')
        installed, path = adapter.is_installed(ProviderConfigModel(binary_path=fake_binary))
        self.assertTrue(installed)
        self.assertEqual(os.path.realpath(path), os.path.realpath(fake_binary))

        # Missing binary
        installed, err = adapter.is_installed(ProviderConfigModel(binary_path='/nonexistent/codex_missing'))
        self.assertFalse(installed)
        self.assertTrue('does not exist' in err or 'not found' in err)

        # Flag injection prevention in binary_path
        installed, err = adapter.is_installed(ProviderConfigModel(binary_path='--bad-flag'))
        self.assertFalse(installed)

    # -------------------------------------------------------------------------
    # 3. Capability Preflight Verification
    # -------------------------------------------------------------------------
    async def test_capabilities_preflight_verifies_required_flags(self):
        """Verify preflight checks version and required security flags in exec --help."""
        adapter = CodexSessionAdapter()

        # Healthy binary with all required flags
        fake_healthy = self._create_fake_executable(
            'codex_cap_ok',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "Usage: codex exec [OPTIONS] [PROMPT]"
                echo "--ignore-user-config"
                echo "--sandbox <SANDBOX_MODE>"
                echo "--ephemeral"
                echo "--json"
                echo "--disable <FEATURE>"
                exit 0
            fi
            exit 1
            """
        )
        ok, msg, ver = await adapter._verify_capabilities(fake_healthy, {}, self.test_dir)
        self.assertTrue(ok)
        self.assertEqual(ver, 'codex-cli 0.166.0')

        # Binary missing --ignore-user-config or --disable
        fake_missing_flags = self._create_fake_executable(
            'codex_cap_missing',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.100.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--sandbox --json"
                exit 0
            fi
            exit 1
            """
        )
        ok, msg, ver = await adapter._verify_capabilities(fake_missing_flags, {}, self.test_dir)
        self.assertFalse(ok)
        self.assertIn('lacks required security flags', msg)
        self.assertIn('--ignore-user-config', msg)

    # -------------------------------------------------------------------------
    # 4. Authentication Status Checking & No Token Leakage
    # -------------------------------------------------------------------------
    async def test_auth_status_not_installed(self):
        """Verify status reporting when binary is missing."""
        adapter = CodexSessionAdapter()
        cfg = ProviderConfigModel(binary_path='/nonexistent/codex')
        status = await adapter.check_auth_status(cfg)
        self.assertFalse(status['installed'])
        self.assertFalse(status['authenticated'])
        self.assertFalse(status['execution_available'])
        self.assertEqual(status['status'], 'not_installed')

    async def test_auth_status_unsupported_cli(self):
        """Verify status reporting when binary fails security capabilities check."""
        adapter = CodexSessionAdapter()
        fake_old = self._create_fake_executable(
            'codex_old',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.1.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "Legacy options only"
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_old)
        status = await adapter.check_auth_status(cfg)
        self.assertTrue(status['installed'])
        self.assertFalse(status['authenticated'])
        self.assertFalse(status['execution_available'])
        self.assertEqual(status['status'], 'unsupported_cli')

    async def test_auth_status_unauthenticated(self):
        """Verify status reporting when official CLI reports Not logged in."""
        adapter = CodexSessionAdapter()
        fake_unauth = self._create_fake_executable(
            'codex_unauth',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Not logged in" >&2
                exit 1
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_unauth)
        status = await adapter.check_auth_status(cfg)
        self.assertTrue(status['installed'])
        self.assertFalse(status['authenticated'])
        self.assertFalse(status['execution_available'])
        self.assertEqual(status['status'], 'unauthenticated')
        self.assertIn('codex login', status['message'])

    async def test_auth_status_api_key_detected(self):
        """Verify status differentiates API key auth from subscription OAuth."""
        adapter = CodexSessionAdapter()
        fake_api_key = self._create_fake_executable(
            'codex_apikey',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "API key configured - sk-proj-12345"
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_api_key)
        status = await adapter.check_auth_status(cfg)
        self.assertTrue(status['installed'])
        self.assertFalse(status['authenticated'])
        self.assertFalse(status['execution_available'])
        self.assertEqual(status['status'], 'api_key_detected')
        self.assertIn('API key or access token instead of a ChatGPT subscription', status['message'])

    async def test_auth_status_chatgpt_authenticated_and_no_email_leakage(self):
        """Verify ChatGPT subscription is recognized and raw emails/tokens are never exposed."""
        adapter = CodexSessionAdapter()

        # Create a dummy auth.json inside a test CODEX_HOME
        codex_home = os.path.join(self.test_dir, 'codex_home')
        os.makedirs(codex_home, exist_ok=True)
        auth_file = os.path.join(codex_home, 'auth.json')
        with open(auth_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'secret_token': 'SUPER_SECRET_DO_NOT_READ'}))

        fake_auth_ok = self._create_fake_executable(
            'codex_chatgpt_ok',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Authentication is checked when you send a request."
                echo "Logged in using ChatGPT as operator@corporate.example.com"
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_auth_ok, config_dir=codex_home)
        status = await adapter.check_auth_status(cfg)

        self.assertTrue(status['installed'])
        self.assertTrue(status['authenticated'])
        self.assertTrue(status['execution_available'])
        self.assertEqual(status['status'], 'authenticated')
        self.assertIsNone(status['reason'])

        # Privacy verification: email and tokens must not be present in message or status fields
        self.assertNotIn('operator@corporate.example.com', status['message'])
        self.assertNotIn('SUPER_SECRET', status['message'])

    # -------------------------------------------------------------------------
    # 5. Model Catalog Discovery & Validation
    # -------------------------------------------------------------------------
    def test_get_supported_models_defaults_and_validation(self):
        """Verify synchronous get_supported_models returns default alias and verified models."""
        adapter = CodexSessionAdapter()
        models = adapter.get_supported_models()
        model_ids = {m['id'] for m in models}

        # Documented 'default' alias and current official models
        self.assertIn('default', model_ids)
        self.assertIn('gpt-6.1-sol', model_ids)
        self.assertIn('gpt-6-astra', model_ids)
        self.assertIn('gpt-5.5', model_ids)

        # Configured models
        cfg = ProviderConfigModel(
            models=[
                ModelDefinition(id='custom-codex-1', name='Custom Codex 1'),
                ModelDefinition(id='--injected-flag', name='Injected Flag'),
            ]
        )
        cfg_models = adapter.get_supported_models(cfg)
        cfg_ids = [m['id'] for m in cfg_models]
        self.assertIn('custom-codex-1', cfg_ids)
        self.assertNotIn('--injected-flag', cfg_ids)

    async def test_discover_models_extracts_bundled_catalog(self):
        """Verify discover_models dynamically inspects offline bundled catalog."""
        adapter = CodexSessionAdapter()
        fake_bundled = self._create_fake_executable(
            'codex_debug_models',
            """
            if [ "$1" = "debug" ] && [ "$2" = "models" ] && [ "$3" = "--bundled" ]; then
                echo '{"models": [{"slug": "gpt-6.1-sol", "display_name": "GPT-6.1-Sol", "visibility": "list"}, {"slug": "hidden-internal", "display_name": "Internal", "visibility": "hide"}]}'
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_bundled)
        discovered = await adapter.discover_models(cfg)
        disc_ids = [m['id'] for m in discovered]

        self.assertIn('default', disc_ids)
        self.assertIn('gpt-6.1-sol', disc_ids)
        self.assertNotIn('hidden-internal', disc_ids)

    # -------------------------------------------------------------------------
    # 6. Command Arguments & Host Isolation Boundaries
    # -------------------------------------------------------------------------
    def test_build_command_args_enforces_all_security_boundaries(self):
        """Verify all isolation flags are passed: --ignore-user-config, --sandbox, --disable, etc."""
        adapter = CodexSessionAdapter()
        scratch_dir = '/tmp/test_scratch_0700'

        # Default model: omits -m
        cmd_default = adapter._build_command_args('/usr/bin/codex', 'default', scratch_dir)
        self.assertEqual(cmd_default[0], '/usr/bin/codex')
        self.assertEqual(cmd_default[1], 'exec')
        self.assertIn('--ignore-user-config', cmd_default)
        self.assertIn('--ignore-rules', cmd_default)
        self.assertIn('--ephemeral', cmd_default)
        self.assertIn('--skip-git-repo-check', cmd_default)
        self.assertIn('--json', cmd_default)
        self.assertIn('--sandbox', cmd_default)
        self.assertEqual(cmd_default[cmd_default.index('--sandbox') + 1], 'read-only')
        self.assertIn('-c', cmd_default)
        self.assertIn('project_doc_max_bytes=0', cmd_default)
        self.assertIn('--enable', cmd_default)
        self.assertIn('skip_host_skill_discovery', cmd_default)

        # Disabled features
        for feat in (
            'shell_tool',
            'hooks',
            'plugins',
            'apps',
            'browser_use',
            'computer_use',
            'image_generation',
            'multi_agent',
            'view_image',
            'skill_search',
            'code_mode_host',
        ):
            self.assertIn(feat, cmd_default)

        self.assertIn('-C', cmd_default)
        self.assertEqual(cmd_default[cmd_default.index('-C') + 1], scratch_dir)

        # Model 'default' must NOT pass -m
        self.assertNotIn('-m', cmd_default)
        # Stdin prompt '-' must be last
        self.assertEqual(cmd_default[-1], '-')

        # Specified model passes -m
        cmd_model = adapter._build_command_args(
            '/usr/bin/codex', 'gpt-6.1-sol', scratch_dir, reasoning_effort='high'
        )
        self.assertIn('-m', cmd_model)
        self.assertEqual(cmd_model[cmd_model.index('-m') + 1], 'gpt-6.1-sol')
        self.assertIn('model_reasoning_effort="high"', cmd_model)
        self.assertEqual(cmd_model[-1], '-')

    # -------------------------------------------------------------------------
    # 7. Non-Streaming Chat Completion
    # -------------------------------------------------------------------------
    async def test_chat_completion_stdin_and_aggregate_agent_text_only(self):
        """Verify prompt is fed via stdin, only agent text is aggregated, and terminal turn.completed is required."""
        adapter = CodexSessionAdapter()

        stdin_dump_file = os.path.join(self.test_dir, 'captured_stdin.txt')
        argv_dump_file = os.path.join(self.test_dir, 'captured_argv.txt')

        fake_codex_exec = self._create_fake_executable(
            'codex_exec_ok',
            f"""
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Logged in using ChatGPT"
                exit 0
            elif [ "$1" = "exec" ]; then
                echo "$@" > "{argv_dump_file}"
                cat > "{stdin_dump_file}"
                # Output realistic Codex JSONL events
                echo '{{"type": "thread.started", "thread_id": "thread-abc"}}'
                echo '{{"type": "turn.started", "turn_id": "turn-xyz"}}'
                # Unrelated progress/command events that MUST NOT be included in response text
                echo '{{"type": "item.completed", "item": {{"type": "command_execution", "command": "ls", "exit_code": 0}}}}'
                echo '{{"type": "item.completed", "item": {{"type": "web_search", "query": "python"}}}}'
                # Genuine agent message chunks
                echo '{{"type": "item.completed", "item": {{"type": "agent_message", "text": "Hello "}}}}'
                echo '{{"type": "item.completed", "item": {{"type": "agent_message", "text": "from Codex!"}}}}'
                # Terminal turn.completed with accurate usage metrics
                echo '{{"type": "turn.completed", "turn_id": "turn-xyz", "usage": {{"input_tokens": 40, "output_tokens": 12, "cached_input_tokens": 10, "reasoning_output_tokens": 4}}}}'
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_codex_exec)
        secret_prompt = 'CLASSIFIED_PROMPT_CONTENT_987'
        resp = await adapter.chat_completion(
            {'messages': [{'role': 'user', 'content': secret_prompt}]},
            model_id='gpt-6.1-sol',
            config=cfg,
        )

        # Content check: only agent text
        content = resp['choices'][0]['message']['content']
        self.assertEqual(content, 'Hello from Codex!')
        self.assertNotIn('command_execution', content)
        self.assertNotIn('web_search', content)

        # Composite model ID check
        self.assertEqual(resp['model'], 'subscription:codex:gpt-6.1-sol')

        # Usage check
        usage = resp['usage']
        self.assertEqual(usage['prompt_tokens'], 40)
        self.assertEqual(usage['completion_tokens'], 12)
        self.assertEqual(usage['total_tokens'], 52)
        self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], 10)
        self.assertEqual(usage['completion_tokens_details']['reasoning_tokens'], 4)

        # Stdin & Argv check: prompt text must NOT leak into argv
        with open(argv_dump_file, 'r', encoding='utf-8') as f:
            captured_argv = f.read()
        self.assertNotIn(secret_prompt, captured_argv)
        self.assertIn('--ignore-user-config', captured_argv)

        with open(stdin_dump_file, 'r', encoding='utf-8') as f:
            captured_stdin = f.read()
        self.assertIn(secret_prompt, captured_stdin)

    async def test_chat_completion_fails_on_missing_terminal_turn(self):
        """Verify nonstream fails closed when turn.completed is missing."""
        adapter = CodexSessionAdapter()
        fake_incomplete = self._create_fake_executable(
            'codex_incomplete',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Logged in using ChatGPT"
                exit 0
            elif [ "$1" = "exec" ]; then
                echo '{"type": "item.completed", "item": {"type": "agent_message", "text": "Partial"}}'
                # Exits 0 but omitted turn.completed
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_incomplete)
        with self.assertRaises(RuntimeError) as ctx:
            await adapter.chat_completion(
                {'messages': [{'role': 'user', 'content': 'Hello'}]},
                model_id='default',
                config=cfg,
            )
        self.assertIn(ERROR_CLI_INVALID_RESPONSE, str(ctx.exception))

    async def test_chat_completion_fails_on_turn_failed_event(self):
        """Verify nonstream fails closed when turn.failed event is reported."""
        adapter = CodexSessionAdapter()
        fake_turn_err = self._create_fake_executable(
            'codex_turn_err',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Logged in using ChatGPT"
                exit 0
            elif [ "$1" = "exec" ]; then
                echo '{"type": "turn.failed", "error": "Model quota exhausted"}'
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_turn_err)
        with self.assertRaises(RuntimeError) as ctx:
            await adapter.chat_completion(
                {'messages': [{'role': 'user', 'content': 'Hello'}]},
                model_id='default',
                config=cfg,
            )
        self.assertIn(ERROR_CLI_EXECUTION_ERROR, str(ctx.exception))
        self.assertNotIn('Model quota exhausted', str(ctx.exception))
        self.assertIn(ERROR_CLI_EXECUTION_ERROR, str(ctx.exception))

    # -------------------------------------------------------------------------
    # 8. Streaming Chat Completion
    # -------------------------------------------------------------------------
    async def test_chat_completion_stream_success(self):
        """Verify streaming chat completion emits deltas, usage, and [DONE]."""
        adapter = CodexSessionAdapter()
        fake_stream_ok = self._create_fake_executable(
            'codex_stream_ok',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Logged in using ChatGPT"
                exit 0
            elif [ "$1" = "exec" ]; then
                echo '{"type": "thread.started", "thread_id": "t1"}'
                echo '{"type": "agent_message_content_delta", "delta": {"text": "Streamed "}}'
                echo '{"type": "agent_message_content_delta", "delta": {"text": "response!"}}'
                echo '{"type": "turn.completed", "usage": {"input_tokens": 15, "output_tokens": 5}}'
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_stream_ok)
        stream = adapter.chat_completion_stream(
            {'messages': [{'role': 'user', 'content': 'Hello'}]},
            model_id='default',
            config=cfg,
        )
        chunks = [c async for c in stream]
        full_sse = ''.join(chunks)

        self.assertIn('"Streamed "', full_sse)
        self.assertIn('"response!"', full_sse)
        self.assertIn('"finish_reason": "stop"', full_sse)
        self.assertIn('[DONE]', full_sse)
        self.assertIn('"total_tokens": 20', full_sse)

    async def test_chat_completion_stream_premature_eof_fails(self):
        """Verify stream premature EOF without turn.completed emits error and NO [DONE]."""
        adapter = CodexSessionAdapter()
        fake_stream_eof = self._create_fake_executable(
            'codex_stream_eof',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Logged in using ChatGPT"
                exit 0
            elif [ "$1" = "exec" ]; then
                echo '{"type": "agent_message_content_delta", "delta": {"text": "Cut off mid-way"}}'
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_stream_eof)
        stream = adapter.chat_completion_stream(
            {'messages': [{'role': 'user', 'content': 'Hello'}]},
            model_id='default',
            config=cfg,
        )
        chunks = [c async for c in stream]
        full_sse = ''.join(chunks)

        # Must NOT emit stop or [DONE]
        self.assertNotIn('"finish_reason": "stop"', full_sse)
        self.assertNotIn('[DONE]', full_sse)
        self.assertIn('subscription_error', full_sse)
        self.assertIn(ERROR_CLI_STREAM_ERROR, full_sse)

    # -------------------------------------------------------------------------
    # 9. Test Connection
    # -------------------------------------------------------------------------
    async def test_test_connection_selects_available_configured_model(self):
        adapter = CodexSessionAdapter()
        cfg = ProviderConfigModel(models=[ModelDefinition(id='restricted-model', name='Restricted')])
        async def complete(request_data, model_id, config, timeout):
            adapter._validate_model_id(model_id, config)
            return {'choices': [{'message': {'content': 'pong'}}]}
        with unittest.mock.patch.object(adapter, 'is_installed', return_value=(True, 'codex')), \
             unittest.mock.patch.object(adapter, 'check_auth_status', new=unittest.mock.AsyncMock(return_value={
                 'installed': True, 'authenticated': True, 'execution_available': True})), \
             unittest.mock.patch.object(adapter, 'chat_completion', side_effect=complete):
            result = await adapter.test_connection(cfg)
            self.assertTrue(result['success'], result)
            self.assertEqual(result['model'], 'restricted-model')
            invalid = await adapter.test_connection(cfg, model_id='default')
            self.assertFalse(invalid['success'])

    async def test_connection_success(self):
        """Verify test_connection returns success dict."""
        adapter = CodexSessionAdapter()
        fake_test = self._create_fake_executable(
            'codex_test_conn',
            """
            if [ "$1" = "--version" ]; then
                echo "codex-cli 0.166.0"
                exit 0
            elif [ "$1" = "exec" ] && [ "$2" = "--help" ]; then
                echo "--ignore-user-config --sandbox --ephemeral --json --disable"
                exit 0
            elif [ "$1" = "login" ] && [ "$2" = "status" ]; then
                echo "Logged in using ChatGPT"
                exit 0
            elif [ "$1" = "exec" ]; then
                echo '{"type": "item.completed", "item": {"type": "agent_message", "text": "pong"}}'
                echo '{"type": "turn.completed"}'
                exit 0
            fi
            exit 1
            """
        )
        cfg = ProviderConfigModel(binary_path=fake_test)
        res = await adapter.test_connection(cfg)
        self.assertTrue(res['success'])
        self.assertEqual(res['response'], 'pong')


    async def test_unsupported_reasoning_rejects_both_modes_without_inference(self):
        adapter = CodexSessionAdapter()
        available = {'installed': True, 'authenticated': True, 'execution_available': True}
        with unittest.mock.patch.object(adapter, 'check_auth_status', new=unittest.mock.AsyncMock(return_value=available)), \
             unittest.mock.patch.object(adapter, 'is_installed', return_value=(True, '/synthetic/codex')), \
             unittest.mock.patch('open_webui.subscriptions.adapters.codex_session.SubprocessExecutor.execute_command', new=unittest.mock.AsyncMock()) as execute:
            for effort in ('minimal', 'extra_high', 'unknown_native_level'):
                request = {'messages': [{'role': 'user', 'content': 'hello'}], 'reasoning_effort': effort}
                with self.assertRaises(HTTPException) as error:
                    await adapter.chat_completion(request, 'default', ProviderConfigModel())
                self.assertEqual(error.exception.status_code, 400)
                chunks = [chunk async for chunk in adapter.chat_completion_stream(request, 'default', ProviderConfigModel())]
                self.assertTrue(any(ERROR_UNSUPPORTED_PARAMETER in chunk for chunk in chunks))
                self.assertFalse(any('[DONE]' in chunk for chunk in chunks))
            execute.assert_not_awaited()
        self.assertEqual(adapter._extract_reasoning_effort({'reasoning_effort': 'low', 'params': {'reasoning_effort': 'high'}}), 'high')

    async def test_native_diagnostics_never_reach_errors_tests_or_logs(self):
        adapter = CodexSessionAdapter()
        secret = 'https://accounts.example.invalid/oauth?state=SYNTHETIC_STATE&code=SYNTHETIC_CODE refresh_token=PRIVATE_REFRESH'
        frame = json.dumps({'type': 'turn.failed', 'error': {'message': secret}})
        available = {'installed': True, 'authenticated': True, 'execution_available': True}
        async def stream(*args, **kwargs):
            yield frame
        with unittest.mock.patch.object(adapter, 'check_auth_status', new=unittest.mock.AsyncMock(return_value=available)), \
             unittest.mock.patch.object(adapter, 'is_installed', return_value=(True, '/synthetic/codex')), \
             unittest.mock.patch('open_webui.subscriptions.adapters.codex_session.SubprocessExecutor.execute_command', new=unittest.mock.AsyncMock(return_value=(0,frame,''))), \
             unittest.mock.patch('open_webui.subscriptions.adapters.codex_session.SubprocessExecutor.execute_stream', stream):
            with self.assertRaises(RuntimeError) as error:
                await adapter.chat_completion({'messages': [{'role':'user','content':'hello'}]}, 'default', ProviderConfigModel())
            tested = await adapter.test_connection(ProviderConfigModel())
            with self.assertLogs('open_webui.subscriptions.adapters.codex_session', level='WARNING') as logs:
                chunks = [chunk async for chunk in adapter.chat_completion_stream(
                    {'messages': [{'role':'user','content':'hello'}]}, 'default', ProviderConfigModel())]
        combined = str(error.exception) + tested['message'] + ''.join(chunks) + ''.join(logs.output)
        for marker in ('SYNTHETIC_STATE', 'SYNTHETIC_CODE', 'PRIVATE_REFRESH', 'accounts.example.invalid'):
            self.assertNotIn(marker, combined)
        self.assertFalse(tested['success'])
        self.assertFalse(any('[DONE]' in chunk for chunk in chunks))

    async def test_bundled_catalog_is_single_flight_cached_and_runtime_scoped(self):
        import shlex
        adapter = CodexSessionAdapter()
        counter = os.path.join(self.test_dir, 'discovery-count')
        def cli(model):
            catalog = json.dumps({'models': [{'slug': model, 'display_name': model, 'visibility': 'list'}]})
            return f"""
if [ "$1" = "debug" ]; then
    echo discovery >> {shlex.quote(counter)}
    echo '{catalog}'
    exit 0
elif [ "$1" = "exec" ]; then
    echo '{{"type":"item.completed","item":{{"type":"agent_message","text":"reply"}}}}'
    echo '{{"type":"turn.completed"}}'
    exit 0
fi
exit 1
"""
        binary = self._create_fake_executable('codex_dynamic_catalog', cli('native-catalog-a'))
        cfg = ProviderConfigModel(binary_path=binary)
        catalogs = await asyncio.gather(*(adapter.discover_models(cfg) for _ in range(4)))
        self.assertTrue(all('native-catalog-a' in {model['id'] for model in catalog} for catalog in catalogs))
        with open(counter) as stream:
            self.assertEqual(len(stream.readlines()), 1)
        available = {'installed': True, 'authenticated': True, 'execution_available': True}
        with unittest.mock.patch.object(adapter, 'check_auth_status', new=unittest.mock.AsyncMock(return_value=available)):
            result = await adapter.chat_completion({'messages':[{'role':'user','content':'hello'}]}, 'native-catalog-a', cfg)
        self.assertEqual(result['model'], 'subscription:codex:native-catalog-a')
        self._create_fake_executable('codex_dynamic_catalog', cli('native-catalog-b'))
        refreshed = await adapter.discover_models(cfg)
        self.assertIn('native-catalog-b', {model['id'] for model in refreshed})
        self.assertNotIn('native-catalog-a', {model['id'] for model in refreshed})
        override = ProviderConfigModel(binary_path=binary, models=[ModelDefinition(id='manual',name='Manual')])
        self.assertEqual(await adapter.discover_models(override), [{'id':'manual','name':'Manual'}])
        with open(counter) as stream:
            self.assertEqual(len(stream.readlines()), 2)


if __name__ == '__main__':
    unittest.main()
