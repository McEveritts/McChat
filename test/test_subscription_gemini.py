from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import HTTPException

from open_webui.subscriptions.acp_catalog import (
    extract_models,
    find_config_option,
    find_thinking_option,
    flatten_option_choices,
)
from open_webui.subscriptions.acp_client import (
    AcpClient,
    AcpError,
    AcpProcessError,
    AcpRequestError,
)
from open_webui.subscriptions.adapters.gemini import (
    CLIENT_INFO,
    CLIENT_CAPABILITIES,
    MAX_OUTPUT_BYTES,
    GeminiAdapter,
    parse_gemini_usage,
    validate_acp_model_id,
)
from open_webui.subscriptions.antigravity_auth import (
    AntigravityAuthController,
    AuthFlowState,
    antigravity_auth_controller,
)
from open_webui.subscriptions.antigravity_profile import (
    REMOVED_ENV_KEYS,
    build_antigravity_environment,
    get_default_profile_dir,
    parse_antigravity_authorization_url,
    prepare_antigravity_profile,
    validate_antigravity_callback_url,
    verify_browser_suppression,
)
from open_webui.subscriptions.config import ModelDefinition, ProviderConfigModel
from open_webui.subscriptions.security import validate_binary_path

TEST_SCRATCH_DIR = os.path.expanduser('~/.tmp/mcchat-subscriptions')


class TestAntigravitySecurityAndProfile(unittest.TestCase):
    """Tests for Antigravity binary validation, profile isolation, and OAuth URL safety."""

    def setUp(self):
        os.makedirs(TEST_SCRATCH_DIR, exist_ok=True)
        self.tmp_dir = tempfile.TemporaryDirectory(dir=TEST_SCRATCH_DIR, prefix='gemini_profile_test_')
        self.test_dir = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_validate_binary_path_official_antigravity_names(self):
        """Binary path validation allows official Antigravity/agy names and rejects gemini CLI fallback."""
        launcher = '/home/betheenvy/.local/libexec/t3-antigravity/agy_acp_server.par'
        if os.path.exists(launcher):
            self.assertEqual(validate_binary_path(launcher, 'gemini'), os.path.realpath(launcher))

        # Allowed official Antigravity/agy executable names
        for allowed_name in ['agy_acp_server', 'agy_acp_server.par', 'agy']:
            test_bin = os.path.join(self.test_dir, allowed_name)
            with open(test_bin, 'w') as f:
                f.write('#!/bin/sh\nexit 0\n')
            os.chmod(test_bin, 0o755)
            self.assertEqual(validate_binary_path(test_bin, 'gemini'), os.path.realpath(test_bin))

        # gemini CLI fallback must be REJECTED (wrong runtime, not official Antigravity)
        gemini_bin = os.path.join(self.test_dir, 'gemini')
        with open(gemini_bin, 'w') as f:
            f.write('#!/bin/sh\nexit 0\n')
        os.chmod(gemini_bin, 0o755)
        with self.assertRaises(ValueError):
            validate_binary_path(gemini_bin, 'gemini')

        # Arbitrary system binaries must be REJECTED
        for invalid_name in ['bash', 'sh', 'python3', 'curl', 'malicious.sh']:
            bad_bin = os.path.join(self.test_dir, invalid_name)
            with open(bad_bin, 'w') as f:
                f.write('#!/bin/sh\nexit 0\n')
            os.chmod(bad_bin, 0o755)
            with self.assertRaises(ValueError):
                validate_binary_path(bad_bin, 'gemini')

    def test_ensure_private_profile_creates_isolated_settings(self):
        """Private profile must have mode 0700 and contain oauth-personal settings."""
        profile_dir = os.path.join(self.test_dir, 'isolated_gemini')
        info = prepare_antigravity_profile(profile_dir)
        self.assertEqual(str(info.gemini_home), os.path.realpath(profile_dir))
        self.assertTrue(os.path.isdir(profile_dir))

        mode = stat.S_IMODE(os.stat(profile_dir).st_mode)
        self.assertEqual(mode, 0o700)

        settings_path = str(info.settings_path)
        self.assertTrue(os.path.isfile(settings_path))
        with open(settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data.get('auth', {}).get('type'), 'oauth-personal')

    def test_build_antigravity_env_purges_ambient_credentials(self):
        """Ambient Google and other API keys must be purged, while launcher HOME is preserved."""
        fake_env = {
            'HOME': '/home/betheenvy',
            'PATH': '/usr/bin:/bin',
            'GEMINI_API_KEY': 'secret-gemini-key',
            'GOOGLE_API_KEY': 'secret-google-key',
            'ANTHROPIC_API_KEY': 'secret-anthropic-key',
            'OPENAI_API_KEY': 'secret-openai-key',
            'T3_SECRET': 'dont-leak',
        }
        profile_dir = os.path.join(self.test_dir, 'profile')
        info = prepare_antigravity_profile(profile_dir)
        clean_env = build_antigravity_environment(info, base_env=fake_env)

        for key in ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', 'GCLOUD_PROJECT']:
            self.assertNotIn(key, clean_env)

        self.assertEqual(clean_env['GEMINI_HOME'], str(info.gemini_home))
        self.assertEqual(clean_env['AGY_ACP_FORCE_FILE_STORAGE'], '1')
        self.assertEqual(clean_env['HOME'], '/home/betheenvy')

    def test_validate_auth_url_safety(self):
        """Google OAuth authorization URL must strictly match accounts.google.com and loopback redirect."""
        valid_url = (
            'https://accounts.google.com/o/oauth2/v2/auth?'
            'client_id=test-client.apps.googleusercontent.com&'
            'response_type=code&'
            'redirect_uri=http%3A%2F%2F127.0.0.1%3A38491%2F&'
            'state=secure_state_token_12345&'
            'scope=email%20profile'
        )
        parsed = parse_antigravity_authorization_url(valid_url)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['state'], 'secure_state_token_12345')
        self.assertEqual(parsed['port'], 38491)

        self.assertIsNone(parse_antigravity_authorization_url(
            'https://evil.com/o/oauth2/v2/auth?state=123&redirect_uri=http://127.0.0.1:4000/&response_type=code'
        ))
        self.assertIsNone(parse_antigravity_authorization_url(
            'http://accounts.google.com/o/oauth2/v2/auth?state=123&redirect_uri=http://127.0.0.1:4000/&response_type=code'
        ))

    def test_validate_callback_url_safety(self):
        """Callback redirect URL must match expected loopback origin, port, and state."""
        expected_state = 'test_state_abc123'
        pending_redirect = 'http://127.0.0.1:45678/'
        valid_cb = f'http://127.0.0.1:45678/?state={expected_state}&code=4/0AX4XfWh...'
        clean_target = validate_antigravity_callback_url(
            pending_redirect_uri=pending_redirect,
            pending_state=expected_state,
            callback_url=valid_cb,
        )
        from urllib.parse import parse_qs, urlparse
        self.assertEqual(parse_qs(urlparse(clean_target).query)['code'], ['4/0AX4XfWh...'])

        with self.assertRaises(ValueError):
            validate_antigravity_callback_url(
                pending_redirect_uri=pending_redirect,
                pending_state='different_state',
                callback_url=valid_cb,
            )

    def test_verify_browser_suppression_helper(self):
        """Browser suppression helper must be verified before launch."""
        cmd = verify_browser_suppression(sys.executable)
        self.assertIn(sys.executable, cmd)
        self.assertIn('__T3_ANTIGRAVITY_AUTH_URL__', cmd)


class TestAcpCatalog(unittest.TestCase):
    """Tests for acp_catalog model extraction and option schema handling."""

    def test_flat_option_choices(self):
        """Extract models from flat options list using official 'value' field."""
        config_options = [
            {
                'id': 'model',
                'category': 'model',
                'type': 'select',
                'currentValue': 'model-alpha',
                'options': [
                    {'value': 'model-alpha', 'name': 'Model Alpha'},
                    {'value': 'model-beta', 'name': 'Model Beta'},
                ],
            }
        ]
        models = extract_models(config_options)
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0], {'id': 'model-alpha', 'name': 'Model Alpha'})
        self.assertEqual(models[1], {'id': 'model-beta', 'name': 'Model Beta'})

    def test_grouped_option_choices(self):
        """Extract models from nested SessionConfigSelectGroup options."""
        config_options = [
            {
                'id': 'model',
                'category': 'model',
                'type': 'select',
                'currentValue': 'm-pro',
                'options': [
                    {
                        'group': 'group-1',
                        'name': 'Flagship Models',
                        'options': [
                            {'value': 'm-pro', 'name': 'Model Pro'},
                            {'value': 'm-flash', 'name': 'Model Flash'},
                        ],
                    },
                    {
                        'group': 'group-2',
                        'name': 'Legacy Models',
                        'options': [
                            {'value': 'm-legacy', 'name': 'Model Legacy'},
                        ],
                    },
                ],
            }
        ]
        models = extract_models(config_options)
        self.assertEqual(len(models), 3)
        self.assertEqual([m['id'] for m in models], ['m-pro', 'm-flash', 'm-legacy'])

    def test_empty_catalog_no_injection(self):
        """Empty or missing model options returns empty list without injecting hardcoded defaults."""
        self.assertEqual(extract_models([]), [])
        self.assertEqual(extract_models(None), [])
        self.assertEqual(extract_models([{'id': 'mode', 'category': 'mode'}]), [])

    def test_find_thinking_option(self):
        """Find thinking level option by category 'thought_level' and id 'thinking_level'."""
        config_options = [
            {'id': 'model', 'category': 'model', 'currentValue': 'm1'},
            {
                'id': 'thinking_level',
                'category': 'thought_level',
                'type': 'select',
                'currentValue': 'low',
                'options': [
                    {'value': 'minimal', 'name': 'Minimal'},
                    {'value': 'low', 'name': 'Low'},
                    {'value': 'high', 'name': 'High'},
                ],
            },
        ]
        opt = find_thinking_option(config_options)
        self.assertIsNotNone(opt)
        self.assertEqual(opt['id'], 'thinking_level')
        choices = flatten_option_choices(opt)
        self.assertEqual([c['value'] for c in choices], ['minimal', 'low', 'high'])

    def test_extract_models_reasoning_efforts_scoped_to_current_model_only(self):
        """Reasoning efforts are derived ONLY for the current model whose thinking options are advertised."""
        config_options = [
            {
                'id': 'model',
                'category': 'model',
                'currentValue': 'family-a',
                'options': [
                    {'value': 'family-a', 'name': 'Family A'},
                    {'value': 'family-b', 'name': 'Family B'},
                ],
            },
            {
                'id': 'thinking_level',
                'category': 'thought_level',
                'currentValue': 'high',
                'options': [
                    {'value': 'minimal', 'name': 'Minimal'},
                    {'value': 'high', 'name': 'High'},
                    {'value': 'extra_high', 'name': 'Extra High'},
                ],
            },
        ]
        models = extract_models(config_options)
        self.assertEqual(len(models), 2)
        model_a = next(m for m in models if m['id'] == 'family-a')
        model_b = next(m for m in models if m['id'] == 'family-b')
        self.assertIn('reasoning_efforts', model_a)
        self.assertEqual(len(model_a['reasoning_efforts']), 3)
        self.assertEqual([e['value'] for e in model_a['reasoning_efforts']], ['minimal', 'high', 'extra_high'])
        self.assertNotIn('reasoning_efforts', model_b)


class TestGeminiUsage(unittest.TestCase):
    """Tests for native ACP Usage parsing."""

    def test_official_wire_usage_parsing(self):
        """Parse native inputTokens, outputTokens, cachedReadTokens, cachedWriteTokens, thoughtTokens, totalTokens."""
        raw = {
            'inputTokens': 100,
            'outputTokens': 20,
            'cachedReadTokens': 30,
            'cachedWriteTokens': 10,
            'thoughtTokens': 5,
            'totalTokens': 125,
        }
        usage = parse_gemini_usage(raw)
        self.assertIsNotNone(usage)
        self.assertEqual(usage['prompt_tokens'], 100)
        self.assertEqual(usage['completion_tokens'], 20)
        self.assertEqual(usage['total_tokens'], 125)
        self.assertEqual(usage['completion_tokens_details']['reasoning_tokens'], 5)
        self.assertEqual(usage['cached_tokens'], 30)
        self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], 30)
        self.assertEqual(usage['prompt_tokens_details']['cached_write_tokens'], 10)

    def test_usage_preserves_total_without_inventing_zeros(self):
        """When breakdown is missing, totalTokens is preserved."""
        raw = {'totalTokens': 50}
        usage = parse_gemini_usage(raw)
        self.assertIsNotNone(usage)
        self.assertEqual(usage['total_tokens'], 50)

    def test_empty_usage_returns_none(self):
        """Empty or invalid usage dict returns None."""
        self.assertIsNone(parse_gemini_usage(None))
        self.assertIsNone(parse_gemini_usage({}))
        self.assertIsNone(parse_gemini_usage({'irrelevant': 'key'}))


class MockAcpServer:
    """Helper to generate a mock ACP server script adhering to official 1.4.5 schemas."""

    @staticmethod
    def create_mock_script(tmp_dir: str, script_name: str = 'agy_acp_server', behavior: dict[str, Any] | None = None) -> str:
        behavior_json = json.dumps(behavior or {})
        script_path = os.path.join(tmp_dir, script_name)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(f'''#!/usr/bin/env python3
import sys
import json
import time

sys.stdin.reconfigure(line_buffering=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

behavior = json.loads({json.dumps(behavior_json)})

if behavior.get("emit_auth_url_on_start"):
    auth_url = behavior.get("auth_url", "https://accounts.google.com/o/oauth2/v2/auth?state=xyz&redirect_uri=http://127.0.0.1:45678/&response_type=code")
    sys.stderr.write('__T3_ANTIGRAVITY_AUTH_URL__' + json.dumps(auth_url) + chr(10))
    sys.stderr.flush()

session_counter = 0
selected_model = "native-model-a"
selected_thinking = "low"

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        agent_ver = behavior.get("agent_version", "1.4.5")
        resp = {{
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {{
                "protocolVersion": 2,
                "agentInfo": {{"name": "antigravity-acp", "version": agent_ver}},
                "agentCapabilities": {{
                    "auth": {{"logout": {{}}}},
                    "promptCapabilities": {{"image": True}}
                }}
            }}
        }}
        sys.stdout.write(json.dumps(resp) + chr(10))

    elif method == "authenticate":
        if behavior.get("auth_required"):
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {{"code": -32000, "message": "Authentication required: visit URL"}}
            }}
        elif behavior.get("sensitive_auth_error"):
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {{"code": -32000, "message": "Failed at https://accounts.google.com/o/oauth2/v2/auth?state=s3cr3t&code=auth_code_123 with Bearer ya29.secret_token"}}
            }}
        elif behavior.get("api_key_error"):
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {{"code": -32001, "message": "Failed due to INVALID_API_KEY with key AIzaSyFakeSecret"}}
            }}
        elif behavior.get("privacy_error"):
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {{"code": -32002, "message": "User terms and PRIVACY policy acceptance required"}}
            }}
        else:
            resp = {{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}
        sys.stdout.write(json.dumps(resp) + chr(10))

    elif method == "session/new":
        session_counter += 1
        sess_id = f"test-session-{{session_counter}}"
        resp = {{
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {{
                "sessionId": sess_id,
                "configOptions": [
                    {{
                        "id": "model",
                        "category": "model",
                        "type": "select",
                        "currentValue": selected_model,
                        "options": [
                            {{"value": "native-model-a", "name": "Model Alpha"}},
                            {{"value": "native-model-b", "name": "Model Beta"}}
                        ]
                    }},
                    {{
                        "id": "thinking_level",
                        "category": "thought_level",
                        "type": "select",
                        "currentValue": selected_thinking,
                        "options": [
                            {{"value": "low", "name": "Low"}},
                            {{"value": "high", "name": "High"}}
                        ]
                    }}
                ]
            }}
        }}
        sys.stdout.write(json.dumps(resp) + chr(10))

    elif method in ("session/setOption", "session/set_config_option"):
        params = req.get("params", {{}})
        config_id = params.get("configId")
        val = params.get("value")
        if config_id == "model":
            selected_model = val
            if behavior.get("thinking_by_model", {{}}).get(selected_model):
                selected_thinking = behavior["thinking_by_model"][selected_model][0]["value"]
        elif config_id == "thinking_level":
            selected_thinking = val
        elif config_id == "thought_level":
            sys.stdout.write(json.dumps({{
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {{"code": -32602, "message": "Unknown configId: thought_level"}}
            }}) + chr(10))
            continue

        resp = {{
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {{
                "configOptions": [
                    {{
                        "id": "model",
                        "category": "model",
                        "type": "select",
                        "currentValue": selected_model,
                        "options": [
                            {{"value": "native-model-a", "name": "Model Alpha"}},
                            {{"value": "native-model-b", "name": "Model Beta"}}
                        ]
                    }},
                    {{
                        "id": "thinking_level",
                        "category": "thought_level",
                        "type": "select",
                        "currentValue": selected_thinking,
                        "options": behavior.get("thinking_by_model", {{}}).get(selected_model, [
                            {{"value": "low", "name": "Low"}},
                            {{"value": "high", "name": "High"}}
                        ])
                    }}
                ]
            }}
        }}
        sys.stdout.write(json.dumps(resp) + chr(10))

    elif method == "session/prompt":
        sess_id = req.get("params", {{}}).get("sessionId", "test-session-1")

        if behavior.get("prompt_sensitive_error"):
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {{
                    "code": -32603,
                    "message": "Crash Bearer ya29.secret_token with OAuth https://accounts.google.com?code=123"
                }}
            }}
            sys.stdout.write(json.dumps(resp) + chr(10))
            continue

        if behavior.get("foreign_update"):
            for_update = {{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": "foreign-session-xyz",
                    "update": {{
                        "sessionUpdate": "config_option_update",
                        "configOptions": [{{"id": "model", "category": "model", "currentValue": "foreign-model", "options": []}}]
                    }}
                }}
            }}
            sys.stdout.write(json.dumps(for_update) + chr(10))

        if behavior.get("model_replacement"):
            rep_model = behavior.get("model_replacement_target", "native-model-b")
            rep_thinking = behavior.get("model_replacement_thinking", "high")
            th_options = behavior.get("model_replacement_th_options", [{{"value": "low", "name": "Low"}}, {{"value": "high", "name": "High"}}])
            rep_opts = [
                {{
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "currentValue": rep_model,
                    "options": [{{"value": rep_model, "name": f"Model {{rep_model}}"}}]
                }}
            ]
            if behavior.get("disable_thinking_in_replacement"):
                pass
            else:
                rep_opts.append({{
                    "id": "thinking_level",
                    "category": "thought_level",
                    "type": "select",
                    "currentValue": rep_thinking,
                    "options": th_options
                }})
            rep_update = {{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": sess_id,
                    "update": {{
                        "sessionUpdate": "config_option_update",
                        "configOptions": rep_opts
                    }}
                }}
            }}
            sys.stdout.write(json.dumps(rep_update) + chr(10))

        if behavior.get("emit_user_echo"):
            user_echo = {{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": sess_id,
                    "update": {{
                        "sessionUpdate": "user_message_chunk",
                        "content": {{"type": "text", "text": "Echoed prompt"}}
                    }}
                }}
            }}
            sys.stdout.write(json.dumps(user_echo) + chr(10))

        if behavior.get("model_replacement_after_chunk"):
            chunk_update = {{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": sess_id,
                    "update": {{
                        "sessionUpdate": "agent_message_chunk",
                        "content": {{"type": "text", "text": "chunk before replacement"}}
                    }}
                }}
            }}
            sys.stdout.write(json.dumps(chunk_update) + chr(10))
            rep_update = {{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": sess_id,
                    "update": {{
                        "sessionUpdate": "config_option_update",
                        "configOptions": [{{"id": "model", "category": "model", "currentValue": "native-model-b", "options": []}}]
                    }}
                }}
            }}
            sys.stdout.write(json.dumps(rep_update) + chr(10))

        if not behavior.get("empty_response"):
            chunks = behavior.get("agent_chunks", [f"reply by {{behavior.get('model_replacement_target', selected_model)}}" if behavior.get("model_replacement") else f"reply by {{selected_model}}"])
            for chunk in chunks:
                agent_update = {{
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {{
                        "sessionId": sess_id,
                        "update": {{
                            "sessionUpdate": "agent_message_chunk",
                            "content": {{"type": "text", "text": chunk}}
                        }}
                    }}
                }}
                sys.stdout.write(json.dumps(agent_update) + chr(10))

        if behavior.get("missing_stop_reason"):
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {{
                    "usage": {{"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}
                }}
            }}
        else:
            stop_reason = behavior.get("stop_reason", "end_turn")
            resp = {{
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {{
                    "stopReason": stop_reason,
                    "usage": {{"inputTokens": 12, "outputTokens": 6, "cachedReadTokens": 4, "thoughtTokens": 2, "totalTokens": 18}}
                }}
            }}
        sys.stdout.write(json.dumps(resp) + chr(10))

    elif method == "session/cancel":
        if req_id is not None:
            resp = {{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}
            sys.stdout.write(json.dumps(resp) + chr(10))

    elif method == "logout":
        resp = {{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}
        sys.stdout.write(json.dumps(resp) + chr(10))
''')
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return script_path


class TestGeminiAdapter(unittest.IsolatedAsyncioTestCase):
    """Tests for GeminiAdapter enforcing official Antigravity ACP contracts."""

    def setUp(self):
        os.makedirs(TEST_SCRATCH_DIR, exist_ok=True)
        self.tmp_dir = tempfile.TemporaryDirectory(dir=TEST_SCRATCH_DIR, prefix='gemini_adapter_test_')
        self.test_dir = self.tmp_dir.name
        self.script_dir = os.path.join(self.test_dir, 'bin')
        os.makedirs(self.script_dir, exist_ok=True)
        self.profile_dir = os.path.join(self.test_dir, 'profile')
        prepare_antigravity_profile(self.profile_dir)
        self.adapter = GeminiAdapter()

        installed, binary = self.adapter.is_installed()
        binary_path = binary if installed else ''
        antigravity_auth_controller.clear_authenticated_state(binary_path=binary_path, config_dir='')
        antigravity_auth_controller.clear_authenticated_state(binary_path='', config_dir='')

    def tearDown(self):
        installed, binary = self.adapter.is_installed()
        binary_path = binary if installed else ''
        antigravity_auth_controller.clear_authenticated_state(binary_path=binary_path, config_dir='')
        antigravity_auth_controller.clear_authenticated_state(binary_path='', config_dir='')
        self.tmp_dir.cleanup()

    def test_unverified_account_models_empty(self):
        """Unverified account returns empty model list; no hardcoded auto/2.5 fallback."""
        cfg = ProviderConfigModel(config_dir=self.profile_dir)
        models = self.adapter.get_supported_models(cfg)
        self.assertEqual(models, [])

    def test_supported_models_account_verified(self):
        """Verified account exposes discovered models; configured restrictions intersect catalog."""
        installed, binary = self.adapter.is_installed()
        discovered = [{'id': 'native-model-a', 'name': 'Alpha'}, {'id': 'native-model-b', 'name': 'Beta'}]

        antigravity_auth_controller.set_authenticated_models(
            discovered,
            binary_path=binary,
            config_dir=self.profile_dir,
        )

        cfg_default = ProviderConfigModel(config_dir=self.profile_dir)
        models = self.adapter.get_supported_models(cfg_default)
        self.assertEqual(models, discovered)

        # Configured restrictions intersect verified catalog
        cfg_restricted = ProviderConfigModel(
            config_dir=self.profile_dir,
            models=[ModelDefinition(id='native-model-b', name='Beta')]
        )
        restricted_models = self.adapter.get_supported_models(cfg_restricted)
        self.assertEqual(restricted_models, [{'id': 'native-model-b', 'name': 'Beta'}])

    async def test_check_auth_status_does_not_spawn_background_session(self):
        """Status check returns 'unknown' without starting ACP session or triggering login."""
        config = ProviderConfigModel(config_dir=self.profile_dir)
        with patch.object(self.adapter, 'is_installed', return_value=(True, '/synthetic/agy_acp_server.par')):
            result = await self.adapter.check_auth_status(config)
            self.assertEqual(result['status'], 'unknown')
            self.assertTrue(result['execution_available'])
            self.assertIn('sign in', result['message'].lower())

    async def test_unsupported_parameters_rejected(self):
        """Subscriptions reject raw API parameters like temperature, top_p, etc."""
        config = ProviderConfigModel(config_dir=self.profile_dir)
        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.chat_completion(
                request_data={'model': 'native-model-a', 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.7},
                model_id='native-model-a',
                config=config,
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn('temperature', ctx.exception.detail)

    async def test_successful_chat_completion_with_model_selection(self):
        """Chat completion selects requested model via choice.value, not silently defaulting."""
        script_path = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        response = await self.adapter.chat_completion(
            request_data={
                'model': 'native-model-b',
                'messages': [{'role': 'user', 'content': 'Hello!'}],
            },
            model_id='native-model-b',
            config=config,
        )

        self.assertEqual(response['choices'][0]['message']['content'], 'reply by native-model-b')
        self.assertEqual(response['model'], 'subscription:gemini:native-model-b')
        self.assertEqual(response['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(response['usage']['prompt_tokens'], 12)
        self.assertEqual(response['usage']['completion_tokens'], 6)
        self.assertEqual(response['usage']['total_tokens'], 18)

    async def test_chat_completion_with_reasoning_effort(self):
        """Reasoning effort sets thinking_level configId with choice value."""
        script_path = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        response = await self.adapter.chat_completion(
            request_data={
                'model': 'native-model-b',
                'messages': [{'role': 'user', 'content': 'Think hard'}],
                'reasoning_effort': 'high',
            },
            model_id='native-model-b',
            config=config,
        )

        self.assertEqual(response['choices'][0]['finish_reason'], 'stop')

    async def test_unavailable_model_rejected_before_prompt(self):
        """Requesting an unadvertised model fails before prompt execution."""
        script_path = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
                model_id='nonexistent-model',
                config=config,
            )
        self.assertIn('not available in Antigravity session', str(ctx.exception))

    async def test_unavailable_reasoning_effort_rejected_before_prompt(self):
        """Requesting an unsupported reasoning effort fails before prompt execution."""
        script_path = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        # 'max' is a valid controls.py parameter but not offered by MockAcpServer (only low/high)
        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={
                    'messages': [{'role': 'user', 'content': 'hi'}],
                    'reasoning_effort': 'max',
                },
                model_id='native-model-a',
                config=config,
            )
        self.assertIn('is not available for model', str(ctx.exception))

    async def test_missing_stop_reason_rejected(self):
        """Prompt response missing terminal stopReason is strictly rejected as an error."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir, behavior={'missing_stop_reason': True}
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
                model_id='native-model-a',
                config=config,
            )
        self.assertIn('missing stopReason', str(ctx.exception))

    async def test_test_connection_empty_text_fails_no_fake_pong(self):
        """Connection test returning empty assistant text returns success=False, not fake pong."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir, behavior={'empty_response': True}
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        res = await self.adapter.test_connection(config, model_id='native-model-a')
        self.assertFalse(res['success'])
        self.assertIn('No assistant text received', res['message'])

    async def test_chat_stream_successful_sse(self):
        """Streaming yields OpenAI-compatible SSE chunks and terminates with [DONE]."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir, behavior={'agent_chunks': ['Chunk1 ', 'Chunk2']}
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        events = []
        async for sse in self.adapter.chat_completion_stream(
            request_data={'messages': [{'role': 'user', 'content': 'Stream me'}]},
            model_id='native-model-a',
            config=config,
        ):
            events.append(sse)

        self.assertTrue(any('Chunk1 ' in e for e in events))
        self.assertTrue(any('Chunk2' in e for e in events))
        self.assertTrue(events[-1].strip().endswith('[DONE]'))

    def test_mixed_profile_cache_isolation(self):
        """Different config_dirs isolate account state independently without crosstalk."""
        installed, binary = self.adapter.is_installed()
        profile_a = os.path.join(self.test_dir, 'profile_a')
        profile_b = os.path.join(self.test_dir, 'profile_b')
        prepare_antigravity_profile(profile_a)
        prepare_antigravity_profile(profile_b)

        models_a = [{'id': 'model-a1', 'name': 'A1'}, {'id': 'model-a2', 'name': 'A2'}]
        models_b = [{'id': 'model-b1', 'name': 'B1'}]

        antigravity_auth_controller.set_authenticated_models(
            models_a, binary_path=binary, config_dir=profile_a
        )

        cfg_a = ProviderConfigModel(config_dir=profile_a)
        cfg_b = ProviderConfigModel(config_dir=profile_b)

        self.assertEqual(self.adapter.get_supported_models(cfg_a), models_a)
        self.assertEqual(self.adapter.get_supported_models(cfg_b), [])

        antigravity_auth_controller.set_authenticated_models(
            models_b, binary_path=binary, config_dir=profile_b
        )

        self.assertEqual(self.adapter.get_supported_models(cfg_a), models_a)
        self.assertEqual(self.adapter.get_supported_models(cfg_b), models_b)

        antigravity_auth_controller.clear_authenticated_state(binary_path=binary, config_dir=profile_a)
        self.assertEqual(self.adapter.get_supported_models(cfg_a), [])
        self.assertEqual(self.adapter.get_supported_models(cfg_b), models_b)

        antigravity_auth_controller.clear_authenticated_state(binary_path=binary, config_dir=profile_b)
        self.assertEqual(self.adapter.get_supported_models(cfg_b), [])

    async def test_chat_stream_slow_consumer_output_cap(self):
        """Streaming enforces cumulative 4MB output ceiling across individually valid frames."""
        # 10 chunks of 500KB each = 5MB > 4MB MAX_OUTPUT_BYTES
        chunk = 'X' * 500_000
        script_path = MockAcpServer.create_mock_script(
            self.script_dir, behavior={'agent_chunks': [chunk] * 10}
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        events = []
        async for sse in self.adapter.chat_completion_stream(
            request_data={'messages': [{'role': 'user', 'content': 'Generate huge output'}]},
            model_id='native-model-a',
            config=config,
        ):
            events.append(sse)

        # Output cap must be enforced and [DONE] must NOT be emitted
        self.assertTrue(any('exceeded limit' in e.lower() or 'output limit exceeded' in e.lower() for e in events))
        self.assertFalse(any('[DONE]' in e for e in events))

    async def test_chat_completion_immediate_auth_failure(self):
        """Auth URL emission during chat completion immediately fails with auth required without 25s wait."""
        start = time.monotonic()
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={
                'emit_auth_url_on_start': True,
                'auth_required': True,
            },
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
                model_id='native-model-a',
                config=config,
                timeout=30,
            )
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 5.0)  # Must fail immediately, not wait 25s or 30s
        self.assertIn('CLI_AUTH_REQUIRED', str(ctx.exception))
        # Ensure auth URL is not exposed in error message
        self.assertNotIn('https://accounts.google.com', str(ctx.exception))

    async def test_chat_stream_cancellation_and_cleanup(self):
        """Consumer cancellation safely breaks stream, cancels task, and closes client."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir, behavior={'agent_chunks': ['Chunk1', 'Chunk2', 'Chunk3']}
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        gen = self.adapter.chat_completion_stream(
            request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
            model_id='native-model-a',
            config=config,
        )

        first_event = await anext(gen)
        self.assertIn('Chunk1', first_event)
        # Abrupt consumer exit / generator close
        await gen.aclose()

    async def test_native_model_replacement_nonstream(self):
        """Native fallback fails before any successful response can mislabel stored history."""
        script_path = MockAcpServer.create_mock_script(self.script_dir, behavior={'model_replacement': True})
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)
        with self.assertRaisesRegex(RuntimeError, 'MODEL_CHANGED'):
            await self.adapter.chat_completion(
                {'messages': [{'role': 'user', 'content': 'Hello'}]}, 'native-model-a', config)

    async def test_native_model_replacement_stream(self):
        script_path = MockAcpServer.create_mock_script(self.script_dir, behavior={'model_replacement': True})
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)
        chunks = [chunk async for chunk in self.adapter.chat_completion_stream(
            {'messages': [{'role': 'user', 'content': 'Hello'}]}, 'native-model-a', config)]
        self.assertTrue(any('MODEL_CHANGED' in chunk for chunk in chunks))
        self.assertFalse(any('[DONE]' in chunk for chunk in chunks))

    async def test_model_replacement_after_chunks_fails_closed(self):
        """Model switch after assistant chunks already yielded fails closed with MODEL_CHANGED."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={'model_replacement_after_chunk': True},
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        # In nonstream: raises RuntimeError with MODEL_CHANGED
        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'Hello'}]},
                model_id='native-model-a',
                config=config,
            )
        self.assertIn('MODEL_CHANGED', str(ctx.exception))

        # In stream: emits error chunk with MODEL_CHANGED and does NOT emit [DONE]
        events = []
        async for sse in self.adapter.chat_completion_stream(
            request_data={'messages': [{'role': 'user', 'content': 'Hello'}]},
            model_id='native-model-a',
            config=config,
        ):
            events.append(sse)
        self.assertTrue(any('MODEL_CHANGED' in e for e in events))
        self.assertFalse(any('[DONE]' in e for e in events))

    async def test_effort_invalidated_by_model_replacement(self):
        """Effort change or thinking disablement that invalidates requested reasoning fails clearly."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={
                'model_replacement': True,
                'model_replacement_target': 'native-model-b',
                'disable_thinking_in_replacement': True,
            },
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                    'reasoning_effort': 'high',
                },
                model_id='native-model-a',
                config=config,
            )
        self.assertTrue('MODEL_CHANGED' in str(ctx.exception) or 'reasoning effort' in str(ctx.exception).lower())

    async def test_foreign_session_update_ignored(self):
        """Session updates for foreign sessionId are strictly ignored and do not alter active model."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={'foreign_update': True},
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        response = await self.adapter.chat_completion(
            request_data={'messages': [{'role': 'user', 'content': 'Hello'}]},
            model_id='native-model-a',
            config=config,
        )

        self.assertEqual(response['model'], 'subscription:gemini:native-model-a')

    async def test_api_key_and_privacy_errors_classified(self):
        """API key and privacy policy errors are classified into safe generic codes without leaking raw keys."""
        # API key error
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={'api_key_error': True},
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
                model_id='native-model-a',
                config=config,
            )
        self.assertIn('API key', str(ctx.exception))
        self.assertNotIn('AIzaSyFakeSecret', str(ctx.exception))

        # Privacy error
        script_path_priv = MockAcpServer.create_mock_script(
            self.script_dir,
            script_name='agy_priv',
            behavior={'privacy_error': True},
        )
        config_priv = ProviderConfigModel(binary_path=script_path_priv, config_dir=self.profile_dir)
        with self.assertRaises(RuntimeError) as ctx_priv:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
                model_id='native-model-a',
                config=config_priv,
            )
        self.assertIn('Privacy policy', str(ctx_priv.exception))

    async def test_oauth_and_secret_redaction(self):
        """SEC8-06: Raw exception text containing OAuth URLs, state, code, or Bearer never leaks."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={'sensitive_auth_error': True},
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        # 1. In test_connection
        res = await self.adapter.test_connection(config, model_id='native-model-a')
        self.assertFalse(res['success'])
        self.assertNotIn('https://accounts.google.com', res['message'])
        self.assertNotIn('s3cr3t', res['message'])
        self.assertNotIn('auth_code_123', res['message'])
        self.assertNotIn('ya29.', res['message'])

        # 2. In chat_completion
        with self.assertRaises(RuntimeError) as ctx:
            await self.adapter.chat_completion(
                request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
                model_id='native-model-a',
                config=config,
            )
        err_str = str(ctx.exception)
        self.assertNotIn('https://accounts.google.com', err_str)
        self.assertNotIn('s3cr3t', err_str)
        self.assertNotIn('auth_code_123', err_str)
        self.assertNotIn('ya29.', err_str)

        # 3. In chat_completion_stream
        events = []
        async for sse in self.adapter.chat_completion_stream(
            request_data={'messages': [{'role': 'user', 'content': 'hi'}]},
            model_id='native-model-a',
            config=config,
        ):
            events.append(sse)
        for e in events:
            self.assertNotIn('https://accounts.google.com', e)
            self.assertNotIn('s3cr3t', e)
            self.assertNotIn('auth_code_123', e)
            self.assertNotIn('ya29.', e)

    def test_missing_bare_binary_not_reported_installed(self):
        """Configured bare runtime name that does not exist in PATH is reported not installed."""
        cfg = ProviderConfigModel(binary_path='agy_acp_server')
        installed, msg = self.adapter.is_installed(cfg)
        self.assertFalse(installed)
        self.assertIn('not found in PATH', msg)

    async def test_test_connection_records_actual_version_and_epoch(self):
        """test_connection records actual initialize agentInfo.version, expected_epoch, and scoped reasoning efforts."""
        script_path = MockAcpServer.create_mock_script(
            self.script_dir,
            behavior={'agent_version': '2.4.0'},
        )
        config = ProviderConfigModel(binary_path=script_path, config_dir=self.profile_dir)

        captured_args = {}
        original_setter = antigravity_auth_controller.set_authenticated_models

        def mock_setter(models, *args, **kwargs):
            captured_args['models'] = models
            captured_args['kwargs'] = kwargs
            return original_setter(models, *args, **kwargs)

        with patch.object(antigravity_auth_controller, 'set_authenticated_models', side_effect=mock_setter):
            res = await self.adapter.test_connection(config)
            self.assertTrue(res['success'])

        self.assertEqual(captured_args['kwargs'].get('version'), '2.4.0')
        self.assertIn('expected_epoch', captured_args['kwargs'])

        # Verify reasoning_efforts attached only to native-model-a (the current model)
        models = captured_args['models']
        model_a = next(m for m in models if m['id'] == 'native-model-a')
        model_b = next(m for m in models if m['id'] == 'native-model-b')
        self.assertIn('reasoning_efforts', model_a)
        self.assertNotIn('reasoning_efforts', model_b)

    async def test_catalog_refresh_cannot_restore_a_revoked_account(self):
        script = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        kwargs = {"binary_path": script, "config_dir": self.profile_dir}
        options = [{"id": "model", "category": "model", "currentValue": "native-model-a",
            "options": [{"value": "native-model-a", "name": "Alpha"}]}]
        import inspect
        for stream in (False, True):
            self.assertTrue(antigravity_auth_controller.set_authenticated_models(
                [{"id": "native-model-a", "name": "Alpha"}], **kwargs))
            async def revoked_prompt(client, session_id, text, **unused):
                self.assertTrue(antigravity_auth_controller.clear_authenticated_state(**kwargs))
                for update in ({"sessionUpdate": "config_option_update", "configOptions": options},
                    {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "reply"}}):
                    result = client.on_session_update({"sessionId": session_id, "update": update})
                    if inspect.isawaitable(result):
                        await result
                return {"stopReason": "end_turn"}
            with patch.object(AcpClient, 'prompt', revoked_prompt):
                request = {"messages": [{"role": "user", "content": "hello"}]}
                if stream:
                    _ = [chunk async for chunk in self.adapter.chat_completion_stream(request, 'native-model-a', config)]
                else:
                    await self.adapter.chat_completion(request, 'native-model-a', config)
            self.assertFalse(antigravity_auth_controller.get_account_state(**kwargs)["authenticated"])

    async def test_stream_setup_failure_closes_private_scratch(self):
        from contextlib import contextmanager
        scratch = Path(self.test_dir) / 'failed-stream-scratch'
        @contextmanager
        def context(**unused):
            scratch.mkdir()
            try:
                yield str(scratch)
            finally:
                scratch.rmdir()
        with patch('open_webui.subscriptions.adapters.gemini.create_request_scratch_dir', context), \
             patch.object(self.adapter, '_build_spawn_command_and_env', side_effect=RuntimeError('synthetic failure')):
            chunks = [chunk async for chunk in self.adapter.chat_completion_stream(
                {"messages": [{"role": "user", "content": "hello"}]}, 'native-model-a', ProviderConfigModel())]
        self.assertFalse(scratch.exists())
        self.assertTrue(any('error' in chunk for chunk in chunks))
        self.assertFalse(any('[DONE]' in chunk for chunk in chunks))


    async def test_selected_model_thinking_catalog_persists_in_all_three_paths(self):
        levels = [{'value':'minimal','name':'Minimal'}, {'value':'extra_high','name':'Extra high'}]
        script = MockAcpServer.create_mock_script(self.script_dir, behavior={'thinking_by_model': {'native-model-b': levels}})
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        kwargs = {'binary_path':script, 'config_dir':self.profile_dir}
        self.assertTrue((await self.adapter.test_connection(config, model_id='native-model-a'))['success'])
        self.assertTrue((await self.adapter.test_connection(config, model_id='native-model-b'))['success'])
        for mode in ('test', 'chat', 'stream'):
            if mode == 'chat':
                await self.adapter.chat_completion({'messages':[{'role':'user','content':'hello'}], 'reasoning_effort':'extra_high'}, 'native-model-b', config)
            elif mode == 'stream':
                chunks = [chunk async for chunk in self.adapter.chat_completion_stream(
                    {'messages':[{'role':'user','content':'hello'}], 'reasoning_effort':'minimal'}, 'native-model-b', config)]
                self.assertTrue(any('[DONE]' in chunk for chunk in chunks))
            models = {model['id']:model for model in antigravity_auth_controller.get_account_state(**kwargs)['models']}
            self.assertEqual(models['native-model-b']['reasoning_efforts'], levels)
            self.assertEqual([item['value'] for item in models['native-model-a']['reasoning_efforts']], ['low','high'])

    async def test_paused_consumer_releases_native_resources_on_completion_or_queue_timeout(self):
        from open_webui.subscriptions.executor import _CONCURRENCY_SEMAPHORE
        for chunk_count in (2, 150):
            script = MockAcpServer.create_mock_script(self.script_dir, behavior={'agent_chunks': ['hello'] * chunk_count})
            config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
            baseline = _CONCURRENCY_SEMAPHORE._value
            stream = self.adapter.chat_completion_stream(
                {'messages': [{'role': 'user', 'content': 'hello'}]}, 'native-model-a', config, timeout=0.5)
            first = await anext(stream)
            self.assertIn('hello', first)
            client = stream.ag_frame.f_locals['client']
            worker = stream.ag_frame.f_locals['stream_task']
            scratch = Path(stream.ag_frame.f_locals['scratch_dir'])
            try:
                await asyncio.sleep(1.0)
                self.assertIsNotNone(client.proc.returncode)
                self.assertTrue(worker.done())
                self.assertFalse(scratch.exists())
                self.assertEqual(_CONCURRENCY_SEMAPHORE._value, baseline)
                remaining = [chunk async for chunk in stream]
                if chunk_count == 150:
                    self.assertTrue(any('timed out' in chunk for chunk in remaining))
                    self.assertFalse(any('[DONE]' in chunk for chunk in remaining))
                else:
                    self.assertTrue(any('[DONE]' in chunk for chunk in remaining))
            finally:
                await stream.aclose()
            self.assertEqual(_CONCURRENCY_SEMAPHORE._value, baseline)

    async def test_accepted_effort_casing_survives_unchanged_native_notification(self):
        script = MockAcpServer.create_mock_script(self.script_dir, behavior={
            'model_replacement': True, 'model_replacement_target': 'native-model-a',
            'model_replacement_thinking': 'high'})
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        for effort in ('high', 'HIGH', ' High '):
            payload = {'messages': [{'role': 'user', 'content': 'hello'}], 'reasoning_effort': effort}
            response = await self.adapter.chat_completion(payload, 'native-model-a', config)
            self.assertTrue(response['choices'][0]['message']['content'])
            chunks = [chunk async for chunk in self.adapter.chat_completion_stream(payload, 'native-model-a', config)]
            self.assertTrue(any('[DONE]' in chunk for chunk in chunks))
            self.assertFalse(any('"error"' in chunk for chunk in chunks))

    async def test_current_model_omitted_thinking_clears_only_its_own_choices(self):
        from open_webui.subscriptions.antigravity_profile import profile_identity
        script = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        kwargs = {'binary_path': script, 'config_dir': self.profile_dir}
        self.assertTrue((await self.adapter.test_connection(config, model_id='native-model-a'))['success'])
        self.assertTrue((await self.adapter.test_connection(config, model_id='native-model-b'))['success'])
        for thinking in (None, {'id': 'thinking_level', 'category': 'thought_level', 'options': []}):
            state = {**antigravity_auth_controller.get_account_state(**kwargs),
                     'profile_identity': profile_identity(**kwargs)}
            epoch = state['epoch']
            options = [{'id': 'model', 'category': 'model', 'currentValue': 'native-model-a', 'options': [
                {'value': 'native-model-a', 'name': 'A'}, {'value': 'native-model-b', 'name': 'B'}]}]
            if thinking is not None:
                options.append(thinking)
            self.assertTrue(self.adapter._refresh_catalog(options, state, **kwargs, version='synthetic'))
            refreshed = antigravity_auth_controller.get_account_state(**kwargs)
            models = {m['id']: m for m in refreshed['models']}
            self.assertNotIn('reasoning_efforts', models['native-model-a'])
            self.assertEqual([e['value'] for e in models['native-model-b']['reasoning_efforts']], ['low', 'high'])
            self.assertEqual(refreshed['epoch'], epoch)
            self.assertEqual(state['epoch'], epoch)

    async def test_concurrent_chat_and_successful_test_merge_catalogs_without_revoking_auth(self):
        levels = [{'value': 'extra_high', 'name': 'Extra high'}]
        script = MockAcpServer.create_mock_script(self.script_dir, behavior={'thinking_by_model': {'native-model-b': levels}})
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        kwargs = {'binary_path': script, 'config_dir': self.profile_dir}
        self.assertTrue((await self.adapter.test_connection(config))['success'])
        original_prompt = AcpClient.prompt
        for stream in (False, True):
            waiting, release = asyncio.Event(), asyncio.Event()
            async def gated_prompt(client, session_id, text, **kw):
                if text == 'TEST_PROBE':
                    waiting.set()
                    await release.wait()
                return await original_prompt(client, session_id, text, **kw)
            initial = antigravity_auth_controller.get_account_state(**kwargs)
            with patch.object(AcpClient, 'prompt', gated_prompt):
                test = asyncio.create_task(self.adapter.test_connection(config, prompt='TEST_PROBE', model_id='native-model-b'))
                try:
                    await asyncio.wait_for(waiting.wait(), 3)
                    request = {'messages': [{'role': 'user', 'content': 'hello'}]}
                    if stream:
                        chunks = [c async for c in self.adapter.chat_completion_stream(request, 'native-model-a', config)]
                        self.assertTrue(any('[DONE]' in c for c in chunks))
                    else:
                        response = await self.adapter.chat_completion(request, 'native-model-a', config)
                        self.assertTrue(response['choices'][0]['message']['content'])
                finally:
                    release.set()
                    result = await asyncio.wait_for(test, 5)
            self.assertTrue(result['success'], result)
            refreshed = antigravity_auth_controller.get_account_state(**kwargs)
            self.assertEqual(refreshed['epoch'], initial['epoch'])
            models = {m['id']: m for m in refreshed['models']}
            self.assertEqual(models['native-model-b']['reasoning_efforts'], levels)
            self.assertEqual([e['value'] for e in models['native-model-a']['reasoning_efforts']], ['low', 'high'])

    async def test_cancellation_during_native_close_removes_scratch_before_task_gc(self):
        script = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        self.assertTrue((await self.adapter.test_connection(config))['success'])
        original_cleanup = AcpClient._close_resources
        for mode in ('test', 'chat', 'stream'):
            began, release = asyncio.Event(), asyncio.Event()
            clients = []
            async def gated_close(client):
                clients.append(client)
                began.set()
                await release.wait()
                await original_cleanup(client)
            import contextlib
            from open_webui.subscriptions.adapters.common import create_request_scratch_dir
            scratches = []
            @contextlib.contextmanager
            def scratch_context(**kw):
                with create_request_scratch_dir(**kw) as scratch:
                    scratches.append(Path(scratch))
                    yield scratch
            async def run():
                if mode == 'test':
                    return await self.adapter.test_connection(config)
                request = {'messages': [{'role': 'user', 'content': 'hello'}]}
                if mode == 'stream':
                    return [c async for c in self.adapter.chat_completion_stream(request, 'native-model-a', config)]
                return await self.adapter.chat_completion(request, 'native-model-a', config)
            with patch.object(AcpClient, '_close_resources', gated_close), \
                 patch('open_webui.subscriptions.adapters.gemini.create_request_scratch_dir', scratch_context):
                task = asyncio.create_task(run())
                await asyncio.wait_for(began.wait(), 5)
                task.cancel()
                release.set()
                result = await asyncio.gather(task, return_exceptions=True)
            self.assertIsInstance(result[0], asyncio.CancelledError)
            self.assertTrue(clients[0]._close_task.done())
            self.assertTrue(scratches)
            self.assertTrue(all(not scratch.exists() for scratch in scratches))

    async def test_runtime_replacement_during_test_cannot_certify_new_binary(self):
        script = MockAcpServer.create_mock_script(self.script_dir)
        config = ProviderConfigModel(binary_path=script, config_dir=self.profile_dir)
        original = AcpClient.initialize
        async def replaced(client, *args, **kwargs):
            result = await original(client, *args, **kwargs)
            path = Path(script)
            path.write_text(path.read_text() + '\n# replacement runtime\n')
            return result
        with patch.object(AcpClient, 'initialize', replaced):
            result = await self.adapter.test_connection(config)
        self.assertFalse(result['success'])
        self.assertFalse(antigravity_auth_controller.get_account_state(binary_path=script, config_dir=self.profile_dir)['authenticated'])



if __name__ == '__main__':
    unittest.main()
