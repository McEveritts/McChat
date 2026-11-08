from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault('WEBUI_SECRET_KEY', 'test-secret-key-for-unit-tests-1234567890')
os.environ.setdefault('ENABLE_DB_MIGRATIONS', 'false')
os.environ.setdefault('USE_SLIM_DOCKER', 'true')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from open_webui.subscriptions.controls import (
    ERROR_UNSUPPORTED_PARAMETER,
    validate_subscription_chat_controls,
)
from open_webui.subscriptions.routing import (
    is_subscription_model,
    is_subscription_owner,
    resolve_arena_model,
)
import importlib.util
import types

def _init_child_environment():
    if importlib.util.find_spec('PIL') is None:
        pil_mod = types.ModuleType('PIL')
        image_mod = types.ModuleType('PIL.Image')
        image_mod.open = MagicMock()
        pil_mod.Image = image_mod
        pil_mod.ImageOps = MagicMock()
        sys.modules['PIL'] = pil_mod
        sys.modules['PIL.Image'] = image_mod
        sys.modules['PIL.ImageOps'] = pil_mod.ImageOps

    if importlib.util.find_spec('pytz') is None:
        import datetime
        pytz_mod = types.ModuleType('pytz')
        pytz_mod.timezone = MagicMock()
        pytz_mod.utc = datetime.timezone.utc
        pytz_mod.UTC = datetime.timezone.utc
        sys.modules['pytz'] = pytz_mod

    if importlib.util.find_spec('pycrdt') is None:
        pycrdt_mod = types.ModuleType('pycrdt')
        pycrdt_mod.Doc = MagicMock
        pycrdt_mod.Text = MagicMock
        pycrdt_mod.Array = MagicMock
        pycrdt_mod.Map = MagicMock
        sys.modules['pycrdt'] = pycrdt_mod

    if importlib.util.find_spec('mcp') is None:
        mcp_mod = types.ModuleType('mcp')
        mcp_mod.Client = MagicMock
        shared_mod = types.ModuleType('mcp.shared')
        auth_mod = types.ModuleType('mcp.shared.auth')
        auth_mod.OAuthClientMetadata = MagicMock
        auth_mod.OAuthMetadata = MagicMock
        mcp_mod.shared = shared_mod
        shared_mod.auth = auth_mod
        client_mod = types.ModuleType('mcp.client')
        client_mod.advertise = MagicMock()
        streamable_mod = types.ModuleType('mcp.client.streamable_http')
        streamable_mod.streamable_http_client = MagicMock()
        client_mod.streamable_http = streamable_mod
        mcp_types_mod = types.ModuleType('mcp.types')
        mcp_types_mod.ElicitRequestFormParams = MagicMock
        mcp_types_mod.ElicitResult = MagicMock
        mcp_mod.client = client_mod
        mcp_mod.types = mcp_types_mod
        sys.modules['mcp'] = mcp_mod
        sys.modules['mcp.shared'] = shared_mod
        sys.modules['mcp.shared.auth'] = auth_mod
        sys.modules['mcp.client'] = client_mod
        sys.modules['mcp.client.streamable_http'] = streamable_mod
        sys.modules['mcp.types'] = mcp_types_mod

    if importlib.util.find_spec('validators') is None:
        validators_mod = types.ModuleType('validators')
        validators_mod.url = lambda u: True
        sys.modules['validators'] = validators_mod

    if importlib.util.find_spec('langchain_core') is None:
        lc_mod = types.ModuleType('langchain_core')
        lc_mod.__path__ = []
        lc_dl = types.ModuleType('langchain_core.document_loaders')
        lc_docs = types.ModuleType('langchain_core.documents')
        lc_cb = types.ModuleType('langchain_core.callbacks')
        lc_ret = types.ModuleType('langchain_core.retrievers')
        lc_utils = types.ModuleType('langchain_core.utils')
        lc_fc = types.ModuleType('langchain_core.utils.function_calling')
        lc_fc.convert_to_openai_function = MagicMock
        lc_utils.function_calling = lc_fc
        lc_ret.BaseRetriever = MagicMock
        lc_cb.CallbackManagerForRetrieverRun = MagicMock
        lc_cb.Callbacks = MagicMock
        lc_dl.BaseLoader = MagicMock
        lc_docs.Document = MagicMock
        lc_docs.BaseDocumentCompressor = MagicMock
        lc_mod.document_loaders = lc_dl
        lc_mod.documents = lc_docs
        lc_mod.callbacks = lc_cb
        lc_mod.retrievers = lc_ret
        lc_mod.utils = lc_utils
        sys.modules['langchain_core'] = lc_mod
        sys.modules['langchain_core.document_loaders'] = lc_dl
        sys.modules['langchain_core.documents'] = lc_docs
        sys.modules['langchain_core.callbacks'] = lc_cb
        sys.modules['langchain_core.retrievers'] = lc_ret
        sys.modules['langchain_core.utils'] = lc_utils
        sys.modules['langchain_core.utils.function_calling'] = lc_fc

    if importlib.util.find_spec('aiofiles') is None:
        aiofiles_mod = types.ModuleType('aiofiles')
        aiofiles_os = types.ModuleType('aiofiles.os')
        aiofiles_mod.os = aiofiles_os
        aiofiles_mod.open = MagicMock
        sys.modules['aiofiles'] = aiofiles_mod
        sys.modules['aiofiles.os'] = aiofiles_os

    if importlib.util.find_spec('aiocache') is None:
        aiocache_mod = types.ModuleType('aiocache')
        aiocache_mod.cached = lambda *a, **k: (lambda fn: fn)
        aiocache_mod.SimpleMemoryCache = MagicMock
        sys.modules['aiocache'] = aiocache_mod

    if importlib.util.find_spec('python_multipart') is None:
        pm_mod = types.ModuleType('python_multipart')
        pm_mod.__version__ = '0.0.20'
        sys.modules['python_multipart'] = pm_mod

    if importlib.util.find_spec('numpy') is None:
        np_mod = types.ModuleType('numpy')
        np_mod.array = MagicMock
        sys.modules['numpy'] = np_mod

    if importlib.util.find_spec('langchain_classic') is None:
        lc_classic = types.ModuleType('langchain_classic')
        lc_retr = types.ModuleType('langchain_classic.retrievers')
        lc_retr.ContextualCompressionRetriever = MagicMock
        lc_retr.EnsembleRetriever = MagicMock
        lc_classic.retrievers = lc_retr
        sys.modules['langchain_classic'] = lc_classic
        sys.modules['langchain_classic.retrievers'] = lc_retr

    if importlib.util.find_spec('tiktoken') is None:
        tt_mod = types.ModuleType('tiktoken')
        tt_mod.get_encoding = MagicMock
        sys.modules['tiktoken'] = tt_mod

    if importlib.util.find_spec('langchain_text_splitters') is None:
        lts_mod = types.ModuleType('langchain_text_splitters')
        lts_mod.MarkdownHeaderTextSplitter = MagicMock
        lts_mod.RecursiveCharacterTextSplitter = MagicMock
        lts_mod.TokenTextSplitter = MagicMock
        sys.modules['langchain_text_splitters'] = lts_mod

    if importlib.util.find_spec('boto3') is None:
        sys.modules['boto3'] = MagicMock()
    if importlib.util.find_spec('botocore') is None:
        bc_mod = types.ModuleType('botocore')
        bc_mod.__path__ = []
        bc_cfg = types.ModuleType('botocore.config')
        bc_cfg.Config = MagicMock
        bc_mod.config = bc_cfg
        sys.modules['botocore'] = bc_mod
        sys.modules['botocore.config'] = bc_cfg
        sys.modules['botocore.exceptions'] = MagicMock()
    if importlib.util.find_spec('re2') is None:
        import re
        sys.modules['re2'] = re

    if importlib.util.find_spec('httpx2') is None:
        import httpx
        sys.modules['httpx2'] = httpx

    if importlib.util.find_spec('jsonschema') is None:
        js_mod = types.ModuleType('jsonschema')
        js_mod.Draft202012Validator = MagicMock
        js_mod.FormatChecker = MagicMock
        sys.modules['jsonschema'] = js_mod

    if importlib.util.find_spec('referencing') is None:
        ref_mod = types.ModuleType('referencing')
        ref_mod.Registry = MagicMock
        sys.modules['referencing'] = ref_mod

    if importlib.util.find_spec('asgiref') is None:
        asgi_mod = types.ModuleType('asgiref')
        asgi_mod.__path__ = []
        asgi_typing = types.ModuleType('asgiref.typing')
        asgi_typing.ASGI3Application = MagicMock
        asgi_typing.ASGIReceiveCallable = MagicMock
        asgi_typing.ASGIReceiveEvent = MagicMock
        asgi_typing.ASGISendCallable = MagicMock
        asgi_typing.ASGISendEvent = MagicMock
        asgi_typing.Scope = MagicMock
        asgi_mod.typing = asgi_typing
        sys.modules['asgiref'] = asgi_mod
        sys.modules['asgiref.typing'] = asgi_typing

    if importlib.util.find_spec('loguru') is None:
        sys.modules['loguru'] = MagicMock()

    if importlib.util.find_spec('websockets') is None:
        sys.modules['websockets'] = MagicMock()

    if importlib.util.find_spec('yaml') is None:
        sys.modules['yaml'] = MagicMock()

    if importlib.util.find_spec('regex') is None:
        import re
        reg_mod = types.ModuleType('regex')
        def fake_compile(pattern, *a, **kw):
            clean = pattern.replace(r'[\p{L}\p{N}\p{M}]+', r'\w+')
            return re.compile(clean, *a, **kw)
        reg_mod.compile = fake_compile
        sys.modules['regex'] = reg_mod

_init_child_environment()

from open_webui.routers.tasks import apply_task_model_params


class TestSubscriptionControls(unittest.TestCase):
    """Test strict centralized controls validation for subscription models."""

    def test_accepts_documented_defaults(self):
        # All default/no-op parameters must pass without exception
        valid_payload = {
            'temperature': 1.0,
            'top_p': 1.0,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'response_format': {'type': 'text'},
            'stop': [],
            'reasoning_effort': 'medium',
        }
        validate_subscription_chat_controls(valid_payload)

    def test_rejects_non_default_temperature(self):
        for temp in (0.7, 0.0, 1.5):
            with self.subTest(temp=temp):
                with self.assertRaises(HTTPException) as ctx:
                    validate_subscription_chat_controls({'temperature': temp})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn(ERROR_UNSUPPORTED_PARAMETER, ctx.exception.detail)

    def test_rejects_custom_max_tokens_and_ceilings(self):
        for key in ('max_tokens', 'max_completion_tokens'):
            with self.subTest(key=key):
                with self.assertRaises(HTTPException) as ctx:
                    validate_subscription_chat_controls({key: 2048})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn(ERROR_UNSUPPORTED_PARAMETER, ctx.exception.detail)

    def test_rejects_structured_response_formats(self):
        for rf in ('json_object', {'type': 'json_object'}, {'type': 'json_schema'}):
            with self.subTest(rf=rf):
                with self.assertRaises(HTTPException) as ctx:
                    validate_subscription_chat_controls({'response_format': rf})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn(ERROR_UNSUPPORTED_PARAMETER, ctx.exception.detail)

    def test_rejects_nested_parameters(self):
        # Parameters hidden inside params or options dict must be caught
        nested_payload = {
            'params': {
                'temperature': 0.5,
            }
        }
        with self.assertRaises(HTTPException) as ctx:
            validate_subscription_chat_controls(nested_payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn(ERROR_UNSUPPORTED_PARAMETER, ctx.exception.detail)

    def test_rejects_invalid_reasoning_effort(self):
        malformed_efforts = (
            123,
            True,
            False,
            [],
            {},
            '',
            '   ',
            'a' * 100,
            'high\n',
            'high\x00',
            'high\x1f',
            'high\r',
            'high\t',
        )
        for bad in malformed_efforts:
            with self.subTest(bad=bad):
                with self.assertRaises(HTTPException) as ctx:
                    validate_subscription_chat_controls({'reasoning_effort': bad})
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn(ERROR_UNSUPPORTED_PARAMETER, ctx.exception.detail)
                # Verify error message does not echo raw input
                if isinstance(bad, str) and bad:
                    self.assertNotIn(bad, ctx.exception.detail)

    def test_accepts_valid_bounded_reasoning_effort_shapes(self):
        valid_efforts = (
            'minimal',
            'low',
            'medium',
            'high',
            'extra_high',
            'max',
            'xhigh',
            'ultra',
        )
        for effort in valid_efforts:
            with self.subTest(effort=effort):
                # Valid shapes must pass shared validation without raising
                validate_subscription_chat_controls({'reasoning_effort': effort})


class TestSubscriptionRouting(unittest.IsolatedAsyncioTestCase):
    """Test model classification and Arena pre-persistence resolution."""

    def test_is_subscription_model_detection(self):
        self.assertTrue(is_subscription_model(None, 'subscription:claude:claude-3-7-sonnet'))
        self.assertTrue(is_subscription_model({'id': 'subscription:codex:default'}))
        self.assertTrue(is_subscription_model({'id': 'custom-1', 'owned_by': 'subscription'}))
        self.assertTrue(is_subscription_model({'id': 'custom-2', 'connection_type': 'subscription'}))
        self.assertTrue(is_subscription_model({'id': 'custom-3', 'base_model_id': 'subscription:gemini:auto'}))
        self.assertFalse(is_subscription_model({'id': 'gpt-4o', 'owned_by': 'openai'}))
        self.assertFalse(is_subscription_model(None, 'llama3:latest'))

    async def test_arena_resolution_filters_subscription_models_for_non_owner(self):
        request = MagicMock()
        request.app.state.MODELS = {
            'gpt-4o': {'id': 'gpt-4o', 'owned_by': 'openai'},
            'subscription:claude:claude-3-7-sonnet': {
                'id': 'subscription:claude:claude-3-7-sonnet',
                'owned_by': 'subscription',
            },
        }

        arena_model = {
            'id': 'arena-1',
            'owned_by': 'arena',
            'info': {'meta': {'model_ids': ['gpt-4o', 'subscription:claude:claude-3-7-sonnet']}},
        }
        non_owner = SimpleNamespace(id='user-non-owner')

        with patch('open_webui.subscriptions.routing.is_subscription_owner', new=AsyncMock(return_value=False)):
            resolved_model, resolved_id = await resolve_arena_model(request, arena_model, non_owner)
            # Must strictly select the non-subscription model
            self.assertEqual(resolved_id, 'gpt-4o')

    async def test_arena_resolution_fails_closed_when_no_candidates_eligible(self):
        request = MagicMock()
        request.app.state.MODELS = {
            'subscription:claude:claude-3-7-sonnet': {
                'id': 'subscription:claude:claude-3-7-sonnet',
                'owned_by': 'subscription',
            },
        }
        arena_model = {
            'id': 'arena-subs-only',
            'owned_by': 'arena',
            'info': {'meta': {'model_ids': ['subscription:claude:claude-3-7-sonnet']}},
        }
        non_owner = SimpleNamespace(id='user-non-owner')

        with patch('open_webui.subscriptions.routing.is_subscription_owner', new=AsyncMock(return_value=False)):
            with self.assertRaises(HTTPException) as ctx:
                await resolve_arena_model(request, arena_model, non_owner)
            self.assertEqual(ctx.exception.status_code, 400)


class TestTasksModelParams(unittest.TestCase):
    """Test server-injected task controls stripping for subscription targets."""

    def test_strips_max_tokens_for_subscription_target(self):
        models = {
            'subscription:codex:default': {
                'id': 'subscription:codex:default',
                'owned_by': 'subscription',
            },
            'gpt-4o': {
                'id': 'gpt-4o',
                'owned_by': 'openai',
            },
        }

        # For standard models, params are preserved
        standard_payload = {'model': 'gpt-4o', 'messages': []}
        res_std = apply_task_model_params(standard_payload, models, 'gpt-4o', {'max_tokens': 1000})
        self.assertEqual(res_std.get('max_tokens'), 1000)

        # For subscription models, server-injected params are stripped
        sub_payload = {'model': 'subscription:codex:default', 'messages': []}
        res_sub = apply_task_model_params(sub_payload, models, 'subscription:codex:default', {'max_tokens': 1000, 'temperature': 0.7})
        self.assertNotIn('max_tokens', res_sub)
        self.assertNotIn('temperature', res_sub)

    async def test_alias_retains_native_reasoning_effort_strings(self):
        """Verify alias parameter application preserves valid native reasoning strings (minimal/extra_high)."""
        from open_webui.utils.chat import generate_chat_completion

        alias_model = {
            'id': 'my-gemini-alias',
            'base_model_id': 'subscription:gemini:gemini-2.5-flash',
            'owned_by': 'subscription',
        }
        fake_model_info = MagicMock()
        fake_model_info.params = MagicMock()
        fake_model_info.params.model_dump.return_value = {
            'reasoning_effort': 'minimal',
            'temperature': 0.7,
        }

        captured_payloads = []

        async def fake_sub_generate(request, form_data, user):
            captured_payloads.append(form_data)
            return {'result': 'ok'}

        with patch('open_webui.models.models.Models.get_model_by_id', new=AsyncMock(return_value=fake_model_info)), \
             patch('open_webui.subscriptions.manager.subscription_manager.generate_chat_completion', new=fake_sub_generate):

            fake_request = MagicMock()
            fake_request.state = MagicMock()
            fake_request.state.bypass_system_prompt = False

            # Case A: Alias default 'minimal' is retained
            form_data = {'model': 'my-gemini-alias', 'messages': [{'role': 'user', 'content': 'hi'}]}
            await generate_chat_completion(
                request=fake_request,
                form_data=form_data,
                user=SimpleNamespace(id='owner', role='admin'),
                models={'my-gemini-alias': alias_model},
            )
            self.assertEqual(captured_payloads[0].get('reasoning_effort'), 'minimal')

            # Case B: Per-chat override in params is NOT shadowed by alias default
            fake_request.state.bypass_system_prompt = False
            form_data_override = {
                'model': 'my-gemini-alias',
                'messages': [{'role': 'user', 'content': 'hi'}],
                'params': {'reasoning_effort': 'high'},
            }
            await generate_chat_completion(
                request=fake_request,
                form_data=form_data_override,
                user=SimpleNamespace(id='owner', role='admin'),
                models={'my-gemini-alias': alias_model},
            )
            self.assertEqual(captured_payloads[1].get('params', {}).get('reasoning_effort'), 'high')
            self.assertNotEqual(captured_payloads[1].get('reasoning_effort'), 'minimal')


class TestUntouchedCallerGuards(unittest.IsolatedAsyncioTestCase):
    """Verify owner gates on untouched callers (actions, embeddings, audio realtime)."""

    async def test_actions_rejects_subscription_models(self):
        from open_webui.utils.actions import chat_action

        request = MagicMock()
        request.app.state.MODELS = {
            'subscription:claude:claude-3-7-sonnet': {
                'id': 'subscription:claude:claude-3-7-sonnet',
                'owned_by': 'subscription',
            }
        }
        request.state.direct = False
        user = SimpleNamespace(id='owner', role='user')

        with patch('open_webui.utils.actions.Functions.get_function_by_id', new=AsyncMock(return_value=SimpleNamespace(type='action', is_active=True))):
            with patch('open_webui.utils.actions.check_model_access', new=AsyncMock()):
                with self.assertRaises(HTTPException) as ctx:
                    await chat_action(request, 'some_action', {'model': 'subscription:claude:claude-3-7-sonnet'}, user)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn('Subscription provider connections do not support custom actions', ctx.exception.detail)

    async def test_embeddings_rejects_subscription_models(self):
        from open_webui.utils.embeddings import generate_embeddings

        request = MagicMock()
        request.app.state.MODELS = {
            'subscription:gemini:auto': {
                'id': 'subscription:gemini:auto',
                'owned_by': 'subscription',
            }
        }
        request.state.direct = False
        user = SimpleNamespace(id='owner', role='user')

        with patch('open_webui.utils.embeddings.check_model_access', new=AsyncMock()):
            with self.assertRaises(HTTPException) as ctx:
                await generate_embeddings(request, {'model': 'subscription:gemini:auto', 'input': 'test'}, user)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn('Subscription provider connections do not support embeddings', ctx.exception.detail)


class TestSubscriptionManagerMetadata(unittest.IsolatedAsyncioTestCase):
    """Test manager.py model enumeration and model_controls exposure."""

    async def test_manager_get_models_exposes_model_controls_metadata(self):
        from open_webui.subscriptions.manager import SubscriptionManager
        from open_webui.subscriptions.base import BaseSubscriptionAdapter
        from open_webui.subscriptions.config import ProviderConfigModel

        manager = SubscriptionManager()
        fake_adapter = MagicMock(spec=BaseSubscriptionAdapter)
        fake_adapter.provider_id = 'test_provider'
        fake_adapter.default_name = 'Test Provider'
        fake_adapter.is_installed.return_value = (True, '/bin/test')
        fake_adapter.execution_available = True
        fake_adapter.get_supported_models.return_value = [
            {
                'id': 'model-with-thinking',
                'name': 'Model With Thinking',
                'reasoning_efforts': [
                    {'value': 'minimal', 'name': 'Minimal'},
                    {'value': 'high', 'name': 'High'},
                    {'value': 'extra_high', 'name': 'Extra High'},
                ],
            },
            {
                'id': 'model-without-thinking',
                'name': 'Model Without Thinking',
            },
        ]
        manager._adapters = {'test_provider': fake_adapter}

        with patch.object(
            manager,
            'get_runtime_config',
            new=AsyncMock(
                return_value=(
                    True,
                    'owner-id',
                    180,
                    {'test_provider': ProviderConfigModel(enabled=True)},
                )
            ),
        ):
            models = await manager.get_models(user=SimpleNamespace(id='owner-id', role='admin'))

        self.assertEqual(len(models), 2)
        m_thinking = next(m for m in models if m['id'] == 'subscription:test_provider:model-with-thinking')
        m_no_thinking = next(m for m in models if m['id'] == 'subscription:test_provider:model-without-thinking')

        # Model with advertised reasoning_efforts must expose model_controls
        mc = m_thinking['info']['params']['model_controls']
        self.assertIn('reasoning_effort', mc)
        self.assertEqual(mc['reasoning_effort']['label'], 'Reasoning Effort')
        opts = mc['reasoning_effort']['options']
        self.assertEqual(len(opts), 3)
        self.assertEqual(opts['minimal']['label'], 'Minimal')
        self.assertEqual(opts['minimal']['params'], {'reasoning_effort': 'minimal'})
        self.assertEqual(opts['high']['label'], 'High')
        self.assertEqual(opts['high']['params'], {'reasoning_effort': 'high'})
        self.assertEqual(opts['extra_high']['label'], 'Extra High')
        self.assertEqual(opts['extra_high']['params'], {'reasoning_effort': 'extra_high'})

        # Model without advertised reasoning_efforts must NOT expose model_controls
        self.assertNotIn('model_controls', m_no_thinking.get('info', {}).get('params', {}))


if __name__ == '__main__':
    unittest.main()


class TestEffectiveSubscriptionParameterSecurity(unittest.TestCase):
    def test_nested_and_json_custom_features_are_rejected_before_normalization(self):
        import json
        from open_webui.subscriptions.security import validate_chat_request_security
        for feature in ('image_generation', 'memory', 'web_search'):
            for value in ({feature: True}, json.dumps({feature: True})):
                body = {'messages': [{'role': 'user', 'content': 'hello'}],
                    'params': {'custom_params': {'features': value}}}
                with self.subTest(feature=feature, value=value), self.assertRaises(HTTPException) as error:
                    validate_chat_request_security(body)
                self.assertEqual(error.exception.status_code, 400)
        for controls in ({'temperature': 0.25}, {'max_tokens': 128}, {'tools': '[{"type":"function"}]'}):
            with self.subTest(controls=controls), self.assertRaises(HTTPException):
                validate_chat_request_security({'messages': [{'role': 'user', 'content': 'hello'}],
                    'params': {'custom_params': controls}})

    def test_native_reasoning_menu_expansion_remains_supported(self):
        from open_webui.subscriptions.security import validate_chat_request_security
        validate_chat_request_security({'messages': [{'role': 'user', 'content': 'hello'}],
            'params': {'custom_params': {'reasoning_effort': 'extra_high'}}})


class TestNativeCatalogProductionCallers(unittest.IsolatedAsyncioTestCase):
    async def test_codex_models_and_status_use_discovery_not_static_catalog(self):
        from open_webui.subscriptions.manager import SubscriptionManager
        from open_webui.subscriptions.config import ProviderConfigModel
        manager = SubscriptionManager()
        discovered = [{'id':'native-offline-model','name':'Native Offline Model'}]
        adapter = SimpleNamespace(execution_available=True,default_name='Codex',
            is_installed=lambda _: (True,'/synthetic/codex'),
            get_supported_models=lambda _: [{'id':'stale-model','name':'Stale'}],
            discover_models=AsyncMock(return_value=discovered),
            check_auth_status=AsyncMock(return_value={'authenticated':True,'execution_available':True,'status':'authenticated'}))
        manager._adapters = {'codex':adapter}
        manager.get_runtime_config = AsyncMock(return_value=(True,'owner',180,{'codex':ProviderConfigModel(enabled=True)}))
        owner = SimpleNamespace(id='owner',role='admin')
        models = await manager.get_models(user=owner)
        status = await manager.get_providers_status(current_user=owner)
        self.assertEqual([model['id'] for model in models], ['subscription:codex:native-offline-model'])
        self.assertEqual(status[0]['models'], discovered)
        self.assertEqual(adapter.discover_models.await_count, 2)
