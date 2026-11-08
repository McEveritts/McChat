from __future__ import annotations

import asyncio
import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _init_child_environment():
    """Initializes stubs strictly inside the isolated child process for packages not installed in .venv."""
    if importlib.util.find_spec('PIL') is None:
        pil_mod = types.ModuleType('PIL')
        image_mod = types.ModuleType('PIL.Image')
        image_mod.open = MagicMock()
        pil_mod.Image = image_mod
        sys.modules['PIL'] = pil_mod
        sys.modules['PIL.Image'] = image_mod

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
        shared_mod = types.ModuleType('mcp.shared')
        auth_mod = types.ModuleType('mcp.shared.auth')
        auth_mod.OAuthClientMetadata = MagicMock
        auth_mod.OAuthMetadata = MagicMock
        mcp_mod.shared = shared_mod
        shared_mod.auth = auth_mod
        sys.modules['mcp'] = mcp_mod
        sys.modules['mcp.shared'] = shared_mod
        sys.modules['mcp.shared.auth'] = auth_mod

    if importlib.util.find_spec('validators') is None:
        validators_mod = types.ModuleType('validators')
        validators_mod.url = lambda u: True
        sys.modules['validators'] = validators_mod

    if importlib.util.find_spec('langchain_core') is None:
        lc_mod = types.ModuleType('langchain_core')
        lc_dl = types.ModuleType('langchain_core.document_loaders')
        lc_docs = types.ModuleType('langchain_core.documents')
        lc_dl.BaseLoader = MagicMock
        lc_docs.Document = MagicMock
        lc_mod.document_loaders = lc_dl
        lc_mod.documents = lc_docs
        sys.modules['langchain_core'] = lc_mod
        sys.modules['langchain_core.document_loaders'] = lc_dl
        sys.modules['langchain_core.documents'] = lc_docs

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


class MockUser:
    def __init__(self, role='user', user_id='user-123', email='test@example.com'):
        self.role = role
        self.id = user_id
        self.email = email


class _ChildChatIntegrationCase(unittest.IsolatedAsyncioTestCase):
    """Executes chat routing, system prompt preservation, and arena filtering inside isolated child process."""
    __test__ = False

    @classmethod
    def setUpClass(cls):
        from open_webui.internal.db import engine
        import open_webui.models.config
        open_webui.models.config.Config.__table__.create(bind=engine, checkfirst=True)

    async def test_system_prompt_preservation_for_base_subscription_model(self):
        """Custom system prompt on base subscription model is preserved without double-prompting (L2-06)."""
        from open_webui.subscriptions.manager import subscription_manager
        from open_webui.utils.chat import generate_chat_completion

        owner = MockUser(role='admin', user_id='owner-1')
        form_data = {
            'model': 'subscription:claude:sonnet',
            'messages': [{'role': 'user', 'content': 'Hello!'}],
            'stream': False,
        }

        mock_model = {
            'id': 'subscription:claude:sonnet',
            'name': 'Claude Sonnet',
            'owned_by': 'subscription',
            'info': {
                'params': {
                    'system': 'You are a specialized security assistant.',
                }
            },
        }

        request = MagicMock()
        request.app.state.MODELS = {'subscription:claude:sonnet': mock_model}
        request.state = MagicMock()
        request.state.direct = False
        request.state.bypass_system_prompt = False
        request.state.metadata = {}

        mock_db_model = MagicMock()
        mock_db_model.params.model_dump.return_value = {
            'system': 'You are a specialized security assistant.'
        }

        captured_payload = None

        async def mock_completion(request, form_data, user=None):
            nonlocal captured_payload
            captured_payload = copy.deepcopy(form_data)
            return {'id': 'chatcmpl-test', 'choices': [{'message': {'role': 'assistant', 'content': 'pong'}}]}

        with patch('open_webui.models.models.Models.get_model_by_id', new_callable=AsyncMock, return_value=mock_db_model):
            with patch('open_webui.utils.chat.check_model_access', new_callable=AsyncMock, return_value=True):
                with patch.object(subscription_manager, 'generate_chat_completion', side_effect=mock_completion):
                    res = await generate_chat_completion(
                        request=request,
                        form_data=form_data,
                        user=owner,
                        bypass_filter=False,
                    )

                    self.assertIsNotNone(captured_payload)
                    messages = captured_payload.get('messages', [])
                    system_msgs = [m for m in messages if m.get('role') == 'system']
                    self.assertEqual(len(system_msgs), 1)
                    self.assertEqual(system_msgs[0]['content'], 'You are a specialized security assistant.')
                    # Verify original form_data was preserved
                    self.assertEqual(form_data['model'], 'subscription:claude:sonnet')

    async def test_system_prompt_preservation_for_custom_alias_model(self):
        """Custom model alias pointing to subscription base model preserves alias prompt."""
        from open_webui.subscriptions.manager import subscription_manager
        from open_webui.utils.chat import generate_chat_completion

        owner = MockUser(role='admin', user_id='owner-1')
        form_data = {
            'model': 'custom-coding-assistant',
            'messages': [{'role': 'user', 'content': 'Write code.'}],
            'stream': False,
        }

        mock_alias_model = {
            'id': 'custom-coding-assistant',
            'name': 'Custom Coding Assistant',
            'base_model_id': 'subscription:claude:sonnet',
            'owned_by': 'custom',
            'info': {
                'params': {
                    'system': 'You write safe and concise Python code.',
                }
            },
        }

        request = MagicMock()
        request.app.state.MODELS = {'custom-coding-assistant': mock_alias_model}
        request.state = MagicMock()
        request.state.direct = False
        request.state.bypass_system_prompt = False
        request.state.metadata = {}

        mock_db_model = MagicMock()
        mock_db_model.params.model_dump.return_value = {
            'system': 'You write safe and concise Python code.'
        }

        captured_payload = None

        async def mock_completion(request, form_data, user=None):
            nonlocal captured_payload
            captured_payload = copy.deepcopy(form_data)
            return {'id': 'chatcmpl-test', 'choices': [{'message': {'role': 'assistant', 'content': 'code'}}]}

        with patch('open_webui.models.models.Models.get_model_by_id', new_callable=AsyncMock, return_value=mock_db_model):
            with patch('open_webui.utils.chat.check_model_access', new_callable=AsyncMock, return_value=True):
                with patch.object(subscription_manager, 'generate_chat_completion', side_effect=mock_completion):
                    res = await generate_chat_completion(
                        request=request,
                        form_data=form_data,
                        user=owner,
                        bypass_filter=False,
                    )

                    self.assertIsNotNone(captured_payload)
                    # Resolved base model id sent to subscription_manager
                    self.assertEqual(captured_payload['model'], 'subscription:claude:sonnet')
                    messages = captured_payload.get('messages', [])
                    system_msgs = [m for m in messages if m.get('role') == 'system']
                    self.assertEqual(len(system_msgs), 1)
                    self.assertEqual(system_msgs[0]['content'], 'You write safe and concise Python code.')

    async def test_arena_filtering_excludes_subscription_models_for_non_owners(self):
        """Arena model resolution must exclude subscription models if requester is not the owner."""
        from open_webui.utils.chat import generate_chat_completion

        other_user = MockUser(role='user', user_id='user-non-owner')
        form_data = {
            'model': 'arena-model',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'stream': False,
        }

        mock_arena_model = {
            'id': 'arena-model',
            'owned_by': 'arena',
            'info': {
                'meta': {
                    'model_ids': ['model-a', 'subscription:claude:sonnet'],
                    'filter_mode': 'include',
                }
            },
        }

        mock_model_a = {
            'id': 'model-a',
            'owned_by': 'openai',
            'info': {},
        }

        request = MagicMock()
        request.app.state.MODELS = {
            'arena-model': mock_arena_model,
            'model-a': mock_model_a,
            'subscription:claude:sonnet': {'id': 'subscription:claude:sonnet', 'owned_by': 'subscription'},
        }
        request.state = MagicMock()
        request.state.direct = False
        request.state.bypass_system_prompt = False
        request.state.metadata = {}

        with patch('open_webui.models.config.Config.get', new_callable=AsyncMock, return_value='owner-admin'):
            with patch('open_webui.utils.chat.check_model_access', new_callable=AsyncMock, return_value=True):
                with patch('open_webui.utils.chat.generate_openai_chat_completion', new_callable=AsyncMock, return_value={'id': 'test'}):
                    res = await generate_chat_completion(
                        request=request,
                        form_data=form_data,
                        user=other_user,
                        bypass_filter=False,
                    )

                    # Only model-a could have been selected since subscription:claude:sonnet was filtered out
                    self.assertEqual(form_data['model'], 'model-a')


class TestSubscriptionChatIntegration(unittest.TestCase):
    """Parent test harness: executes heavyweight chat integration tests in fully isolated child processes.

    Zero module-level sys.modules or os.environ pollution during collection or execution.
    """

    def _run_in_isolated_process(self, method_name: str):
        child_env = os.environ.copy()
        child_env['_ISOLATED_SUBSCRIPTION_CHILD'] = '1'
        proc = subprocess.run(
            [sys.executable, __file__, method_name],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            self.fail(
                f"Isolated test {method_name} failed with code {proc.returncode}:\n"
                f"--- STDOUT ---\n{proc.stdout}\n"
                f"--- STDERR ---\n{proc.stderr}"
            )

    def test_system_prompt_preservation_for_base_subscription_model(self):
        self._run_in_isolated_process('test_system_prompt_preservation_for_base_subscription_model')

    def test_system_prompt_preservation_for_custom_alias_model(self):
        self._run_in_isolated_process('test_system_prompt_preservation_for_custom_alias_model')

    def test_arena_filtering_excludes_subscription_models_for_non_owners(self):
        self._run_in_isolated_process('test_arena_filtering_excludes_subscription_models_for_non_owners')


def _child_entrypoint(method_name: str):
    _init_child_environment()
    test_base = os.path.expanduser('~/.tmp/mcchat-subscriptions')
    os.makedirs(test_base, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=test_base, prefix='test_subs_child_') as tmp_dir:
        os.environ['USE_SLIM_DOCKER'] = 'true'
        os.environ['WEBUI_SECRET_KEY'] = 'test-secret-key-12345678901234567890'
        os.environ['ENABLE_DB_MIGRATIONS'] = 'false'
        os.environ['DATA_DIR'] = tmp_dir
        os.environ['DATABASE_URL'] = f'sqlite:///{tmp_dir}/test_webui.db'

        suite = unittest.TestSuite()
        suite.addTest(_ChildChatIntegrationCase(method_name))
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
        res = runner.run(suite)
        if not res.wasSuccessful():
            sys.exit(1)
        sys.exit(0)


if __name__ == '__main__':
    if os.environ.get('_ISOLATED_SUBSCRIPTION_CHILD') == '1':
        method = sys.argv[1] if len(sys.argv) > 1 else 'test_system_prompt_preservation_for_base_subscription_model'
        _child_entrypoint(method)
    else:
        unittest.main()
