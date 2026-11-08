from __future__ import annotations

import copy
import logging
import time
from typing import Any
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from open_webui.subscriptions.adapters.claude import ClaudeCodeAdapter
from open_webui.subscriptions.adapters.codex_session import CodexSessionAdapter
from open_webui.subscriptions.adapters.gemini import GeminiAdapter
from open_webui.subscriptions.base import BaseSubscriptionAdapter
from open_webui.subscriptions.config import (
    DEFAULT_ENABLE_SUBSCRIPTIONS,
    DEFAULT_PROVIDERS_CONFIG,
    DEFAULT_SUBSCRIPTIONS_OWNER_ID,
    DEFAULT_SUBSCRIPTIONS_REQUEST_TIMEOUT,
    SUBSCRIPTIONS_CONFIG_KEYS,
    ProviderConfigModel,
    SubscriptionsConfigForm,
)
from open_webui.subscriptions.controls import (
    is_valid_reasoning_effort_shape,
    validate_subscription_chat_controls,
)
from open_webui.subscriptions.security import (
    check_subscription_access,
    sanitize_error_message,
    validate_chat_request_security,
    validate_owner_assignment,
)

log = logging.getLogger(__name__)

SAFE_SUBSCRIPTION_CAPABILITIES = {
    'builtin_tools': False,
    'tools': False,
    'terminal': False,
    'code_interpreter': False,
    'knowledge': False,
    'web_search': False,
    'skills': False,
    'vision': False,
    'file_upload': False,
    'citations': False,
    'memory': False,
    'usage': False,
    'file_context': False,
}


class SubscriptionManager:
    """Central orchestrator for subscription-based CLI model providers (Bring Your Own Subscription).

    Manages adapter registration, database-persisted configuration, model enumeration,
    and single-owner request authorization.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BaseSubscriptionAdapter] = {}
        self.register_adapter(ClaudeCodeAdapter())
        self.register_adapter(CodexSessionAdapter())
        self.register_adapter(GeminiAdapter())


    def register_adapter(self, adapter: BaseSubscriptionAdapter) -> None:
        self._adapters[adapter.provider_id] = adapter

    def get_adapter(self, provider_id: str) -> BaseSubscriptionAdapter | None:
        return self._adapters.get(provider_id)

    async def get_runtime_config(self) -> tuple[bool, str, int, dict[str, ProviderConfigModel]]:
        """Fetch runtime configuration from persistent database storage with env fallbacks."""
        from open_webui.models.config import Config

        db_configs = await Config.get_many(
            SUBSCRIPTIONS_CONFIG_KEYS['ENABLE_SUBSCRIPTIONS'],
            SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_OWNER_ID'],
            SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_REQUEST_TIMEOUT'],
            SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_PROVIDERS'],
        )

        enabled = db_configs.get(
            SUBSCRIPTIONS_CONFIG_KEYS['ENABLE_SUBSCRIPTIONS'],
            DEFAULT_ENABLE_SUBSCRIPTIONS,
        )
        owner_id = db_configs.get(
            SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_OWNER_ID'],
            DEFAULT_SUBSCRIPTIONS_OWNER_ID,
        ) or ''
        timeout = db_configs.get(
            SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_REQUEST_TIMEOUT'],
            DEFAULT_SUBSCRIPTIONS_REQUEST_TIMEOUT,
        )
        providers_raw = db_configs.get(
            SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_PROVIDERS'],
            DEFAULT_PROVIDERS_CONFIG,
        )

        providers: dict[str, ProviderConfigModel] = {}
        if isinstance(providers_raw, dict):
            for pid, pdata in providers_raw.items():
                if isinstance(pdata, dict):
                    providers[pid] = ProviderConfigModel(**pdata)
                elif isinstance(pdata, ProviderConfigModel):
                    providers[pid] = pdata

        # Ensure all registered adapters have an entry
        for pid in self._adapters:
            if pid not in providers:
                default_data = DEFAULT_PROVIDERS_CONFIG.get(pid, {})
                providers[pid] = ProviderConfigModel(**default_data)

        return bool(enabled), str(owner_id), int(timeout), providers

    async def get_config_dict(self) -> dict[str, Any]:
        """Return full configuration dictionary for settings management."""
        enabled, owner_id, timeout, providers = await self.get_runtime_config()
        return {
            'ENABLE_SUBSCRIPTIONS': enabled,
            'SUBSCRIPTIONS_OWNER_ID': owner_id,
            'SUBSCRIPTIONS_REQUEST_TIMEOUT': timeout,
            'SUBSCRIPTIONS_PROVIDERS': {pid: p.model_dump() for pid, p in providers.items()},
        }

    async def update_config(self, form_data: SubscriptionsConfigForm, current_user: Any = None) -> dict[str, Any]:
        """Save updated configuration to persistent storage enforcing single-user ownership atomically."""
        import json
        from open_webui.internal.db import get_async_db
        from open_webui.models.config import Config
        from sqlalchemy import cast, select, text, update, Text
        from sqlalchemy.exc import IntegrityError, OperationalError

        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Authentication required.',
            )

        caller_id = getattr(current_user, 'id', None)
        caller_role = getattr(current_user, 'role', None)
        if caller_role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Only administrators can configure subscription ownership.',
            )

        if not Config.PERSISTENT_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Subscription settings require persistent configuration for atomic account ownership. Enable persistent configuration before changing these settings.',
            )

        owner_key = SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_OWNER_ID']
        enable_key = SUBSCRIPTIONS_CONFIG_KEYS['ENABLE_SUBSCRIPTIONS']
        timeout_key = SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_REQUEST_TIMEOUT']
        providers_key = SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_PROVIDERS']

        dumped_providers = {pid: p.model_dump() for pid, p in form_data.SUBSCRIPTIONS_PROVIDERS.items()}

        async with get_async_db() as db:
            try:
                async with db.begin():
                    # Attempt SQLite BEGIN IMMEDIATE or Postgres with_for_update before reading
                    dialect_obj = getattr(getattr(db, 'bind', None), 'dialect', None)
                    dialect_name = getattr(dialect_obj, 'name', '') if dialect_obj else ''

                    if dialect_name == 'sqlite':
                        try:
                            await db.execute(text('BEGIN IMMEDIATE'))
                        except Exception:
                            pass

                    if dialect_name and dialect_name != 'sqlite':
                        res = await db.execute(select(Config).where(Config.key == owner_key).with_for_update())
                        row = res.scalar_one_or_none()
                    else:
                        row = await db.get(Config, owner_key)

                    if row is not None:
                        existing_owner = str(row.value).strip() if (row.value is not None) else ''
                        expected_raw = existing_owner
                    else:
                        # Respect effective environment owner default when DB row absent
                        existing_owner = str(DEFAULT_SUBSCRIPTIONS_OWNER_ID).strip()
                        expected_raw = None

                    validate_owner_assignment(
                        new_owner_id=form_data.SUBSCRIPTIONS_OWNER_ID,
                        current_user=current_user,
                        existing_owner_id=existing_owner,
                        enabled=form_data.ENABLE_SUBSCRIPTIONS,
                    )

                    now = int(time.time())
                    target_owner = existing_owner if existing_owner else form_data.SUBSCRIPTIONS_OWNER_ID

                    if row is not None:
                        # CAS conditional update verifying owner value has not changed concurrently
                        allowed_matches = [expected_raw, json.dumps(expected_raw)]
                        if not expected_raw:
                            allowed_matches.extend(['""', '', 'null', json.dumps(None), json.dumps("")])

                        stmt = (
                            update(Config)
                            .where(Config.key == owner_key)
                            .where(cast(Config.value, Text).in_(allowed_matches))
                            .values(value=target_owner, updated_at=now)
                        )
                        res = await db.execute(stmt)
                        if res.rowcount == 0:
                            await db.rollback()
                            raise HTTPException(
                                status_code=status.HTTP_409_CONFLICT,
                                detail='Subscription ownership was concurrently claimed by another administrator.',
                            )
                    else:
                        # Missing row case: insert and handle concurrent unique conflict
                        db.add(Config(key=owner_key, value=target_owner, updated_at=now))
                        try:
                            await db.flush()
                        except IntegrityError:
                            await db.rollback()
                            raise HTTPException(
                                status_code=status.HTTP_409_CONFLICT,
                                detail='Subscription ownership was concurrently claimed by another administrator.',
                            )

                    # Update remaining configuration keys atomically in same transaction
                    other_updates = {
                        enable_key: form_data.ENABLE_SUBSCRIPTIONS,
                        timeout_key: form_data.SUBSCRIPTIONS_REQUEST_TIMEOUT,
                        providers_key: dumped_providers,
                    }

                    for k, v in other_updates.items():
                        existing_entry = await db.get(Config, k)
                        if existing_entry is not None:
                            existing_entry.value = v
                            existing_entry.updated_at = now
                        else:
                            db.add(Config(key=k, value=v, updated_at=now))

                    await db.flush()
            except (IntegrityError, OperationalError) as exc:
                try:
                    await db.rollback()
                except Exception:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail='Subscription ownership was concurrently claimed by another administrator or database is locked.',
                ) from exc
            except HTTPException:
                try:
                    await db.rollback()
                except Exception:
                    pass
                raise
            except Exception:
                try:
                    await db.rollback()
                except Exception:
                    pass
                raise

        return await self.get_config_dict()

    async def get_providers_status(self, current_user: Any = None) -> list[dict[str, Any]]:
        """Check detection, executable paths, and auth status across all providers.

        Gated exclusively to the designated owner. Unclaimed/blank owner status checks are denied.
        """
        enabled, owner_id, timeout, providers = await self.get_runtime_config()

        if not owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Subscription provider status is unavailable before ownership is claimed.',
            )

        if current_user:
            user_id = getattr(current_user, 'id', None)
            if user_id != owner_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='Only the subscription account owner can inspect provider status.',
                )

        results: list[dict[str, Any]] = []

        for pid, adapter in self._adapters.items():
            cfg = providers.get(pid, ProviderConfigModel())
            installed, binary = adapter.is_installed(cfg)
            auth_info = await adapter.check_auth_status(cfg)
            exec_avail = auth_info.get('execution_available', False)
            reason = auth_info.get('reason', '')
            models = (await adapter.discover_models(cfg) if pid == 'codex' else adapter.get_supported_models(cfg)) if exec_avail else []

            results.append(
                {
                    'id': pid,
                    'name': cfg.name or adapter.default_name,
                    'enabled': cfg.enabled,
                    'installed': installed,
                    'binary_path': binary if installed else '',
                    'config_dir': cfg.config_dir,
                    'authenticated': auth_info.get('authenticated', False),
                    'execution_available': exec_avail,
                    'reason': reason,
                    'status': auth_info.get('status', 'unknown'),
                    'version': auth_info.get('version'),
                    'message': auth_info.get('message', ''),
                    'models': models,
                }
            )

        return results

    async def test_provider(
        self,
        provider_id: str,
        prompt: str = 'Hello',
        model_id: str | None = None,
        current_user: Any = None,
    ) -> dict[str, Any]:
        """Run an isolated test prompt through the specified provider adapter."""
        adapter = self.get_adapter(provider_id)
        if not adapter:
            raise HTTPException(status_code=404, detail=f'Unknown provider "{provider_id}".')

        enabled, owner_id, timeout, providers = await self.get_runtime_config()

        # Enforce owner authorization on test calls; denied unconditionally if owner is blank
        check_subscription_access(current_user, owner_id=owner_id)

        cfg = providers.get(provider_id, ProviderConfigModel())

        # Fail closed for disabled/unavailable providers
        if getattr(adapter, 'execution_available', True) is False:
            auth_info = await adapter.check_auth_status(cfg)
            if not auth_info.get('execution_available', False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=auth_info.get('reason', 'Provider execution is unavailable for safety reasons.'),
                )

        return await adapter.test_connection(cfg, prompt=prompt, model_id=model_id, timeout=30)

    async def get_models(self, user: Any = None) -> list[dict[str, Any]]:
        """Return base models for Open WebUI model discovery.

        If user is None (internal canonical warmup), returns canonical base model catalog.
        Disabled/unavailable providers have NO models.
        If user is provided, strictly gates discovery to the designated owner;
        blank owner always returns empty list.
        """
        enabled, owner_id, _, providers = await self.get_runtime_config()
        if not enabled:
            return []

        # If owner is blank, discovery is denied unconditionally, even for admin
        if not owner_id:
            return []

        # When queried in user context, gate strictly to the designated owner
        if user is not None:
            user_id = getattr(user, 'id', None)
            if user_id != owner_id:
                return []

        models: list[dict[str, Any]] = []
        now = int(time.time())

        for pid, adapter in self._adapters.items():
            # Disabled or safety-blocked providers have NO selectable models
            if getattr(adapter, 'execution_available', True) is False:
                continue


            cfg = providers.get(pid)
            if not cfg or not cfg.enabled:
                continue

            installed, _ = adapter.is_installed(cfg)
            if not installed:
                continue

            supported = await adapter.discover_models(cfg) if pid == 'codex' else adapter.get_supported_models(cfg)
            for m in supported:
                full_id = f'subscription:{pid}:{m["id"]}'
                model_entry = {
                    'id': full_id,
                    'name': m['name'],
                    'object': 'model',
                    'created': now,
                    'owned_by': 'subscription',
                    'subscription': {
                        'provider': pid,
                        'model_id': m['id'],
                        'connection_type': 'subscription',
                    },
                    'connection_type': 'subscription',
                    'tags': [{'name': 'subscription', 'color': 'emerald'}],
                    'info': {
                        'meta': {
                            'capabilities': copy.deepcopy(SAFE_SUBSCRIPTION_CAPABILITIES)
                        }
                    },
                }

                # Expose optional reasoning_efforts in model_controls schema
                # Scoped ONLY to the model with verified choices
                reasoning_efforts = m.get('reasoning_efforts')
                if isinstance(reasoning_efforts, list) and reasoning_efforts:
                    options: dict[str, Any] = {}
                    for opt in reasoning_efforts:
                        if isinstance(opt, dict) and 'value' in opt:
                            val = str(opt['value']).strip()
                            if val and is_valid_reasoning_effort_shape(val):
                                opt_name = str(opt.get('name') or val).strip()
                                options[val] = {
                                    'label': opt_name,
                                    'params': {'reasoning_effort': val},
                                }
                    if options:
                        if 'params' not in model_entry['info']:
                            model_entry['info']['params'] = {}
                        model_entry['info']['params']['model_controls'] = {
                            'reasoning_effort': {
                                'label': 'Reasoning Effort',
                                'options': options,
                            }
                        }

                models.append(model_entry)

        return models

    def parse_model_id(self, model_id: str) -> tuple[str, str]:
        """Parse composite model id 'subscription:<provider>:<model>' -> (provider, model)."""
        if not model_id.startswith('subscription:'):
            raise ValueError(f'Invalid subscription model ID: {model_id}')

        parts = model_id.split(':', 2)
        if len(parts) < 3:
            raise ValueError(f'Malformed subscription model ID: {model_id}')

        return parts[1], parts[2]

    async def generate_chat_completion(
        self,
        request: Request,
        form_data: dict[str, Any],
        user: Any,
    ) -> Response | StreamingResponse | dict[str, Any]:
        """Route chat completion request to the appropriate subscription provider adapter."""
        enabled, owner_id, timeout, providers = await self.get_runtime_config()
        if not enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Subscription provider connections are currently disabled.',
            )

        # 1. Enforce owner-only security gate; fail closed if owner blank
        check_subscription_access(user, owner_id=owner_id)

        # 2. Reject unsupported tools, function calls, features, terminal, skills, and attachments
        meta = getattr(request.state, 'metadata', None) if hasattr(request, 'state') else None
        validate_chat_request_security(form_data, metadata=meta, request=request)

        raw_model = form_data.get('model', '')
        try:
            provider_id, model_name = self.parse_model_id(raw_model)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        adapter = self.get_adapter(provider_id)
        if not adapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Subscription provider "{provider_id}" is not registered.',
            )

        # Fail closed for unavailable execution
        if getattr(adapter, 'execution_available', True) is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Subscription provider execution is unavailable for safety reasons.',
            )

        cfg = providers.get(provider_id)
        if not cfg or not cfg.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Subscription provider "{provider_id}" is disabled by administrator.',
            )

        stream = form_data.get('stream', False)
        catalog_before = adapter.get_supported_models(cfg) if provider_id == 'gemini' else None

        async def refresh_changed_catalog():
            if provider_id == 'gemini' and adapter.get_supported_models(cfg) != catalog_before:
                from open_webui.subscriptions.cache import invalidate_subscription_models_cache
                await invalidate_subscription_models_cache(request, user, preserve_active_models=True)

        # Pass a shallow copy so adapter does not mutate caller's form_data
        request_payload = dict(form_data)

        # Flatten nested params/options into request_payload so adapters reliably receive controls,
        # ensuring per-chat nested parameters take precedence over ambient/top-level defaults.
        for nested_key in ('options', 'params'):
            nested_dict = request_payload.get(nested_key)
            if isinstance(nested_dict, dict):
                for k, v in nested_dict.items():
                    if v is not None:
                        if k == 'reasoning_effort' or k not in request_payload or request_payload[k] is None:
                            request_payload[k] = v

        # Validate controls before constructing StreamingResponse so errors raise 400 immediately
        validate_subscription_chat_controls(request_payload, metadata=meta)

        if stream:
            stream_gen = adapter.chat_completion_stream(
                request_data=request_payload,
                model_id=model_name,
                config=cfg,
                timeout=timeout,
            )
            async def stream_with_catalog_refresh():
                try:
                    async for chunk in stream_gen:
                        yield chunk
                finally:
                    try:
                        await stream_gen.aclose()
                    finally:
                        await refresh_changed_catalog()

            return StreamingResponse(
                stream_with_catalog_refresh(),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                },
            )
        else:
            try:
                res = await adapter.chat_completion(
                    request_data=request_payload,
                    model_id=model_name,
                    config=cfg,
                    timeout=timeout,
                )
                return res
            except HTTPException:
                raise
            except Exception as e:
                err_msg = sanitize_error_message(str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=err_msg,
                )
            finally:
                await refresh_changed_catalog()



# Global singleton instance
subscription_manager = SubscriptionManager()
