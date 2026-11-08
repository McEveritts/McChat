from __future__ import annotations

import logging
from typing import Any
from fastapi import Request

log = logging.getLogger(__name__)

async def invalidate_subscription_models_cache(request: Request, user: Any, data: dict[str, Any] | None = None, *, preserve_active_models: bool = False) -> None:
    """Clear shared in-memory and Redis model caches and publish invalidation event across workers."""
    try:
        from open_webui.utils.models import get_all_models, BASE_MODELS_CACHE_KEY
        if hasattr(get_all_models, 'cache') and hasattr(get_all_models.cache, 'clear'):
            await get_all_models.cache.clear()
        redis = getattr(request.app.state, 'redis', None)
        if redis is not None:
            await redis.delete(BASE_MODELS_CACHE_KEY)
        request.app.state.BASE_MODELS = []
        if not preserve_active_models:
            models = getattr(request.app.state, 'MODELS', None)
            if hasattr(models, 'clear'):
                models.clear()
            else:
                request.app.state.MODELS = {}
    except Exception as e:
        log.debug('Failed to clear models cache: %s', type(e).__name__)

    try:
        from open_webui.events import EVENTS, publish_event

        await publish_event(
            request,
            EVENTS.MODEL_PROVIDER_CONFIG_UPDATED,
            actor=user,
            subject_id='subscriptions',
            subject_type='model.provider_config',
            data=data or {
                'provider': 'subscriptions',
                'action': 'state_change',
            },
        )
    except Exception as e:
        log.debug('Skipping event publish: %s', type(e).__name__)


    try:
        from open_webui.socket.main import emit_to_users
        user_id = getattr(user, 'id', None)
        if user_id:
            await emit_to_users('events', {'data': {'type': 'models:refresh', 'data': {}}}, [user_id])
    except Exception as exc:
        log.debug('Skipping owner model refresh: %s', type(exc).__name__)
