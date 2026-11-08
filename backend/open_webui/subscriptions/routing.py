from __future__ import annotations

import logging
import random
from typing import Any
from fastapi import HTTPException, Request, status

from open_webui.models.config import Config
from open_webui.subscriptions.config import SUBSCRIPTIONS_CONFIG_KEYS

log = logging.getLogger(__name__)


def is_subscription_model(model: dict[str, Any] | None = None, model_id: str = '') -> bool:
    """Classify whether a model dictionary or ID resolves to a subscription model.

    Centralized classification helper that checks direct subscription identifiers,
    provider ownership tags, and custom model preset aliases that bind to a subscription
    base model. Avoids redundant DB queries or N+1 lookups across callers.
    """
    if model_id and model_id.startswith('subscription:'):
        return True

    if not isinstance(model, dict):
        return False

    mid = model.get('id') or ''
    if mid.startswith('subscription:'):
        return True

    if model.get('owned_by') == 'subscription':
        return True

    if model.get('connection_type') == 'subscription':
        return True

    if model.get('subscription'):
        return True

    base_id = model.get('base_model_id')
    if not base_id:
        info = model.get('info')
        if isinstance(info, dict):
            base_id = info.get('base_model_id')
            if not base_id and isinstance(info.get('meta'), dict):
                base_id = info['meta'].get('base_model_id')

    if base_id and str(base_id).startswith('subscription:'):
        return True

    return False


async def is_subscription_owner(user: Any) -> bool:
    """Check if the given user is the registered subscription account owner."""
    if not user:
        return False
    user_id = getattr(user, 'id', None)
    if not user_id:
        return False
    owner_id = await Config.get(SUBSCRIPTIONS_CONFIG_KEYS['SUBSCRIPTIONS_OWNER_ID'], '')
    return bool(owner_id and user_id == owner_id)


async def resolve_arena_model(
    request: Request,
    model: dict[str, Any],
    user: Any,
    form_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve an Arena model wrapper to an eligible concrete target model.

    Enforces single-owner subscription filtering BEFORE pre-DB checks, chat creation,
    file linking, and background tasks. If the filtered candidate pool is empty,
    fails closed immediately with HTTP 400 (never falls back to unfiltered re-randomization).
    Carries the resolved model selection into form_data and metadata.
    """
    if not isinstance(model, dict) or model.get('owned_by') != 'arena':
        target_id = (form_data.get('model') if form_data else None) or (model.get('id') if isinstance(model, dict) else '') or ''
        return model, target_id

    meta = (model.get('info', {}) or {}).get('meta', {})
    model_ids = meta.get('model_ids')
    filter_mode = meta.get('filter_mode')

    app_state = getattr(getattr(request, 'app', None), 'state', None)
    available_models = getattr(app_state, 'MODELS', {}) if app_state else {}

    if model_ids and filter_mode == 'exclude':
        candidates = [
            mid for mid, m in available_models.items()
            if m.get('owned_by') != 'arena' and mid not in model_ids
        ]
    elif isinstance(model_ids, list) and model_ids:
        candidates = [
            mid for mid in model_ids
            if mid in available_models and available_models[mid].get('owned_by') != 'arena'
        ]
    else:
        candidates = [
            mid for mid, m in available_models.items()
            if m.get('owned_by') != 'arena'
        ]

    # Enforce owner-only access for subscription models in the Arena pool
    sub_owner = await is_subscription_owner(user)
    if not sub_owner:
        candidates = [
            mid for mid in candidates
            if not is_subscription_model(available_models.get(mid), mid)
        ]

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No eligible models available in Arena.',
        )

    selected_id = random.choice(candidates)
    selected_model = available_models.get(selected_id, {'id': selected_id})

    if form_data is not None:
        form_data['model'] = selected_id
    if metadata is not None:
        metadata['selected_model_id'] = selected_id

    return selected_model, selected_id
