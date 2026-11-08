from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status

from open_webui.subscriptions.config import ProviderTestRequest, SubscriptionsConfigForm
from open_webui.subscriptions.manager import subscription_manager
from open_webui.utils.auth import get_admin_user, get_verified_user

log = logging.getLogger(__name__)

router = APIRouter()


@router.get('/config', response_model=dict)
async def get_subscriptions_config(
    request: Request,
    user=Depends(get_admin_user),
):
    """Retrieve subscription provider configuration (Owner only; fresh claim allowed for admin)."""
    cfg = await subscription_manager.get_config_dict()
    owner_id = cfg.get('SUBSCRIPTIONS_OWNER_ID', '')
    user_id = getattr(user, 'id', None)

    # If already claimed, non-owner admins receive a redacted read-only stub
    if owner_id and user_id != owner_id:
        return {
            'ENABLE_SUBSCRIPTIONS': cfg.get('ENABLE_SUBSCRIPTIONS', False),
            'SUBSCRIPTIONS_OWNER_ID': owner_id,
            'SUBSCRIPTIONS_PROVIDERS': {},
            'is_owner': False,
            'current_user_id': user_id,
        }

    # If unclaimed (fresh setup), redact private paths so non-owner cannot read them before claim
    if not owner_id:
        providers = cfg.get('SUBSCRIPTIONS_PROVIDERS', {})
        for pdata in providers.values():
            if isinstance(pdata, dict):
                pdata['binary_path'] = ''
                pdata['config_dir'] = ''

    cfg['is_owner'] = True
    cfg['current_user_id'] = user_id
    return cfg


from open_webui.subscriptions.cache import invalidate_subscription_models_cache as _invalidate_models_cache


@router.post('/config/update', response_model=dict)
async def update_subscriptions_config(
    request: Request,
    form_data: SubscriptionsConfigForm,
    user=Depends(get_admin_user),
):
    """Update subscription provider configuration enforcing single-user ownership atomically."""
    result = await subscription_manager.update_config(form_data, current_user=user)

    await _invalidate_models_cache(
        request,
        user,
        data={
            'provider': 'subscriptions',
            'enabled': form_data.ENABLE_SUBSCRIPTIONS,
            'owner_id': form_data.SUBSCRIPTIONS_OWNER_ID,
            'providers_count': len(form_data.SUBSCRIPTIONS_PROVIDERS),
        },
    )

    result['current_user_id'] = getattr(user, 'id', None)
    return result


@router.get('/providers', response_model=list[dict])
async def get_subscription_providers(
    request: Request,
    user=Depends(get_admin_user),
):
    """Inspect installation and auth status of supported subscription CLI providers (Owner only)."""
    return await subscription_manager.get_providers_status(current_user=user)


@router.post('/providers/{provider_id}/test', response_model=dict)
async def test_subscription_provider(
    request: Request,
    provider_id: str,
    form_data: ProviderTestRequest,
    user=Depends(get_admin_user),
):
    """Run an isolated test completion against the specified provider (Owner only)."""
    result = await subscription_manager.test_provider(
        provider_id=provider_id,
        prompt=form_data.prompt,
        model_id=form_data.model,
        current_user=user,
    )
    if result.get('success'):
        await _invalidate_models_cache(request, user)
    return result


from pydantic import BaseModel, Field

from open_webui.subscriptions.antigravity_auth import antigravity_auth_controller


class ProviderAuthCallbackForm(BaseModel):
    flow_id: str = Field(min_length=1, max_length=128)
    callback_url: str = Field(min_length=1, max_length=16384)


class ProviderAuthCancelForm(BaseModel):
    flow_id: str = Field(min_length=1, max_length=128)


async def _require_owner(user: Any) -> str:
    """Verify that the caller is the designated subscription owner."""
    cfg = await subscription_manager.get_config_dict()
    owner_id = cfg.get('SUBSCRIPTIONS_OWNER_ID', '')
    user_id = getattr(user, 'id', None)
    if not owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Subscription ownership must be claimed before managing authentication.',
        )
    if user_id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Only the designated subscription owner can perform authentication actions.',
        )
    return owner_id


@router.get('/models', response_model=list[dict])
async def get_subscription_models(
    request: Request,
    user=Depends(get_verified_user),
):
    """Return available subscription models for the current user, enforcing owner-only security."""
    return await subscription_manager.get_models(user=user)


@router.post('/providers/{provider_id}/signin', response_model=dict)
async def start_provider_signin(
    provider_id: str,
    user=Depends(get_admin_user),
):
    """Start an explicit browser sign-in flow for supported subscription providers (Owner only)."""
    owner_id = await _require_owner(user)
    if provider_id != 'gemini':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Interactive in-browser sign-in is not supported for provider "{provider_id}". Log in from the host terminal instead.',
        )

    _, _, _, providers = await subscription_manager.get_runtime_config()
    cfg = providers.get('gemini')
    config_dir = cfg.config_dir if cfg else ''
    adapter = subscription_manager.get_adapter('gemini')
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Google Gemini subscription adapter not registered.',
        )

    installed, resolved_binary = adapter.is_installed(cfg)
    if not installed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Google Antigravity ACP runtime not found: {resolved_binary}',
        )

    try:
        return await antigravity_auth_controller.start_sign_in(
            owner_user_id=owner_id,
            binary_path=resolved_binary,
            config_dir=config_dir,
        )
    except Exception as e:
        log.warning("Failed to initiate Google sign-in: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate Google sign-in.",
        )


@router.get('/providers/{provider_id}/auth', response_model=dict)
async def get_provider_auth_status(
    provider_id: str,
    user=Depends(get_admin_user),
):
    """Poll the active authentication flow state for a provider (Owner only)."""
    owner_id = await _require_owner(user)
    if provider_id != 'gemini':
        if provider_id in ('claude', 'codex'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Interactive in-browser authentication polling is not supported for provider "{provider_id}".',
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Subscription provider "{provider_id}" not found.',
        )

    _, _, _, providers = await subscription_manager.get_runtime_config()
    cfg = providers.get('gemini')
    config_dir = cfg.config_dir if cfg else ''
    adapter = subscription_manager.get_adapter('gemini')
    resolved_binary = ''
    if adapter:
        installed, res = adapter.is_installed(cfg)
        if installed:
            resolved_binary = res

    return antigravity_auth_controller.get_flow_snapshot(
        owner_id,
        binary_path=resolved_binary,
        config_dir=config_dir,
    )


@router.post('/providers/{provider_id}/callback', response_model=dict)
async def complete_provider_signin(
    request: Request,
    provider_id: str,
    form_data: ProviderAuthCallbackForm,
    user=Depends(get_admin_user),
):
    """Relay the Google OAuth redirect URL to complete the sign-in flow (Owner only)."""
    owner_id = await _require_owner(user)
    if provider_id != 'gemini':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Callback completion is not supported for provider "{provider_id}".',
        )

    try:
        res = await antigravity_auth_controller.complete_sign_in(
            owner_user_id=owner_id,
            flow_id=form_data.flow_id,
            callback_url=form_data.callback_url,
        )
        if res.get('authenticated') or res.get('phase') == 'succeeded':
            await _invalidate_models_cache(request, user)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log.warning("Failed to complete Google sign-in: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete Google sign-in.",
        )


@router.post('/providers/{provider_id}/cancel', response_model=dict)
async def cancel_provider_signin(
    provider_id: str,
    form_data: ProviderAuthCancelForm,
    user=Depends(get_admin_user),
):
    """Cancel an in-progress authentication flow (Owner only)."""
    owner_id = await _require_owner(user)
    if provider_id != 'gemini':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Cancel is not supported for provider "{provider_id}".',
        )

    try:
        return await antigravity_auth_controller.cancel_sign_in(
            owner_user_id=owner_id,
            flow_id=form_data.flow_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post('/providers/{provider_id}/logout', response_model=dict)
async def logout_provider(
    request: Request,
    provider_id: str,
    user=Depends(get_admin_user),
):
    """Sign out from the specified subscription provider and clear saved credentials (Owner only)."""
    owner_id = await _require_owner(user)
    if provider_id != 'gemini':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Sign-out via web interface is not supported for provider "{provider_id}". Log out from the host terminal instead.',
        )

    _, _, _, providers = await subscription_manager.get_runtime_config()
    cfg = providers.get('gemini')
    config_dir = cfg.config_dir if cfg else ''
    adapter = subscription_manager.get_adapter('gemini')
    resolved_binary = ''
    if adapter:
        installed, res = adapter.is_installed(cfg)
        if installed:
            resolved_binary = res

    res = await antigravity_auth_controller.sign_out(
        owner_user_id=owner_id,
        binary_path=resolved_binary,
        config_dir=config_dir,
    )
    await _invalidate_models_cache(request, user)
    return res
