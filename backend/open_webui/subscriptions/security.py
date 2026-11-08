from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Any
from fastapi import HTTPException, status

log = logging.getLogger(__name__)

# Disallowed characters in executable binary paths and config directories
DANGEROUS_SHELL_CHARS = re.compile(r'[;&|`$()<>\n\r\x00]')

# Disallowed binary names that must never be executed via provider runner
FORBIDDEN_BINARIES = {
    'sh', 'bash', 'zsh', 'dash', 'csh', 'tcsh', 'ksh',
    'sudo', 'su', 'doas',
    'rm', 'dd', 'mkfs', 'shutdown', 'reboot', 'kill', 'pkill',
    'curl', 'wget', 'nc', 'netcat', 'socat', 'ncat',
    'python', 'python3', 'node', 'perl', 'ruby', 'lua', 'php',
    'eval', 'exec', 'systemctl', 'service',
}

ALLOWED_PROVIDER_NAMES = {'claude', 'codex', 'gemini'}
ALLOWED_MESSAGE_ROLES = {'system', 'user', 'assistant'}
MAX_MESSAGE_COUNT = 500
MAX_MESSAGE_CONTENT_LENGTH = 1_000_000


def _matches_provider_binary(name: str, provider_id: str) -> bool:
    name = name.lower()
    if provider_id == 'gemini':
        # Only official Antigravity / agy ACP runtime binaries are permitted
        if name in {'agy_acp_server', 'agy_acp_server.par', 'agy_acp_server.exe', 'agy', 'antigravity'}:
            return True
        if name.startswith('agy_acp_server') or name.startswith('agy-') or name.startswith('agy_') or name.startswith('antigravity'):
            return True
        return False
    return name == provider_id or name.startswith(f'{provider_id}_') or name.startswith(f'{provider_id}-')


def validate_binary_path(binary_path: str, provider_id: str) -> str:
    """Validate a binary path for a subscription provider.

    Ensures the path does not contain shell injection characters, is not a forbidden
    system executable, and strictly matches the provider's known executable name.

    If an absolute path is provided, it must exist, be executable, and have the provider's
    known binary name as its basename.
    """
    if not binary_path or not binary_path.strip():
        if provider_id == 'gemini':
            launcher = '/home/betheenvy/.local/libexec/t3-antigravity/agy_acp_server.par'
            if os.path.isfile(launcher) and os.access(launcher, os.X_OK):
                return launcher
            return 'agy_acp_server.par'
        # Fall back to default provider binary name
        return provider_id

    cleaned = binary_path.strip()

    if DANGEROUS_SHELL_CHARS.search(cleaned):
        raise ValueError('Binary path contains invalid or dangerous shell characters.')

    basename = os.path.basename(cleaned).lower()
    if basename in FORBIDDEN_BINARIES:
        raise ValueError(f'Execution of system utility "{basename}" is strictly forbidden.')

    # The executable must match the designated provider harness binary name
    if not _matches_provider_binary(basename, provider_id):
        raise ValueError(
            f'Binary name "{basename}" does not match the required executable "{provider_id}" for this provider.'
        )

    # If an absolute or relative path is provided, it must exist, be executable, and match provider_id
    if os.path.isabs(cleaned) or os.path.sep in cleaned:
        resolved = os.path.realpath(os.path.expanduser(cleaned))
        if not os.path.isfile(resolved):
            raise ValueError(f'Specified binary file does not exist: {cleaned}')
        if not os.access(resolved, os.X_OK):
            raise ValueError(f'Specified binary file is not executable: {cleaned}')
        res_base = os.path.basename(resolved).lower()
        if not _matches_provider_binary(res_base, provider_id):
            raise ValueError(
                f'Resolved binary "{resolved}" does not have expected basename "{provider_id}".'
            )
        return resolved

    # Otherwise, verify standard PATH lookup resolves to an executable matching provider_id
    resolved = shutil.which(cleaned)
    if resolved:
        res_basename = os.path.basename(resolved).lower()
        if res_basename in FORBIDDEN_BINARIES or not _matches_provider_binary(res_basename, provider_id):
            raise ValueError(f'Resolved executable "{res_basename}" is not "{provider_id}".')
        if not os.access(resolved, os.X_OK):
            raise ValueError(f'Resolved executable "{resolved}" is not executable.')
        return resolved

    return cleaned


def validate_config_dir(config_dir: str) -> str:
    """Validate a custom configuration directory path.

    Prevents shell injection and returns an expanded, resolved absolute path.
    """
    if not config_dir or not config_dir.strip():
        return ''

    cleaned = config_dir.strip()
    if DANGEROUS_SHELL_CHARS.search(cleaned):
        raise ValueError('Configuration directory contains invalid or dangerous shell characters.')

    resolved = os.path.realpath(os.path.expanduser(cleaned))
    return resolved


def sanitize_error_message(error: str) -> str:
    """Sanitize error messages before exposing them to users or API clients.

    Redacts user home directory paths, API tokens, session credentials, and internal stack traces.
    """
    if not error:
        return 'An unknown error occurred during subscription execution.'

    text = str(error)

    # Redact home directory paths: /home/<username> or /mnt/.../username
    text = re.sub(r'/(?:home|mnt/[^/]+)/[a-zA-Z0-9_\-]+', '~', text)

    # Redact potential API keys (sk-..., ant-..., bearer tokens)
    text = re.sub(r'sk-[a-zA-Z0-9_-]{20,}', '[REDACTED_API_KEY]', text)
    text = re.sub(r'ant-[a-zA-Z0-9_-]{20,}', '[REDACTED_AUTH_TOKEN]', text)
    text = re.sub(r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'Bearer [REDACTED_TOKEN]', text, flags=re.IGNORECASE)

    # Clean up excess whitespace
    return text.strip()


def _collect_metadata_objects(form_data: dict, metadata: dict | None = None, request: Any = None) -> list[dict]:
    """Collect all direct and recursive metadata dictionaries from request and payload."""
    metas: list[dict] = []

    def _add(item: Any):
        if isinstance(item, dict) and item not in metas:
            metas.append(item)
            # Check for nested user_message metadata
            um = item.get('user_message')
            if isinstance(um, dict):
                _add(um)
                _add(um.get('meta'))
                _add(um.get('metadata'))

    if metadata is not None:
        _add(metadata)
    _add(form_data.get('metadata'))

    if request is not None and hasattr(request, 'state'):
        state_meta = getattr(request.state, 'metadata', None)
        if isinstance(state_meta, dict):
            _add(state_meta)

    um = form_data.get('user_message') or form_data.get('parent_message')
    if isinstance(um, dict):
        _add(um)
        _add(um.get('meta'))
        _add(um.get('metadata'))

    return metas


def validate_chat_request_security(
    form_data: dict,
    metadata: dict | None = None,
    request: Any = None,
    require_complete_messages: bool = True,
) -> None:
    """Enforce strict security constraints for subscription-provider chat requests.

    Subscription models run on the host CLI harness without autonomous tool sandboxing.
    Therefore, host file execution, external tool calls, function calling, terminal sessions,
    skill integrations, and multimodal attachments must be rejected explicitly with clear 400
    errors before and after preprocessing. Direct and recursive metadata are inspected separately.

    When require_complete_messages is False (early phased validation on confirmed saved chat paths),
    messages may be empty or system-only while user_message is validated for content, roles, and safety
    prior to database turn insertion and middleware reconstruction.
    """
    params = form_data.get('params') or {}
    all_metas = _collect_metadata_objects(form_data, metadata=metadata, request=request)
    from open_webui.subscriptions.controls import iter_subscription_parameter_sources
    parameter_sources = list(iter_subscription_parameter_sources(form_data, metadata))
    for meta in tuple(all_metas):
        parameter_sources.extend(iter_subscription_parameter_sources(meta))
    all_metas.extend(parameter_sources)

    # 1. Reject tools, tool_ids, and tool_servers across top-level, params, and all metadata
    tool_keys = ('tools', 'tool_ids', 'tool_servers', 'functions', 'function_call', 'tool_choice')
    for key in tool_keys:
        if form_data.get(key) or (isinstance(params, dict) and params.get(key)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    'Subscription provider connections do not support autonomous host tool execution '
                    'or function calling. Please disable tools or use standard API connections.'
                ),
            )
        for meta in all_metas:
            if meta.get(key):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        'Subscription provider connections do not support autonomous host tool execution '
                        'or function calling in metadata. Please disable tools or use standard API connections.'
                    ),
                )

    # 2. Reject terminal_id across form_data, params, request.state, and metadata
    if form_data.get('terminal_id') or (isinstance(params, dict) and params.get('terminal_id')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Subscription provider connections do not support terminal integrations.',
        )
    if request is not None and hasattr(request, 'state') and getattr(request.state, 'terminal_id', None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Subscription provider connections do not support terminal integrations.',
        )
    for meta in all_metas:
        if meta.get('terminal_id'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Subscription provider connections do not support terminal integrations.',
            )

    # 3. Reject skill_ids and skills across form_data, params, request.state, and metadata
    skill_keys = ('skill_ids', 'skills', 'skill_id')
    for key in skill_keys:
        if form_data.get(key) or (isinstance(params, dict) and params.get(key)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Subscription provider connections do not support skill integrations.',
            )
        if request is not None and hasattr(request, 'state') and getattr(request.state, key, None):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Subscription provider connections do not support skill integrations.',
            )
        for meta in all_metas:
            if meta.get(key):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Subscription provider connections do not support skill integrations.',
                )

    # 4. Reject file attachments across top-level form_data, params, and all metadata
    has_files = bool(form_data.get('files') or (isinstance(params, dict) and params.get('files')))
    if not has_files:
        for meta in all_metas:
            if meta.get('files'):
                has_files = True
                break
    if has_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                'Subscription provider connections do not support file attachments on the host system. '
                'Please provide inline text or use standard API connections.'
            ),
        )

    # 5. Reject active features that trigger host/external execution
    unsupported_feature_names = {
        'web_search',
        'code_interpreter',
        'terminal',
        'image_generation',
        'tools',
        'memory',
    }
    feature_dicts = []
    if isinstance(form_data.get('features'), dict):
        feature_dicts.append(form_data['features'])
    if isinstance(params, dict) and isinstance(params.get('features'), dict):
        feature_dicts.append(params['features'])
    for meta in all_metas:
        if isinstance(meta.get('features'), dict):
            feature_dicts.append(meta['features'])

    for f_dict in feature_dicts:
        violating = [feat for feat, val in f_dict.items() if val and feat in unsupported_feature_names]
        if violating:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f'Subscription provider connections do not support feature(s): {", ".join(sorted(violating))}. '
                    'Please disable these features or use standard API connections.'
                ),
            )

    # 6. Reject unsupported nondefault chat controls in top-level, params, and all metadata
    from open_webui.subscriptions.controls import validate_subscription_chat_controls
    validate_subscription_chat_controls(form_data, metadata=metadata)


    # 7. Deep inspection of user_message if provided in form_data or metadata
    user_msg = (
        (metadata.get('user_message') if isinstance(metadata, dict) else None)
        or form_data.get('user_message')
        or form_data.get('parent_message')
    )
    if user_msg is not None:
        if not isinstance(user_msg, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='user_message must be an object.',
            )
        um_role = user_msg.get('role', 'user')
        if not um_role or um_role not in ALLOWED_MESSAGE_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f'user_message has unsupported role "{um_role}". '
                    'Only "system", "user", and "assistant" roles are supported.'
                ),
            )
        if user_msg.get('tool_calls') or user_msg.get('function_call'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='user_message contains tool_calls or function_call, which are not supported.',
            )
        if user_msg.get('files'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='user_message contains file attachments, which are not supported.',
            )
        um_content = user_msg.get('content')
        if um_content is not None:
            if isinstance(um_content, str):
                if len(um_content) > MAX_MESSAGE_CONTENT_LENGTH:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail='user_message exceeds maximum content length limit.',
                    )
            elif isinstance(um_content, list):
                for b_idx, block in enumerate(um_content):
                    if not isinstance(block, dict):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'user_message block {b_idx} is malformed.',
                        )
                    b_type = block.get('type')
                    if b_type in ('image_url', 'image', 'audio', 'file', 'input_audio') or any(k in block for k in ('image_url', 'audio', 'file', 'input_audio')):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f'user_message block {b_idx} contains multimodal attachments. '
                                'Subscription providers support plain text conversations only.'
                            ),
                        )
                    if b_type != 'text':
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f'user_message block {b_idx} contains unsupported content type "{b_type}". '
                                'Subscription providers support plain text conversations only.'
                            ),
                        )
                    text_val = block.get('text', '')
                    if not isinstance(text_val, str):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'user_message block {b_idx} text must be a string.',
                        )
                    if len(text_val) > MAX_MESSAGE_CONTENT_LENGTH:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'user_message block {b_idx} exceeds maximum content length limit.',
                        )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='user_message contains invalid content format (must be string or text blocks).',
                )

    # 8. Deep inspection of message contents: role bounds, tool_calls, empty prompts, and multimodal blocks
    messages = form_data.get('messages')
    if require_complete_messages:
        if messages is None or not isinstance(messages, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Chat completion request must contain a "messages" array.',
            )
        if len(messages) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Chat completion request messages array cannot be empty.',
            )
    else:
        if messages is not None and not isinstance(messages, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Chat completion request "messages" must be an array when provided.',
            )

    has_user_prompt = False
    if isinstance(messages, list):
        if len(messages) > MAX_MESSAGE_COUNT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Messages array exceeds maximum supported length ({MAX_MESSAGE_COUNT}).',
            )

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Message at index {idx} is malformed (must be an object).',
                )

            role = msg.get('role')
            if not role or role not in ALLOWED_MESSAGE_ROLES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f'Message at index {idx} has unsupported role "{role}". '
                        'Only "system", "user", and "assistant" roles are supported.'
                    ),
                )

            if msg.get('tool_calls') or msg.get('function_call'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f'Message at index {idx} contains tool_calls or function_call. '
                        'Function calling is not supported in subscription provider chats.'
                    ),
                )

            if msg.get('files'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Message at index {idx} contains file attachments, which are not supported.',
                )

            content = msg.get('content')
            if content is None:
                if role == 'user':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'Message at index {idx} has missing content.',
                    )
                continue

            if isinstance(content, str):
                if len(content) > MAX_MESSAGE_CONTENT_LENGTH:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'Message at index {idx} exceeds maximum content length limit.',
                    )
                if role == 'user' and content.strip():
                    has_user_prompt = True
            elif isinstance(content, list):
                for b_idx, block in enumerate(content):
                    if not isinstance(block, dict):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Message {idx} block {b_idx} is malformed.',
                        )
                    b_type = block.get('type')
                    if b_type != 'text':
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f'Message {idx} block {b_idx} contains unsupported content type "{b_type}". '
                                'Subscription providers support plain text conversations only.'
                            ),
                        )
                    if any(k in block for k in ('image_url', 'audio', 'file', 'input_audio')):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f'Message {idx} block {b_idx} contains multimodal attachments. '
                                'Subscription providers support plain text conversations only.'
                            ),
                        )
                    text_val = block.get('text', '')
                    if not isinstance(text_val, str):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Message {idx} block {b_idx} text must be a string.',
                        )
                    if len(text_val) > MAX_MESSAGE_CONTENT_LENGTH:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Message {idx} block {b_idx} exceeds maximum content length limit.',
                        )
                    if role == 'user' and text_val.strip():
                        has_user_prompt = True
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Message at index {idx} contains invalid content format (must be string or text blocks).',
                )

    if require_complete_messages:
        if not has_user_prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Chat completion requires at least one non-empty user message.',
            )
    else:
        # Early phase check: user prompt in messages OR user_message OR saved regeneration
        is_regeneration = bool(
            (metadata.get('assistant_message_id') if isinstance(metadata, dict) else None)
            or (metadata.get('regeneration_message_id') if isinstance(metadata, dict) else None)
            or form_data.get('assistant_message_id')
            or form_data.get('regeneration_message_id')
        )
        has_um_prompt = False
        if user_msg and isinstance(user_msg, dict):
            um_content = user_msg.get('content')
            if isinstance(um_content, str) and um_content.strip():
                has_um_prompt = True
            elif isinstance(um_content, list):
                has_um_prompt = any(
                    isinstance(b, dict) and b.get('type') == 'text' and bool(str(b.get('text', '')).strip())
                    for b in um_content
                )

        if not has_user_prompt and not has_um_prompt and not is_regeneration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Chat completion requires at least one non-empty user message.',
            )


def check_subscription_access(user: Any, owner_id: str | None = None) -> None:
    """Enforce single-user host subscription binding.

    In compliance with provider Terms of Service (no account intermediation/sharing),
    host subscription credentials must bind exclusively to one authenticated owner user.
    If no owner is configured (blank), all execution, test, and discovery requests are
    denied unconditionally, even for administrators.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authentication required to access host subscription provider.',
        )

    user_id = getattr(user, 'id', None)

    # Fresh install mode: no owner bound yet -> always deny execution/test/discovery
    if not owner_id or not str(owner_id).strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                'Host subscription provider is not configured or claimed. '
                'An administrator must claim ownership and enable providers before use.'
            ),
        )

    # Bound owner mode: user must match the persisted owner ID exactly
    if user_id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                'Host subscription is exclusively bound to the authenticated owner account. '
                'Cross-user sharing of host plan credentials is prohibited by provider terms.'
            ),
        )


def validate_owner_assignment(
    new_owner_id: str,
    current_user: Any,
    existing_owner_id: str | None = None,
    enabled: bool = False,
) -> None:
    """Validate that subscription ownership is properly claimed and cannot be transferred or cleared.

    - Only administrators can configure subscription ownership.
    - If ownership was previously claimed, it CANNOT be cleared (set to empty) or transferred to another user.
    - Admins can only claim ownership for their own account (new_owner_id == current_user.id).
    - If subscriptions are enabled, owner_id must be non-empty and equal to current_user.id.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authentication required.',
        )

    current_id = getattr(current_user, 'id', None)
    current_role = getattr(current_user, 'role', None)

    if current_role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Only administrators can configure subscription ownership.',
        )

    existing_owner = str(existing_owner_id).strip() if existing_owner_id else ''
    new_owner = str(new_owner_id).strip() if new_owner_id else ''

    if existing_owner:
        if existing_owner != current_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Only the designated subscription owner can modify subscription configuration.',
            )
        if new_owner != existing_owner:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    'Subscription ownership cannot be cleared or transferred once claimed. '
                    'To stop using subscriptions, disable providers or the subscriptions master switch.'
                ),
            )
    else:
        # Fresh setup claim
        if new_owner and new_owner != current_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Administrators can only claim subscription ownership for their own account.',
            )

    if enabled and (not new_owner or new_owner != current_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Enabling subscriptions requires an explicit ownership claim by the current administrator.',
        )
