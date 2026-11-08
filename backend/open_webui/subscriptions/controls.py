from __future__ import annotations

import logging
import json
from typing import Any
from fastapi import HTTPException, status

log = logging.getLogger(__name__)

# Standard subscription error codes shared between security validation and adapters
ERROR_UNSUPPORTED_PARAMETER = 'UNSUPPORTED_PARAMETER'
ERROR_CLI_EXECUTION_ERROR = 'CLI_EXECUTION_ERROR'
ERROR_CLI_STREAM_ERROR = 'CLI_STREAM_ERROR'
ERROR_CLI_NOT_INSTALLED = 'CLI_NOT_INSTALLED'
ERROR_CLI_UNSUPPORTED_VERSION = 'CLI_UNSUPPORTED_VERSION'
ERROR_CLI_AUTH_REQUIRED = 'CLI_AUTH_REQUIRED'
ERROR_CLI_API_KEY_NOT_SUPPORTED = 'CLI_API_KEY_NOT_SUPPORTED'
ERROR_CLI_INVALID_RESPONSE = 'CLI_INVALID_RESPONSE'
ERROR_CLI_PREMATURE_EOF = 'CLI_PREMATURE_EOF'
ERROR_SAFETY_RESTRICTION = 'SAFETY_RESTRICTION'


VALID_REASONING_EFFORTS = frozenset({'low', 'medium', 'high', 'xhigh', 'max', 'ultra'})

MAX_REASONING_EFFORT_LEN = 64


def is_valid_reasoning_effort_shape(val: Any) -> bool:
    """Validate generic bounded string shape for reasoning effort.

    Enforces:
    - Must be a string (reject non-string types such as int, bool, list, dict)
    - Must be non-empty (after stripping)
    - No control characters (0x00-0x1F, 0x7F) or non-printable characters anywhere in string
    - Must not exceed maximum allowed length (64 chars)
    """
    if not isinstance(val, str) or isinstance(val, bool):
        return False
    if not val or len(val) > MAX_REASONING_EFFORT_LEN:
        return False
    for ch in val:
        code = ord(ch)
        if code < 32 or code == 127 or not ch.isprintable():
            return False
    trimmed = val.strip()
    if not trimmed:
        return False
    return True

UNSUPPORTED_DEFAULT_KEYS = frozenset({
    'temperature', 'top_p', 'top_k', 'max_tokens', 'max_completion_tokens',
    'frequency_penalty', 'presence_penalty', 'seed', 'stop', 'response_format',
    'logit_bias', 'logprobs', 'top_logprobs', 'n', 'best_of',
})


def filter_subscription_defaults(params: dict[str, Any]) -> dict[str, Any]:
    """Remove API-only defaults injected by the server, preserving caller validation."""
    visited = 0

    def filtered(source, depth=0, custom=False):
        nonlocal visited
        visited += 1
        if depth > 8 or visited > 128:
            raise HTTPException(400, f'Parameter nesting exceeds supported bounds (code: {ERROR_UNSUPPORTED_PARAMETER}).')
        result = {}
        for key, value in source.items():
            if key in UNSUPPORTED_DEFAULT_KEYS:
                continue
            if custom and isinstance(value, str):
                if len(value) > 1_048_576:
                    raise HTTPException(400, f'Custom parameter exceeds supported bounds (code: {ERROR_UNSUPPORTED_PARAMETER}).')
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    pass
                except RecursionError:
                    raise HTTPException(400, f'Custom parameter nesting exceeds supported bounds (code: {ERROR_UNSUPPORTED_PARAMETER}).')
            if key in ('params', 'options', 'custom_params') and isinstance(value, dict):
                value = filtered(value, depth + 1, key == 'custom_params')
            result[key] = value
        return result

    return filtered(params)



def iter_subscription_parameter_sources(request_data: dict, metadata: dict | None = None):
    """Inspect the same nested parameter sources which payload normalization can promote."""
    pending = [(request_data, 0, False)]
    if isinstance(metadata, dict):
        pending.append((metadata, 0, False))
    seen = {}
    while pending:
        source, depth, custom = pending.pop()
        if not isinstance(source, dict) or id(source) in seen:
            continue
        if depth > 8 or len(seen) >= 128:
            raise HTTPException(400, f'Parameter nesting exceeds supported bounds (code: {ERROR_UNSUPPORTED_PARAMETER}).')
        seen[id(source)] = source
        effective = source
        if custom:
            effective = dict(source)
            for key, value in source.items():
                # apply_params_to_form_data decodes JSON-valued custom parameters.
                if isinstance(value, str):
                    if len(value) > 1_048_576:
                        raise HTTPException(400, f'Custom parameter exceeds supported bounds (code: {ERROR_UNSUPPORTED_PARAMETER}).')
                    try:
                        effective[key] = json.loads(value)
                    except (ValueError, TypeError):
                        pass
                    except RecursionError:
                        raise HTTPException(400, f'Custom parameter nesting exceeds supported bounds (code: {ERROR_UNSUPPORTED_PARAMETER}).')
        yield effective
        for key in ('params', 'options', 'custom_params', 'metadata', 'meta'):
            nested = effective.get(key)
            if key in ('params', 'options', 'custom_params') and nested is not None and not isinstance(nested, dict):
                raise HTTPException(400, f'Parameter containers must be objects (code: {ERROR_UNSUPPORTED_PARAMETER}).')
            if isinstance(nested, dict):
                pending.append((nested, depth + 1, key == 'custom_params'))


def validate_subscription_chat_controls(
    request_data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Validate chat controls for subscription-based CLI model providers.

    Centralized single validator used early (pre-DB persistence in security.py)
    and by all provider adapters. Inspects top-level and nested parameter dictionaries
    (params, options, metadata).

    Permits documented no-op default values (e.g. temperature=1.0, top_p=1.0, penalties=0.0).
    Rejects custom sampling, ceiling limits (max_tokens/max_completion_tokens),
    structured response formats, stop sequences, and invalid reasoning_effort with HTTP 400.
    Malformed parameter types fail with HTTP 400 rather than being silently ignored.
    """
    for d in iter_subscription_parameter_sources(request_data, metadata):
        # 1. Temperature: reject custom non-default values (None or default 1.0 is accepted)
        if 'temperature' in d and d['temperature'] is not None:
            try:
                temp_val = float(d['temperature'])
                if abs(temp_val - 1.0) > 1e-4:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f'Subscription providers operate with fixed model parameters. '
                            f'Custom temperature={temp_val} is not supported (code: {ERROR_UNSUPPORTED_PARAMETER}).'
                        ),
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid temperature parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 2. Top-p: reject custom non-default values (None or default 1.0 is accepted)
        if 'top_p' in d and d['top_p'] is not None:
            try:
                top_p_val = float(d['top_p'])
                if abs(top_p_val - 1.0) > 1e-4:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f'Subscription providers operate with fixed model parameters. '
                            f'Custom top_p={top_p_val} is not supported (code: {ERROR_UNSUPPORTED_PARAMETER}).'
                        ),
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid top_p parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 3. Top-k: reject custom non-default values
        if 'top_k' in d and d['top_k'] is not None:
            try:
                k_val = int(d['top_k'])
                if k_val > 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f'Subscription providers do not support custom top_k={k_val} controls '
                            f'(code: {ERROR_UNSUPPORTED_PARAMETER}).'
                        ),
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid top_k parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 4. Max tokens / Max completion tokens: reject custom token ceilings
        for mt_key in ('max_tokens', 'max_completion_tokens'):
            if mt_key in d and d[mt_key] is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f'Subscription providers do not support custom {mt_key}. '
                        f'Maximum tokens are managed by provider CLI defaults (code: {ERROR_UNSUPPORTED_PARAMETER}).'
                    ),
                )

        # 5. Response format: reject structured outputs / json schema mode
        if 'response_format' in d and d['response_format'] is not None:
            rf = d['response_format']
            rf_type = rf.get('type') if isinstance(rf, dict) else str(rf)
            if rf_type not in (None, 'text', ''):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f'Subscription providers do not support custom response_format="{rf_type}". '
                        f'Conversations are plain text only (code: {ERROR_UNSUPPORTED_PARAMETER}).'
                    ),
                )

        # 6. Stop sequences
        if 'stop' in d and d['stop'] is not None:
            stop_val = d['stop']
            if stop_val and stop_val != [] and stop_val != '':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Subscription providers do not support custom stop sequences (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 7. Seed
        if 'seed' in d and d['seed'] is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Subscription providers do not support deterministic seed parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
            )

        # 8. Penalties: presence_penalty and frequency_penalty (None or default 0.0 is accepted)
        for p_key in ('presence_penalty', 'frequency_penalty'):
            if p_key in d and d[p_key] is not None:
                try:
                    val = float(d[p_key])
                    if abs(val) > 1e-4:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Subscription providers do not support {p_key} (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                        )
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'Invalid {p_key} parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                    )

        # 9. Logit bias
        if 'logit_bias' in d and d['logit_bias'] is not None:
            if d['logit_bias'] != {}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Subscription providers do not support logit_bias (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 10. Multiple completions (n)
        if 'n' in d and d['n'] is not None:
            try:
                n_val = int(d['n'])
                if n_val != 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'Subscription providers only support single completion (n=1) (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid n parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 11. Best of
        if 'best_of' in d and d['best_of'] is not None:
            try:
                b_val = int(d['best_of'])
                if b_val != 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'Subscription providers do not support best_of (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid best_of parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 12. Logprobs
        if 'logprobs' in d and d['logprobs'] is not None:
            if bool(d['logprobs']):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Subscription providers do not support logprobs output controls (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )

        # 13. Top logprobs
        if 'top_logprobs' in d and d['top_logprobs'] is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Subscription providers do not support top_logprobs controls (code: {ERROR_UNSUPPORTED_PARAMETER}).',
            )

        # 14. Reasoning effort: validate bounded string shape, reject invalid without echoing raw input
        if 'reasoning_effort' in d and d['reasoning_effort'] is not None:
            raw_val = d['reasoning_effort']
            if not is_valid_reasoning_effort_shape(raw_val):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid reasoning_effort parameter (code: {ERROR_UNSUPPORTED_PARAMETER}).',
                )
