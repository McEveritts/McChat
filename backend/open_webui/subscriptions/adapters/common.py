from __future__ import annotations

import contextlib
import html
import json
import logging
import os
import re
import shutil
import tempfile
from typing import Any, Iterator
from fastapi import HTTPException, status

log = logging.getLogger(__name__)

# Strict allowlist of environment variables permitted to cross into provider CLI subprocesses.
# Never copy os.environ directly; only explicitly allowlisted system and localization keys are passed.
SAFE_ENV_KEYS = frozenset({
    'PATH',
    'HOME',
    'LANG',
    'LC_ALL',
    'LC_CTYPE',
    'TMPDIR',
    'TZ',
    'SHELL',
    # Necessary runtime library / loader paths for node/rust CLI environments
    'LD_LIBRARY_PATH',
    'NODE_PATH',
    'SSL_CERT_FILE',
    'SSL_CERT_DIR',
})

# Diagnostic codes for structured, privacy-preserving error reporting.
# No raw CLI output, bearer tokens, or internal filesystem paths are exposed.
ERROR_CLI_NOT_INSTALLED = 'CLI_NOT_INSTALLED'
ERROR_CLI_UNSUPPORTED_VERSION = 'CLI_UNSUPPORTED_VERSION'
ERROR_CLI_AUTH_REQUIRED = 'CLI_AUTH_REQUIRED'
ERROR_CLI_API_KEY_NOT_SUPPORTED = 'CLI_API_KEY_NOT_SUPPORTED'
ERROR_CLI_EXECUTION_ERROR = 'CLI_EXECUTION_ERROR'
ERROR_CLI_INVALID_RESPONSE = 'CLI_INVALID_RESPONSE'
ERROR_CLI_STREAM_ERROR = 'CLI_STREAM_ERROR'
ERROR_CLI_PREMATURE_EOF = 'CLI_PREMATURE_EOF'
ERROR_UNSUPPORTED_PARAMETER = 'UNSUPPORTED_PARAMETER'
ERROR_SAFETY_RESTRICTION = 'SAFETY_RESTRICTION'


def build_sanitized_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Construct an allowlisted, sanitized environment dictionary for CLI subprocesses.

    Strictly excludes sensitive host environment variables (e.g. ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, OPENAI_API_KEY, AWS_*, WEBUI_*, DATABASE_*).
    """
    env: dict[str, str] = {}
    for key in SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    if extra_env:
        for k, v in extra_env.items():
            if v is not None:
                env[k] = str(v)

    # Belt-and-suspenders: explicitly purge any accidental sensitive keys
    dangerous_prefixes = (
        'ANTHROPIC_',
        'OPENAI_',
        'CODEX_',
        'AWS_',
        'CLAUDE_CODE_USE_',
        'WEBUI_',
        'DATABASE_',
        'SECRET_',
    )
    for k in list(env.keys()):
        if any(k.startswith(pfx) for pfx in dangerous_prefixes):
            env.pop(k, None)

    return env


@contextlib.contextmanager
def create_request_scratch_dir(prefix: str = 'sub_scratch_') -> Iterator[str]:
    """Create a private mode 0700 per-request scratch working directory.

    Owned by the backend runtime under DATA_DIR or ~/.tmp/mcchat-subscriptions/scratch.
    Guarantees clean isolation and automatic cleanup upon request completion or cancellation.
    """
    data_dir = os.environ.get('DATA_DIR')
    if data_dir and os.path.isdir(data_dir):
        scratch_base = os.path.join(data_dir, 'subscription_scratch')
    else:
        scratch_base = os.path.expanduser('~/.tmp/mcchat-subscriptions/scratch')

    os.makedirs(scratch_base, mode=0o700, exist_ok=True)
    try:
        os.chmod(scratch_base, 0o700)
    except OSError:
        pass

    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=scratch_base)
    try:
        os.chmod(temp_dir, 0o700)
    except OSError:
        pass

    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def validate_model_identifier(model_id: str) -> str:
    """Validate model ID to prevent flag injection and control syntax."""
    if not model_id or not isinstance(model_id, str):
        raise ValueError('Model identifier must be a non-empty string.')
    cleaned = model_id.strip()
    if not (1 <= len(cleaned) <= 64):
        raise ValueError(f'Model identifier length must be between 1 and 64 characters (got {len(cleaned)}).')
    if cleaned.startswith('-'):
        raise ValueError('Model identifier cannot start with "-" (flag injection prevention).')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', cleaned):
        raise ValueError(f"Model identifier contains invalid characters: '{cleaned}'.")
    return cleaned


from open_webui.subscriptions.controls import (
    validate_subscription_chat_controls as _validate_controls,
)


def validate_subscription_chat_controls(request_data: dict[str, Any]) -> None:
    """Validate chat controls for subscription-based CLI model providers."""
    _validate_controls(request_data)



def format_structured_prompt(messages: list[dict[str, Any]]) -> str:
    """Format conversation messages into an unambiguous structured prompt.

    Preserves role boundaries and system instructions without interpreting embedded
    role labels (e.g. 'Human:', 'Assistant:') in user text.
    """
    system_parts: list[str] = []
    turns: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if isinstance(content, list):
            content = '\n'.join(
                item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text'
            )
        elif not isinstance(content, str):
            content = str(content)

        content = html.escape(content, quote=False)

        if role == 'system':
            system_parts.append(content)
        elif role == 'assistant':
            turns.append(f'<turn role="assistant">\n{content}\n</turn>')
        else:
            turns.append(f'<turn role="user">\n{content}\n</turn>')

    output_lines: list[str] = []
    if system_parts:
        output_lines.append('<system_instructions>')
        output_lines.append('\n\n'.join(system_parts))
        output_lines.append('</system_instructions>\n')

    output_lines.append('<conversation_history>')
    output_lines.extend(turns)
    output_lines.append('</conversation_history>\n')
    output_lines.append('<instruction>Please respond to the conversation above as the assistant.</instruction>')

    return '\n'.join(output_lines)


def parse_usage_dict(raw_usage: Any) -> dict[str, Any] | None:
    """Parse usage statistics including cache tokens.

    Returns None if usage is unknown or missing, avoiding fabricated zeros.
    """
    if not isinstance(raw_usage, dict):
        return None

    in_tok = raw_usage.get('input_tokens')
    out_tok = raw_usage.get('output_tokens')

    if in_tok is None and out_tok is None:
        return None

    prompt_tokens = int(in_tok or 0)
    completion_tokens = int(out_tok or 0)
    cache_read = raw_usage.get('cache_read_input_tokens')
    cache_creation = raw_usage.get('cache_creation_input_tokens')
    prompt_tokens += int(cache_read or 0) + int(cache_creation or 0)
    total_tokens = prompt_tokens + completion_tokens

    usage: dict[str, Any] = {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
    }

    if cache_read is not None or cache_creation is not None:
        details: dict[str, Any] = {}
        if cache_read is not None:
            details['cached_tokens'] = int(cache_read)
        usage['prompt_tokens_details'] = details

        if cache_read is not None:
            usage['cache_read_input_tokens'] = int(cache_read)
        if cache_creation is not None:
            usage['cache_creation_input_tokens'] = int(cache_creation)

    return usage
