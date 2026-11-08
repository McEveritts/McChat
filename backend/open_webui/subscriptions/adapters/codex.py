from __future__ import annotations

"""Legacy codex adapter compatibility module.

Re-exports CodexSessionAdapter from codex_session.py as both
CodexSessionAdapter and CodexAdapter for backwards compatibility.
"""

from open_webui.subscriptions.adapters.codex_session import (
    CodexSessionAdapter,
    SUPPORTED_CODEX_MODELS,
    SUPPORTED_MODEL_IDS,
    VALID_REASONING_EFFORTS,
    parse_codex_usage,
)

# Alias for backwards compatibility with existing imports
CodexAdapter = CodexSessionAdapter
CODEX_UNAVAILABLE_REASON = 'Codex subscription execution is unavailable'

__all__ = [
    'CodexAdapter',
    'CodexSessionAdapter',
    'CODEX_UNAVAILABLE_REASON',
    'SUPPORTED_CODEX_MODELS',
    'SUPPORTED_MODEL_IDS',
    'VALID_REASONING_EFFORTS',
    'parse_codex_usage',
]
