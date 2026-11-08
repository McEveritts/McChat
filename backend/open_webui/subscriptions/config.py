from __future__ import annotations

import os
import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Persistent database config keys for subscription management
SUBSCRIPTIONS_CONFIG_KEYS = {
    'ENABLE_SUBSCRIPTIONS': 'subscriptions.enable',
    'SUBSCRIPTIONS_OWNER_ID': 'subscriptions.owner_id',
    'SUBSCRIPTIONS_REQUEST_TIMEOUT': 'subscriptions.request_timeout',
    'SUBSCRIPTIONS_PROVIDERS': 'subscriptions.providers',
}

# Environment variable defaults
DEFAULT_ENABLE_SUBSCRIPTIONS = os.getenv('ENABLE_SUBSCRIPTIONS', 'false').lower() == 'true'
DEFAULT_SUBSCRIPTIONS_OWNER_ID = os.getenv('SUBSCRIPTIONS_OWNER_ID', '')
DEFAULT_SUBSCRIPTIONS_REQUEST_TIMEOUT = int(os.getenv('SUBSCRIPTIONS_REQUEST_TIMEOUT', '180'))

# Default provider definitions with maintained official models
DEFAULT_PROVIDERS_CONFIG = {
    'claude': {
        'enabled': False,
        'name': 'Claude Code (Anthropic Subscription)',
        'binary_path': '',
        'config_dir': '',
        'models': [
            {'id': 'sonnet', 'name': 'Claude Sonnet (Subscription)'},
            {'id': 'opus', 'name': 'Claude Opus (Subscription)'},
            {'id': 'haiku', 'name': 'Claude Haiku (Subscription)'},
        ],
    },
    'codex': {
        'enabled': False,
        'name': 'OpenAI Codex (ChatGPT Subscription)',
        'binary_path': '',
        'config_dir': '',
        'models': [],
    },
    'gemini': {
        'enabled': False,
        'name': 'Google Gemini (Antigravity ACP Subscription)',
        'binary_path': '',
        'config_dir': '',
        'models': [],
    },
}

ALLOWED_PROVIDER_IDS = {'claude', 'codex', 'gemini'}
MODEL_ID_REGEX = re.compile(r'^[a-zA-Z0-9_.-]+$')



class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)


ModelConfigModel = ModelDefinition


class ProviderConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    name: str | None = Field(default=None, max_length=128)
    binary_path: str = Field(default='', max_length=512)
    config_dir: str = Field(default='', max_length=512)
    models: list[ModelDefinition] = Field(default_factory=list, max_length=50)

    @field_validator('models')
    @classmethod
    def validate_unique_models(cls, v: list[ModelDefinition]) -> list[ModelDefinition]:
        seen = set()
        for m in v:
            if m.id in seen:
                raise ValueError(f'Duplicate model id "{m.id}" found in provider configuration.')
            seen.add(m.id)
        return v


class SubscriptionsConfigForm(BaseModel):
    model_config = ConfigDict(extra='forbid')

    ENABLE_SUBSCRIPTIONS: bool = False
    SUBSCRIPTIONS_OWNER_ID: str = Field(default='', max_length=256)
    SUBSCRIPTIONS_REQUEST_TIMEOUT: int = Field(default=180, ge=1, le=600)
    SUBSCRIPTIONS_PROVIDERS: dict[str, ProviderConfigModel] = Field(default_factory=dict)

    @field_validator('SUBSCRIPTIONS_PROVIDERS')
    @classmethod
    def validate_providers(cls, v: dict[str, ProviderConfigModel]) -> dict[str, ProviderConfigModel]:
        unknown = set(v.keys()) - ALLOWED_PROVIDER_IDS
        if unknown:
            raise ValueError(f'Unsupported provider id(s): {", ".join(sorted(unknown))}. Only "claude", "codex", and "gemini" are supported.')
        for p_id, p_cfg in v.items():
            for m in p_cfg.models:
                if m.id.startswith('-'):
                    raise ValueError(f'Model id "{m.id}" cannot start with a dash.')
                if not MODEL_ID_REGEX.match(m.id):
                    raise ValueError(f'Model id "{m.id}" contains invalid characters.')
        return v


class ProviderTestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt: str = Field(default='Hello, reply with "pong" and nothing else.', max_length=4096)
    model: str | None = Field(default=None, max_length=64)

