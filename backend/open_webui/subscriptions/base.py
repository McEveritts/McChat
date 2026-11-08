from __future__ import annotations

import abc
import time
import uuid
from typing import Any, AsyncGenerator
from open_webui.subscriptions.config import ProviderConfigModel


class BaseSubscriptionAdapter(abc.ABC):
    """Abstract base adapter for a CLI-owned subscription provider."""

    def __init__(self, provider_id: str | None = None, default_name: str | None = None):
        self.provider_id = provider_id or getattr(self, 'provider_id', '')
        self.default_name = default_name or getattr(self, 'default_name', '')

    @abc.abstractmethod
    def is_installed(self, config: ProviderConfigModel) -> tuple[bool, str]:
        """Check if the provider CLI binary is installed and executable on the host.

        Returns (is_installed, binary_path_or_error).
        """
        pass

    @abc.abstractmethod
    async def check_auth_status(self, config: ProviderConfigModel) -> dict[str, Any]:
        """Check if the provider CLI is authenticated on the host.

        Does NOT extract or inspect OAuth token files directly.
        Returns dict with keys: 'installed', 'authenticated', 'status', 'version',
        'message', 'execution_available', 'reason'.
        """
        pass

    @abc.abstractmethod
    def get_supported_models(self, config: ProviderConfigModel) -> list[dict[str, Any]]:
        """Return the list of models supported by this provider."""
        pass

    @abc.abstractmethod
    async def chat_completion(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> dict[str, Any]:
        """Generate a non-streaming chat completion in standard OpenAI format."""
        pass

    @abc.abstractmethod
    async def chat_completion_stream(
        self,
        request_data: dict[str, Any],
        model_id: str,
        config: ProviderConfigModel,
        timeout: int = 180,
    ) -> AsyncGenerator[str, None]:
        """Generate an SSE stream of chat completion chunks in standard OpenAI format."""
        pass

    @abc.abstractmethod
    async def test_connection(
        self,
        config: ProviderConfigModel,
        prompt: str = 'Hello',
        model_id: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Run a test prompt to verify end-to-end functionality."""
        pass

    def build_openai_completion_dict(
        self,
        content: str,
        model_id: str,
        completion_id: str | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str = 'stop',
    ) -> dict[str, Any]:
        """Construct an OpenAI-compatible completion dictionary.

        Omits usage if unknown instead of fabricating zero token counts.
        """
        cid = completion_id or f'chatcmpl-{uuid.uuid4().hex[:12]}'
        resp: dict[str, Any] = {
            'id': cid,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': f'subscription:{self.provider_id}:{model_id}',
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': content,
                    },
                    'finish_reason': finish_reason,
                }
            ],
        }
        if usage is not None:
            resp['usage'] = usage
        return resp
