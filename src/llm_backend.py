from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import AssistantResponse

DEFAULT_OLLAMA_HOST = 'http://localhost:11434'
DEFAULT_OLLAMA_MODEL = 'qwen2.5-coder:7b'
DEFAULT_OPENAI_COMPAT_HOST = 'http://localhost:8080'
DEFAULT_OPENAI_COMPAT_MODEL = 'local-model'


class LLMBackendError(RuntimeError):
    """Raised when the local LLM backend cannot complete a request."""


@dataclass(frozen=True)
class OllamaBackend:
    """Backend for Ollama running at localhost:11434 (native Ollama API)."""

    model: str = DEFAULT_OLLAMA_MODEL
    host: str = DEFAULT_OLLAMA_HOST

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        payload: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'stream': False,
        }
        if response_format is not None:
            payload['format'] = response_format
        if options:
            payload['options'] = options
        body = json.dumps(payload).encode('utf-8')
        request = Request(
            url=f'{self.host.rstrip("/")}/api/chat',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with urlopen(request) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace').strip()
            message = detail or str(error)
            raise LLMBackendError(f'Ollama request failed: {message}') from error
        except URLError as error:
            raise LLMBackendError(
                f'Could not reach Ollama at {self.host}. '
                'Make sure Ollama is running and listening on that host.'
            ) from error
        except json.JSONDecodeError as error:
            raise LLMBackendError('Ollama returned invalid JSON.') from error

        return normalize_ollama_response(payload)


def normalize_ollama_response(payload: dict[str, Any]) -> AssistantResponse:
    message = payload.get('message')
    if not isinstance(message, dict):
        raise LLMBackendError('Ollama response did not include a message object.')

    content = message.get('content')
    if not isinstance(content, str):
        raise LLMBackendError('Ollama response did not include assistant content.')

    return AssistantResponse(content=content, raw=payload)


@dataclass(frozen=True)
class OpenAICompatBackend:
    """Backend for any OpenAI-compatible API endpoint.

    Works with llama.cpp server, LM Studio, Ollama /v1 endpoint,
    text-generation-webui, vLLM, TabbyAPI, and any other server that
    implements the OpenAI /v1/chat/completions wire format.

    The Rust claw harness uses the same provider pattern internally
    (see rust/crates/api/src/providers/openai_compat.rs).

    Examples::

        # llama.cpp server (default port 8080)
        OpenAICompatBackend(host='http://localhost:8080', model='local-model')

        # LM Studio (default port 1234)
        OpenAICompatBackend(host='http://localhost:1234', model='lmstudio-model')

        # Ollama OpenAI-compat endpoint
        OpenAICompatBackend(host='http://localhost:11434', model='qwen2.5-coder:7b')

        # Remote model with API key
        OpenAICompatBackend(host='https://api.openai.com', model='gpt-4o', api_key='sk-...')
    """

    model: str = DEFAULT_OPENAI_COMPAT_MODEL
    host: str = DEFAULT_OPENAI_COMPAT_HOST
    api_key: str = 'local'

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        payload: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'stream': False,
        }
        if response_format == 'json':
            payload['response_format'] = {'type': 'json_object'}
        elif isinstance(response_format, dict):
            payload['response_format'] = response_format
        if options:
            if 'temperature' in options:
                payload['temperature'] = options['temperature']
            if 'max_tokens' in options:
                payload['max_tokens'] = options['max_tokens']

        base = self.host.rstrip('/')
        # Normalise: if host already ends in /v1 don't double it
        url = f'{base}/chat/completions' if base.endswith('/v1') else f'{base}/v1/chat/completions'
        body = json.dumps(payload).encode('utf-8')
        request = Request(
            url=url,
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )

        try:
            with urlopen(request) as response:
                raw = json.loads(response.read().decode('utf-8'))
        except HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace').strip()
            message = detail or str(error)
            raise LLMBackendError(f'OpenAI-compat request failed ({url}): {message}') from error
        except URLError as error:
            raise LLMBackendError(
                f'Could not reach OpenAI-compat server at {self.host}. '
                'Make sure the server is running (llama.cpp, LM Studio, Ollama /v1, etc.).'
            ) from error
        except json.JSONDecodeError as error:
            raise LLMBackendError('OpenAI-compat server returned invalid JSON.') from error

        return normalize_openai_compat_response(raw)


def normalize_openai_compat_response(payload: dict[str, Any]) -> AssistantResponse:
    choices = payload.get('choices')
    if not isinstance(choices, list) or not choices:
        raise LLMBackendError('OpenAI-compat response did not include any choices.')
    message = choices[0].get('message')
    if not isinstance(message, dict):
        raise LLMBackendError('OpenAI-compat response choice did not include a message object.')
    content = message.get('content')
    if not isinstance(content, str):
        raise LLMBackendError('OpenAI-compat response message did not include string content.')
    return AssistantResponse(content=content, raw=payload)


# Union type for all supported backends
LLMBackend = Union[OllamaBackend, OpenAICompatBackend]


def create_backend(
    backend_kind: str = 'ollama',
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
) -> LLMBackend:
    """Factory that constructs the right backend from CLI/env arguments.

    Args:
        backend_kind: ``"ollama"`` or ``"openai-compat"``
        model: model name (falls back to per-backend defaults)
        host: base URL (falls back to per-backend defaults)
        api_key: API key for openai-compat backends (defaults to ``"local"``,
                 or ``OPENAI_API_KEY`` env var when set)
    """
    if backend_kind == 'openai-compat':
        resolved_key = api_key or os.environ.get('OPENAI_API_KEY', 'local')
        return OpenAICompatBackend(
            model=model or DEFAULT_OPENAI_COMPAT_MODEL,
            host=host or DEFAULT_OPENAI_COMPAT_HOST,
            api_key=resolved_key,
        )
    # default: ollama
    return OllamaBackend(
        model=model or DEFAULT_OLLAMA_MODEL,
        host=host or DEFAULT_OLLAMA_HOST,
    )
