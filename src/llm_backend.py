from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import AssistantResponse

DEFAULT_OLLAMA_HOST = 'http://localhost:11434'
DEFAULT_OLLAMA_MODEL = 'qwen2.5-coder:7b'


class LLMBackendError(RuntimeError):
    """Raised when the local LLM backend cannot complete a request."""


@dataclass(frozen=True)
class OllamaBackend:
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
