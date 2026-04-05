from __future__ import annotations

from dataclasses import dataclass

from .llm_backend import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, OllamaBackend

DEFAULT_SYSTEM_PROMPT = (
    'You are a local coding assistant running inside a minimal harness. '
    'Be concise, accurate, and explicit about uncertainty.'
)


@dataclass(frozen=True)
class PortingTask:
    name: str
    description: str


def build_chat_messages(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    return messages


def run_local_task(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    backend = OllamaBackend(model=model, host=host)
    response = backend.chat(build_chat_messages(prompt=prompt, system_prompt=system_prompt))
    return response.content


__all__ = ['DEFAULT_SYSTEM_PROMPT', 'PortingTask', 'build_chat_messages', 'run_local_task']
