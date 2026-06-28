"""Thin provider-agnostic interface so the identical harness runs across models.

MockProvider needs no API key and lets the whole pipeline run offline (wiring
test). Real providers are lazy-imported so missing SDKs never break imports.
"""
from __future__ import annotations
import requests
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Optional


class Provider(ABC):
    name = "base"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...


class MockProvider(Provider):
    """Deterministic offline stand-in. Reports one issue grounded in the first
    changed line it finds in the diff, so the harness produces a real table.
    Swap for a real provider to get meaningful numbers."""

    name = "mock"

    def __init__(self, canned: Optional[str] = None):
        self._canned = canned

    def complete(self, system: str, user: str) -> str:
        if self._canned is not None:
            return self._canned
        file_m = re.search(r"\+\+\+ b/(\S+)", user)
        line_m = re.search(r"@@ .*?\+(\d+)", user)
        file = file_m.group(1) if file_m else "unknown"
        line = int(line_m.group(1)) if line_m else 1
        return json.dumps(
            {
                "summary": "Mock summary: this PR modifies one or more functions.",
                "issues": [
                    {
                        "file": file,
                        "line_start": line,
                        "line_end": line,
                        "type": "likely_bug",
                        "severity": "major",
                        "confidence": 0.5,
                        "explanation": "Mock finding. Replace MockProvider with a real provider.",
                        "suggested_fix": None,
                        "is_question": False,
                    }
                ],
            }
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 2000):
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                            max_retries=8)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


class OpenAIProvider(Provider):
    """Works with OpenAI and any OpenAI-compatible endpoint (Groq, Gemini's
    compat API, GitHub Models, OpenRouter, Ollama, ...) via base_url."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 2000,
                 base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 name: Optional[str] = None):
        from openai import OpenAI  # lazy

        key = os.environ.get(api_key_env)
        if not key and "ollama" not in (base_url or ""):
            raise RuntimeError(
                f"Set the {api_key_env} environment variable for this provider.")
        self._client = OpenAI(api_key=key or "ollama", base_url=base_url,
                              max_retries=8, timeout=60.0)
        self._model = model
        self._max_tokens = max_tokens
        if name:
            self.name = name

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    
class SelfHostedProvider(Provider):
    """Provider for a self-hosted Ollama instance exposed over HTTP."""

    name = "selfhosted"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self._model = model
        self._timeout = timeout

        self._base_url = (
            base_url
            or os.environ.get(
                "SELFHOSTED_BASE_URL",
                "https://sabbath-crazily-gigantic.ngrok-free.dev/api/generate",
            )
        )

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            self._base_url,
            json=payload,
            timeout=self._timeout,
        )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "")

# Free / OpenAI-compatible presets. base_url and model can be overridden with
# the matching *_BASE_URL / *_MODEL env vars if a default goes stale.
PRESETS = {
    # name:    (base_url,                                              default_model,                     key_env)
    "gemini":  ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash",          "GEMINI_API_KEY"),
    "groq":    ("https://api.groq.com/openai/v1",                       "llama-3.3-70b-versatile",         "GROQ_API_KEY"),
    "github":  ("https://models.github.ai/inference",                  "openai/gpt-4o-mini",              "GITHUB_TOKEN"),
    "openrouter": ("https://openrouter.ai/api/v1",                     "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY"),
    "ollama":  ("http://localhost:11434/v1",                           "llama3.1",                        "OLLAMA_API_KEY"),
}


def _make_preset(name: str) -> Provider:
    base_url, model, key_env = PRESETS[name]
    base_url = os.environ.get(f"{name.upper()}_BASE_URL", base_url)
    model = os.environ.get(f"{name.upper()}_MODEL", model)
    return OpenAIProvider(model=model, base_url=base_url, api_key_env=key_env, name=name)


_REGISTRY = {
    "mock": MockProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "selfhosted": SelfHostedProvider,
}


def get_provider(name: str, **kwargs) -> Provider:
    if name in PRESETS:
        return _make_preset(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown provider '{name}', choose from "
            f"{list(_REGISTRY) + list(PRESETS)}")
    return _REGISTRY[name](**kwargs)
