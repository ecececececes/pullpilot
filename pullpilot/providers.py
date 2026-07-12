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

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 2000,
                 api_key: Optional[str] = None):
        import anthropic  # lazy

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "Provide an API key or set the ANTHROPIC_API_KEY environment variable.")
        self._client = anthropic.Anthropic(api_key=key, max_retries=8)
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
                 name: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI  # lazy

        key = api_key or os.environ.get(api_key_env)
        if not key and "ollama" not in (base_url or ""):
            raise RuntimeError(
                f"Provide an API key or set the {api_key_env} environment variable.")
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
    """Provider for a self-hosted Ollama instance exposed over HTTP.

    Defaults to a local Ollama (http://localhost:11434) so nothing ever
    leaves the machine; point SELFHOSTED_BASE_URL at any Ollama-compatible
    /api/generate endpoint on your own infrastructure to share one instance."""

    name = "selfhosted"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._model = model or os.environ.get("SELFHOSTED_MODEL", "llama3.2:3b")
        self._timeout = timeout

        self._base_url = (
            base_url
            or os.environ.get(
                "SELFHOSTED_BASE_URL",
                "http://localhost:11434/api/generate",
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


def _make_preset(name: str, api_key: Optional[str] = None,
                 model: Optional[str] = None) -> Provider:
    base_url, default_model, key_env = PRESETS[name]
    base_url = os.environ.get(f"{name.upper()}_BASE_URL", base_url)
    model = model or os.environ.get(f"{name.upper()}_MODEL", default_model)
    return OpenAIProvider(model=model, base_url=base_url, api_key_env=key_env,
                          name=name, api_key=api_key)


_REGISTRY = {
    "mock": MockProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "selfhosted": SelfHostedProvider,
}


def get_provider(name: str, api_key: Optional[str] = None,
                 model: Optional[str] = None, **kwargs) -> Provider:
    """api_key/model override the environment (e.g. pasted into the web UI);
    they are only forwarded to providers that accept them."""
    if name in PRESETS:
        return _make_preset(name, api_key=api_key, model=model)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown provider '{name}', choose from "
            f"{list(_REGISTRY) + list(PRESETS)}")
    if api_key and name in ("anthropic", "openai"):
        kwargs.setdefault("api_key", api_key)
    if model and name in ("anthropic", "openai", "selfhosted"):
        kwargs.setdefault("model", model)
    return _REGISTRY[name](**kwargs)
