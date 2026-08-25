"""Model-provider adapters for the topic-matched experiment.

The simulation core is provider-agnostic.  This module is the only place that
knows whether a free-agent model is served by OpenAI, OpenRouter, or DeepInfra.
OpenRouter and DeepInfra expose OpenAI-compatible Chat Completions endpoints,
so all four free-agent models use the same OpenAI Python client interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Dict, Iterable, Mapping, Sequence

from openai import OpenAI

from protocol import EMBEDDING_MODEL, REFEREE_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    provider: str
    model_id: str
    api_key_env: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt4omini": ModelSpec(
        key="gpt4omini",
        label="GPT-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    "gpt56luna": ModelSpec(
        key="gpt56luna",
        label="GPT-5.6 Luna",
        provider="openai",
        model_id="gpt-5.6-luna",
        api_key_env="OPENAI_API_KEY",
    ),
    "deepseekv3": ModelSpec(
        key="deepseekv3",
        label="DeepSeek-V3",
        provider="openrouter",
        model_id="deepseek/deepseek-chat-v3-0324",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "phi4": ModelSpec(
        key="phi4",
        label="Phi-4",
        provider="deepinfra",
        model_id="microsoft/phi-4",
        api_key_env="DEEPINFRA_API_KEY",
    ),
}


def normalize_messages(messages: Sequence[object]) -> list[dict]:
    """Convert the simulation's lightweight message objects to OpenAI format."""
    normalized: list[dict] = []
    for message in messages:
        if isinstance(message, dict):
            normalized.append({"role": message["role"], "content": message["content"]})
            continue
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None or content is None:
            raise TypeError(f"Unsupported message object: {type(message)!r}")
        normalized.append({"role": str(role), "content": str(content)})
    return normalized


class ProviderRegistry:
    """Lazy provider clients plus a single provider-neutral generation interface."""

    def __init__(
        self,
        *,
        selected_model_keys: Iterable[str],
        client_factory: Callable[..., object] = OpenAI,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.selected_model_keys = tuple(dict.fromkeys(selected_model_keys))
        self._client_factory = client_factory
        self._environ = dict(os.environ if environ is None else environ)
        self._clients: dict[str, object] = {}
        self._validate_keys()

    def _required_key_names(self) -> set[str]:
        # OpenAI is always required: the referee and RAG embeddings use OpenAI.
        required = {"OPENAI_API_KEY"}
        for key in self.selected_model_keys:
            required.add(MODEL_SPECS[key].api_key_env)
        return required

    def _validate_keys(self) -> None:
        missing = [name for name in sorted(self._required_key_names()) if not self._environ.get(name, "").strip()]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )

    def _client(self, provider: str):
        if provider in self._clients:
            return self._clients[provider]
        if provider == "openai":
            client = self._client_factory(api_key=self._environ["OPENAI_API_KEY"])
        elif provider == "openrouter":
            client = self._client_factory(
                api_key=self._environ["OPENROUTER_API_KEY"],
                base_url=OPENROUTER_BASE_URL,
            )
        elif provider == "deepinfra":
            client = self._client_factory(
                api_key=self._environ["DEEPINFRA_API_KEY"],
                base_url=DEEPINFRA_BASE_URL,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self._clients[provider] = client
        return client

    def _spec_for_model_id(self, model_id: str) -> ModelSpec:
        for spec in MODEL_SPECS.values():
            if spec.model_id == model_id:
                return spec
        raise ValueError(f"Unknown free-agent model id: {model_id}")

    def generate(self, model_id: str, messages: Sequence[object], **kwargs) -> str:
        """Generate one completion without changing simulation-level call semantics."""
        normalized = normalize_messages(messages)

        # The referee is always GPT-4o-mini on OpenAI.  GPT-4o-mini free agents use
        # the same transport and are distinguished only by their call parameters.
        if model_id == REFEREE_MODEL:
            client = self._client("openai")
            response = client.chat.completions.create(
                model=model_id,
                messages=normalized,
                **kwargs,
            )
            return (response.choices[0].message.content or "").strip()

        spec = self._spec_for_model_id(model_id)
        client = self._client(spec.provider)
        call_kwargs = dict(kwargs)

        if spec.key == "gpt56luna":
            # GPT-5.6 Luna is called through Chat Completions with reasoning disabled,
            # matching the experiment configuration.  The current OpenAI interface
            # names the output cap `max_completion_tokens` for this model family.
            if "max_tokens" in call_kwargs and "max_completion_tokens" not in call_kwargs:
                call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")
            call_kwargs.setdefault("reasoning_effort", "none")

        response = client.chat.completions.create(
            model=spec.model_id,
            messages=normalized,
            **call_kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    def embed(self, text: str, *, model: str = EMBEDDING_MODEL) -> list[float]:
        client = self._client("openai")
        response = client.embeddings.create(model=model, input=text)
        return list(response.data[0].embedding)


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {model_key}") from exc
