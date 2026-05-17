from __future__ import annotations
import openai
import os
import warnings
from typing import List

from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceResponseError


# --- API Keys (placeholders) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# GPT
GPT_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

# DeepSeek
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEP_CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Phi-4 (Azure Inference API)
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
AZURE_KEY      = os.environ.get("AZURE_KEY", "")
AZURE_CLIENT = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY),
    api_version="2024-05-01-preview"
)

model_name = "Phi-4-3"





def _normalize_messages_for_openai(msgs):
    norm = []
    for m in msgs:
        # Azure Inference message objects
        if hasattr(m, "content"):
            t = type(m).__name__.lower()
            role = "system" if "system" in t else ("assistant" if "assistant" in t else "user")
            norm.append({"role": role, "content": m.content})
        elif isinstance(m, dict) and "role" in m and "content" in m:
            norm.append(m)
        else:
            norm.append({"role": "user", "content": str(m)})
    return norm

def call_llm(model: str, messages: List[dict], **kwargs) -> str:
    if model.startswith("gpt"):
        messages = _normalize_messages_for_openai(messages)
        resp = GPT_CLIENT.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content.strip()
    if model.startswith("deepseek"):
        messages = _normalize_messages_for_openai(messages)
        resp = DEEP_CLIENT.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content.strip()
    if model.startswith("Phi"):
        resp = AZURE_CLIENT.complete(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content.strip()
    raise ValueError(f"Unknown model: {model}")





client_embed   = OpenAI(api_key=OPENAI_API_KEY)

def _get_embedding(text: str, *,
                   model: str = "text-embedding-3-large") -> List[float]:
    try:
        resp = client_embed.embeddings.create(
            model=model,
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        warnings.warn(f"[MemoryStore] Embedding failed: {e!r}")
        return [0.0] * 1536
