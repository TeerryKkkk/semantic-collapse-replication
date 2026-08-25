"""History-Conditioned Avoidance Decoding for conversational agents.

This is a conversational adaptation of Avoidance Decoding, not an exact
unmodified reproduction of its multi-branch story-generation setting. Recent
visible utterances replace negative story branches. The decoder retains the
source method's adaptive candidate construction, contextual and narrative
similarity penalties, and probability-minus-history token score.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import random
from typing import Any, Sequence


@dataclass
class HCADConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantization: str = "nf4"
    seed: int = 42
    max_new_tokens: int = 200

    adaptive_q: float = 1.0
    candidate_min: int = 5
    candidate_max: int = 15
    candidate_cap: int = 15
    candidate_batch_size: int = 15

    avoidance_beta: float = 2.0
    schedule_delta: float = 0.5
    schedule_t0: float = 25.0
    guidance_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    guidance_device: str = "cpu"
    negative_max_tokens: int = 256

    def validate(self) -> None:
        if self.max_new_tokens != 200:
            raise ValueError("HC-AD requires exactly max_new_tokens=200.")
        if not (
            1
            <= self.candidate_min
            <= self.candidate_max
            <= self.candidate_cap
        ):
            raise ValueError("Invalid adaptive candidate bounds.")
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive.")
        if not 0.0 <= self.schedule_delta <= 1.0:
            raise ValueError("schedule_delta must be in [0, 1].")


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _safe_atanh(value: float) -> float:
    return math.atanh(max(-1.0 + 1e-7, min(1.0 - 1e-7, value)))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(item) for item in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def adaptive_value(
    entropy: float,
    previous_entropies: Sequence[float],
    maximum_entropy: float,
    *,
    q: float,
) -> float:
    if maximum_entropy <= 0.0:
        return 0.5
    center = (
        float(entropy)
        if not previous_entropies
        else float(_median(previous_entropies))
    )
    normalized = (float(entropy) - center) / maximum_entropy
    return stable_sigmoid(q * _safe_atanh(normalized))


def adaptive_candidate_count(
    entropy: float,
    previous_entropies: Sequence[float],
    vocab_size: int,
    *,
    q: float = 1.0,
    minimum: int = 5,
    maximum: int = 15,
    cap: int = 15,
) -> int:
    ratio = adaptive_value(
        entropy,
        previous_entropies,
        math.log(max(vocab_size, 2)),
        q=q,
    )
    value = int(round(minimum + (maximum - minimum) * ratio))
    return max(minimum, min(maximum, cap, value))


def schedule_csp_weight(
    step: int,
    *,
    delta: float = 0.5,
    t0: float = 25.0,
) -> float:
    """Use the source paper's prose-consistent CSP-early orientation."""

    return delta + (1.0 - delta) * stable_sigmoid(t0 - float(step))


def _as_eos_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (str, bytes)):
        raise TypeError("EOS token IDs must be integers, not text.")
    try:
        return {int(item) for item in value}
    except TypeError:
        return {int(value)}


def resolve_eos_token_ids(tokenizer: Any, model: Any) -> set[int]:
    """Resolve and deduplicate all tokenizer/model termination IDs."""

    sources = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "eos_token_ids", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_ids", None),
        getattr(
            getattr(model, "generation_config", None),
            "eos_token_id",
            None,
        ),
        getattr(
            getattr(model, "generation_config", None),
            "eos_token_ids",
            None,
        ),
    )
    resolved: set[int] = set()
    for value in sources:
        resolved.update(_as_eos_set(value))
    if not resolved:
        raise RuntimeError(
            "No EOS token IDs were found in the tokenizer or model configuration."
        )
    return resolved


def resolve_context_limit(model: Any) -> int:
    value = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "The loaded model does not declare a valid max_position_embeddings."
        ) from exc
    if limit <= 0:
        raise RuntimeError(
            "The loaded model declares a non-positive max_position_embeddings."
        )
    return limit


def validate_context_lengths(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    negative_max_tokens: int,
    supported_context_tokens: int,
    avoidance_bank_size: int,
) -> None:
    """Check generation and one-negative paths without changing the prompt."""

    prompt = int(prompt_tokens)
    supported = int(supported_context_tokens)
    generation_required = prompt + int(max_new_tokens)
    negative_required = prompt + int(negative_max_tokens)
    violations: list[str] = []
    if generation_required > supported:
        violations.append(f"generation_required={generation_required}")
    if int(avoidance_bank_size) > 0 and negative_required > supported:
        violations.append(f"negative_history_required={negative_required}")
    if violations:
        raise ValueError(
            "HC-AD context length exceeds the loaded model limit: "
            f"prompt_tokens={prompt}, "
            + ", ".join(violations)
            + f", supported_context_tokens={supported}. "
            "No prompt or history text was truncated."
        )


def append_selected_token_unless_eos(
    generated_ids: list[int],
    selected_token: int,
    eos_token_ids: set[int],
) -> bool:
    if int(selected_token) in eos_token_ids:
        return True
    generated_ids.append(int(selected_token))
    return False


class HCADDecoder:
    """One HC-AD model shared by the host simulation's free agents."""

    def __init__(self, config: HCADConfig | None = None):
        self.config = config or HCADConfig()
        self.config.validate()
        self.torch: Any = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.guidance_encoder: Any = None
        self.device: Any = None

    def load(self) -> None:
        if self.model is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "HC-AD dependencies are missing; install requirements.txt."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("HC-AD requires an NVIDIA CUDA GPU.")

        self.torch = torch
        self.device = torch.device("cuda:0")
        quantization = self.config.quantization.lower()
        if quantization == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The BF16 configuration requires GPU BF16 support.")
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

        model_kwargs: dict[str, Any] = {
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
            "torch_dtype": compute_dtype,
        }
        if quantization == "nf4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization not in {"none", "bf16"}:
            raise ValueError(f"Unsupported quantization: {self.config.quantization}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=False,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for HC-AD NSP."
            ) from exc
        self.guidance_encoder = SentenceTransformer(
            self.config.guidance_model_name,
            device=self.config.guidance_device,
        )
        self.guidance_encoder.eval()

    @staticmethod
    def normalize_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                )
                continue
            content = str(getattr(message, "content", message))
            type_name = type(message).__name__.lower()
            if "system" in type_name:
                role = "system"
            elif "assistant" in type_name:
                role = "assistant"
            else:
                role = "user"
            normalized.append({"role": role, "content": content})
        return normalized

    def render_prompt(self, messages: Sequence[Any]) -> tuple[str, Any]:
        self.load()
        normalized = self.normalize_messages(messages)
        prompt_text = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return prompt_text, input_ids.to(self.device)

    def generate(
        self,
        messages: Sequence[Any],
        *,
        avoidance_texts: Sequence[str] = (),
    ) -> str:
        _, input_ids = self.render_prompt(messages)
        validate_context_lengths(
            prompt_tokens=int(input_ids.shape[-1]),
            max_new_tokens=self.config.max_new_tokens,
            negative_max_tokens=self.config.negative_max_tokens,
            supported_context_tokens=resolve_context_limit(self.model),
            avoidance_bank_size=len(avoidance_texts),
        )
        with self.torch.inference_mode():
            generated_ids = self._generate(input_ids, avoidance_texts)
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _generate(
        self,
        input_ids: Any,
        avoidance_texts: Sequence[str],
    ) -> list[int]:
        torch = self.torch
        backbone = self._backbone()
        lm_head = self.model.get_output_embeddings()

        prompt_output = backbone(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prompt_output.past_key_values
        next_logits = lm_head(prompt_output.last_hidden_state[:, -1, :])

        negative_hidden, negative_sentence = (
            self._cache_avoidance_representations(
                avoidance_texts,
                shared_context_cache=past_key_values,
            )
            if avoidance_texts
            else ([], None)
        )

        full_entropies: list[float] = []
        topk_entropies: list[float] = []
        generated_ids: list[int] = []
        eos_ids = resolve_eos_token_ids(self.tokenizer, self.model)

        for zero_step in range(self.config.max_new_tokens):
            step = zero_step + 1
            probabilities = torch.softmax(next_logits.float(), dim=-1)
            entropy = float(
                (
                    -probabilities
                    * torch.log(probabilities.clamp_min(1e-20))
                ).sum().item()
            )
            candidate_count = adaptive_candidate_count(
                entropy,
                full_entropies,
                int(probabilities.shape[-1]),
                q=self.config.adaptive_q,
                minimum=self.config.candidate_min,
                maximum=self.config.candidate_max,
                cap=self.config.candidate_cap,
            )
            full_entropies.append(entropy)

            candidate_probs, candidate_ids = torch.topk(
                probabilities,
                k=candidate_count,
                dim=-1,
            )
            renormalized = candidate_probs / candidate_probs.sum(
                dim=-1,
                keepdim=True,
            )
            topk_entropy = float(
                (
                    -renormalized
                    * torch.log(renormalized.clamp_min(1e-20))
                ).sum().item()
            )
            alpha = adaptive_value(
                topk_entropy,
                topk_entropies,
                math.log(max(candidate_count, 2)),
                q=self.config.adaptive_q,
            )
            topk_entropies.append(topk_entropy)

            candidate_hidden = self._candidate_hidden_states(
                backbone,
                candidate_ids[0],
                past_key_values,
            )
            candidate_norm = torch.nn.functional.normalize(
                candidate_hidden.float(),
                dim=-1,
            )
            history = self._history_scores(
                candidate_norm,
                candidate_ids[0],
                generated_ids,
                negative_hidden,
                negative_sentence,
                step=step,
            )
            final_scores = (
                (1.0 - alpha) * candidate_probs[0].float()
                - alpha * history
            )
            selected_index = int(torch.argmax(final_scores).item())
            selected_token = int(candidate_ids[0, selected_index].item())

            if append_selected_token_unless_eos(
                generated_ids,
                selected_token,
                eos_ids,
            ):
                break

            selected_tensor = torch.tensor(
                [[selected_token]],
                dtype=torch.long,
                device=self.device,
            )
            selected_output = backbone(
                input_ids=selected_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = selected_output.past_key_values
            next_logits = lm_head(selected_output.last_hidden_state[:, -1, :])

        return generated_ids

    def _backbone(self) -> Any:
        backbone = getattr(self.model, "model", None)
        if backbone is None:
            backbone = getattr(self.model, "base_model", None)
        if backbone is None:
            raise RuntimeError("Could not locate the causal LM backbone.")
        return backbone

    def _candidate_hidden_states(
        self,
        backbone: Any,
        candidate_ids: Any,
        past_key_values: Any,
    ) -> Any:
        chunks: list[Any] = []
        for start in range(
            0,
            int(candidate_ids.shape[0]),
            self.config.candidate_batch_size,
        ):
            chunk = candidate_ids[
                start : start + self.config.candidate_batch_size
            ]
            repeated_cache = _repeat_cache(
                past_key_values,
                int(chunk.shape[0]),
            )
            output = backbone(
                input_ids=chunk.reshape(-1, 1),
                past_key_values=repeated_cache,
                use_cache=False,
                return_dict=True,
            )
            chunks.append(output.last_hidden_state[:, -1, :])
            del output, repeated_cache
        return self.torch.cat(chunks, dim=0)

    def _cache_avoidance_representations(
        self,
        avoidance_texts: Sequence[str],
        *,
        shared_context_cache: Any,
    ) -> tuple[list[Any], Any]:
        """Re-encode history as hypothetical responses under the current prompt."""

        hidden_by_text: list[Any] = []
        backbone = self._backbone()
        for text in avoidance_texts:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.negative_max_tokens,
                return_tensors="pt",
            )
            negative_ids = encoded["input_ids"].to(self.device)
            negative_cache = _repeat_cache(shared_context_cache, 1)
            output = backbone(
                input_ids=negative_ids,
                past_key_values=negative_cache,
                use_cache=False,
                return_dict=True,
            )
            hidden_by_text.append(
                self.torch.nn.functional.normalize(
                    output.last_hidden_state[0].float(),
                    dim=-1,
                )
            )
            del output, negative_ids, negative_cache

        sentence_embeddings = self.guidance_encoder.encode(
            list(avoidance_texts),
            batch_size=len(avoidance_texts),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.config.guidance_device,
        )
        return hidden_by_text, sentence_embeddings.cpu().float()

    def _history_scores(
        self,
        candidate_norm: Any,
        candidate_ids: Any,
        generated_ids: Sequence[int],
        negative_hidden: Sequence[Any],
        negative_sentence: Any,
        *,
        step: int,
    ) -> Any:
        candidate_count = int(candidate_ids.shape[0])
        zeros = self.torch.zeros(
            candidate_count,
            device=self.device,
            dtype=self.torch.float32,
        )
        if not negative_hidden or negative_sentence is None:
            return zeros

        csp_columns = []
        for hidden in negative_hidden:
            csp_columns.append(
                self.torch.max(
                    candidate_norm @ hidden.transpose(0, 1),
                    dim=-1,
                ).values
            )
        csp_by_item = self.torch.stack(csp_columns, dim=-1)

        candidate_texts = [
            self.tokenizer.decode(
                list(generated_ids) + [int(token_id)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for token_id in candidate_ids.tolist()
        ]
        candidate_sentence = self.guidance_encoder.encode(
            candidate_texts,
            batch_size=len(candidate_texts),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.config.guidance_device,
        ).cpu().float()
        nsp_by_item = (
            candidate_sentence @ negative_sentence.transpose(0, 1)
        ).to(self.device)

        gamma = schedule_csp_weight(
            step,
            delta=self.config.schedule_delta,
            t0=self.config.schedule_t0,
        )
        hybrid_by_item = (
            gamma * csp_by_item + (1.0 - gamma) * nsp_by_item
        )
        return self.config.avoidance_beta * self.torch.max(
            hybrid_by_item,
            dim=-1,
        ).values


def _repeat_cache(cache: Any, repeats: int) -> Any:
    """Return a storage-independent cache with a repeated batch dimension."""

    if repeats < 1:
        raise ValueError("Cache repetition count must be positive.")
    if hasattr(cache, "batch_repeat_interleave"):
        cloned = copy.deepcopy(cache)
        if repeats > 1:
            cloned.batch_repeat_interleave(repeats)
        return cloned
    if isinstance(cache, (tuple, list)):
        repeated_layers = []
        for layer in cache:
            repeated_layers.append(
                tuple(
                    tensor.repeat_interleave(repeats, dim=0)
                    for tensor in layer
                )
            )
        return tuple(repeated_layers)
    raise TypeError(
        f"Unsupported KV cache type for candidate batching: {type(cache)!r}"
    )
