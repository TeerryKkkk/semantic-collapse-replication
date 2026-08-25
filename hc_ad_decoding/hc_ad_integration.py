"""Narrow HC-AD adapter for the bundled baseline multi-agent simulation.

The host owns prompts, RAG, memory writes, judge behavior, routing, and round
structure. This adapter replaces only the free-agent language-model call and
writes only the finally selected HC-AD utterance into host state.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from avoidance_history import (
    AVOIDANCE_BANK_SIZE,
    MEMORY_ROUNDS,
    build_avoidance_bank_from_log,
)
from hc_ad_decoder import HCADDecoder


def configure_host_runtime(host: ModuleType, decoder: HCADDecoder) -> None:
    """Preserve and verify the bundled baseline RAG configuration."""

    if decoder.tokenizer is None:
        raise RuntimeError("Load HCADDecoder before configuring the host.")
    if int(getattr(host, "MAX_ROUNDS_MEMORY", -1)) != MEMORY_ROUNDS:
        raise ValueError(
            f"The host must use MAX_ROUNDS_MEMORY={MEMORY_ROUNDS}."
        )
    if getattr(getattr(host, "enc", None), "name", None) != "cl100k_base":
        raise RuntimeError(
            "The bundled host must use the cl100k_base token counter."
        )
    embedding_defaults = getattr(
        getattr(host, "_get_embedding", None),
        "__kwdefaults__",
        {},
    ) or {}
    if embedding_defaults.get("model") != "text-embedding-3-large":
        raise RuntimeError(
            "The bundled host must default to text-embedding-3-large."
        )


def make_hc_ad_agent_class(
    host: ModuleType,
    decoder: HCADDecoder,
    *,
    memory_log_dir: Path,
    avoidance_bank_size: int = AVOIDANCE_BANK_SIZE,
) -> type:
    """Override only decide-next/reaction while preserving host state logic."""

    class HCADFreeAgent(host.FreeAgent):
        all_agent_names: tuple[str, ...] = ()

        def __init__(
            self,
            name: str,
            model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
            *,
            reset_log: bool = True,
        ) -> None:
            super().__init__(name=name, model=model, reset_log=False)
            memory_log_dir.mkdir(parents=True, exist_ok=True)
            self.memory.txt_path = str(memory_log_dir / f"{name}_log.txt")
            if reset_log and Path(self.memory.txt_path).exists():
                Path(self.memory.txt_path).unlink()

        def _decoder_bank(
            self,
            *,
            current_round: int,
            current_trigger: str,
        ) -> tuple[str, ...]:
            bank = build_avoidance_bank_from_log(
                self.memory.txt_path,
                current_round=current_round,
                memory_rounds=MEMORY_ROUNDS,
                bank_size=avoidance_bank_size,
                current_trigger=current_trigger,
                agent_names=self.all_agent_names,
            )
            return bank.texts

        def decide_next(
            self,
            current_round: int,
            context_info: str,
        ) -> str:
            messages = self._build_chat_messages_for_main(
                current_round=current_round,
                ctx=context_info,
            )
            selected_text = decoder.generate(
                messages,
                avoidance_texts=self._decoder_bank(
                    current_round=current_round,
                    current_trigger="Describe what you do next.",
                ),
            )
            self.write_log(current_round, "main_output", selected_text)
            return selected_text

        def decide_reaction(
            self,
            current_round: int,
            actor_name: str,
            action_name: str,
            referee_desc: str,
            display_text: str,
        ) -> str:
            messages = self._build_chat_messages_for_reaction(
                current_round=current_round,
                actor_name=actor_name,
                action_name=action_name,
                referee_desc=referee_desc,
                display_text=display_text,
            )
            selected_text = decoder.generate(
                messages,
                avoidance_texts=self._decoder_bank(
                    current_round=current_round,
                    current_trigger=display_text,
                ),
            )
            self.write_log(current_round, "reaction_output", selected_text)
            return selected_text

    return HCADFreeAgent


def make_hc_ad_environment_class(host: ModuleType, agent_class: type) -> type:
    class HCADEnvironment(host.Environment):
        def add_free_agents(self, agent_configs: list[tuple]) -> None:
            names = tuple(str(config[0]) for config in agent_configs)
            for config in agent_configs:
                if len(config) == 3:
                    name, model, reset = config
                else:
                    name, model = config
                    reset = True
                agent = agent_class(
                    name=name,
                    model=model,
                    reset_log=reset,
                )
                agent.all_agent_names = names
                self.agents.append(agent)

    return HCADEnvironment
