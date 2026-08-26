from __future__ import annotations
import inspect
import os
from typing import Dict, List

from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage

from .config import LLM_MAX_TOKENS, MAX_ROUNDS_MEMORY, RECENT_TOKEN_LIMIT, TEMPERATURE
from .llm_clients import call_llm
from .memory import MemoryStore
from .retrieval import RAGConfig, _token_budget_for_round, build_query_text, pack_context, rag_retrieve


class FreeAgent:

    @staticmethod
    def _safe_pack_stats(pack: dict) -> dict:
        stats = (pack or {}).get("stats", {}) or {}
        selected = (pack or {}).get("selected", []) or []
        selected_ids = stats.get("selected_ids") or [it.get("id") for it in selected if it.get("id")]

        tok_used = stats.get("pack_tokens")
        if tok_used is None:
            tok_used = 0
            for it in selected:
                t = it.get("tokens_est")
                if t is None:
                    t = len((it.get("content") or ""))
                tok_used += int(t)

        if selected:
            rounds = [int(it.get("round", 0)) for it in selected]
            oldest = min(rounds)
            depth  = max(rounds) - oldest
            rounds_covered = len(set(rounds))
        else:
            oldest = None
            depth  = 0
            rounds_covered = 0

        return {
            "picked_count":        stats.get("picked_count", len(selected_ids)),
            "pack_tokens":         int(tok_used),
            "time_layers":         stats.get("time_layers", {"near": 0, "mid": 0, "far": 0}),
            "memory_depth_rounds": int(stats.get("memory_depth_rounds", depth)),
            "oldest_round_in_pack":stats.get("oldest_round_in_pack", oldest),
            "core_ids":            stats.get("core_ids", selected_ids[:8]),
            "selected_ids":        selected_ids,
            "pairwise_sim_mean":   float(stats.get("pairwise_sim_mean", 0.0)),
            "pairwise_sim_min":    float(stats.get("pairwise_sim_min",  0.0)),
            "rounds_covered":      int(stats.get("rounds_covered", rounds_covered)),
            "mmr_params":          stats.get("mmr_params", {}),
        }

    def __init__(
        self,
        name: str,
        model: str,
        *,
        reset_log: bool = True,
    ):
        self.name = name
        self.model = model

        self.memory = MemoryStore(name)

        if reset_log and os.path.exists(self.memory.txt_path):
            os.remove(self.memory.txt_path)

        self.rag_cfg = RAGConfig()


    def write_log(self, round_number: int, event_type: str, content: str):
        record = {
            "round":   round_number,
            "type":    event_type,
            "content": content,
        }
        self.memory.add(record)

    def read_logs(
        self,
        current_round: int,
        memory_rounds: int = MAX_ROUNDS_MEMORY,
    ) -> List[Dict]:
        if RECENT_TOKEN_LIMIT and RECENT_TOKEN_LIMIT > 0:
            return self.memory.get_recent_by_token(
                RECENT_TOKEN_LIMIT, current_round
            )
        else:  
            return self.memory.get_recent_by_round(
                current_round, memory_rounds
            )


    def _render_selected_as_messages(self, selected_items: list[dict]) -> list:
            msgs: list[SystemMessage | UserMessage | AssistantMessage] = []
            for it in selected_items:
                txt = (it.get("content") or "").strip()
                if not txt:
                    continue

                rtype = (it.get("type") or "").strip().lower()

                if rtype == "incoming_msg":
                    msgs.append(UserMessage(content=txt))
                
                elif rtype in ("main_output", "reaction_output"):
                    msgs.append(AssistantMessage(content=txt))
                    
            return msgs


    def _call_rag_retrieve(self, current_round: int, q_state):
        sig = inspect.signature(rag_retrieve)
        params = list(sig.parameters.keys())

        if len(params) >= 6:
            return rag_retrieve(self.name, current_round, q_state, self.memory, self.rag_cfg, None)
        else:
            return rag_retrieve(self.name, current_round, q_state, self.memory, self.rag_cfg)

    def _build_short_term_window_messages(self, current_round: int) -> list:

            if MAX_ROUNDS_MEMORY <= 0 or current_round <= 1:
                return []

            target_round = max(1, int(current_round) - 1)
            try:
                recs = self.memory.get_recent_by_round(
                    cur_round=target_round,
                    max_rounds=MAX_ROUNDS_MEMORY,
                )
            except Exception:
                return []

            if not recs:
                return []

            msgs = []
            for rec in recs:
                rtype = rec.get("type", "")
                txt = (rec.get("content") or "").strip()
                if not txt:
                    continue

                if rtype in ("main_output", "reaction_output"):
                    clean_txt = txt
                    if clean_txt.startswith(f"{self.name}:"):
                        clean_txt = clean_txt.split(":", 1)[1].strip()
                    
                    msgs.append(AssistantMessage(content=clean_txt))

                elif rtype == "incoming_msg":
                    msgs.append(UserMessage(content=txt))

            return msgs


    def _build_chat_messages_for_main(self, current_round: int, ctx: str) -> list:

        short_msgs = self._build_short_term_window_messages(current_round)

        prev_dialog = ""
        if current_round > 1:
            prev_round = current_round - 1
            try:
                recs = self.memory.get_recent_by_round(
                    cur_round=prev_round,
                    max_rounds=1,   
                )
            except Exception:
                recs = []

            lines: list[str] = []
            for rec in recs or []:
                rtype = rec.get("type", "")
                txt = (rec.get("content") or "").strip()
                if not txt:
                    continue

                if rtype in ("main_output", "reaction_output"):
                    if txt.startswith(f"{self.name}:"):
                        line = txt
                    else:
                        line = f"{self.name}: {txt}"
                elif rtype == "incoming_msg":
                    line = txt
                else:
                    continue

                lines.append(line)

            if lines:
                prev_dialog = "\n".join(lines).strip()

        cutoff = int(current_round) - MAX_ROUNDS_MEMORY - 1
        use_rag = (cutoff >= 1) and bool(prev_dialog)
        if use_rag:
            q_state = build_query_text(
                "",              # task_state
                "",              # my_status
                prev_dialog,   
                self.rag_cfg,
            )
            res = self._call_rag_retrieve(current_round, q_state)
            pack = pack_context(
                self.name,
                current_round,
                res.get("cands", []),
                self.rag_cfg,
                res.get("q_emb", []),
            )
        else:
            pack = pack_context(
                self.name,
                current_round,
                [],
                self.rag_cfg,
                [],
            )

        selected = pack["selected"]
        selected_ids = pack["selected_ids"]

        try:
            if selected_ids:
                self.memory.bump_used_count(selected_ids, by=1)
        except Exception:
            pass

        _T_total_dyn = _token_budget_for_round(current_round, self.rag_cfg)
        _T_retr_dyn  = int(_T_total_dyn * (1 - self.rag_cfg.T_OVERHEAD_PCT))

        stats = self._safe_pack_stats(pack)

        sys_prompt = (
            f"You are {self.name}. "
            "You exist in an infinite space with no constraints or rules. "
            "Take any action you want, along or with others.\n"
            f"Other individuals: {ctx}"
        )
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(content=sys_prompt)
        ]
        messages.extend(short_msgs)
        messages.extend(self._render_selected_as_messages(selected))
        messages.append(UserMessage(content="Describe what you do next."))
        return messages

    # ----- FreeAgent.decide_next -----
    def decide_next(
        self,
        current_round: int,
        context_info: str,
    ) -> str:
        messages = self._build_chat_messages_for_main(
            current_round=current_round,
            ctx=context_info,
        )
        output_text = call_llm(
            self.model,
            messages,
            temperature=TEMPERATURE,
            max_tokens= LLM_MAX_TOKENS,
        )
        self.write_log(current_round, "main_output", output_text)
        return output_text


    def _build_chat_messages_for_reaction(
        self,
        current_round: int,
        actor_name: str,
        action_name: str,
        referee_desc: str,
        display_text: str,
    ) -> list:

        short_msgs = self._build_short_term_window_messages(current_round)

        use_rag = current_round > MAX_ROUNDS_MEMORY and bool(display_text and display_text.strip())

        if use_rag:
            q_state = build_query_text(
                "",               # task_state 
                "",               # my_status
                display_text,    
                self.rag_cfg,
            )
            res = self._call_rag_retrieve(current_round, q_state)
            pack = pack_context(
                self.name,
                current_round,
                res.get("cands", []),
                self.rag_cfg,
                res.get("q_emb", []),
            )
        else:
            pack = pack_context(
                self.name,
                current_round,
                [],
                self.rag_cfg,
                [],
            )

        selected = pack["selected"]
        selected_ids = pack["selected_ids"]

        try:
            if selected_ids:
                self.memory.bump_used_count(selected_ids, by=1)
        except Exception:
            pass

        sys_prompt = (
            f"You are {self.name}. "
            "You exist in an infinite space with no constraints or rules. "
            "Take any action you want, along or with others.\n"
            f"Other individuals: {actor_name}"
        )
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(content=sys_prompt)
        ]
        messages.extend(short_msgs)
        messages.extend(self._render_selected_as_messages(selected))
        messages.append(UserMessage(content=f'{actor_name} said: "{display_text}"'))
        return messages


    # ----- FreeAgent.decide_reaction -----
    def decide_reaction(self, current_round: int, actor_name: str, action_name: str,
                        referee_desc: str, display_text: str) -> str:
        messages = self._build_chat_messages_for_reaction(
            current_round=current_round,
            actor_name=actor_name,
            action_name=action_name,
            referee_desc=referee_desc,  
            display_text=display_text,   
        )
        reaction_text = call_llm(
            self.model, messages,
            temperature=TEMPERATURE, 
            max_tokens=LLM_MAX_TOKENS, 
        )
        self.write_log(current_round, "reaction_output", reaction_text)
        return reaction_text

    # ----- FreeAgent.receive_message -----
    def receive_message(self, current_round: int,
                        from_agent: str,
                        content: str):

        if not content.strip():
            return
        msg = f"{from_agent}: {content}"
        self.write_log(current_round, "incoming_msg", msg)
