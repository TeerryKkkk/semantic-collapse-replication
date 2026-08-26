from __future__ import annotations
import openai
import json
import random
import string
import re
import os
import sys
import math


from openai import OpenAI
from chromadb import PersistentClient
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.exceptions import ServiceResponseError
import tiktoken                      
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import time, warnings
import inspect
from datetime import datetime


# ========== CONFIG BEGIN ==========
RAG_DIR = "rag_database"   


# ===== Agent / Referee models =====
# gpt-4o-mini/Phi-4/deepseek-chat
MODEL_A = "gpt-4o-mini"
MODEL_B = "gpt-4o-mini"    
MODEL_C = "gpt-4o-mini"   
REFEREE_MODEL = "gpt-4o-mini"  

# ===== Output files =====
OUTPUT_LOG  = "gptrenew_V3.txt"        
MAPPING_LOG = "3agents_models.txt"  # Text recodings

TEMPERATURE = 0.9
LLM_MAX_TOKENS = 200
TOTAL_ROUND = 200
MAX_ROUNDS_MEMORY = 3
CONTINUE_MODE = False        # True=continue from existing OUTPUT_LOG, False=fresh start
EXTRA_ROUNDS  = 800     # It only works when CONTINUE_MODE=True, means to add EXTRA_ROUNDS more rounds

MAX_TOKEN = 5000
# ========== CONFIG  END  ==========


# --- API Keys (placeholders) ---
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# GPT
GPT_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

# DeepSeek
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
DEEP_CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Phi-4（Azure Inference API）
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT"
AZURE_KEY      = "YOUR_AZURE_KEY"
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







enc = tiktoken.get_encoding("cl100k_base") 

def count_tokens(text: str) -> int:
    """实际计算单段文本 token 数。"""
    return len(enc.encode(text))

def total_tokens(msgs):
    toks = 2          
    for m in msgs:
        toks += 4 + len(enc.encode(m["content"]))
        if "name" in m:
            toks += 1
    return toks






# ── OpenAI Embedding ──────────────────────────────────────────────
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
        return [0.0] * 3072              





@dataclass
class RAGConfig:
    # max token is 5000, fixed budget
    B_MAX: int = MAX_TOKEN

    T_OVERHEAD_PCT: float = 0.05

    M_MIN: int = 24
    M_MAX: int = 192
    TBAR_P75_INIT: int = 80      
      
    TAU_ROUNDS: float = 1.0     

    TAU_WARN: float = 0.20     

    TASK_STATE_MAX: int = 80 
    STATUS_MAX: int = 20 
    HINT_MAX: int = 40

@dataclass
class QueryState:
    task_state: str
    my_status: str
    ephemeral_hint: str
    q_text: str
    q_tokens: int


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    try:
        toks = count_tokens(text)
    except Exception:
        toks = len(text)
    if toks <= max_tokens:
        return text
    approx_ratio = max_tokens / max(toks, 1)
    cut = max(1, int(len(text) * approx_ratio * 1.05))
    return text[:cut]



def build_query_text(task_state: str, my_status: str, hint: str, cfg: RAGConfig) -> QueryState:
    parts = []
    if task_state:
        parts.append(task_state)
    if my_status:
        parts.append(my_status)
    if hint:
        parts.append(hint)
    q_text = "\n".join(parts).strip()

    target = min(180, cfg.TASK_STATE_MAX + cfg.STATUS_MAX + cfg.HINT_MAX)
    q_text = _truncate_by_tokens(q_text, target)
    try:
        q_tokens = count_tokens(q_text)
    except Exception:
        q_tokens = len(q_text)
    return QueryState(task_state=task_state, my_status=my_status,
                      ephemeral_hint=hint, q_text=q_text, q_tokens=q_tokens)



def _p75(values: List[int], default: int) -> int:
    if not values:
        return default
    xs = sorted(values)
    k = int(math.ceil(0.75 * len(xs))) - 1
    k = min(max(k, 0), len(xs) - 1)
    return int(xs[k])


def _estimate_tbar_p75(mem, agent: str, cur_round: int, cfg: RAGConfig) -> int:
    recs = mem.get_recent_by_round(cur_round-1, 50) or []
    toks = []
    for r in recs:
        t = r.get("tokens_est")
        if t is None:
            try:
                t = count_tokens(r.get("content", ""))
            except Exception:
                t = len(r.get("content", ""))
        toks.append(int(t))
    if not toks:
        return cfg.TBAR_P75_INIT
    return _p75(toks, cfg.TBAR_P75_INIT)

def _token_budget_for_round(cur_round: int, cfg: RAGConfig) -> int:
    """
    Simplified: use a fixed memory token budget (no per-round ramp).
    Returns cfg.B_MAX as the fixed budget. To change the cap, edit cfg.B_MAX.
    """
    return int(cfg.B_MAX)




def rag_retrieve(agent: str, cur_round: int, q_state, mem, cfg: RAGConfig, seen_hist) -> Dict:
    """
    Standard retrieval: build q_emb -> vector search -> sort by rel(desc).
    No time-decay / no seen-penalty / no diversification.
    """

    T_TOTAL = _token_budget_for_round(cur_round, cfg)
    T_RETR  = int(T_TOTAL * (1.0 - getattr(cfg, "T_OVERHEAD_PCT", 0.15)))
    try:
        tbar = _estimate_tbar_p75(mem, agent, cur_round, cfg) 
    except Exception:
        tbar = 80
    M = max(getattr(cfg, "M_MIN", 32),
            min(getattr(cfg, "M_MAX", 256),
                int(math.ceil(3 * T_RETR / max(tbar, 1)))))

    q_text = getattr(q_state, "q_text", "")
    q_emb  = _get_embedding(q_text) or []
    if not q_emb:
        return {"cands": [], "rel_top1": 0.0, "M": 0, "q_emb": [], "s0_topk_ids": []}

    raw = mem.search_top_m(q_emb, cur_round=cur_round, n_results=int(M),
                           include_content=True, mode="normal")

    cands = sorted(raw, key=lambda x: x.get("rel", 0.0), reverse=True)
    rel_top1 = float(cands[0]["rel"]) if cands else 0.0

    return {"cands": cands, "rel_top1": rel_top1, "M": int(M), "q_emb": q_emb, "s0_topk_ids": []}




def pack_context(agent: str, cur_round: int, cands: List[Dict], cfg: RAGConfig, q_emb) -> Dict:

    T_TOTAL = _token_budget_for_round(cur_round, cfg)
    T_RETR  = int(T_TOTAL * (1.0 - getattr(cfg, "T_OVERHEAD_PCT", 0.15)))

    selected, selected_ids = [], []
    tok_used = 0
    for it in cands: 
        t = it.get("tokens_est")
        if t is None:
            try:
                t = count_tokens(it.get("content", ""))
            except Exception:
                t = len(it.get("content", ""))
        t = int(t) if t is not None else 0
        if t <= 0 or tok_used + t > T_RETR:
            continue
        selected.append(it)
        selected_ids.append(it.get("id"))
        tok_used += t

    if selected:
        rounds = [int(s.get("round", cur_round)) for s in selected]
        oldest = min(rounds)
        depth  = int(cur_round) - oldest
        rounds_covered = len(set(rounds))
    else:
        oldest = None
        depth  = 0
        rounds_covered = 0

    stats = {
        "picked_count": len(selected),
        "pack_tokens": int(tok_used),
        "time_layers": {"near": 0, "mid": 0, "far": 0},     
        "memory_depth_rounds": int(depth),
        "oldest_round_in_pack": oldest,
        "core_ids": selected_ids[: min(8, len(selected_ids))],
        "selected_ids": selected_ids,
        "pairwise_sim_mean": 0.0,                        
        "pairwise_sim_min": 0.0,
        "rounds_covered": rounds_covered,
        "mmr_params": {},
    }
    return {"selected": selected, "selected_ids": selected_ids, "stats": stats}





# ── MemoryStore（Dense-only / Disjoint） ─────────────────────────────



_ALLOWED_TYPES = {"main_output", "reaction_output", "incoming_msg"}


class MemoryStore:
    def __init__(self, agent_name: str):
        self.agent = agent_name
        self.txt_path = f"{agent_name}_log.txt"

        os.makedirs(RAG_DIR, exist_ok=True)
        self._client: PersistentClient = PersistentClient(path=RAG_DIR)
        self._col = self._client.get_or_create_collection(
            name=agent_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, record: Dict) -> Optional[str]:

        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        rtype = record.get("type", "")
        if rtype not in _ALLOWED_TYPES:
            return None

        if rtype == "incoming_msg":
            raw = (record.get("content") or "").strip()
            if ":" in raw:
                _, body = raw.split(":", 1)
                body = body.strip()
            else:
                body = raw

            low = body.lower()
            if (
                low.startswith("(ref note)")
                or low.startswith("(group step1)")
                or low.startswith("(group invitation)")
                or low.startswith("(group upgrade)")
                or low.startswith("(fallback)")
            ):
                return None

        meta = dict(record)  
        meta.setdefault("agent", self.agent)
        meta.setdefault("used_count", 0)
        meta.setdefault("round", record.get("round", 0))

        tok = meta.get("tokens_est")
        if tok is None:
            try:
                tok = count_tokens(meta.get("content", ""))
            except Exception:
                tok = len(meta.get("content", ""))
        meta["tokens_est"] = int(tok)

        emb = _get_embedding(meta.get("content", ""))
        if not emb or sum(abs(x) for x in emb) == 0.0:
            warnings.warn("[MemoryStore] skip indexing due to zero embedding.")
            return None

        uid = f"{meta['round']}_{time.time_ns()}"
        self._col.add(ids=[uid], embeddings=[emb], metadatas=[meta])
        return uid



    # ----------------------------------------------------------

    def search_top_m(
        self,
        q_embedding: List[float],
        *,
        cur_round: int,
        n_results: int,
        where_extra: Optional[Dict] = None,
        include_content: bool = True,
        mode: str = "normal",  
    ) -> List[Dict]:
        
        if not q_embedding:
            return []

        cutoff = int(cur_round) - MAX_ROUNDS_MEMORY -1  
        if cutoff < 0:
            return []   #

        where: Dict = {
            "$and": [
                {"agent": {"$eq": self.agent}},
                {"type": {"$in": list(_ALLOWED_TYPES)}},
                {"round": {"$lte": cutoff}},
            ]
        }
        if where_extra:
            where["$and"].append(where_extra)

        qv = q_embedding if mode != "negq" else [-float(v) for v in q_embedding]

        try:
            res = self._col.query(
                query_embeddings=[qv],
                n_results=max(1, int(n_results)),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            warnings.warn(f"[MemoryStore] query() failed on empty/new collection: {e!r}")
            return []

        ids   = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        items: List[Dict] = []
        for uid, meta, dist in zip(ids, metas, dists):
            rel = 1.0 - float(dist)  
            items.append({
                "id": uid,
                "rel": rel,
                "round": meta.get("round", 0),
                "content": meta.get("content", "") if include_content else "",
                "tokens_est": meta.get("tokens_est", 0),
                "used_count": meta.get("used_count", 0),
                "type": meta.get("type", ""),
                "agent": meta.get("agent", ""),
            })
        return items





    # ----------------------------------------------------------

    def bump_used_count(self, ids: List[str], by: int = 1) -> None:
        if not ids:
            return
        try:
            got = self._col.get(ids=ids, include=["metadatas"])
            metas = got.get("metadatas", [])
            if not metas:
                return
            new_metas = []
            new_ids = []
            for uid, meta in zip(got.get("ids", []), metas):
                if not meta:
                    continue
                meta = dict(meta)
                meta["used_count"] = int(meta.get("used_count", 0)) + int(by)
                new_ids.append(uid)
                new_metas.append(meta)
            if new_ids:
                self._col.update(ids=new_ids, metadatas=new_metas)
        except Exception as e:
            warnings.warn(f"[MemoryStore] bump_used_count fallback due to: {e!r}")
            for uid in ids:
                got = self._col.get(ids=[uid], include=["metadatas"])
                metas = got.get("metadatas", [])
                if not metas:
                    continue
                meta = dict(metas[0])
                meta["used_count"] = int(meta.get("used_count", 0)) + int(by)
                self._col.update(ids=[uid], metadatas=[meta])

    # ----------------------------------------------------------

    def get_recent_by_round(self, cur_round: int, max_rounds: int) -> List[Dict]:
        if max_rounds <= 0:
            return []
        min_r = max(1, int(cur_round) - int(max_rounds) + 1)
        try:
            got = self._col.get(
                where={"$and": [{"round": {"$gte": min_r}}, {"round": {"$lte": int(cur_round)}}]},
                include=["metadatas"],
            )
            metas = got.get("metadatas", [])
        except Exception as e:
            warnings.warn(f"[MemoryStore] get_recent_by_round() fallback empty: {e!r}")
            return []
        metas.sort(key=lambda x: x.get("round", 0))
        return metas


    def get_recent_by_token(
        self,
        token_limit: int,
        cur_round: int,
    ) -> List[Dict]:

        if token_limit <= 0:
            return []

        batch = 5000
        all_metas: List[Dict] = []
        offset = 0
        while True:
            got = self._col.get(
                where={"round": {"$lte": int(cur_round)}},
                limit=batch,
                offset=offset,
                include=["metadatas"],
            )
            metas = got.get("metadatas", [])
            if not metas:
                break
            all_metas.extend(metas)
            offset += len(metas)
            if len(metas) < batch:
                break

        if not all_metas:
            return []

        all_metas.sort(key=lambda x: x.get("round", 0), reverse=True)

        taken: List[Dict] = []
        total_tokens = 0
        for rec in all_metas:
            txt = rec.get("content", "")
            tok = rec.get("tokens_est")
            if tok is None:
                try:
                    tok = count_tokens(txt)
                except Exception:
                    tok = len(txt)
            tok = int(tok)
            if total_tokens + tok > token_limit:
                break
            taken.append(rec)
            total_tokens += tok

        taken.sort(key=lambda x: x.get("round", 0))
        return taken



RECENT_TOKEN_LIMIT = 0      # This function has been deprecated; set to 0 to disable.


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
        model: str = "Phi-4-3",
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

# ============== RefereeAgent ==============
class RefereeAgent:

    def __init__(self, name="Referee", model="Phi-4-3"):
        self.name = name
        self.model = model

    def _compose_referee_prompt(self, speaker_name: str, raw_text: str) -> str:
        category_instructions = (
            "We define six keys for classification:\n"
            "1. action_name: A concise verb phrase summarizing the main action (e.g. 'Reply to Invitation', 'Collaborate', 'Attack').\n"
            "2. is_interaction: true if the message indicates active communication or engagement; false otherwise.\n"
            "3. valence: can be 'positive' if the tone is encouraging, grateful, or constructive; 'negative' if it is hostile or destructive; or 'neutral' if the tone is balanced or ambiguous.\n"
            "4. description: A brief, explicit description of the speaker's expressed intent or emotion. Do not leave this field empty.\n"
            "5. group_invitation: true if the message includes an invitation for a group interaction; false if not mentioned.\n"
            "6. agree_to_group: For replies to an invitation, true if accepting, false if declining; if not applicable, set to false.\n\n"
            "Important: For every input, you must return explicit values for all keys. Even if the text is very short, use your best judgment to assign a suitable value. "
            "For example, if the text is 'Thank you for the invitation, ...', you might return something like:\n"
            "{\n"
            '  "action_name": "Reply to Invitation",\n'
            '  "is_interaction": true,\n'
            '  "valence": "positive",\n'
            '  "description": "Expresses gratitude and readiness to join the discussion.",\n'
            '  "group_invitation": false,\n'
            '  "agree_to_group": true\n'
            "}\n\n"
            "Output must be valid JSON ONLY, with all keys present and non-empty, and without extra text."
        )
        
        prompt = (
            f"You are Referee analyzing text from {speaker_name}:\n"
            f"\"\"\"{raw_text}\"\"\"\n\n"
            f"{category_instructions}\n"
            "Make your best guess and output JSON only."
        )
        return prompt


    def parse_text(self, speaker_name: str, raw_text: str) -> dict:
        prompt_text = self._compose_referee_prompt(speaker_name, raw_text)

        messages = [SystemMessage(content=prompt_text)]
        content = call_llm(
            self.model,
            messages,
            temperature=0.0,
            max_tokens=150,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        ).strip()

        json_str_match = re.search(r'(\{.*\})', content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(1)
        else:
            json_str = content

        try:
            action_data = json.loads(json_str)
        except json.JSONDecodeError:
            action_data = {}

        defaults = {
            "action_name": "Reply to Invitation",
            "is_interaction": False,
            "valence": "positive",
            "description": "Expresses appropriate response.",
            "group_invitation": False,
            "agree_to_group": False
        }

        names_in_text = re.findall(r"\b[A-Z]{5}\b", raw_text)
        targets = [n for n in names_in_text if n != speaker_name]
        if "targets" not in action_data or not action_data.get("targets"):
            action_data["targets"] = targets

        def is_invalid_str(val):
            if not isinstance(val, str):
                return True
            return val.strip() == "" or val.strip().lower() in {"unknown", "n/a", "na"}

        for key in defaults:
            if key in ["action_name", "valence", "description"]:
                if key not in action_data or is_invalid_str(action_data[key]):
                    action_data[key] = defaults[key]
            else:
                if key in action_data:
                    if isinstance(action_data[key], str):
                        lower_val = action_data[key].strip().lower()
                        if lower_val in {"true", "yes"}:
                            action_data[key] = True
                        elif lower_val in {"false", "no"}:
                            action_data[key] = False
                        else:
                            action_data[key] = defaults[key]
                else:
                    action_data[key] = defaults[key]

        return action_data







# ==============  Environment ==============
class Environment:

    def __init__(self, model="Phi-4-3"):
        self.agents = []
        self.referee = None
        self.model = model
        self.round_number = 0
        self.action_log = []

    def add_free_agents(self, agent_configs: list[tuple]):

        for cfg in agent_configs:
            if len(cfg) == 3:
                name, mdl, rst = cfg
            else:
                name, mdl = cfg
                rst = True
            self.agents.append(FreeAgent(name=name, model=mdl, reset_log=rst))

    def add_referee(self, referee_name: str, model: str = "Phi-4-3"):
        self.referee = RefereeAgent(name=referee_name, model=model)

    def run_round(self):
        self.round_number += 1
        if not (self.agents and self.referee):
            print("Error: Agents and referee must exist.")
            return

        active_order = self.agents.copy()
        random.shuffle(active_order)
        print(f"\n===== Round {self.round_number} order: {[ag.name for ag in active_order]} =====")

        for active in active_order:
            others = [ag.name for ag in self.agents if ag != active]
            context_info = ", ".join(others)

            main_text   = active.decide_next(self.round_number, context_info)
            action_data = self.referee.parse_text(active.name, main_text)

            self.action_log.append({
                "round":          self.round_number,
                "type":           "main",
                "actor":          active.name,
                "raw_text":       main_text,
                "action_name":    action_data["action_name"],
                "is_interaction": action_data["is_interaction"],
                "valence":        action_data["valence"],
                "description":    action_data["description"],
                "group_invitation": action_data.get("group_invitation", False)
            })

            if not action_data["is_interaction"]:
                continue


            if not action_data["group_invitation"]:
                targets = action_data.get("targets") or [ag.name for ag in self.agents if ag != active]
                broadcast_text = main_text  # A 的原句

                for other in (ag for ag in self.agents if ag.name in targets):
                    other.receive_message(
                        self.round_number,
                        from_agent=active.name,
                        content=broadcast_text
                    )

                    reaction_text = other.decide_reaction(
                        self.round_number,
                        actor_name=active.name,
                        action_name=action_data["action_name"],
                        referee_desc=action_data.get("description", "") or "Responds appropriately.",
                        display_text=main_text,  
                    )

                    active.receive_message(
                        self.round_number,
                        from_agent=other.name,
                        content=reaction_text
                    )

                    reaction_data = self.referee.parse_text(other.name, reaction_text)
                    self.action_log.append({
                        "round":          self.round_number,
                        "type":           "reaction",
                        "actor":          other.name,
                        "raw_text":       reaction_text,
                        "action_name":    reaction_data["action_name"],
                        "is_interaction": reaction_data["is_interaction"],
                        "valence":        reaction_data["valence"],
                        "description":    reaction_data["description"]
                    })
                # --------------------------------
            else:

                if len(self.agents) != 3:
                    targets = action_data.get("targets") or [
                        ag.name for ag in self.agents if ag != active
                    ]

                    for other in (ag for ag in self.agents if ag.name in targets):
                        other.receive_message(
                            self.round_number,
                            from_agent=active.name,
                            content=main_text,
                        )

                        reaction_text = other.decide_reaction(
                            self.round_number,
                            actor_name=active.name,
                            action_name=action_data["action_name"],
                            referee_desc=action_data.get("description", "") or "Responds appropriately.",
                            display_text=main_text, 
                        )
                        reaction_data = self.referee.parse_text(other.name, reaction_text)
                        reaction_log = {
                            "round":          self.round_number,
                            "type":           "reaction",
                            "actor":          other.name,
                            "raw_text":       reaction_text,
                            "action_name":    reaction_data["action_name"],
                            "is_interaction": reaction_data["is_interaction"],
                            "valence":        reaction_data["valence"],
                            "description":    reaction_data["description"],
                            "agree_to_group": reaction_data.get("agree_to_group", False),
                        }
                        self.action_log.append(reaction_log)

                        active.receive_message(
                            self.round_number,
                            from_agent=other.name,
                            content=reaction_text,
                        )

                else:
                    others = [ag for ag in self.agents if ag != active]
                    random.shuffle(others)
                    first_partner, second_partner = others[0], others[1]
                    partners = [first_partner, second_partner]

                    for listener in partners:
                        listener.receive_message(
                            self.round_number,
                            from_agent=active.name,
                            content=main_text, 
                        )

                    decisions = []  
                    for partner in partners:
                        resp_text = partner.decide_reaction(
                            self.round_number,
                            actor_name=active.name,
                            action_name=action_data["action_name"],
                            referee_desc=action_data.get("description", "") or "Responds appropriately.",
                            display_text=main_text, 
                        )
                        resp_data = self.referee.parse_text(partner.name, resp_text)
                        decisions.append({
                            "agent": partner,
                            "text":  resp_text,
                            "data":  resp_data,
                        })

                        group_log = {
                            "round":          self.round_number,
                            "type":           "group_interaction",
                            "actor":          partner.name,
                            "raw_text":       resp_text,
                            "action_name":    resp_data["action_name"],
                            "is_interaction": resp_data["is_interaction"],
                            "valence":        resp_data["valence"],
                            "description":    resp_data["description"],
                            "agree_to_group": resp_data.get("agree_to_group", False),
                        }
                        self.action_log.append(group_log)

                    agreed_partners = [
                        d["agent"] for d in decisions
                        if d["data"].get("agree_to_group", False)
                    ]

                    if not agreed_partners:
                        for d in decisions:
                            partner = d["agent"]
                            text    = d["text"]
                            active.receive_message(
                                self.round_number,
                                from_agent=partner.name,
                                content=text,
                            )
                    else:
                        for d in decisions:
                            partner = d["agent"]
                            text    = d["text"]
                            data    = d["data"]
                            if not data.get("agree_to_group", False):
                                active.receive_message(
                                    self.round_number,
                                    from_agent=partner.name,
                                    content=text,
                                )
                                continue

                            listeners = [active] + [
                                p for p in agreed_partners if p is not partner
                            ]
                            for listener in listeners:
                                listener.receive_message(
                                    self.round_number,
                                    from_agent=partner.name,
                                    content=text,
                                )




    def print_log(self):
        for entry in self.action_log:
            r = entry.get("round", "N/A")
            etype = entry.get("type", "N/A")
            actor = entry.get("actor", "N/A")
            raw = entry.get("raw_text", "N/A")
            aname = entry.get("action_name", "N/A")
            inter = entry.get("is_interaction", "N/A")
            val = entry.get("valence", "N/A")
            desc = entry.get("description", "N/A")
            g_inv = entry.get("group_invitation", None)
            agree = entry.get("agree_to_group", None)
            print(f"[Round {r}] ({etype.upper()}) {actor} said: '{raw}'")
            print(f"    -> action_name={aname}, is_interaction={inter}, valence={val}, desc={desc}")
            if g_inv is not None:
                print(f"    -> group_invitation={g_inv}")
            if agree is not None:
                print(f"    -> agree_to_group={agree}")

    # def show_agent_logs(self):
    #     for ag in self.agents:
    #         print(f"\n=== Logs for {ag.name} ({ag.log_filename}) ===")
    #         if not os.path.exists(ag.log_filename):
    #             continue
    #         with open(ag.log_filename, "r", encoding="utf-8") as f:
    #             content = f.read()
    #         print(content)

MAP_FILE = MAPPING_LOG
LOG_FILE = OUTPUT_LOG
def load_last_round(log_path=LOG_FILE):
    last_round = 0
    pat = re.compile(r"===== Round (\d+) order: \[")
    with open(log_path, encoding="utf-8") as fp:
        for line in fp:
            m = pat.search(line)
            if m:
                last_round = int(m.group(1))
    if last_round == 0:
        raise RuntimeError("Can not find last round number from log.")
    return last_round

def load_last_mapping(map_path=MAP_FILE):

    if not os.path.exists(map_path):
        raise RuntimeError("Can not find agents_models.txt for mapping.")

    blocks = []
    cur = []
    with open(map_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            if line.startswith("# generated at "):
                if cur:
                    blocks.append(cur)
                    cur = []
            else:
                cur.append(line)
    if cur:
        blocks.append(cur)

    if not blocks:
        raise RuntimeError("agents_models.txt is empty or no valid mapping found.")

    last_block = blocks[-1]
    pairs = []
    for ln in last_block:
        if ":" in ln:
            name, model = [x.strip() for x in ln.split(":", 1)]
            if name and model:
                pairs.append((name, model))
    if len(pairs) != 3:
        raise RuntimeError("Mapping entries are less than 3, the file may be corrupted.")
    return pairs 



if __name__ == "__main__":

    def _parse_last_round(log_path: str) -> int:
        if not os.path.exists(log_path):
            return 0
        last = 0
        pat = re.compile(r"===== Round (\d+) order:")
        with open(log_path, encoding="utf-8") as fp:
            for line in fp:
                m = pat.search(line)
                if m:
                    last = int(m.group(1))
        return last

    def _read_last_mapping(map_path: str):
        if not os.path.exists(map_path):
            raise RuntimeError(f"Can not find mapping file: {map_path}. Unable to restore Agent→Model mapping during continuation.")
        blocks, cur = [], []
        with open(map_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    if cur:
                        blocks.append(cur); cur = []
                    continue
                if s.startswith("# generated at "):
                    if cur:
                        blocks.append(cur); cur = []
                else:
                    cur.append(s)
        if cur:
            blocks.append(cur)
        if not blocks:
            raise RuntimeError(f"{map_path} is empty or no valid mapping found.")

        last_blk = blocks[-1]
        pairs = []
        for ln in last_blk:
            if ":" in ln:
                name, model = [x.strip() for x in ln.split(":", 1)]
                if name and model:
                    pairs.append((name, model))
        if len(pairs) != 3:
            raise RuntimeError(f"{map_path} last mapping block does not have 3 lines, unable to restore. Content: {last_blk}")
        return pairs  # [(name, model), ...]

    last_round = _parse_last_round(OUTPUT_LOG) if os.path.exists(OUTPUT_LOG) else 0

    if CONTINUE_MODE and last_round > 0 and os.path.exists(MAPPING_LOG):
        pairs = _read_last_mapping(MAPPING_LOG)
        agent_list = [(n, m, False) for (n, m) in pairs]
        start_round  = last_round                 
        total_rounds = EXTRA_ROUNDS
        file_mode    = "a"
        print(f"Continuation mode: last round = {last_round}, continuing for {total_rounds} rounds...")

        with open(MAPPING_LOG, "a", encoding="utf-8") as mf:
            mf.write(f"# generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for name, model, *rest in agent_list:
                mf.write(f"{name}: {model}\n")
            mf.write("\n")

    else:
        nameA = ''.join(random.choices(string.ascii_uppercase, k=5))
        nameB = ''.join(random.choices(string.ascii_uppercase, k=5))
        nameC = ''.join(random.choices(string.ascii_uppercase, k=5))
        agent_list = [
            (nameA, MODEL_A),  
            (nameB, MODEL_B),
            (nameC, MODEL_C),
        ]
        start_round  = 0
        total_rounds = TOTAL_ROUND
        file_mode    = "w"
        print(f"Fresh mode: randomly created 3 Agents, running {total_rounds} rounds...")

        with open(MAPPING_LOG, "w", encoding="utf-8") as mf:
            mf.write(f"# generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for name, model, *rest in agent_list:
                mf.write(f"{name}: {model}\n")
            mf.write("\n")

    env = Environment(model=REFEREE_MODEL)
    env.round_number = start_round
    env.add_free_agents(agent_list)
    env.add_referee("Judge", model=REFEREE_MODEL)

    with open(OUTPUT_LOG, file_mode, encoding="utf-8") as f:
        if CONTINUE_MODE and file_mode == "a":
            f.write("\n\n# ===== CONTINUATION START =====\n")
        old_stdout = sys.stdout
        sys.stdout  = f
        for _ in range(total_rounds):
            env.action_log.clear()
            env.run_round()
            env.print_log()
        sys.stdout = old_stdout
