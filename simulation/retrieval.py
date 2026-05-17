from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List

from .config import MAX_TOKEN
from .llm_clients import _get_embedding
from .token_utils import count_tokens


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
