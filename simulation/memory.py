from __future__ import annotations
import json
import os
import time
import warnings
from typing import Dict, List, Optional

from chromadb import PersistentClient

from .config import MAX_ROUNDS_MEMORY, RAG_DIR
from .llm_clients import _get_embedding
from .token_utils import count_tokens


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
