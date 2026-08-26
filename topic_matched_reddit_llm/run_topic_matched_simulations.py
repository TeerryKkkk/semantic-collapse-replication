"""Run the topic-matched three-agent simulations for the four manuscript models.

Parallelism is restricted to independent model batches.  Within each model, Reddit
questions are processed serially; within each question, agent turns, referee calls,
reactions, routing, and memory operations remain serial and follow the simulation core.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import string
import time
from typing import List

import tiktoken

import multiagent_simulation as sim
from model_providers import MODEL_SPECS, ProviderRegistry, get_model_spec
from protocol import (
    CONTINUATION_TOKENS,
    MAX_TOKENS_PER_GENERATION,
    N_AGENTS,
    RAG_TOKEN_BUDGET,
    REFEREE_MODEL,
    SHORT_TERM_MEMORY_ROUNDS,
    SIMULATION_SEED,
    TEMPERATURE,
    TOKENIZER_NAME,
)

ROOT = Path(__file__).resolve().parent


class BudgetReached(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, default=ROOT / "outputs" / "selection" / "selection_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected questions per model (0=all).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def load_selection(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Selection manifest not found: {path}. Run sample_reddit_discussions.py first.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row.get("selected_rank") or 10**9))
    if not rows:
        raise RuntimeError("Selection manifest is empty.")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class BudgetTracker:
    def __init__(self, *, run_dir: Path, budget: int, model_key: str):
        self.run_dir = run_dir
        self.budget = int(budget)
        self.model_key = model_key
        self.encoder = tiktoken.get_encoding(TOKENIZER_NAME)
        self.utterances: list[dict] = []
        self.texts: list[str] = []
        self.current_tokens = 0
        self.jsonl_path = run_dir / "llm_utterances.jsonl"
        self.progress_path = run_dir / "progress.json"
        self._write_progress(round_number=0, status="running")

    @property
    def complete(self) -> bool:
        return self.current_tokens >= self.budget

    def raise_if_complete(self) -> None:
        if self.complete:
            raise BudgetReached

    def _write_progress(self, *, round_number: int, status: str) -> None:
        _atomic_json(self.progress_path, {
            "status": status,
            "model_key": self.model_key,
            "tokens": min(self.current_tokens, self.budget),
            "raw_tokens_so_far": self.current_tokens,
            "budget": self.budget,
            "utterances": len(self.utterances),
            "round": int(round_number),
            "updated_unix": time.time(),
        })

    def add(self, *, round_number: int, event_type: str, actor: str, text: str) -> None:
        clean = (text or "").strip()
        if not clean:
            return
        self.texts.append(clean)
        token_ids = self.encoder.encode("\n".join(self.texts))
        self.current_tokens = len(token_ids)
        record = {
            "occurrence": len(self.utterances) + 1,
            "round": int(round_number),
            "type": event_type,
            "actor": actor,
            "text": clean,
            "stream_tokens_after": self.current_tokens,
        }
        self.utterances.append(record)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_progress(round_number=round_number, status="running")

    def finalize(self) -> dict:
        raw_text = "\n".join(self.texts)
        raw_ids = self.encoder.encode(raw_text)
        if len(raw_ids) < self.budget:
            raise RuntimeError(f"Run ended before reaching the {self.budget}-token continuation budget.")
        analysis_ids = raw_ids[: self.budget]
        (self.run_dir / "llm_raw_stream.txt").write_text(raw_text, encoding="utf-8")
        (self.run_dir / "llm_raw_stream.token_ids.json").write_text(json.dumps(raw_ids), encoding="utf-8")
        (self.run_dir / f"llm_analysis_stream_{self.budget}.txt").write_text(self.encoder.decode(analysis_ids), encoding="utf-8")
        (self.run_dir / f"llm_analysis_stream_{self.budget}.token_ids.json").write_text(json.dumps(analysis_ids), encoding="utf-8")
        self._write_progress(round_number=self.utterances[-1]["round"], status="complete")
        return {
            "utterance_count": len(self.utterances),
            "raw_tokens": len(raw_ids),
            "analysis_tokens": len(analysis_ids),
        }


def deterministic_agent_names(seed: int) -> List[str]:
    rng = random.Random(seed)
    names: list[str] = []
    while len(names) < N_AGENTS:
        name = "".join(rng.choices(string.ascii_uppercase, k=5))
        if name not in names:
            names.append(name)
    return names


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def inject_seed_at_round_one(messages: list, *, current_round: int, seed_text: str) -> list:
    """Insert the Reddit title once, immediately before the ordinary terminal user message."""
    if int(current_round) != 1:
        return messages
    insertion = sim.UserMessage(content=seed_text)
    if messages:
        messages.insert(len(messages) - 1, insertion)
    else:
        messages.append(insertion)
    return messages


def make_matched_environment(*, seed_text: str, tracker: BudgetTracker):
    class MatchedFreeAgent(sim.FreeAgent):
        def __init__(self, name: str, model: str, *, reset_log: bool = True):
            super().__init__(name=name, model=model, reset_log=reset_log)
            self._matched_seed = seed_text
            self._tracker = tracker
            # The seed is an ordinary round-1 incoming memory record.  It is not
            # re-injected on later rounds; subsequent access follows the standard
            # short-term-memory and RAG mechanisms.
            self.write_log(1, "incoming_msg", f"InitialPrompt: {self._matched_seed}")

        def _build_chat_messages_for_main(self, current_round: int, ctx: str) -> list:
            messages = super()._build_chat_messages_for_main(current_round, ctx)
            return inject_seed_at_round_one(messages, current_round=current_round, seed_text=self._matched_seed)

        def _build_chat_messages_for_reaction(self, current_round: int, actor_name: str, action_name: str, referee_desc: str, display_text: str) -> list:
            messages = super()._build_chat_messages_for_reaction(current_round, actor_name, action_name, referee_desc, display_text)
            return inject_seed_at_round_one(messages, current_round=current_round, seed_text=self._matched_seed)

        def decide_next(self, current_round: int, context_info: str) -> str:
            self._tracker.raise_if_complete()
            text = super().decide_next(current_round, context_info)
            self._tracker.add(round_number=current_round, event_type="main", actor=self.name, text=text)
            return text

        def decide_reaction(self, current_round: int, actor_name: str, action_name: str, referee_desc: str, display_text: str) -> str:
            self._tracker.raise_if_complete()
            text = super().decide_reaction(current_round, actor_name, action_name, referee_desc, display_text)
            self._tracker.add(round_number=current_round, event_type="reaction", actor=self.name, text=text)
            return text

    class MatchedEnvironment(sim.Environment):
        def add_free_agents(self, agent_configs: list[tuple]):
            for config in agent_configs:
                name, model = config[:2]
                reset = config[2] if len(config) == 3 else True
                self.agents.append(MatchedFreeAgent(name=name, model=model, reset_log=reset))

    return MatchedEnvironment


def status_matches(status: dict, *, model_key: str, seed_text: str) -> bool:
    spec = get_model_spec(model_key)
    return (
        status.get("status") == "complete"
        and status.get("model_key") == model_key
        and status.get("model") == spec.model_id
        and status.get("seed") == seed_text
        and status.get("tokenizer") == TOKENIZER_NAME
        and int(status.get("continuation_token_budget", -1)) == CONTINUATION_TOKENS
        and float(status.get("temperature", -1)) == TEMPERATURE
        and int(status.get("max_tokens_per_generation", -1)) == MAX_TOKENS_PER_GENERATION
        and status.get("referee_model") == REFEREE_MODEL
    )


def run_one(row: dict, *, model_key: str, output_root: Path, providers: ProviderRegistry) -> dict:
    spec = get_model_spec(model_key)
    thread_id = row["thread_id"]
    seed_text = row["title"].strip()
    run_dir = output_root / model_key / thread_id
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            if not status_matches(status, model_key=model_key, seed_text=seed_text):
                raise RuntimeError(f"Refusing to overwrite protocol-mismatched completed run: {run_dir}")
            return {"thread_id": thread_id, "status": "skipped_complete"}
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    tracker = BudgetTracker(run_dir=run_dir, budget=CONTINUATION_TOKENS, model_key=model_key)
    sim.configure_runtime(
        llm_caller=providers.generate,
        embedding_caller=providers.embed,
        rag_dir=str((run_dir / "rag_database").resolve()),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_PER_GENERATION,
        short_term_memory_rounds=SHORT_TERM_MEMORY_ROUNDS,
        rag_token_budget=RAG_TOKEN_BUDGET,
    )
    MatchedEnvironment = make_matched_environment(seed_text=seed_text, tracker=tracker)
    random.seed(SIMULATION_SEED)
    names = deterministic_agent_names(SIMULATION_SEED)
    agent_list = [(name, spec.model_id) for name in names]
    metadata = {
        "status": "running",
        "protocol": "topic_matched_reddit_llm_v1",
        "thread_id": thread_id,
        "seed": seed_text,
        "model_key": model_key,
        "model_label": spec.label,
        "provider": spec.provider,
        "model": spec.model_id,
        "referee_model": REFEREE_MODEL,
        "rag_embedding_model": "text-embedding-3-large",
        "temperature": TEMPERATURE,
        "max_tokens_per_generation": MAX_TOKENS_PER_GENERATION,
        "continuation_token_budget": CONTINUATION_TOKENS,
        "tokenizer": TOKENIZER_NAME,
        "simulation_random_seed": SIMULATION_SEED,
        "agent_names": names,
        "seed_injection": "shared Reddit title inserted as a round-1 user message and written once to ordinary memory",
        "topic_reminder_after_round_1": False,
        "simulation_core_sha256": _sha256(ROOT / "multiagent_simulation.py"),
        "provider_adapter_sha256": _sha256(ROOT / "model_providers.py"),
    }
    _atomic_json(status_path, metadata)

    start = time.time()
    environment = None
    try:
        with pushd(run_dir):
            random.seed(SIMULATION_SEED)
            environment = MatchedEnvironment()
            environment.add_free_agents(agent_list)
            environment.add_referee("Judge", model=REFEREE_MODEL)
            with (run_dir / "transcript.txt").open("w", encoding="utf-8") as transcript, contextlib.redirect_stdout(transcript):
                while not tracker.complete:
                    environment.action_log.clear()
                    try:
                        environment.run_round()
                    except BudgetReached:
                        environment.print_log()
                        break
                    environment.print_log()
    except Exception as exc:
        metadata.update({
            "status": "incomplete_restart_required",
            "error": f"{type(exc).__name__}: {exc}",
            "generated_tokens_checkpoint": tracker.current_tokens,
            "utterance_count_checkpoint": len(tracker.utterances),
            "elapsed_seconds": time.time() - start,
        })
        _atomic_json(status_path, metadata)
        tracker._write_progress(round_number=int(environment.round_number if environment else 0), status="failed")
        raise

    final = tracker.finalize()
    metadata.update(final)
    metadata.update({
        "status": "complete",
        "rounds_entered": int(environment.round_number if environment else 0),
        "elapsed_seconds": time.time() - start,
    })
    _atomic_json(status_path, metadata)
    return {"thread_id": thread_id, "status": "complete", **final}


def run_model_batch(payload: dict) -> dict:
    model_key = payload["model_key"]
    providers = ProviderRegistry(selected_model_keys=[model_key])
    output_root = Path(payload["output_root"])
    results, failures = [], []
    for row in payload["rows"]:
        try:
            results.append(run_one(row, model_key=model_key, output_root=output_root, providers=providers))
        except Exception as exc:
            failures.append({"thread_id": row.get("thread_id"), "error": f"{type(exc).__name__}: {exc}"})
    return {"model_key": model_key, "results": results, "failures": failures}


def _progress(output_root: Path, model_key: str, thread_id: str) -> tuple[int, str]:
    run_dir = output_root / model_key / thread_id
    status_path = run_dir / "status.json"
    progress_path = run_dir / "progress.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                return CONTINUATION_TOKENS, "complete"
            if status.get("status") == "incomplete_restart_required":
                return min(int(status.get("generated_tokens_checkpoint", 0)), CONTINUATION_TOKENS), "failed"
        except Exception:
            pass
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            return min(int(progress.get("tokens", 0)), CONTINUATION_TOKENS), str(progress.get("status", "running"))
        except Exception:
            pass
    return 0, "pending"


def monitor(futures, *, model_keys: list[str], rows: list[dict], output_root: Path, poll_seconds: float) -> list[dict]:
    jobs = [(model_key, row["thread_id"]) for model_key in model_keys for row in rows]
    total = len(jobs) * CONTINUATION_TOKENS
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    bar = tqdm(total=total, unit="tok", unit_scale=True, desc="topic-matched runs") if tqdm else None
    pending = set(futures)
    future_to_model = {future: model_keys[i] for i, future in enumerate(futures)}
    summaries = []
    last = 0
    while pending:
        done, pending = wait(pending, timeout=max(0.2, poll_seconds), return_when=FIRST_COMPLETED)
        for future in done:
            model_key = future_to_model[future]
            try:
                summaries.append(future.result())
            except Exception as exc:
                summaries.append({"model_key": model_key, "results": [], "failures": [{"thread_id": "<model_batch>", "error": repr(exc)}]})
        current = 0
        complete = failed = 0
        by_model = {key: 0 for key in model_keys}
        for model_key, thread_id in jobs:
            tokens, state = _progress(output_root, model_key, thread_id)
            current += tokens
            by_model[model_key] += tokens
            complete += state == "complete"
            failed += state == "failed"
        if bar:
            if current > last:
                bar.update(current - last)
            last = current
            per_model = " ".join(f"{key}={100*by_model[key]/(len(rows)*CONTINUATION_TOKENS):.0f}%" for key in model_keys)
            bar.set_postfix_str(f"{per_model} complete={complete}/{len(jobs)} failed={failed}")
    if bar:
        bar.close()
    return summaries


def main() -> None:
    args = parse_args()
    rows = load_selection(args.selection_manifest.resolve())
    if args.limit > 0:
        rows = rows[: args.limit]
    model_keys = list(dict.fromkeys(args.models))
    print("Frozen protocol: N=3, temperature=0.9, max_tokens=200, 20,000 cl100k_base continuation tokens")
    print("Topic condition: Reddit title is supplied once at round 1; no later explicit topic reminder")
    print("Models:")
    for key in model_keys:
        spec = get_model_spec(key)
        print(f"  - {spec.label}: {spec.provider} | {spec.model_id}")
    print(f"Questions/model: {len(rows)}; total runs: {len(rows) * len(model_keys)}")
    if args.dry_run:
        print("DRY RUN: no API calls made.")
        return
    # Validate credentials in the parent process before starting long jobs.
    ProviderRegistry(selected_model_keys=model_keys)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = [{"model_key": key, "rows": rows, "output_root": str(output_root)} for key in model_keys]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=context) as pool:
        futures = [pool.submit(run_model_batch, payload) for payload in payloads]
        summaries = monitor(futures, model_keys=model_keys, rows=rows, output_root=output_root, poll_seconds=args.poll_seconds)
    failures = [failure for summary in summaries for failure in summary.get("failures", [])]
    _atomic_json(output_root / "run_summary.json", {"summaries": summaries, "completed_unix": time.time()})
    for summary in sorted(summaries, key=lambda row: row["model_key"]):
        complete = sum(result.get("status") == "complete" for result in summary.get("results", []))
        skipped = sum(result.get("status") == "skipped_complete" for result in summary.get("results", []))
        print(f"{get_model_spec(summary['model_key']).label}: complete={complete}, skipped={skipped}, failed={len(summary.get('failures', []))}")
    if failures:
        for failure in failures:
            print(f"FAILED {failure['thread_id']}: {failure['error']}")
        raise SystemExit(f"{len(failures)} run(s) failed. Re-run the same command to restart only incomplete runs.")


if __name__ == "__main__":
    main()
