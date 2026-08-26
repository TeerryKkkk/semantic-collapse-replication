"""Run the conversational HC-AD intervention.

The bundled, hash-verified baseline supplies prompts, memory/RAG, judge,
routing, and round structure. This runner replaces only free-agent generation
with HC-AD.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import ModuleType

from avoidance_history import AVOIDANCE_BANK_SIZE
from hc_ad_decoder import HCADConfig, HCADDecoder
from hc_ad_integration import (
    configure_host_runtime,
    make_hc_ad_agent_class,
    make_hc_ad_environment_class,
)


BUNDLED_HOST_PATH = Path(__file__).with_name("standard_run.py")
STANDARD_RUN_SHA256 = (
    "609aea51c29e629a24e04f0d9ee9c2f0f7587069eb513a3c0af6213daa851d50"
)


def load_host_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Host simulation not found: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != STANDARD_RUN_SHA256:
        raise RuntimeError(
            "Host simulation hash mismatch. "
            f"Expected {STANDARD_RUN_SHA256}, got {actual_hash}: {path}"
        )
    if not hasattr(builtins, "AZURE_ENDPOINT"):
        builtins.AZURE_ENDPOINT = os.getenv(
            "AZURE_ENDPOINT",
            "https://example.invalid",
        )
    spec = importlib.util.spec_from_file_location(
        "_hc_ad_baseline_host_simulation",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load host simulation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        module.OPENAI_API_KEY = key
        module.GPT_CLIENT = module.OpenAI(api_key=key)
        module.client_embed = module.OpenAI(api_key=key)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HC-AD intervention."
    )
    parser.add_argument(
        "--host-simulation",
        type=Path,
        default=BUNDLED_HOST_PATH,
        help=(
            "Hash-matching baseline simulation module; defaults to the "
            "bundled standard_run.py."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Fresh directory for runtime artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--quantization",
        choices=("nf4", "bf16", "none"),
        default="nf4",
    )
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Independent replicate seed; use a different value for each run.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "YOUR_OPENAI_API_KEY":
        raise RuntimeError(
            "OPENAI_API_KEY is required by the bundled judge and RAG host."
        )
    if args.rounds != 200:
        raise ValueError("The manuscript HC-AD condition uses exactly 200 rounds.")
    if args.run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; use a fresh path: {args.run_dir}"
        )

    args.run_dir.mkdir(parents=True)
    transcript_path = args.run_dir / "transcript.txt"
    mapping_path = args.run_dir / "agents_models.txt"
    rag_path = args.run_dir / "rag_database"
    memory_path = args.run_dir / "memory_logs"

    host = load_host_module(args.host_simulation.resolve())
    host.OUTPUT_LOG = str(transcript_path)
    host.MAPPING_LOG = str(mapping_path)
    host.MODEL_A = args.model_name
    host.MODEL_B = args.model_name
    host.MODEL_C = args.model_name
    host.REFEREE_MODEL = "gpt-4o-mini"
    host.RAG_DIR = str(rag_path)

    decoder = HCADDecoder(
        HCADConfig(
            model_name=args.model_name,
            quantization=args.quantization,
            seed=args.seed,
            max_new_tokens=200,
            adaptive_q=1.0,
            candidate_min=5,
            candidate_max=15,
            candidate_cap=15,
            candidate_batch_size=15,
            avoidance_beta=2.0,
            schedule_delta=0.5,
            schedule_t0=25.0,
            guidance_model_name="sentence-transformers/all-MiniLM-L6-v2",
            guidance_device="cpu",
            negative_max_tokens=256,
        )
    )
    decoder.load()
    configure_host_runtime(host, decoder)

    agent_class = make_hc_ad_agent_class(
        host,
        decoder,
        memory_log_dir=memory_path,
        avoidance_bank_size=AVOIDANCE_BANK_SIZE,
    )
    environment_class = make_hc_ad_environment_class(host, agent_class)

    random.seed(args.seed)
    decoder.torch.manual_seed(args.seed)
    decoder.torch.cuda.manual_seed_all(args.seed)
    names = [
        "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
        for _ in range(3)
    ]
    agent_list = [(name, args.model_name) for name in names]
    with mapping_path.open("w", encoding="utf-8") as handle:
        for name, model in agent_list:
            handle.write(f"{name}: {model}\n")

    environment = environment_class(model="gpt-4o-mini")
    environment.add_free_agents(agent_list)
    environment.add_referee("Judge", model="gpt-4o-mini")

    with transcript_path.open("w", encoding="utf-8") as output:
        previous_stdout = sys.stdout
        try:
            sys.stdout = output
            for _ in range(args.rounds):
                environment.action_log.clear()
                environment.run_round()
                environment.print_log()
        finally:
            sys.stdout = previous_stdout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
