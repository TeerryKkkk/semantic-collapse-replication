from __future__ import annotations
import os
import random
import re
import string
import sys
from datetime import datetime

from .config import (
    CONTINUE_MODE,
    EXTRA_ROUNDS,
    MAPPING_LOG,
    MODEL_A,
    MODEL_B,
    MODEL_C,
    OUTPUT_LOG,
    REFEREE_MODEL,
    TOTAL_ROUND,
)
from .environment import Environment


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
        raise RuntimeError(f"Can not find mapping file: {map_path}.")

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
        raise RuntimeError(f"{map_path} is empty or no valid mapping found.")

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


def main():

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

    env = Environment()
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
