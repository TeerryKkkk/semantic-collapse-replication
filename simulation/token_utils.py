from __future__ import annotations
import tiktoken


enc = tiktoken.get_encoding("cl100k_base") 

def count_tokens(text: str) -> int:
    """Count tokens for one text segment."""
    return len(enc.encode(text))

def total_tokens(msgs):
    toks = 2          
    for m in msgs:
        toks += 4 + len(enc.encode(m["content"]))
        if "name" in m:
            toks += 1
    return toks
