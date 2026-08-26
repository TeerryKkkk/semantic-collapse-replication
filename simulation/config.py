from __future__ import annotations

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

RECENT_TOKEN_LIMIT = 0      # Set to 0 to disable this feature.
