from .rope import RotaryEmbeddingESM

from .em_llm import em_llm_attn_forward, em_llm_causal_lm_forward

ATTN_FORWARD = {
    "em-llm": em_llm_attn_forward,
}

CAUSAL_LM_FORWARD = {
    "em-llm": em_llm_causal_lm_forward,
}

__all__ = ["RotaryEmbeddingESM", "ATTN_FORWARD"]