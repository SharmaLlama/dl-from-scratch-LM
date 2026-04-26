"""
Model factory: builds a LanguageModel from an ExperimentConfig.

Add new paper implementations by importing them and extending _ATTENTION_REGISTRY
and _PE_REGISTRY. Each attention class must subclass BaseMultiHeadAttention;
each PE class must subclass BasePositionalEncoding.
"""

from __future__ import annotations

from core.attention.base import BaseMultiHeadAttention
from core.layers.utils import Projection
from core.model.decoder import Decoder
from core.model.decoder_block import DecoderBlock
from core.model.language_model import LanguageModel
from core.positional_encoding.base import BasePositionalEncoding, LearnedPE, SinusoidalPE
from papers.vanilla_attention.attention import VanillaMultiHeadAttention
from papers.big_bird.attention import SparseMultiHeadAttention
from training.configs.schemas import ExperimentConfig

# ── Registries ────────────────────────────────────────────────────────────────
# Add entries here as each paper's attention.py is implemented.

_ATTENTION_REGISTRY: dict[str, type[BaseMultiHeadAttention]] = {
    "vanilla": VanillaMultiHeadAttention,
    # "rope":          RoPEMultiHeadAttention,
    # "alibi":         ALiBiAttention,
    "big_bird":      SparseMultiHeadAttention,
    # "performer":     PerformerAttention,
    # "flash":         FlashAttention,
    # "nope":          NoPEAttention,
}

_PE_REGISTRY: dict[str, type[BasePositionalEncoding]] = {
    "sinusoidal": SinusoidalPE,
    "learned": LearnedPE,
    # "rope":   RoPEPositionalEncoding,   # returns None from _get_position_encoding
    # "alibi":  ALiBiPositionalEncoding,  # returns None; bias is in-attention
    # "none":   NoPE,
}


# ── Public entry point ────────────────────────────────────────────────────────

def build_model(cfg: ExperimentConfig) -> LanguageModel:
    m = cfg.model

    attn_cls = _get_registry_entry(_ATTENTION_REGISTRY, m.attention_type, "attention")
    pe_cls = _get_registry_entry(_PE_REGISTRY, m.pe_type, "positional encoding")

    embedding = pe_cls(
        vocab_size=m.vocab_size,
        d_model=m.d_model,
        max_seq_len=m.max_seq_len,
        dropout=m.dropout,
    )

    # Each layer gets its own attention instance (weights are NOT shared)
    blocks = [
        DecoderBlock(
            attention=attn_cls(
                n_heads=m.n_heads,
                d_model=m.d_model,
                dk=m.dk,
                dv=m.dv,
                dropout=m.dropout,
                **m.attention_kwargs,
            ),
            d_model=m.d_model,
            d_ff=m.d_ff,
            dropout=m.dropout,
            pre_norm=m.pre_norm,
            ffn_type=m.ffn_type,
        )
        for _ in range(m.n_layers)
    ]

    decoder = Decoder(blocks, m.d_model)
    projection = Projection(m.d_model, m.vocab_size)
    model = LanguageModel(embedding, decoder, projection)
    model.initialise()
    return model


def _get_registry_entry(registry: dict, key: str, label: str):
    if key not in registry:
        available = list(registry.keys())
        raise NotImplementedError(
            f"Unknown {label} type '{key}'. "
            f"Available: {available}. "
            f"Add '{key}' to _{'_'.join(label.upper().split())}_REGISTRY in training/factory.py."
        )
    return registry[key]
