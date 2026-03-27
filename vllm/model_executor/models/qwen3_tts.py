# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The Qwen team, Alibaba Group.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3TTS Talker model compatible with HuggingFace weights."""

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.compilation.backends import set_model_tag
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    create_forward_context, get_forward_context, override_forward_context, BatchDescriptor
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# ── RoPE helpers for the native code predictor ──────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (standard RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard 1-D rotary position embeddings to Q and K.

    Args:
        q, k: [batch, num_heads, seq_len, head_dim]
        cos, sin: [1, 1, seq_len, head_dim]  (broadcastable)
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3TTSNativeRotaryEmbedding(nn.Module):
    """Simple 1-D rotary position embedding for the native code predictor.

    Matches the ``Qwen3TTSRotaryEmbedding`` in the original HF code, but
    simplified: no dynamic-rope, no MRoPE – just standard RoPE with a
    configurable ``rope_theta``.
    """

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # Use nn.Parameter so vLLM natively handles device/dtype casting.
        # requires_grad=False because this is deterministic and not trained.
        # The weight-loader already skips "rotary_emb.inv_freq".
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tensors for positions ``[0 .. seq_len)``.

        Returns:
            cos: [1, 1, seq_len, head_dim]
            sin: [1, 1, seq_len, head_dim]
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # [seq_len] x [head_dim/2] → [seq_len, head_dim/2]
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin


def _sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample tokens from logits with temperature, top-k, top-p, and repetition penalty.
    
    Args:
        logits: [..., vocab_size] - logits for token prediction
        do_sample: Whether to sample or use argmax
        temperature: Sampling temperature
        top_k: Top-k sampling (keep top k tokens)
        top_p: Top-p (nucleus) sampling
        repetition_penalty: Penalty for repeated tokens
        previous_tokens: [..., seq_len] - previously generated tokens for penalty
        
    Returns:
        tokens: [...] - sampled token indices
    """
    if repetition_penalty != 1.0 and previous_tokens is not None:
        # Apply repetition penalty
        # Create a copy to avoid modifying original logits in place if needed elsewhere
        # logits = logits.clone() 
        # But we can probably modify in place here as it's the last step
        
        # Check if previous_tokens has the same batch dim
        if previous_tokens.dim() == logits.dim(): # [batch, seq] vs [batch, vocab]
             # Handle standard case
             pass
        
        # We need to gather scores for the tokens that have appeared
        # This is a bit expensive if history is long, but for code predictor (16 tokens) it's fine.
        # For full history, we might skip implementation if context is missing.
        
        # Simple implementation for small context (like code predictor loop):
        # Gather scores for each token in previous_tokens
        score = torch.gather(logits, -1, previous_tokens)
        
        # Apply penalty: if score < 0 then score * penalty else score / penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        
        # Scatter back
        logits.scatter_(-1, previous_tokens, score)

    if not do_sample:
        return logits.argmax(dim=-1)
    
    logits = logits / max(temperature, 1e-6)
    
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < v[..., [-1]], float('-inf'), logits)
    
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = torch.where(indices_to_remove, float('-inf'), logits)
    
    probs = torch.softmax(logits, dim=-1)
    # Flatten for multinomial, then reshape back
    orig_shape = probs.shape[:-1]
    probs_flat = probs.view(-1, probs.size(-1))
    tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    return tokens_flat.view(orig_shape)


class Qwen3TTSTalkerMLP(nn.Module):
    """MLP for Qwen3TTS Talker - standard SwiGLU architecture."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3TTSTalkerResizeMLP(nn.Module):
    """Resize MLP for text projection in Qwen3TTS Talker.
    
    Maps from text_hidden_size to hidden_size with an intermediate layer.
    """

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            input_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
        )
        self.linear_fc2 = RowParallelLinear(
            intermediate_size,
            output_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
        )
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_fc1(x)
        x = self.act_fn(x)
        x, _ = self.linear_fc2(x)
        return x


class Qwen3TTSNativeAttention(nn.Module):
    """Native attention for Qwen3TTS using torch SDPA.
    
    Used for the code predictor which has deterministic shapes and doesn't
    benefit from KV caching. Can be captured in CUDA graphs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim else hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=qkv_bias)
        
        # QK normalization
        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass using torch SDPA.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_embeddings: Optional (cos, sin) tuple from rotary
                embedding, each [1, 1, seq_len, head_dim].
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose to [batch, num_heads, seq, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary position embeddings (standard 1-D RoPE)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand KV heads if using GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Apply scaled dot product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,  # Use causal if no mask provided
            scale=self.scaling,
        )
        
        # Reshape back to [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        output = self.o_proj(attn_output)
        return output


class Qwen3TTSTalkerAttention(nn.Module):
    """Multi-headed attention for Qwen3TTS Talker with MRoPE support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        max_position: int = 32768,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = head_dim if head_dim else hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        
        # QK normalization (like Qwen3)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Rotary embeddings with MRoPE support
        is_mrope = rope_scaling is not None and "mrope_section" in rope_scaling
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply QK normalization per head
        q = q.view(*q.shape[:-1], -1, self.head_dim)
        q = self.q_norm(q)
        q = q.view(*q.shape[:-2], -1)
        
        k = k.view(*k.shape[:-1], -1, self.head_dim)
        k = self.k_norm(k)
        k = k.view(*k.shape[:-2], -1)
        
        # Apply rotary embeddings
        # Expand positions to 3D for MRoPE (all dims get same values for TTS)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        q, k = self.rotary_emb(positions, q, k)
        
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3TTSNativeMLP(nn.Module):
    """Native MLP for Qwen3TTS Code Predictor using standard PyTorch layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TTSCodePredictorDecoderLayer(nn.Module):
    """Native decoder layer for Qwen3TTS Code Predictor.
    
    Uses native PyTorch attention (SDPA) instead of vLLM attention.
    This is more efficient for the code predictor since:
    - Shapes are deterministic (fixed 15 steps)
    - No KV cache benefit
    - Can be captured in CUDA graphs
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen3TTSNativeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
        )
        
        self.mlp = Qwen3TTSNativeMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self Attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask, position_embeddings
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    """Decoder layer for Qwen3TTS Talker."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        
        self.self_attn = Qwen3TTSTalkerAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        self.mlp = Qwen3TTSTalkerMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        self.input_layernorm = RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Keys whose values must stay as plain dicts (expected by downstream code)
_KEEP_AS_DICT_KEYS = {"rope_scaling"}


def _dict_to_namespace(d, _key: Optional[str] = None):
    """Recursively convert a dict to SimpleNamespace for attribute access.

    Certain keys (e.g. ``rope_scaling``) are kept as plain dicts because
    downstream code (``get_rope``, ``"mrope_section" in rope_scaling``, etc.)
    expects dict-like objects.
    """
    if isinstance(d, dict):
        if _key in _KEEP_AS_DICT_KEYS:
            return d  # keep as plain dict
        return SimpleNamespace(
            **{k: _dict_to_namespace(v, _key=k) for k, v in d.items()}
        )
    return d


def _get_tts_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    """Get the full TTS config if available, otherwise return None."""
    if hasattr(hf_config, "talker_config"):
        return hf_config
    return None


def _get_talker_config(hf_config: PretrainedConfig):
    """Get the talker config from either full TTS config or talker config directly.

    If talker_config is stored as a plain dict (from Qwen3TTSConfig),
    convert it to a namespace so attribute access (config.hidden_size etc.) works.
    """
    if hasattr(hf_config, "talker_config"):
        tc = hf_config.talker_config
        if isinstance(tc, dict):
            return _dict_to_namespace(tc)
        return tc
    # Otherwise assume this is already the talker config
    return hf_config


@support_torch_compile
class Qwen3TTSTalkerModel(nn.Module):
    """Qwen3TTS Talker Model - transformer backbone with text embeddings.

    The codec embedding lives in the code predictor; this module only
    keeps the text embedding (needed on the first PP rank for input
    processing).
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        
        config = _get_talker_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        
        # Text embedding for text tokens (codec embedding is in code_predictor)
        if get_pp_group().is_first_rank:
            self.text_embedding = VocabParallelEmbedding(
                config.text_vocab_size,
                config.text_hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.text_embedding",
            )
        else:
            self.text_embedding = PPMissingLayer()
        
        # Decoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3TTSTalkerDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        
        # Final layer norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings for input ids."""
        return self.text_embedding(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        combined_embeddings: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = combined_embeddings
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3TTSTalkerCodePredictorModel(nn.Module):
    """Native PyTorch code predictor model for Qwen3TTS Talker.
    
    Uses native attention (SDPA) instead of vLLM attention since:
    - Runs for fixed 15 steps per global time step
    - Shapes are deterministic
    - No benefit from KV caching
    - Can be captured in CUDA graphs for efficiency
    """

    def __init__(self, config: PretrainedConfig, embedding_dim: int) -> None:
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_code_groups = config.num_code_groups
        
        # Codec embeddings for groups 1 to N-1 (group 0 uses main model embedding)
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(config.vocab_size, embedding_dim)
            for _ in range(config.num_code_groups - 1)
        ])
        
        # Decoder layers using native attention
        self.layers = nn.ModuleList([
            Qwen3TTSCodePredictorDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Standard 1-D rotary position embeddings (matches HF code predictor)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.rotary_emb = Qwen3TTSNativeRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=getattr(config, "rope_theta", 1_000_000.0),
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        """Get codec embedding layers for all groups."""
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs_embeds: [batch_size, seq_len, hidden_size]
            attention_mask: Optional causal mask
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        hidden_states = inputs_embeds
        
        # Compute position embeddings shared across all decoder layers.
        # Positions are simply [0, 1, ..., seq_len-1] since we
        # recompute from scratch each call (no KV cache).
        seq_len = hidden_states.shape[1]
        position_embeddings = self.rotary_emb(
            seq_len, hidden_states.device, hidden_states.dtype
        )
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, position_embeddings
            )
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


@support_torch_compile
class Qwen3TTSTalkerCodePredictor(nn.Module):
    """Code predictor for Qwen3TTS Talker.
    
    Predicts all codec groups: group 0 via ``codec_head`` (from the talker
    hidden states) and groups 1..N-1 via the native code-predictor
    transformer.
    
    This module uses native PyTorch operations instead of vLLM attention
    since the code predictor:
    - Runs independently for each global time step
    - Has deterministic shapes (fixed 15 steps)
    - Doesn't benefit from KV cache
    - Can be captured in CUDA graphs
    
    Owns:
    - ``codec_embedding`` – VocabParallelEmbedding for group-0 codec tokens
      (shared with the outer model for input embedding lookups).
    - ``codec_head`` – lm head for group-0 prediction.
    - ``suppress_mask`` – precomputed bool mask for suppressing reserved tokens.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        
        hf_config = vllm_config.model_config.hf_config
        talker_config = _get_talker_config(hf_config)
        config = talker_config.code_predictor_config
        if isinstance(config, dict):
            config = _dict_to_namespace(config)
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.num_code_groups = config.num_code_groups
        self.hidden_size = config.hidden_size
        self.talker_hidden_size = talker_config.hidden_size
        
        # ── Group-0 codec embedding (moved from Qwen3TTSTalkerModel) ────
        # Available on all ranks so the outer model can look up codec
        # embeddings on the first PP rank and the code predictor can use
        # them for generation on the last PP rank.
        self.codec_embedding = VocabParallelEmbedding(
            talker_config.vocab_size,
            talker_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.codec_embedding",
        )
        
        # ── Group-0 prediction head (moved from outer model) ────────────
        if get_pp_group().is_last_rank:
            self.codec_head = ParallelLMHead(
                talker_config.vocab_size,
                talker_config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.codec_head",
            )
        else:
            self.codec_head = PPMissingLayer()
        
        # Precomputed suppress mask (True for the top-1024 reserved token IDs
        # except EOS).  Static tensor – CUDA-graph safe.
        self.suppress_mask = nn.Parameter(
            torch.zeros(talker_config.vocab_size, dtype=torch.bool),
            requires_grad=False,
        )
        
        self.logits_processor = LogitsProcessor(talker_config.vocab_size)
        
        # ── Code-predictor transformer backbone ─────────────────────────
        self.model = Qwen3TTSTalkerCodePredictorModel(
            config, self.talker_hidden_size
        )
        
        # Projection from talker hidden size to code predictor hidden size
        if config.hidden_size != self.talker_hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                self.talker_hidden_size, config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = nn.Identity()
        
        # LM heads for each code group (1 to N-1)
        self.lm_head = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ])
        
        # ── Persistent scratch buffers for generate_all_groups ───────────
        # Pre-allocated once to avoid per-call allocation overhead and to
        # ensure **constant memory addresses** for PIECEWISE / CUDA-graph
        # capture.  Sized to max_num_tokens (the maximum seq_len the model
        # runner will ever pass).  generate_all_groups slices these by
        # seq_len each call; under CUDA graph, seq_len is constant so
        # the slices are always the same view.
        #
        # Plain attributes (not register_buffer / nn.Parameter) because
        # vLLM does not call .to(dtype) on the model after construction --
        # it loads weights directly.  The device context manager active
        # during __init__ places these on the correct GPU, and we set
        # dtype explicitly from the model config.
        max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        N = config.num_code_groups  # typically 16
        hidden = talker_config.hidden_size
        cp_hidden = config.hidden_size
        dtype = vllm_config.model_config.dtype
        self._max_cp_len = 1 + N  # talker ctx + N codec groups

        # Input-embedding buffer: [max_tokens, 1+N, hidden]
        # Position 0 = talker context; positions 1..N = codec group embeds.
        # Zeroed so unfilled positions don't introduce NaN; the causal mask
        # in SDPA prevents them from affecting filled positions' outputs.
        self._cp_inputs_embeds = torch.zeros(
            max_num_tokens, self._max_cp_len, hidden, dtype=dtype
        )
        # Output buffer from code-predictor forward: [max_tokens, 1+N, cp_hidden]
        # Pre-allocated so the compiled forward always writes to the same memory.
        self._cp_hidden_states = torch.empty(
            max_num_tokens, self._max_cp_len, cp_hidden, dtype=dtype
        )
        # Codec token IDs: [max_tokens, N]
        self._cp_all_codecs = torch.empty(
            max_num_tokens, N, dtype=torch.long
        )

    def get_group0_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up group-0 codec embeddings (used by the outer model for
        input embedding on the first PP rank)."""
        return self.codec_embedding(input_ids)

    def get_group_embeddings(self) -> nn.ModuleList:
        """Get codec embedding layers for groups 1..N-1."""
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the code predictor transformer.

        The input must always be the **full** pre-allocated buffer
        ``_cp_inputs_embeds[:seq_len]`` with shape
        ``[seq_len, 1 + num_code_groups, talker_hidden_size]``.
        Unfilled positions should be zero; the native SDPA uses causal
        masking (``is_causal=True``) so they cannot contaminate filled
        positions.

        Passing a fixed-shape tensor on every call ensures a single
        compiled graph under PIECEWISE and a single CUDA-graph capture.

        Args:
            inputs_embeds: [batch_size, max_cp_len, talker_hidden_size]
                Always the full code-predictor sequence length.

        Returns:
            hidden_states: [batch_size, max_cp_len, cp_hidden_size]
        """
        # Project embeddings to code predictor hidden size
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        # No attention_mask → SDPA uses is_causal=True (lower-triangular)
        hidden_states = self.model(inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        generation_step: int,
    ) -> torch.Tensor:
        """Compute logits for a specific code group (1..N-1).
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            generation_step: Which code group to predict (0 to num_code_groups-2)
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        if generation_step >= len(self.lm_head):
            raise ValueError(
                f"generation_step {generation_step} exceeds number of "
                f"code groups {len(self.lm_head)}"
            )
        return self.lm_head[generation_step](hidden_states)

    def generate_codes(
        self,
        talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        prev_group0_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate **all** codec groups given the talker hidden states.

        First predicts group-0 from ``talker_hidden`` using ``codec_head``
        (with suppress-mask and sampling), then autoregressively generates
        groups 1..N-1 via the code-predictor transformer.

        Uses **persistent scratch buffers** that were pre-allocated in the
        constructor to ``max_num_tokens``.  Every call to the compiled
        ``forward()`` receives the **full** ``_cp_inputs_embeds`` buffer
        (fixed shape ``[S, 1+N, H]``), ensuring a single compiled graph
        under PIECEWISE and a single CUDA-graph capture.  Unfilled
        positions are zero; causal masking in SDPA prevents them from
        affecting filled positions.  The correct output position is
        extracted *after* the compiled forward returns.

        Args:
            talker_hidden: [seq_len, hidden_size] - hidden states from talker
            do_sample: Whether to sample or use argmax
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repeated tokens
            prev_group0_tokens: [seq_len, window] - recent group-0 token
                history for cross-step repetition penalty (optional)

        Returns:
            all_codecs: [seq_len, num_code_groups] - all codec tokens
        """
        seq_len = talker_hidden.shape[0]
        N = self.num_code_groups  # typically 16

        # ── Slice persistent buffers to the actual seq_len ──────────────
        # Under CUDA graph, seq_len is constant so these are always the
        # same views (same memory address, same shape).
        inputs_embeds = self._cp_inputs_embeds[:seq_len]       # [S, 1+N, H]
        all_codecs = self._cp_all_codecs[:seq_len]             # [S, N]

        # Zero the input buffer so unfilled positions are clean (no NaN).
        # The running-sum buffer also needs zeroing.
        inputs_embeds.zero_()

        # Fill position 0 with the talker context
        inputs_embeds[:, 0, :] = talker_hidden

        # ── Predict group-0 codec using the talker's hidden states ──────
        logits = self.logits_processor(self.codec_head, talker_hidden)
        logits = logits.masked_fill(self.suppress_mask.bool(), float('-inf'))

        first_codec = _sample_from_logits(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=prev_group0_tokens,
        )

        all_codecs[:, 0] = first_codec
        first_embed = self.codec_embedding(first_codec)  # [seq_len, hidden]
        inputs_embeds[:, 1, :] = first_embed

        # ── Generate groups 1 through N-1 ───────────────────────────────
        for step in range(N - 1):
            # Always pass the FULL buffer (fixed shape for compiled fwd).
            # Positions 0..current_len-1 are filled; the rest are zero.
            # Causal masking ensures filled positions' outputs are correct.
            hidden_states = self(inputs_embeds)  # [S, 1+N, cp_hidden]

            # Extract logits from the last FILLED position
            current_len = step + 2  # talker ctx + groups 0..step
            logits = self.compute_logits(
                hidden_states[:, current_len - 1, :], step
            )  # [seq_len, vocab]

            # Repetition penalty context: only prior inner-group tokens
            # (skip group 0 — it uses a different, larger vocab)
            if repetition_penalty != 1.0 and step > 0:
                current_context = all_codecs[:, 1: step + 1]
            else:
                current_context = None

            next_token = _sample_from_logits(
                logits, do_sample, temperature, top_k, top_p,
                repetition_penalty, current_context,
            )
            all_codecs[:, step + 1] = next_token

            # Embed the predicted token and accumulate
            next_embed = self.get_group_embeddings()[step](
                next_token
            )  # [seq_len, hidden]

            # Write embedding into buffer for the next iteration
            # (every position is written, including the last -- the
            # forward still reads it via causal masking even though we
            # don't need another forward after the last step).
            inputs_embeds[:, current_len, :] = next_embed

        return all_codecs


@ignore_torch_compile
@support_torch_compile
class Qwen3TTSTalkerForConditionalGeneration(nn.Module, SupportsPP):
    """Qwen3TTS Talker for conditional generation.
    
    Top-level model that orchestrates text-to-codec generation.  The
    ``code_predictor`` sub-module is compiled via ``@support_torch_compile``
    (it uses native PyTorch SDPA and benefits from compilation/CUDA-graph
    capture).  The transformer backbone (``model``) is *not* compiled
    because it relies on vLLM's paged ``Attention`` which is already
    optimised and whose custom-op fake kernels can produce stride
    mismatches under ``torch.compile`` for certain GQA configurations.
    """
    
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        
        hf_config = vllm_config.model_config.hf_config
        tts_config = _get_tts_config(hf_config)
        config = _get_talker_config(hf_config)
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.tts_config = tts_config
        self.quant_config = quant_config
        
        # Sampling parameters from config (shared for talker and code predictor)
        if tts_config is not None:
            self.do_sample = getattr(tts_config, "do_sample", True)
            self.temperature = getattr(tts_config, "temperature", 1.0)
            self.top_k = getattr(tts_config, "top_k", 50)
            self.top_p = getattr(tts_config, "top_p", 1.0)
            self.repetition_penalty = getattr(tts_config, "repetition_penalty", 1.0)
        else:
            self.do_sample = True
            self.temperature = 1.0
            self.top_k = 50
            self.top_p = 1.0
            self.repetition_penalty = 1.0
        
        # Transformer backbone (not compiled – uses vLLM paged Attention)
        with set_model_tag("talker"):
            self.model = Qwen3TTSTalkerModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )
        
        # Text projection MLP: maps text embeddings from text_hidden_size
        # to talker hidden_size.  Weights loaded from the checkpoint;
        # used at runtime to compute input embeddings from text tokens.
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            input_size=config.text_hidden_size,
            intermediate_size=config.text_hidden_size,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text_projection"),
        )

        # Compiled code predictor (native PyTorch SDPA, benefits from
        # torch.compile + CUDA-graph capture).  Owns codec_head,
        # suppress_mask, codec_embedding, and the code-predictor
        # transformer.
        with set_model_tag("code_predictor"):
            self.code_predictor = Qwen3TTSTalkerCodePredictor(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )
        
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Persistent buffers for outputs and intermediate results.
        # Fixed memory addresses are required for CUDA graph replay.
        max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        codes_num = self.code_predictor.num_code_groups
        dtype = vllm_config.model_config.dtype
        self._out_codes = torch.empty(
            max_num_tokens, codes_num, dtype=torch.long
        )
        self._combined_embeddings = torch.empty(
            max_num_tokens, config.hidden_size, dtype=dtype
        )


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get group-0 codec embeddings for input ids."""
        return self.code_predictor.get_group0_embeddings(input_ids)

    def _get_logits_indices(self) -> tuple[Optional[torch.Tensor], int, int]:
        """Extract logits_indices from the forward context.

        Returns the indices of positions that need sampling (last token
        per request), or ``None`` during profile / dummy runs where
        attention metadata is not yet available.
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata

        if attn_metadata is None:
            # Profile / dummy run — no real batch to sample from.
            return None, 0, 0

        # In v1 attn_metadata is dict[layer_name, per_layer_metadata].
        # All layers share the same query_start_loc; grab any one.
        if isinstance(attn_metadata, dict):
            any_layer_meta = next(iter(attn_metadata.values()))
        else:
            any_layer_meta = attn_metadata

        query_start_loc = any_layer_meta.query_start_loc
        indices = query_start_loc[1:] - 1

        num_requests = indices.shape[0]
        if self.vllm_config.compilation_config.use_cudagraph:
            padded_num_requests = self.vllm_config.pad_for_cudagraph(num_requests)
        else:
            padded_num_requests = num_requests
        if num_requests != padded_num_requests:
            # need to pad indices so we run on known cuda kernel size
            indices = torch.nn.functional.pad(indices, (0, padded_num_requests - num_requests))
        return indices, num_requests, padded_num_requests

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        acoustic_ids: Optional[torch.Tensor] = None,
        prev_group0_tokens: Optional[torch.Tensor] = None,
    ) -> Union[IntermediateTensors, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through the talker model.
        
        Runs the compiled transformer backbone, then delegates all codec
        prediction (group 0 through group N-1) to the code predictor.
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs for rotary embeddings
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed input embeddings
            text_ids: Text token IDs
            acoustic_ids: 0th acoustic token IDs
            prev_group0_tokens: Previous group 0 tokens
            
        Returns:
            For non-last PP rank: IntermediateTensors for pipeline parallelism
            For last PP rank: tuple of (hidden_states, codes)
        """

        # Compute combined embeddings from text and acoustic token IDs.
        # text_projection maps text_hidden_size → hidden_size; the group-0
        # codec embedding is the same hidden_size.  Inactive acoustic
        # positions use acoustic_zero_token_id whose embedding was zeroed
        # during checkpoint conversion, so no explicit masking is needed.
        #
        # acoustic_ids shape: [seq_len, num_code_groups]
        #   Column 0 → group-0 codec embedding (talker vocab)
        #   Columns 1..N-1 → code-predictor codec embeddings
        # The sum of all group embeddings replicates the original HF
        # embedding logic where all codebook embeddings are summed.
        if get_pp_group().is_first_rank:
            text_embed = self.text_projection(
                self.model.get_text_embeddings(text_ids)
            )
            codec_embed = self.get_input_embeddings(acoustic_ids[:, 0])
            group_embeddings = self.code_predictor.get_group_embeddings()
            for i in range(len(group_embeddings)):
                codec_embed = codec_embed + group_embeddings[i](
                    acoustic_ids[:, i + 1]
                )
            seq_len = text_embed.shape[0]
            combined_embeddings = self._combined_embeddings[:seq_len]
            torch.add(text_embed, codec_embed, out=combined_embeddings)
        else:
            combined_embeddings = None

        # Forward through the compiled transformer backbone
        hidden_states = self.model(
            input_ids, 
            positions, 
            intermediate_tensors, 
            inputs_embeds,
            combined_embeddings,
        )
        
        # Intermediate PP ranks return tensors for next rank
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        

        logits_indices, num_requests, padded_num_requests = self._get_logits_indices()
        num_tokens = hidden_states.shape[0]
        if logits_indices is None or logits_indices.shape[0] == num_tokens:
            # either dummy run or decode-only, run code predictor without slicing
            all_codecs = self.code_predictor.generate_codes(
                talker_hidden=hidden_states,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                prev_group0_tokens=prev_group0_tokens,
            )
            return hidden_states, all_codecs
        else:
            # run code predictor only for tokens that are to be decoded
            selected_states = hidden_states[logits_indices].contiguous()
            selected_prev_g0 = (prev_group0_tokens[logits_indices]
                                if prev_group0_tokens is not None else None)
            ctx = get_forward_context()
            old_batch_descriptor = ctx.batch_descriptor
            old_mode = ctx.cudagraph_runtime_mode
            ctx.batch_descriptor = BatchDescriptor(
                num_tokens=padded_num_requests,
                uniform_decode=False,
            )
            ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            try:
                codes = self.code_predictor.generate_codes(
                    talker_hidden=selected_states,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    prev_group0_tokens=selected_prev_g0,
                )
            finally:
                ctx.batch_descriptor = old_batch_descriptor
                ctx.cudagraph_runtime_mode = old_mode
            # scatter results into buffer
            self._out_codes[logits_indices[:num_requests]] = codes[:num_requests]
            return hidden_states, self._out_codes[:num_tokens]
        
        

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return hidden_states for compatibility.
        
        Note: Actual logits computation and sampling happens inside forward().
        This method exists for compatibility with vLLM's model runner interface.
        """
        return hidden_states

    # Weight-name prefixes that moved during the refactoring.
    # Maps old (HF / pre-conversion) prefix → new (vLLM model) prefix.
    # Applied after stripping the "talker." wrapper prefix.
    _weight_remap_prefixes: list[tuple[str, str]] = [
        # codec_embedding moved from model → code_predictor
        ("model.codec_embedding.", "code_predictor.codec_embedding."),
        # codec_head moved from root → code_predictor
        ("codec_head.", "code_predictor.codec_head."),
        # suppress_mask moved from root → code_predictor
        ("suppress_mask", "code_predictor.suppress_mask"),
    ]

    @staticmethod
    def build_prefill_tokens(
        tokenizer,
        text: str,
        speaker: Union[str, int],
        language: Union[str, int, None],
        config: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``text_ids`` and ``acoustic_ids`` for the prefill stage.

        Pure-function, thread-safe helper that assembles the token sequences
        the talker model expects as prefill input.  Can be called from any
        thread without holding model state.

        Prefill layout::

            A. Role prefix        (3)     role token IDs      | zero_token
            B+C. Ctrl + speaker   (N-1)   tts_pad / tts_bos   | codec ctrl tokens
            D. Synth text + EOS   (T+1)   content token IDs   | codec_pad
            E. Final BOS          (1)     tts_pad              | codec_bos

        Args:
            tokenizer: HuggingFace tokenizer for the model.
            text: Text to synthesize.
            speaker: Speaker name (``str``, looked up in
                ``config["talker_config"]["spk_id"]``) or raw speaker
                codec token ID (``int``).
            language: Language name (``str``, looked up in
                ``config["talker_config"]["codec_language_id"]``), raw
                codec token ID (``int``), or ``None`` to omit language
                (uses the nothink control path).
            config: The **full** config dict from the converted checkpoint's
                ``config.json`` (as produced by
                ``convert_qwen3tts_checkpoint.py``).

        Returns:
            text_ids:     ``[L]`` int64 – text token IDs.
            acoustic_ids: ``[L, num_code_groups]`` int64 – acoustic token IDs.
                Column 0 holds group-0 control/codec tokens; columns
                1..N-1 are filled with ``acoustic_zero_token_id`` (whose
                embedding is zeroed in the converted checkpoint).
        """
        tc = config["talker_config"]

        tts_bos = config["tts_bos_token_id"]
        tts_eos = config["tts_eos_token_id"]
        tts_pad = config["tts_pad_token_id"]
        codec_pad = tc["codec_pad_id"]
        codec_bos = tc["codec_bos_id"]
        codec_nothink = tc["codec_nothink_id"]
        codec_think = tc["codec_think_id"]
        codec_think_bos = tc["codec_think_bos_id"]
        codec_think_eos = tc["codec_think_eos_id"]
        zero_token = tc["acoustic_zero_token_id"]
        num_code_groups = tc.get("num_code_groups", 16)

        # Resolve speaker name → codec token ID
        if isinstance(speaker, str):
            spk_map = tc.get("spk_id", {})
            if speaker not in spk_map:
                available = ", ".join(sorted(spk_map)) if spk_map else "(none)"
                raise ValueError(
                    f"Unknown speaker '{speaker}'. Available: {available}"
                )
            speaker_id = spk_map[speaker]
        else:
            speaker_id = int(speaker)

        # Resolve language name → codec token ID
        if isinstance(language, str):
            lang_map = tc.get("codec_language_id", {})
            if language not in lang_map:
                available = ", ".join(sorted(lang_map)) if lang_map else "(none)"
                raise ValueError(
                    f"Unknown language '{language}'. Available: {available}"
                )
            language_id: Optional[int] = lang_map[language]
        elif language is not None:
            language_id = int(language)
        else:
            language_id = None

        text_list: list[int] = []
        g0_list: list[int] = []

        # A. Role prefix
        role_tokens = tokenizer.encode("<|im_start|>assistant\n")[:3]
        for rid in role_tokens:
            text_list.append(rid)
            g0_list.append(zero_token)

        # B+C. Control header + speaker
        if language_id is None or language_id < 0:
            codec_ctrl = [
                codec_nothink, codec_think_bos, codec_think_eos,
                speaker_id, codec_pad, codec_bos,
            ]
        else:
            codec_ctrl = [
                codec_think, codec_think_bos, language_id, codec_think_eos,
                speaker_id, codec_pad, codec_bos,
            ]
        n_ctrl = len(codec_ctrl)
        for i in range(n_ctrl - 1):
            text_list.append(tts_pad if i < n_ctrl - 2 else tts_bos)
            g0_list.append(codec_ctrl[i])

        # D. Synth text + EOS
        synth_full = (
            f"<|im_start|>assistant\n{text}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        full_ids = tokenizer.encode(synth_full)
        content_ids = full_ids[3:-5]

        for tid in content_ids:
            text_list.append(tid)
            g0_list.append(codec_pad)
        text_list.append(tts_eos)
        g0_list.append(codec_pad)

        # E. Final BOS
        text_list.append(tts_pad)
        g0_list.append(codec_bos)

        text_ids = torch.tensor(text_list, dtype=torch.long)

        # Build [L, num_code_groups] acoustic_ids: group 0 gets control
        # tokens, groups 1..N-1 get zero_token (zeroed embedding).
        seq_len = len(g0_list)
        acoustic_ids = torch.full(
            (seq_len, num_code_groups), zero_token, dtype=torch.long
        )
        acoustic_ids[:, 0] = torch.tensor(g0_list, dtype=torch.long)

        return text_ids, acoustic_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        
        # Mark deterministically-initialized params as already loaded so the
        # strict weight-loading check doesn't complain about them missing
        # from the checkpoint.
        for pname in params_dict:
            if "rotary_emb.inv_freq" in pname:
                loaded_params.add(pname)
        
        for name, loaded_weight in weights:
            # The HF checkpoint stores weights under
            # Qwen3TTSForConditionalGeneration which wraps the talker as
            # ``self.talker = Qwen3TTSTalkerForConditionalGeneration(...)``.
            # Strip the "talker." prefix so names align with this model.
            if name.startswith("talker."):
                name = name[len("talker."):]
            
            # Skip weights that don't belong to the talker (e.g.
            # speaker_encoder weights).
            if name.startswith("speaker_encoder."):
                continue
            
            if "rotary_emb.inv_freq" in name:
                continue
            
            # Remap old (HF / pre-conversion) weight names to the new
            # locations.  Converted checkpoints already use the new names
            # so these replacements are no-ops for them.
            for old_pfx, new_pfx in self._weight_remap_prefixes:
                if name.startswith(old_pfx):
                    name = new_pfx + name[len(old_pfx):]
                    break
            
            # Handle stacked parameters (for vLLM parallel layers in the
            # talker backbone).  The code predictor uses native nn.Linear
            # layers, so the stacked name won't exist in params_dict --
            # we must NOT mutate `name` so the fallback path can still
            # load the weight directly.
            stacked_loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                
                # Skip loading extra bias for GPTQ models
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                
                if mapped_name not in params_dict:
                    continue
                    
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                stacked_loaded = True
                break
            
            if stacked_loaded:
                continue
            
            # Direct parameter loading (native layers and non-stacked params)
            # Skip loading extra bias for GPTQ models
            if name.endswith(".bias") and name not in params_dict:
                continue
            
            if name not in params_dict:
                continue
                
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        
        return loaded_params
