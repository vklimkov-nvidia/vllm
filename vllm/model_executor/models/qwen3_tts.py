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
from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context
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


def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-max trick: equivalent to categorical sampling.

    Uses only uniform RNG + log + argmax — all CUDA-graph safe.
    Unlike ``torch.multinomial``, this degrades gracefully on degenerate
    inputs (all-zero probs / all-``-inf`` logits) instead of triggering
    a device-side assert that poisons the CUDA context.  Also ~2.5x
    faster than multinomial in graph replay benchmarks.
    """
    u = torch.empty_like(logits).uniform_(1e-20, 1.0 - 1e-20)
    return (logits - torch.log(-torch.log(u))).argmax(dim=-1)


def _multinomial_sample(logits: torch.Tensor) -> torch.Tensor:
    """Standard softmax + multinomial sampling.

    CUDA-graph capturable on PyTorch >= 2.8, but will crash with a
    device-side assert if any row has all-zero probabilities (e.g.
    during graph warmup with uninitialised buffers).
    """
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def _sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[torch.Tensor] = None,
    use_gumbel: bool = True,
) -> torch.Tensor:
    """Sample tokens from logits (CUDA-graph safe).

    All operations are legal inside ``torch.cuda.graph()`` capture on
    PyTorch >= 2.8 (``topk``, ``sort``, ``multinomial``, ``uniform_``,
    ``argmax``, ``gather``, ``scatter_``, ``masked_fill``).

    The only patterns that remain **unsafe** during capture are
    host-to-device copies such as ``torch.tensor(scalar, device=cuda)``
    and ``torch.full_like(t, val)`` for some values — use
    ``masked_fill`` or pre-allocated buffers instead.

    Args:
        use_gumbel: If ``True`` (default), use the Gumbel-max trick for
            the final categorical draw.  Gumbel-max is ~2.5x faster
            than ``multinomial`` and robust to degenerate warmup data.
            Set ``False`` to use ``softmax → multinomial`` instead.
    """
    if repetition_penalty != 1.0 and previous_tokens is not None:
        score = torch.gather(logits, -1, previous_tokens)
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )
        logits.scatter_(-1, previous_tokens, score)

    if not do_sample:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-6)

    # ── Top-k filtering ─────────────────────────────────────────────
    if top_k is not None and top_k > 0:
        vals, idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)

        # ── Top-p (nucleus) within the top-k slice ──────────────────
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_vals, sort_idx = torch.sort(vals, dim=-1, descending=True)
            probs = torch.softmax(sorted_vals, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            remove = (cum_probs - probs) > top_p
            sorted_vals = sorted_vals.masked_fill(remove, -1e10)
            # Unsort back to topk order
            unsort_idx = sort_idx.argsort(dim=-1)
            vals = sorted_vals.gather(-1, unsort_idx)

        sampled_in_k = (_gumbel_sample(vals) if use_gumbel
                        else _multinomial_sample(vals))
        return idxs.gather(-1, sampled_in_k.unsqueeze(-1)).squeeze(-1)

    # ── Top-p only (no top-k) ───────────────────────────────────────
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, dim=-1, descending=True
        )
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        remove = (cum_probs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -1e10)

        sampled_sorted = (_gumbel_sample(sorted_logits) if use_gumbel
                          else _multinomial_sample(sorted_logits))
        return sorted_indices.gather(
            -1, sampled_sorted.unsqueeze(-1)
        ).squeeze(-1)

    # ── No filtering — sample from full distribution ────────────────
    if use_gumbel:
        return _gumbel_sample(logits)
    return _multinomial_sample(logits)


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
    """Code predictor for Qwen3TTS Talker — groups 1..N-1 only.

    Given the previous step's backbone hidden state and the group-0 token
    (sampled by vLLM), autoregressively predicts codec groups 1 through
    N-1 using a small native-attention transformer.

    Group-0 prediction (``codec_head``, ``suppress_mask``) is handled by
    the outer model's ``compute_logits()`` + vLLM sampler.

    Also owns ``codec_embedding`` (group-0 codebook), shared with the
    outer model for input-embedding lookups.
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

        # Group-0 codec embedding (shared with outer model)
        self.codec_embedding = VocabParallelEmbedding(
            talker_config.vocab_size,
            talker_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.codec_embedding",
        )

        # Code-predictor transformer backbone
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

        # Sampling parameters for the internal groups-1..N-1 loop,
        # read from code_predictor_config. Fallback defaults match the
        # original HF implementation's subtalker_* arguments.
        self.do_sample = getattr(config, "do_sample", True)
        self.temperature = getattr(config, "temperature", 0.9)
        self.top_k = getattr(config, "top_k", 50)
        self.top_p = getattr(config, "top_p", 1.0)
        self.repetition_penalty = getattr(config, "repetition_penalty", 1.0)
        self.use_gumbel = getattr(config, "use_gumbel", True)

        # ── Persistent scratch buffers ──────────────────────────────────
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        N = config.num_code_groups
        hidden = talker_config.hidden_size
        cp_hidden = config.hidden_size
        dtype = vllm_config.model_config.dtype
        self._max_cp_len = 1 + N  # prev_hidden ctx + group0 + groups 1..N-1

        self._cp_inputs_embeds = torch.zeros(
            max_num_tokens, self._max_cp_len, hidden, dtype=dtype
        )
        self._cp_hidden_states = torch.empty(
            max_num_tokens, self._max_cp_len, cp_hidden, dtype=dtype
        )
        # Only groups 1..N-1 (N-1 columns)
        self._cp_all_codecs = torch.empty(
            max_num_tokens, N - 1, dtype=torch.long
        )

    def get_group0_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up group-0 codec embeddings."""
        return self.codec_embedding(input_ids)

    def get_group_embeddings(self) -> nn.ModuleList:
        """Get codec embedding layers for groups 1..N-1."""
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the code predictor transformer."""
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        hidden_states = self.model(inputs_embeds)
        return hidden_states

    def _compute_inner_logits(
        self,
        hidden_states: torch.Tensor,
        generation_step: int,
    ) -> torch.Tensor:
        """Compute logits for a specific inner code group (1..N-1)."""
        if generation_step >= len(self.lm_head):
            raise ValueError(
                f"generation_step {generation_step} exceeds number of "
                f"code groups {len(self.lm_head)}"
            )
        return self.lm_head[generation_step](hidden_states)

    def generate_groups_1_15(
        self,
        prev_hidden: torch.Tensor,
        group0_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Generate codec groups 1..N-1 given previous hidden state and group0.

        Args:
            prev_hidden: [seq_len, hidden_size] backbone output from previous step
            group0_tokens: [seq_len] group-0 tokens (from vLLM sampling)

        Returns:
            codes_1_15: [seq_len, num_code_groups - 1]
        """
        seq_len = prev_hidden.shape[0]
        N = self.num_code_groups

        inputs_embeds = self._cp_inputs_embeds[:seq_len]  # Batch x Books x Dim
        all_codecs = self._cp_all_codecs[:seq_len]

        inputs_embeds.zero_()

        # Position 0: previous backbone hidden state
        inputs_embeds[:, 0, :] = prev_hidden

        # Position 1: group-0 codec embedding
        inputs_embeds[:, 1, :] = self.codec_embedding(group0_tokens)

        for step in range(N - 1):
            hidden_states = self(inputs_embeds)

            current_len = step + 2
            logits = self._compute_inner_logits(
                hidden_states[:, current_len - 1, :], step
            )

            if self.repetition_penalty != 1.0 and step > 0:
                current_context = all_codecs[:, :step]
            else:
                current_context = None

            next_token = _sample_from_logits(
                logits,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                previous_tokens=current_context,
                use_gumbel=self.use_gumbel,
            )
            all_codecs[:, step] = next_token

            next_embed = self.get_group_embeddings()[step](next_token)
            inputs_embeds[:, current_len, :] = next_embed

        return all_codecs


@ignore_torch_compile
@support_torch_compile
class Qwen3TTSTalkerForConditionalGeneration(nn.Module, SupportsPP):
    """Qwen3TTS Talker for conditional generation.

    Per-step flow:

    1. **Code predictor** (conditional): given the previous step's backbone
       hidden state (``prev_hidden``, custom input) and the group-0 token
       (``input_ids``, sampled by vLLM at the previous step), predict codec
       groups 1..N-1.  Skipped when ``prev_hidden`` is all-zero (prefill).
    2. **Embedding**: text_projection(text_embed) + codec_embed(group0)
       + sum of groups-1..N-1 embeddings from the code predictor.
    3. **Backbone**: transformer with vLLM paged attention and KV cache.
    4. **Logits**: ``compute_logits()`` projects backbone output through
       ``codec_head`` and applies ``suppress_mask``.  vLLM's standard
       sampler then samples the next group-0 token.

    Custom I/O:
      Inputs:  ``text_ids`` (int64), ``prev_hidden`` (float, dim=hidden_size)
      Outputs: ``codes`` (int64, dim=N-1), ``hidden`` (float, dim=hidden_size)
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
        config = _get_talker_config(hf_config)
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vllm_config = vllm_config

        # Transformer backbone (not compiled – uses vLLM paged Attention)
        with set_model_tag("talker"):
            self.model = Qwen3TTSTalkerModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        # Text projection MLP
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            input_size=config.text_hidden_size,
            intermediate_size=config.text_hidden_size,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text_projection"),
        )

        # Compiled code predictor (groups 1..N-1 only)
        with set_model_tag("code_predictor"):
            self.code_predictor = Qwen3TTSTalkerCodePredictor(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )

        # Group-0 prediction head + suppress mask (used by compute_logits
        # so vLLM's standard sampler can sample group-0).
        if get_pp_group().is_last_rank:
            self.codec_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "codec_head"),
            )
        else:
            self.codec_head = PPMissingLayer()

        self.suppress_mask = nn.Parameter(
            torch.zeros(config.vocab_size, dtype=torch.bool),
            requires_grad=False,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Persistent buffers — addresses must be stable across CUDA graph
        # replays.  The piecewise CUDAGraphWrapper does NOT copy inputs on
        # replay; it expects the same ``data_ptr()`` that was recorded during
        # capture.  Any tensor created transiently in ``forward()`` (like
        # ``text_embed + codec_embed``) would have a new address each call,
        # causing the replayed graph to read stale memory.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        codes_num = self.code_predictor.num_code_groups - 1  # groups 1..N-1
        dtype = vllm_config.model_config.dtype
        self._out_codes = torch.zeros(
            max_num_tokens, codes_num, dtype=torch.long
        )
        self._combined_embeddings = torch.zeros(
            max_num_tokens, config.hidden_size, dtype=dtype
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get group-0 codec embeddings for input ids."""
        return self.code_predictor.get_group0_embeddings(input_ids)

    def _get_decode_idxs(self):
        """
        helper function that returns indices of decoding tokens,
        that's where exactly the local transformer should be
        applied. 

        Returns:
            decode_idx: indices of decoder requests, if None returned,
                        local transformer should be applied everywhere
            num_requests: number of decoding requests, before padding
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            # when attention metadata is not provided (capturing, dummy run)
            # then we should apply the local transformer everywhere
            return None, 0

        if isinstance(attn_metadata, dict):
            any_layer_meta = next(iter(attn_metadata.values()))
        else:
            any_layer_meta = attn_metadata

        if any_layer_meta.max_query_len == 1:
            # all requests in the batch a decode-only,
            # apply local transformer everywhere
            return None, 0
        
        start_loc = any_layer_meta.query_start_loc
        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = (tokens_per_req == 1)  # shape: (num_reqs,)
        decode_token_indices = start_loc[:-1][is_decode]

        num_requests = decode_token_indices.shape[0]
        padded_num_requests = num_requests
        if self.vllm_config.compilation_config.use_cudagraph:
            padded_num_requests = self.vllm_config.pad_for_cudagraph(num_requests)
        if padded_num_requests != num_requests:
            decode_token_indices = torch.nn.functional.pad(decode_token_indices, (0, padded_num_requests - num_requests))
        return decode_token_indices, num_requests

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> Union[IntermediateTensors,
               tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass: code predictor -> embedding -> backbone.

        Handles three regimes transparently:

        * **Profile / dummy run** (``attn_metadata is None``): the code-
          predictor path runs on every token so it is captured in the
          compiled CUDA graph.
        * **Decode-only batch**: every token is a decode token — the
          compiled / CUDA-graphed path replays directly.
        * **Mixed prefill + decode**: only decode-token positions are
          extracted and fed through the code predictor (eager); group
          embeddings are scattered back into ``codec_embed`` at those
          positions.

        Args:
            input_ids: Group-0 codec tokens  ``[num_tokens]``.
            positions: Position IDs for rotary embeddings.
            intermediate_tensors: For pipeline parallelism.
            inputs_embeds: Pre-computed input embeddings (unused).
            text_ids: Text token IDs (custom input).
            prev_hidden: Backbone hidden state from the previous step
                (custom input; all-zero during prefill).

        Returns:
            Non-last PP rank: IntermediateTensors.
            Last PP rank: ``(hidden_states, codes_1_15, hidden_states)``.
        """
        text_embed = self.text_projection(
            self.model.get_text_embeddings(text_ids)
        )
        codec_embed = self.get_input_embeddings(input_ids)

        decode_idx, num_req = self._get_decode_idxs()
        group_embeddings = self.code_predictor.get_group_embeddings()
        if decode_idx is None:
            codes_1_15 = self.code_predictor.generate_groups_1_15(
                prev_hidden=prev_hidden,
                group0_tokens=input_ids,
            )
            self._out_codes[:codes_1_15.shape[0]] = codes_1_15
            for i in range(len(group_embeddings)):
                codec_embed.add_(group_embeddings[i](codes_1_15[:, i]))
        elif num_req > 0:
            # need to overwrite the batch batch descriptor since we are slicing the inputs
            ctx = get_forward_context()
            orig_batch_descriptor = ctx.batch_descriptor
            ctx.batch_descriptor = BatchDescriptor(
                num_tokens=num_req,
                uniform_decode=False,
            )
            codes_1_15 = self.code_predictor.generate_groups_1_15(
                prev_hidden=prev_hidden[decode_idx],
                group0_tokens=input_ids[decode_idx],
            )
            # restore original batch descriptor
            ctx.batch_descriptor = orig_batch_descriptor
            valid_dec_idx = decode_idx[:num_req]
            self._out_codes[valid_dec_idx] = codes_1_15[:num_req]
            for i in range(len(group_embeddings)):
                codec_embed[valid_dec_idx] = (
                    codec_embed[valid_dec_idx]
                    + group_embeddings[i](codes_1_15[:num_req, i])
                )

        num_tokens = input_ids.shape[0]
        combined_embeddings = self._combined_embeddings[:num_tokens]
        torch.add(text_embed, codec_embed, out=combined_embeddings)

        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            combined_embeddings,
        )

        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        out_codes = self._out_codes[:num_tokens]
        return hidden_states, out_codes, hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute group-0 logits for vLLM sampling.

        Projects backbone hidden states through ``codec_head``, applies the
        ``suppress_mask`` to block reserved token IDs, and returns logits
        of shape ``[batch, vocab_size]``.
        """
        logits = self.logits_processor(self.codec_head, hidden_states)
        logits = logits.masked_fill(self.suppress_mask.bool(), float('-inf'))
        return logits

    # Weight-name remapping applied after stripping the "talker." prefix.
    # Handles both original HF names and previously-converted checkpoint
    # names where codec_head / suppress_mask lived under code_predictor.
    _weight_remap_prefixes: list[tuple[str, str]] = [
        ("model.codec_embedding.", "code_predictor.codec_embedding."),
        ("code_predictor.codec_head.", "codec_head."),
        ("code_predictor.suppress_mask", "suppress_mask"),
    ]

    @staticmethod
    def build_prefill_tokens(
        tokenizer,
        text: str,
        speaker: Union[str, int],
        language: Union[str, int, None],
        config: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``text_ids`` and ``group0_ids`` for the prefill stage.

        Returns:
            text_ids:    ``[L]`` int64 – text token IDs.
            group0_ids:  ``[L]`` int64 – group-0 codec control / token IDs
                (used as ``prompt_token_ids`` for vLLM).
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
        group0_ids = torch.tensor(g0_list, dtype=torch.long)

        return text_ids, group0_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for pname in params_dict:
            if "rotary_emb.inv_freq" in pname:
                loaded_params.add(pname)

        for name, loaded_weight in weights:
            if name.startswith("talker."):
                name = name[len("talker."):]

            if name.startswith("speaker_encoder."):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            for old_pfx, new_pfx in self._weight_remap_prefixes:
                if name.startswith(old_pfx):
                    name = new_pfx + name[len(old_pfx):]
                    break

            stacked_loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)

                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue

                if mapped_name not in params_dict:
                    continue

                param = params_dict[mapped_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                stacked_loaded = True
                break

            if stacked_loaded:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
