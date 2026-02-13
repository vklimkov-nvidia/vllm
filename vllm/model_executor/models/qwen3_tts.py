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
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
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
    ) -> torch.Tensor:
        """Forward pass using torch SDPA.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
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
    ) -> torch.Tensor:
        # Self Attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
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


class Qwen3TTSTalkerModel(nn.Module):
    """Qwen3TTS Talker Model - transformer backbone with codec and text embeddings."""

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
        
        # Codec embedding for audio tokens
        if get_pp_group().is_first_rank:
            self.codec_embedding = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.codec_embedding",
            )
            # Text embedding for text tokens
            self.text_embedding = VocabParallelEmbedding(
                config.text_vocab_size,
                config.text_hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.text_embedding",
            )
        else:
            self.codec_embedding = PPMissingLayer()
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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get codec embeddings for input ids."""
        return self.codec_embedding(input_ids)
    
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
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3TTSTalkerCodePredictor(nn.Module):
    """Native PyTorch code predictor for Qwen3TTS Talker.
    
    Predicts codec groups 1 to N-1 given the hidden states from the main talker
    and previous codec groups.
    
    This module uses native PyTorch operations instead of vLLM abstractions
    since the code predictor:
    - Runs independently for each global time step
    - Has deterministic shapes (fixed 15 steps)
    - Doesn't benefit from KV cache
    - Can be captured in CUDA graphs
    """

    def __init__(self, config: PretrainedConfig, talker_hidden_size: int) -> None:
        super().__init__()
        
        self.config = config
        self.num_code_groups = config.num_code_groups
        self.hidden_size = config.hidden_size
        
        # Model backbone
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_hidden_size)
        
        # Projection from talker hidden size to code predictor hidden size
        if config.hidden_size != talker_hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                talker_hidden_size, config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = nn.Identity()
        
        # LM heads for each code group (1 to N-1)
        self.lm_head = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ])

    def get_input_embeddings(self) -> nn.ModuleList:
        """Get codec embedding layers."""
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the code predictor.
        
        Args:
            inputs_embeds: [batch_size, seq_len, talker_hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        # Project embeddings to code predictor hidden size
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        
        hidden_states = self.model(inputs_embeds, attention_mask)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        generation_step: int,
    ) -> torch.Tensor:
        """Compute logits for a specific code group.
        
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

    def generate_all_groups(
        self,
        talker_hidden: torch.Tensor,
        first_codec: torch.Tensor,
        talker_codec_embedding: nn.Module,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate all codec groups given talker hidden state and first codec.
        
        This runs the full code predictor loop for all 15 additional groups.
        Treats the first dimension as batch (works with packed sequences where
        seq_len becomes the batch dimension).
        
        Args:
            talker_hidden: [seq_len, hidden_size] - hidden states from talker (packed)
            first_codec: [seq_len] - first codec token (group 0) from talker
            talker_codec_embedding: Embedding layer for group 0 codec
            do_sample: Whether to sample or use argmax
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            all_codecs: [seq_len, num_code_groups] - all 16 codec tokens
        """
        # Prepare initial input: [seq_len, 2, hidden]
        # - talker_hidden: context from main model (provides position 0)
        # - first codec embedding: the group 0 token (provides position 1)
        first_embed = talker_codec_embedding(first_codec)  # [seq_len, hidden]
        inputs_embeds = torch.stack([talker_hidden, first_embed], dim=1)  # [seq_len, 2, hidden]
        
        all_codecs = [first_codec]  # Start with group 0
        
        # Generate groups 1 through num_code_groups-1 (15 groups)
        for step in range(self.num_code_groups - 1):
            # Forward through code predictor model
            #print(f"code predictor #{step}: inputs_embeds: {inputs_embeds.shape}: {torch.max(inputs_embeds)}", flush=True)
            hidden_states = self.forward(inputs_embeds)  # [seq_len, seq_so_far, hidden]
            
            # Get logits from the appropriate head (last position)
            # step=0 -> lm_head[0] predicts group 1
            # step=1 -> lm_head[1] predicts group 2, etc.
            logits = self.compute_logits(hidden_states[:, -1, :], step)  # [seq_len, vocab]
            #print(f"code predictor #{step}: logits: {logits.shape}: {torch.max(logits)}", flush=True)
            
            # Sample next token
            # Prepare context for repetition penalty (all previously generated codecs in this frame)
            # We stack them to get [seq_len, num_generated]
            if repetition_penalty != 1.0:
                 current_context = torch.stack(all_codecs, dim=1)
            else:
                 current_context = None

            next_token = _sample_from_logits(
                logits, do_sample, temperature, top_k, top_p, repetition_penalty, current_context
            )
            #print(f"code predictor #{step}: sampled code: {next_token}", flush=True)
            all_codecs.append(next_token)
            
            # Prepare embedding for next step (if not last step)
            if step < self.num_code_groups - 2:
                # step=0 -> get_input_embeddings()[0] embeds group 1's token
                # step=1 -> get_input_embeddings()[1] embeds group 2's token, etc.
                next_embed = self.get_input_embeddings()[step](next_token)  # [seq_len, hidden]
                inputs_embeds = torch.cat([
                    inputs_embeds, 
                    next_embed.unsqueeze(1)
                ], dim=1)
        
        return torch.stack(all_codecs, dim=1)  # [seq_len, num_code_groups]


class Qwen3TTSTalkerForConditionalGeneration(nn.Module, SupportsPP):
    """Qwen3TTS Talker for conditional generation.
    
    This model generates codec tokens conditioned on text input.
    It contains:
    - model: Qwen3TTSTalkerModel (transformer backbone)
    - text_projection: MLP to project text embeddings to hidden size
    - codec_head: Linear head for codec token prediction
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
            # Fall back to defaults if no TTS config
            self.do_sample = True
            self.temperature = 1.0
            self.top_k = 50
            self.top_p = 1.0
            self.repetition_penalty = 1.0
        
        # Transformer model
        self.model = Qwen3TTSTalkerModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        
        # Text projection MLP
        if get_pp_group().is_first_rank:
            self.text_projection = Qwen3TTSTalkerResizeMLP(
                input_size=config.text_hidden_size,
                intermediate_size=config.text_hidden_size,
                output_size=config.hidden_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "text_projection"),
            )
        else:
            self.text_projection = PPMissingLayer()
        
        # Codec head for token prediction (first code group)
        if get_pp_group().is_last_rank:
            self.codec_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "codec_head"),
            )
        else:
            self.codec_head = PPMissingLayer()
        
        # Code predictor for additional code groups (native PyTorch, no vLLM attention)
        self.code_predictor = Qwen3TTSTalkerCodePredictor(
            config=config.code_predictor_config,
            talker_hidden_size=config.hidden_size,
        )
        
        self.logits_processor = LogitsProcessor(config.vocab_size)
        
        # Precomputed weights (loaded from checkpoint, computed by the
        # conversion script).  These must be static tensors so they
        # are CUDA-graph safe (no dynamic creation or in-place mutation
        # during forward).  Using nn.Parameter(requires_grad=False) so
        # vLLM natively handles dtype casting.
        #
        # tts_pad_embed: text_projection(text_embedding(tts_pad_token_id))
        #   Added to codec embeddings at every autoregressive step to
        #   maintain the dual-stream text+codec architecture.
        self.tts_pad_embed = nn.Parameter(
            torch.zeros(config.hidden_size),
            requires_grad=False,
        )
        # suppress_mask: bool mask [vocab_size] – True for the top 1024
        #   token IDs (except EOS) that the original model suppresses.
        self.suppress_mask = nn.Parameter(
            torch.zeros(config.vocab_size, dtype=torch.bool),
            requires_grad=False,
        )
        
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get codec embeddings for input ids."""
        return self.model.get_input_embeddings(input_ids)
    
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings for input ids."""
        return self.model.get_text_embeddings(input_ids)
    
    def project_text_embeddings(
        self, 
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Project text embeddings to hidden size."""
        return self.text_projection(text_embeds)

    def _embed_codecs(self, all_codecs: torch.Tensor) -> torch.Tensor:
        """Embed predicted codec tokens and sum them for next autoregressive step.
        
        Replicates the original Qwen3TTS embedding logic: each codebook group
        is embedded with its own embedding layer, then all embeddings are summed
        to produce the input for the next iteration.
        
        The original model additionally adds tts_pad_embed (the projected text
        embedding of the pad token) to maintain the dual-stream text+codec
        architecture.  In non-streaming mode all text is consumed during
        prefill, so every generation step adds the same tts_pad_embed.
        
        Args:
            all_codecs: [seq_len, num_code_groups] - all codec tokens
                        Column 0 is group 0 (main talker), columns 1.. are
                        from the code predictor.
                        
        Returns:
            next_input_embeds: [seq_len, hidden_size] - summed embeddings
                               ready to be fed as inputs_embeds for the next
                               autoregressive step.
        """
        # Embed group 0 using main codec embedding
        cb0_embed = self.model.codec_embedding(
            all_codecs[:, 0]
        )  # [seq_len, hidden_size]
        
        # Embed groups 1..N-1 using code predictor embeddings
        cp_embeddings = self.code_predictor.get_input_embeddings()
        cb_embeds = [cb0_embed]
        for i in range(len(cp_embeddings)):
            cb_embed = cp_embeddings[i](
                all_codecs[:, i + 1]
            )  # [seq_len, hidden_size]
            cb_embeds.append(cb_embed)
        
        # Stack and sum across codebook groups
        # [seq_len, num_code_groups, hidden_size] -> [seq_len, hidden_size]
        codec_hiddens = torch.stack(cb_embeds, dim=1)
        next_input_embeds = codec_hiddens.sum(dim=1)
        
        # Add tts_pad_embed to maintain dual-stream text+codec architecture.
        # In the original model this is:
        #   if generation_step < trailing_text_hidden.shape[1]:
        #       inputs_embeds += trailing_text_hidden[:, generation_step]
        #   else:
        #       inputs_embeds += tts_pad_embed
        # For non-streaming mode (all text consumed in prefill),
        # trailing_text_hidden == tts_pad_embed at every step.
        # tts_pad_embed is a precomputed buffer (loaded from weights).
        next_input_embeds = next_input_embeds + self.tts_pad_embed.unsqueeze(0)
        
        return next_input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        combined_embeddings: Optional[torch.Tensor] = None,
    ) -> Union[IntermediateTensors, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through the talker model.
        
        Runs the main transformer model, samples the first codec (group 0),
        then runs the code predictor to generate remaining 15 codec groups.
        Finally embeds all predicted codecs and sums them so the result can
        be fed directly as input embeddings for the next autoregressive step.
        
        Sampling parameters are taken from the config (do_sample, temperature,
        top_k, top_p) and are shared between the talker and code predictor.
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs for rotary embeddings
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed input embeddings
            combined_embeddings: Pre-computed combined embeddings
            
        Returns:
            For non-last PP rank: IntermediateTensors for pipeline parallelism
            For last PP rank: tuple of (hidden_states, all_codecs, next_input_embeds)
                - hidden_states: [seq_len, hidden_size] - final hidden states
                - all_codecs: [seq_len, num_code_groups] - all 16 codec tokens
                - next_input_embeds: [seq_len, 1, hidden_size] - summed codec
                  embeddings for the next autoregressive iteration
        """
        # Forward through main transformer model
        hidden_states = self.model(
            input_ids, 
            positions, 
            intermediate_tensors, 
            inputs_embeds,
            combined_embeddings
        )
        
        # Handle pipeline parallelism - intermediate ranks return tensors for next rank
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        
        # Compute logits for first codec (group 0)
        logits = self.logits_processor(self.codec_head, hidden_states)
        
        # Suppress reserved tokens using precomputed mask (CUDA-graph safe).
        # The mask is True for the top 1024 token IDs (except EOS).
        logits = logits.masked_fill(self.suppress_mask.bool(), float('-inf'))
        
        # Sample first codec token using config sampling params
        first_codec = _sample_from_logits(
            logits,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            previous_tokens=None, # No history access for first token
        )
        
        # Generate remaining codec groups (1-15) using code predictor
        # Uses same sampling params from config
        all_codecs = self.code_predictor.generate_all_groups(
            talker_hidden=hidden_states,
            first_codec=first_codec,
            talker_codec_embedding=self.model.codec_embedding,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        
        # Embed all predicted codecs and sum for next autoregressive step
        #print(f">>>>>all_codecs: {torch.min(all_codecs)} <-> {torch.max(all_codecs)}", flush=True)
        next_input_embeds = self._embed_codecs(all_codecs)
        
        return hidden_states, all_codecs, next_input_embeds

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return hidden_states for compatibility.
        
        Note: Actual logits computation and sampling happens inside forward().
        This method exists for compatibility with vLLM's model runner interface.
        """
        return hidden_states

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
