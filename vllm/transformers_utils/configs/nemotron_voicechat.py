# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from transformers import PretrainedConfig


class NemotronVoicechatConfig(PretrainedConfig):
    model_type = "nemotron_voicechat"
    
    def __init__(
        self,
        # custom config, related sampling
        hidden_size: int = 1152,
        context_hidden_size: int = 1536,
        intermediate_size: int = 4608,
        num_quantizers: int = 31,
        codebook_size: int = 1024,
        num_iter: int = 8,
        top_p_or_k: float = 0.8,
        noise_scale: float = 0.8,
        exponent: float = 3.0,
        latent_size: int = 512,
        mog_low_rank: int = 64,
        mog_num_layers: int = 3,
        mog_num_predictions: int = 1024,
        mog_min_log_std: float = -4.0,
        mog_eps: float = 1e-6,

        # subword encoding config
        emb_backbone_config: Optional[dict] = None,
        emb_backbone_type: str = "t5gemma",
        max_char_len: int = 128,
        emb_char_vocab_size: int = 256,
        emb_vocab_size: int = 151936,

        gemma3_config: Optional[dict] = None,
        nemotron_h_config: Optional[dict] = None,

        **kwargs,
    ):
        # custom config, related sampling
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.intermediate_size = intermediate_size
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.num_iter = num_iter
        self.top_p_or_k = top_p_or_k
        self.noise_scale = noise_scale
        self.exponent = exponent
        self.latent_size = latent_size
        self.mog_low_rank = mog_low_rank
        self.mog_num_layers = mog_num_layers
        self.mog_num_predictions = mog_num_predictions
        self.mog_min_log_std = mog_min_log_std
        self.mog_eps = mog_eps

        # subword encoding config
        self.emb_backbone_config = emb_backbone_config
        self.emb_backbone_type = emb_backbone_type
        self.max_char_len = max_char_len
        self.emb_char_vocab_size = emb_char_vocab_size
        self.emb_vocab_size = emb_vocab_size

        self.gemma3_config = gemma3_config
        self.nemotron_h_config = nemotron_h_config

        super().__init__(**kwargs)
    
    def get_text_config(self, *args, **kwargs):
        """
        This is used to get global params of the model.
        Use bigger model's config.
        """
        from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig
        
        # If nemotron_h_config is a dict, convert it to NemotronHConfig
        if isinstance(self.nemotron_h_config, dict):
            return NemotronHConfig(**self.nemotron_h_config)
        return self.nemotron_h_config

