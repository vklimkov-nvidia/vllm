# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from transformers import PretrainedConfig


class Qwen3TTSConfig(PretrainedConfig):
    """Configuration for Qwen3TTS model (Qwen3TTSForConditionalGeneration).
    
    This config contains all parameters including nested talker_config,
    code_predictor_config, and speaker_encoder_config as attribute dictionaries.
    """
    
    model_type = "qwen3_tts"
    
    def __init__(
        self,
        # Special token IDs
        assistant_token_id: int = 77091,
        im_end_token_id: int = 151645,
        im_start_token_id: int = 151644,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        tts_pad_token_id: int = 151671,
        
        # Model info
        tokenizer_type: str = "qwen3_tts_tokenizer_12hz",
        tts_model_size: str = "1b7",
        tts_model_type: str = "base",
        
        # Sampling parameters (shared between talker and code predictor)
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        
        # Nested configs (stored as dicts; PretrainedConfig handles attribute access)
        talker_config: Optional[dict] = None,
        speaker_encoder_config: Optional[dict] = None,
        
        **kwargs,
    ):
        # Special token IDs
        self.assistant_token_id = assistant_token_id
        self.im_end_token_id = im_end_token_id
        self.im_start_token_id = im_start_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.tts_pad_token_id = tts_pad_token_id
        
        # Model info
        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type
        
        # Sampling parameters
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Store nested configs as plain dicts.
        # PretrainedConfig.from_dict() automatically provides attribute access
        # on nested dicts, so no custom namespace wrapper is needed.
        self.talker_config = talker_config or self._default_talker_config()
        self.speaker_encoder_config = (
            speaker_encoder_config or self._default_speaker_encoder_config()
        )
        
        # Build a PretrainedConfig from talker_config so that
        # vLLM's get_hf_text_config() → config.get_text_config() returns an
        # object with num_attention_heads, hidden_size, etc.
        #
        # IMPORTANT: We strip "mrope_section" from rope_scaling in text_config.
        # The talker uses MRoPE internally (all 3 dims get sequential positions),
        # but vLLM's infrastructure interprets mrope_section as "this model needs
        # VL-style MRoPE position computation" (image/video grid positions),
        # which would fail because TTS has no image_token_id etc.
        # The actual MRoPE handling is done inside Qwen3TTSTalkerAttention.
        text_cfg = dict(self.talker_config)
        if isinstance(text_cfg.get("rope_scaling"), dict):
            rs = dict(text_cfg["rope_scaling"])
            rs.pop("mrope_section", None)
            text_cfg["rope_scaling"] = rs or None
        self.text_config = PretrainedConfig(**text_cfg)

        super().__init__(**kwargs)
    
    @staticmethod
    def _default_talker_config() -> dict:
        """Default talker config values."""
        return {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "vocab_size": 3072,
            "num_code_groups": 16,
            "text_hidden_size": 2048,
            "text_vocab_size": 151936,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_theta": 1000000.0,
            "rope_scaling": {
                "interleaved": True,
                "mrope_interleaved": True,  # vLLM uses this key
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
                "type": "default",
            },
            "sliding_window": None,
            "use_sliding_window": False,
            "use_cache": True,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_think_id": 2154,
            "codec_nothink_id": 2155,
            "codec_pad_id": 2148,
            "codec_think_bos_id": 2156,
            "codec_think_eos_id": 2157,
            "codec_language_id": {},
            "spk_id": {},
            "spk_is_dialect": {},
            "position_id_per_seconds": 13,
            "code_predictor_config": {
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 5,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "hidden_act": "silu",
                "max_position_embeddings": 65536,
                "rms_norm_eps": 1e-6,
                "vocab_size": 2048,
                "num_code_groups": 16,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "rope_theta": 1000000.0,
                "rope_scaling": None,
                "use_cache": True,
            },
        }
    
    @staticmethod
    def _default_speaker_encoder_config() -> dict:
        """Default speaker encoder config values."""
        return {
            "enc_dim": 2048,
            "sample_rate": 24000,
        }
