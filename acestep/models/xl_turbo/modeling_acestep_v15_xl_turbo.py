# Copyright 2025 The ACESTEO Team. All rights reserved.
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
import copy
import math
import time
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

# Transformers imports (sorted by submodule, then alphabetically)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from vector_quantize_pytorch import ResidualFSQ

# Local config import with fallback
try:
    from .configuration_acestep_v15 import AceStepConfig
except ImportError:
    from configuration_acestep_v15 import AceStepConfig

# Differential correction in wavelet domain (DCW).
from acestep.models.common.dcw_correction import DCWCorrector


logger = logging.get_logger(__name__)

from acestep.models.common.modeling_layers import (
    create_4d_mask,
    TimestepEmbedding,
    AceStepAttention,
    AceStepEncoderLayer,
    AceStepDiTLayer,
    AceStepPreTrainedModel,
    AceStepLyricEncoder,
    AttentionPooler,
    AudioTokenDetokenizer,
    Lambda,
    _repaint_step_injection,
    _repaint_boundary_blend,
)


def pack_sequences(hidden1: torch.Tensor, hidden2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
    """
    Pack two sequences by concatenating and sorting them based on mask values.

    Args:
        hidden1: First hidden states tensor of shape [B, L1, D]
        hidden2: Second hidden states tensor of shape [B, L2, D]
        mask1: First mask tensor of shape [B, L1]
        mask2: Second mask tensor of shape [B, L2]

    Returns:
        Tuple of (packed_hidden_states, new_mask) where:
        - packed_hidden_states: Packed hidden states with valid tokens (mask=1) first, shape [B, L1+L2, D]
        - new_mask: New mask tensor indicating valid positions, shape [B, L1+L2]
    """
    # Step 1: Concatenate hidden states and masks along sequence dimension
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)  # [B, L, D]
    mask_cat = torch.cat([mask1, mask2], dim=1)  # [B, L]

    B, L, D = hidden_cat.shape

    # Step 2: Sort indices so that mask values of 1 come before 0
    sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)  # [B, L]

    # Step 3: Reorder hidden states using sorted indices
    hidden_left = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(B, L, D))

    # Step 4: Create new mask based on valid sequence lengths
    lengths = mask_cat.sum(dim=1)  # [B]
    new_mask = (torch.arange(L, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1))

    return hidden_left, new_mask


class AceStepTimbreEncoder(AceStepPreTrainedModel):
    """
    Encoder for extracting timbre embeddings from reference audio.
    
    Processes packed reference audio acoustic features to extract timbre
    representations. Uses a special token (CLS-like) to aggregate information
    from the entire reference audio sequence. Outputs are unpacked back to
    batch format for use in conditioning.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Project acoustic features to model hidden size
        self.embed_tokens = nn.Linear(config.timbre_hidden_dim, config.hidden_size)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Special token for aggregating timbre information (prepended to sequence)
        self.special_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.layers = nn.ModuleList(
            [AceStepEncoderLayer(config, layer_idx) for layer_idx in range(config.num_timbre_encoder_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def unpack_timbre_embeddings(self, timbre_embs_packed, refer_audio_order_mask):
        """
        Unpack packed timbre embeddings into batch format.

        Args:
            timbre_embs_packed: Packed timbre embeddings of shape [N, d]
            refer_audio_order_mask: Order mask indicating batch assignment for each packed embedding

        Returns:
            Tuple of (unpacked_embeddings, mask):
            - unpacked_embeddings: Unpacked embeddings of shape [B, max_count, d]
            - new_mask: Mask indicating valid positions, shape [B, max_count]
        """
        N, d = timbre_embs_packed.shape
        device = timbre_embs_packed.device
        dtype = timbre_embs_packed.dtype
        
        # Get batch size
        B = int(refer_audio_order_mask.max().item() + 1)
        
        # Calculate element count and positions for each batch
        counts = torch.bincount(refer_audio_order_mask, minlength=B)
        max_count = counts.max().item()
        
        # Calculate positions within batch
        sorted_indices = torch.argsort(refer_audio_order_mask * N + torch.arange(N, device=device), stable=True)
        sorted_batch_ids = refer_audio_order_mask[sorted_indices]
        
        positions = torch.arange(N, device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), 
                                torch.cumsum(counts, dim=0)[:-1]])
        positions_in_sorted = positions - batch_starts[sorted_batch_ids]
        
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(N, device=device)
        positions_in_batch = positions_in_sorted[inverse_indices]
        
        # Use one-hot encoding and matrix multiplication (gradient-friendly approach)
        # Create one-hot encoding
        indices_2d = refer_audio_order_mask * max_count + positions_in_batch  # (N,)
        one_hot = F.one_hot(indices_2d, num_classes=B * max_count).to(dtype)  # (N, B*max_count)
        
        # Rearrange using matrix multiplication
        timbre_embs_flat = one_hot.t() @ timbre_embs_packed  # (B*max_count, d)
        timbre_embs_unpack = timbre_embs_flat.reshape(B, max_count, d)
        
        # Create mask indicating valid positions
        mask_flat = (one_hot.sum(dim=0) > 0).long()  # (B*max_count,)
        new_mask = mask_flat.reshape(B, max_count)
        
        return timbre_embs_unpack, new_mask

    @can_return_tuple
    def forward(
        self,
        refer_audio_acoustic_hidden_states_packed: Optional[torch.FloatTensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        inputs_embeds = refer_audio_acoustic_hidden_states_packed
        # Project embeddings: N x T x timbre_hidden_dim -> N x T x hidden_size
        inputs_embeds = self.embed_tokens(inputs_embeds)
        # Prepend special token for timbre aggregation (CLS-like token)
        inputs_embeds = torch.cat([self.special_token.expand(inputs_embeds.shape[0], 1, -1), inputs_embeds], dim=1)
        # Expand attention mask to account for the prepended CLS token
        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask,
            ], dim=1)
        # Cache position: only used for mask construction (not for actual caching)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        # Positional IDs
        position_ids = cache_position.unsqueeze(0)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        
        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None

        # SDPA and eager attention masks.
        # 必须手动生成 4D Mask [B, 1, L, L]
        
        # 1. Full Attention (Bidirectional, Global)
        full_attn_mask = create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            attention_mask=attention_mask,     # [B, L]
            sliding_window=None,
            is_sliding_window=False,
            is_causal=False                    # <--- 关键：双向注意力
        )

        # 2. Sliding Attention (Bidirectional, Local)
        if self.config.use_sliding_window:
            sliding_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask, # [B, L]
                sliding_window=self.config.sliding_window,
                is_sliding_window=True,        # <--- 开启滑动窗口
                is_causal=False                # <--- 关键：双向注意力
            )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }
        
        # Initialize hidden states
        hidden_states = inputs_embeds

        # Create position embeddings to be shared across all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through transformer layers
        for layer_module in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        # Extract special token output (first position) as timbre embedding: N x T x D -> N x D
        hidden_states = hidden_states[:, 0, :]
        # Unpack packed embeddings back to batch format
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(hidden_states, refer_audio_order_mask)
        return timbre_embs_unpack, timbre_embs_mask


class AceStepAudioTokenizer(AceStepPreTrainedModel):
    """
    Audio tokenizer module.
    
    Converts continuous acoustic features into discrete quantized tokens.
    Process: project -> pool patches -> quantize. Used for converting audio
    representations into discrete tokens for processing by the diffusion model.
    """
    def __init__(self, config):
        super().__init__(config)
        # Project acoustic features to hidden size
        self.audio_acoustic_proj = nn.Linear(config.audio_acoustic_hidden_dim, config.hidden_size)
        # Pool patches into sequence-level representations
        self.attention_pooler = AttentionPooler(config)
        # Quantize continuous representations into discrete tokens
        self.quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=config.fsq_input_levels,
            num_quantizers=config.fsq_input_num_quantizers
        )
        self.pool_window_size = config.pool_window_size
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        
        # Project acoustic features to hidden size
        hidden_states = self.audio_acoustic_proj(hidden_states)
        # Pool sequences: N x T//pool_window_size x pool_window_size x d -> N x T//pool_window_size x d
        hidden_states = self.attention_pooler(hidden_states)
        # Quantize continuous representations into discrete tokens: N x T//pool_window_size x d
        # ResidualFSQ internally disables autocast (forces float32 in FSQ layers),
        # which causes dtype mismatch with project_in/project_out when model is in bf16/fp16.
        # Solution: temporarily cast quantizer to float32, run, then restore.
        input_dtype = hidden_states.dtype
        quantizer_dtype = next(self.quantizer.parameters()).dtype
        if quantizer_dtype != torch.float32:
            self.quantizer.float()
        quantized, indices = self.quantizer(hidden_states.float())
        if quantizer_dtype != torch.float32:
            self.quantizer.to(quantizer_dtype)
        quantized = quantized.to(input_dtype)
        return quantized, indices

    def tokenize(self, x):
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=self.pool_window_size)
        quantized, indices = self.forward(x)
        return quantized, indices


class AceStepDiTModel(AceStepPreTrainedModel):
    """
    DiT (Diffusion Transformer) model for AceStep.
    
    Main diffusion model that generates audio latents conditioned on text, lyrics,
    and timbre. Uses patch-based processing with transformer layers, timestep
    conditioning, and cross-attention to encoder outputs.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        # Rotary position embeddings for transformer layers
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        # Stack of DiT transformer layers
        self.layers = nn.ModuleList(
            [AceStepDiTLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        in_channels = config.in_channels
        inner_dim = config.hidden_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        
        # Input projection: patch embedding using 1D convolution
        # Converts sequence into patches for efficient processing
        self.proj_in = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),  # [B, T, C] -> [B, C, T]
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(lambda x: x.transpose(1, 2)),  # [B, C, T//patch_size] -> [B, T//patch_size, C]
        )

        # Timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        
        # Project encoder hidden states to model dimension
        condition_dim = config.encoder_hidden_size
        self.condition_embedder = nn.Linear(condition_dim, inner_dim, bias=True)

        # Output normalization and projection
        # Adaptive layer norm with scale-shift modulation, then de-patchify
        self.norm_out = Qwen3RMSNorm(inner_dim, eps=config.rms_norm_eps)
        self.proj_out = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),  # [B, T//patch_size, inner_dim] -> [B, inner_dim, T//patch_size]
            nn.ConvTranspose1d(
                in_channels=inner_dim,
                out_channels=config.audio_acoustic_hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(lambda x: x.transpose(1, 2)),  # [B, out_channels, T] -> [B, T, out_channels]
        )
        # Scale-shift table for adaptive output normalization (2 values: shift, scale)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        return_hidden_states: int = None,
        custom_layers_config: Optional[dict] = None,
        enable_early_exit: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Disable cache during training or when gradient checkpointing is enabled
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if self.training:
            use_cache = False
    
        # Initialize cache if needed (only during inference for auto-regressive generation)
        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        # Compute timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        # Combine embeddings
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents (source latents + chunk masks) with hidden states
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        # Record original sequence length for later restoration after padding
        original_seq_len = hidden_states.shape[1]
        # Apply padding if sequence length is not divisible by patch_size
        # This ensures proper patch extraction
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode='constant', value=0)

        # Project input to patches and project encoder states
        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)
        
        # Cache positions
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
        
        # Position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        seq_len = hidden_states.shape[1]
        encoder_seq_len = encoder_hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None
        encoder_attention_mask = None
        attention_mask = None
        # SDPA and eager attention masks.
        # 必须手动生成 4D Mask [B, 1, L, L]
        
        # 1. Full Attention (Bidirectional, Global)
        full_attn_mask = create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            attention_mask=attention_mask,     # [B, L]
            sliding_window=None,
            is_sliding_window=False,
            is_causal=False                    # <--- 关键：双向注意力
        )
        max_len = max(seq_len, encoder_seq_len)
        
        encoder_attention_mask = create_4d_mask(
            seq_len=max_len,
            dtype=dtype,
            device=device,
            attention_mask=attention_mask,     # [B, L]
            sliding_window=None,
            is_sliding_window=False,
            is_causal=False                    # <--- 关键：双向注意力
        )
        encoder_attention_mask = encoder_attention_mask[:, :, :seq_len, :encoder_seq_len]
        # 2. Sliding Attention (Bidirectional, Local)
        if self.config.use_sliding_window:
            sliding_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask, # [B, L]
                sliding_window=self.config.sliding_window,
                is_sliding_window=True,        # <--- 开启滑动窗口
                is_causal=False                # <--- 关键：双向注意力
            )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_cross_attentions = () if output_attentions else None

        # Handle early exit for custom layer configurations
        max_needed_layer = float('inf')
        if custom_layers_config is not None and enable_early_exit:
            max_needed_layer = max(custom_layers_config.keys())
            # Force output_attentions to True when early exit is enabled for attention extraction
            output_attentions = True
            if all_cross_attentions is None:
                all_cross_attentions = ()

        # Process through transformer layers
        for index_block, layer_module in enumerate(self.layers):

            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                encoder_hidden_states,
                self_attn_mask_mapping["encoder_attention_mask"],
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and self.layers[index_block].use_cross_attention:
                # layer_outputs structure: (hidden_states, self_attn_weights, cross_attn_weights)
                # Extract the last element which is cross_attn_weights
                if len(layer_outputs) >= 3:
                    all_cross_attentions += (layer_outputs[2],)
        
        if return_hidden_states:
            return hidden_states

        # Extract scale-shift parameters for adaptive output normalization
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer norm: norm(x) * (1 + scale) + shift
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        # Project output: de-patchify back to original sequence format
        hidden_states = self.proj_out(hidden_states)
        
        # Crop back to original sequence length to ensure exact length match (remove padding)
        hidden_states = hidden_states[:, :original_seq_len, :]
        
        outputs = (hidden_states, past_key_values)

        if output_attentions:
            outputs += (all_cross_attentions,)
        return outputs


class AceStepConditionEncoder(AceStepPreTrainedModel):
    """
    Condition encoder for AceStep model.
    
    Encodes multiple conditioning inputs (text, lyrics, timbre) and packs them
    into a single sequence for cross-attention in the diffusion model. Handles
    projection, encoding, and sequence packing.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        self.config = config
        # Project text embeddings to model hidden size
        self.text_projector = nn.Linear(config.text_hidden_dim, config.hidden_size, bias=False)
        # Encoder for lyric text
        self.lyric_encoder = AceStepLyricEncoder(config)
        # Encoder for timbre from reference audio
        self.timbre_encoder = AceStepTimbreEncoder(config)

    def forward(
        self,
        # Text inputs
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        # Lyric inputs
        lyric_hidden_states: Optional[torch.LongTensor] = None,
        lyric_attention_mask: Optional[torch.Tensor] = None,
        # Reference audio for timbre
        refer_audio_acoustic_hidden_states_packed: Optional[torch.Tensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
    ):
        # Project and encode text
        text_hidden_states = self.text_projector(text_hidden_states)
        # Encode lyrics
        lyric_encoder_outputs = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask,
        )
        lyric_hidden_states = lyric_encoder_outputs.last_hidden_state
        # Encode timbre from reference audio
        timbre_embs_unpack, timbre_embs_mask = self.timbre_encoder(refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask)

        # Pack sequences: combine lyrics and timbre, then add text
        # This creates a single sequence with all conditioning information
        encoder_hidden_states, encoder_attention_mask = pack_sequences(lyric_hidden_states, timbre_embs_unpack, lyric_attention_mask, timbre_embs_mask)
        encoder_hidden_states, encoder_attention_mask = pack_sequences(encoder_hidden_states, text_hidden_states, encoder_attention_mask, text_attention_mask)
        return encoder_hidden_states, encoder_attention_mask


class AceStepConditionGenerationModel(AceStepPreTrainedModel):
    """
    Main conditional generation model for AceStep.
    
    End-to-end model for generating audio conditioned on text, lyrics, and timbre.
    Combines encoder (for conditioning), decoder (diffusion model), tokenizer
    (for discrete tokenization), and detokenizer (for reconstruction).
    Supports flow matching training and inference with various sampling methods.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        self.config = config
        # Diffusion model components
        self.decoder = AceStepDiTModel(config)  # Main diffusion transformer
        # Build encoder config: XL models use separate encoder dimensions
        # (encoder_hidden_size=2048 != decoder hidden_size=2560)
        encoder_config = copy.deepcopy(config)
        encoder_config.hidden_size = config.encoder_hidden_size
        encoder_config.intermediate_size = config.encoder_intermediate_size
        encoder_config.num_attention_heads = config.encoder_num_attention_heads
        encoder_config.num_key_value_heads = config.encoder_num_key_value_heads
        self.encoder = AceStepConditionEncoder(encoder_config)  # Condition encoder
        self.tokenizer = AceStepAudioTokenizer(encoder_config)  # Audio tokenizer
        self.detokenizer = AudioTokenDetokenizer(encoder_config)  # Audio detokenizer
        # Null condition embedding for classifier-free guidance
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, encoder_config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def tokenize(self, x, silence_latent, attention_mask):
        if x.shape[1] % self.config.pool_window_size != 0:
            pad_len = self.config.pool_window_size - (x.shape[1] % self.config.pool_window_size)
            x = torch.cat([x,  silence_latent[:1,:pad_len].repeat(x.shape[0],1,1)], dim=1)
            attention_mask = F.pad(attention_mask, (0, pad_len), mode='constant', value=0)
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=self.config.pool_window_size)
        seq_len = x.shape[1]
        chunk = math.ceil(attention_mask.shape[1] / seq_len)
        attention_mask = attention_mask.to(x.dtype)
        attention_mask = F.max_pool1d(attention_mask.unsqueeze(1), kernel_size=chunk, stride=chunk, ceil_mode=True).squeeze(1)
        quantized, indices = self.tokenizer(x)
        return quantized, indices, attention_mask

    def detokenize(self, quantized):
        """
        Detokenize quantized audio tokens back to continuous representations.

        Args:
            quantized: Quantized tokens of shape [N, T//pool_window_size, d]

        Returns:
            Detokenized hidden states of shape [N, T, d]
        """
        hidden_states = self.detokenizer(quantized)
        return hidden_states

    @torch.no_grad()
    def prepare_condition(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.Tensor,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
        silence_latent: torch.FloatTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.Tensor,
        is_covers: torch.Tensor,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: torch.FloatTensor = None,
    ):
        
        dtype = hidden_states.dtype
        encoder_hidden_states, encoder_attention_mask = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # N x T x d -> N x T//pool_window_size x pool_window_size x d
        # tokenize and detokenize to get LM hints for cover songs (when is_covers=True)
        # Use precomputed hints if provided (e.g., from audio codes), otherwise tokenize hidden_states
        if precomputed_lm_hints_25Hz is not None:
            print("Using precomputed LM hints")
            lm_hints_25Hz = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]
        else:
            if audio_codes is not None:
                lm_hints_5Hz = self.tokenize.quantizer.get_output_from_indices(audio_codes)
            else:
                lm_hints_5Hz, indices, llm_mask = self.tokenize(hidden_states, silence_latent, attention_mask)
            lm_hints_25Hz = self.detokenize(lm_hints_5Hz)
            # Crop lm_hints_25Hz to match src_latents length (tokenize may have added padding)
            lm_hints_25Hz = lm_hints_25Hz[:, :src_latents.shape[1], :]
        src_latents = torch.where(is_covers.unsqueeze(-1).unsqueeze(-1) > 0, lm_hints_25Hz, src_latents)
        # Concatenate source latents with chunk masks as context
        context_latents = torch.cat([src_latents, chunk_masks.to(dtype)], dim=-1)
        return encoder_hidden_states, encoder_attention_mask, context_latents

    def forward(
        self,
        # Diffusion inputs
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
        # Encoder inputs
        # Text
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        # Lyric
        lyric_hidden_states: Optional[torch.LongTensor] = None,
        lyric_attention_mask: Optional[torch.Tensor] = None,
        # Reference audio for timbre
        refer_audio_acoustic_hidden_states_packed: Optional[torch.Tensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
        src_latents: torch.FloatTensor = None,
        chunk_masks: torch.FloatTensor = None,
        is_covers: torch.Tensor = None,
        silence_latent: torch.FloatTensor = None,
        cfg_ratio: float = 0.15,
    ):
        """Training is intentionally unavailable in this inference-only distribution."""
        raise NotImplementedError("Use generate_audio() for CLI inference; training is not supported.")

    def prepare_noise(self, context_latents: torch.FloatTensor, seed: Union[int, List[int], None] = None):
        """
        Prepare noise tensor for generation with optional seeding.

        Args:
            context_latents: Context latents to determine noise shape
            seed: Can be int, List[int], or None. If None, uses random noise.

        Returns:
            Noise tensor of appropriate shape
        """
        bsz = context_latents.shape[0]
        device = context_latents.device
        dtype = context_latents.dtype
        # Handle seed: can be int, List[int], or None
        src_latents_shape = (context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1] // 2)
        if seed is None:
            # No seed provided - use random
            noise = torch.randn(src_latents_shape, device=device, dtype=dtype)
        elif isinstance(seed, list):
            # List of seeds - generate noise for each sample separately
            noise_list = []
            for i, s in enumerate(seed):
                if s is None or s < 0:
                    # Random seed for this sample
                    noise_i = torch.randn(1, src_latents_shape[1], src_latents_shape[2], device=device, dtype=dtype)
                else:
                    # Use specific seed for this sample
                    generator = torch.Generator(device=device).manual_seed(int(s))
                    noise_i = torch.randn(1, src_latents_shape[1], src_latents_shape[2], generator=generator, device=device, dtype=dtype)
                noise_list.append(noise_i)
            noise = torch.cat(noise_list, dim=0)
        else:
            # Single seed for all samples
            generator = torch.Generator(device=device).manual_seed(int(seed))
            noise = torch.randn(src_latents_shape, generator=generator, device=device, dtype=dtype)

        return noise

    def get_x0_from_noise(self, zt, vt, t):
        return zt - vt * t.unsqueeze(-1).unsqueeze(-1)

    def renoise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        if isinstance(t, torch.Tensor) and t.ndim != x.ndim:
            t = t.unsqueeze(-1).unsqueeze(-1)
        xt = t * noise + (1 - t) * x
        return xt

    def flowedit_generate_audio(self, **kwargs):
        """Flow-edit overlay (#1156) on the CFG-distilled XL-turbo variant.

        Force ``diffusion_guidance_scale=1.0`` because XL-turbo bakes
        CFG into the velocity head — paired-CFG would amplify a delta
        the model wasn't trained to produce.  Otherwise delegate to
        the shared pipeline.
        """
        from acestep.models.common.flow_edit_pipeline import (
            flowedit_generate_audio as _flowedit_impl,
        )

        gs = kwargs.get("diffusion_guidance_scale", 1.0)
        if gs != 1.0:
            from loguru import logger
            logger.info(
                "[xl-turbo flowedit] xl-turbo is CFG-distilled; forcing "
                "diffusion_guidance_scale=1.0 (was {:.2f}).", gs,
            )
            kwargs["diffusion_guidance_scale"] = 1.0
        return _flowedit_impl(self, **kwargs)

    def generate_audio(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.FloatTensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.FloatTensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.LongTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.FloatTensor,
        is_covers: torch.Tensor,
        silence_latent: Optional[torch.FloatTensor] = None,
        attention_mask: torch.Tensor = None,
        seed: int = None,
        retake_seed: Optional[Union[int, List[int]]] = None,
        retake_variance: float = 0.0,
        fix_nfe: int = 8,
        infer_steps: Optional[int] = None,
        infer_method: str = "ode",
        use_cache: bool = True,
        audio_cover_strength: float = 1.0,
        non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
        non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: Optional[torch.FloatTensor] = None,
        shift: float = 3.0,
        timesteps: Optional[torch.Tensor] = None,
        cover_noise_strength: float = 0.0,
        repaint_mask: Optional[torch.Tensor] = None,
        clean_src_latents: Optional[torch.FloatTensor] = None,
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        # --- CFG-related params accepted for API parity but unused on turbo.
        # Turbo models bake classifier-free guidance into the distillation
        # weights and do NOT run a twin unconditional forward pass.  Handlers
        # upstream already force guidance_scale=1.0 before it reaches us (see
        # `AceStepHandler.generate_music`), and the other CFG knobs have
        # no corresponding code path in this sampler.  We declare them
        # explicitly so they don't disappear into `**kwargs` silently, and
        # log an info when a caller sets a non-default value so it's visible
        # that the knob is intentionally a no-op here.
        diffusion_guidance_scale: float = 1.0,
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        **kwargs,
    ):
        if diffusion_guidance_scale != 1.0 or use_adg or cfg_interval_start != 0.0 or cfg_interval_end != 1.0:
            logger.info(
                "XL-Turbo DiT ignores CFG params (guidance_scale=%.2f, use_adg=%s, "
                "cfg_interval=[%.2f, %.2f]); turbo is CFG-distilled.",
                diffusion_guidance_scale, use_adg, cfg_interval_start, cfg_interval_end,
            )

        # Valid shifts: only discrete values 1, 2, 3 are supported
        VALID_SHIFTS = [1.0, 2.0, 3.0]
        
        # Valid timesteps: all unique timesteps from shift=1,2,3 with fix_nfe=8 (total 20 values)
        VALID_TIMESTEPS = [
            1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875, 
            0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75, 
            0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454, 
            0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125
        ]
        
        # Pre-defined timestep schedules for each valid shift (fix_nfe=8, excluding final 0)
        SHIFT_TIMESTEPS = {
            1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
            3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
        }
        
        # Determine the timestep schedule to use
        t_schedule_list = None
        
        if timesteps is not None:
            # Process custom timesteps: map each value to nearest valid timestep
            timesteps_list = timesteps.tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps)
            
            # Remove trailing zeros
            while len(timesteps_list) > 0 and timesteps_list[-1] == 0:
                timesteps_list.pop()
            
            # Validate length: 1-20
            if len(timesteps_list) < 1:
                logger.warning(f"timesteps length is too short after removing trailing zeros, using default shift={shift}")
            elif len(timesteps_list) > 20:
                logger.warning(f"timesteps length={len(timesteps_list)} exceeds maximum 20, truncating to 20")
                timesteps_list = timesteps_list[:20]
                t_schedule_list = timesteps_list
            else:
                t_schedule_list = timesteps_list
            
            if t_schedule_list is not None:
                # Map each timestep to nearest valid timestep
                original_timesteps = t_schedule_list.copy()
                mapped_timesteps = []
                for t in t_schedule_list:
                    nearest = min(VALID_TIMESTEPS, key=lambda x: abs(x - t))
                    mapped_timesteps.append(nearest)
                
                if original_timesteps != mapped_timesteps:
                    logger.warning(f"timesteps mapped to nearest valid values: {original_timesteps} -> {mapped_timesteps}")
                
                t_schedule_list = mapped_timesteps

        # Variable-step schedule: respect `infer_steps` (UI exposes 1–20).
        # Mirrors the base model's `torch.linspace` + shift transform so users
        # can trade quality for speed on turbo models too.  Only fires when
        # the caller did NOT pass an explicit `timesteps` list.
        if t_schedule_list is None and infer_steps is not None and infer_steps > 0:
            n = min(int(infer_steps), 20)
            if n != int(infer_steps):
                logger.warning(
                    "infer_steps=%d exceeds maximum 20, clamping to 20.", infer_steps
                )
            raw = [1.0 - i / n for i in range(n)]
            if shift != 1.0:
                raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
            t_schedule_list = raw

        if t_schedule_list is None:
            # Use shift-based schedule: round to nearest valid shift
            original_shift = shift
            shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
            if original_shift != shift:
                logger.warning(f"shift={original_shift} not supported, rounded to nearest valid shift={shift}")
            t_schedule_list = SHIFT_TIMESTEPS[shift]
        
        if attention_mask is None:
            latent_length = src_latents.shape[1]
            attention_mask = torch.ones(src_latents.shape[0], latent_length, device=src_latents.device, dtype=src_latents.dtype)
        
        time_costs = {}
        start_time = time.time()
        total_start_time = start_time
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes,
        )
        
        encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = None, None, None
        if audio_cover_strength < 1.0:
            non_is_covers = torch.zeros_like(is_covers, device=is_covers.device, dtype=is_covers.dtype)
            # Use silence_latent for non-cover condition to simulate text2music mode (no reference audio)
            silence_latent_expanded = silence_latent[:, :src_latents.shape[1], :].expand(src_latents.shape[0], -1, -1)
            encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = self.prepare_condition(
            text_hidden_states=non_cover_text_hidden_states,
            text_attention_mask=non_cover_text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_latent_expanded,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=silence_latent_expanded,
            chunk_masks=chunk_masks,
            is_covers=non_is_covers,
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )
        
        end_time = time.time()
        time_costs["encoder_time_cost"] = end_time - start_time
        start_time = end_time
        
        noise = self.prepare_noise(context_latents, seed)
        # Retake mixing: variance-preserving blend with an independent noise draw.
        # v=0 -> noise unchanged; v=1 -> equivalent to using retake_seed as the main seed.
        if retake_variance > 0.0:
            retake_noise = self.prepare_noise(context_latents, retake_seed)
            v_rad = retake_variance * (math.pi / 2.0)
            noise = math.cos(v_rad) * noise + math.sin(v_rad) * retake_noise
        bsz, device, dtype = context_latents.shape[0], context_latents.device, context_latents.dtype
        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        
        # Cover noise initialization: blend noise with src_latents
        if cover_noise_strength > 0.0:
            # cover_noise_strength=1 means closest to src, so noise_level should be low
            effective_noise_level = 1.0 - cover_noise_strength
            # Find nearest valid timestep from t_schedule
            nearest_t = min(t_schedule_list, key=lambda x: abs(x - effective_noise_level))
            # xt = nearest_t * noise + (1 - nearest_t) * src_latents
            xt = self.renoise(src_latents, nearest_t, noise)
            # Truncate t_schedule to start from nearest_t
            start_idx = t_schedule_list.index(nearest_t)
            t_schedule_list = t_schedule_list[start_idx:]
            logger.info(
                f"[generate_audio] Cover mode: cover_noise_strength={cover_noise_strength}, "
                f"effective_noise_level={effective_noise_level:.4f}, nearest_t={nearest_t:.4f}, "
                f"remaining_steps={len(t_schedule_list)}"
            )
        else:
            xt = noise
        
        # Use pre-computed t_schedule_list (already validated and mapped to valid timesteps)
        t_schedule = torch.tensor(t_schedule_list, device=device, dtype=dtype)
        num_steps = len(t_schedule)
        
        # Recalculate cover_steps based on actual num_steps
        use_heun = sampler_mode == "heun"
        use_norm_clamp = velocity_norm_threshold > 0.0
        use_ema = velocity_ema_factor > 0.0
        prev_vt = None
        if use_heun and infer_method == "sde":
            logger.warning("Heun sampler is not compatible with SDE; falling back to Euler.")
            use_heun = False

        # DCW — opt-in per-band wavelet-domain correction (CVPR 2026).
        # No-op unless `dcw_enabled=True` and a non-zero scaler is configured.
        dcw_corrector = DCWCorrector(
            enabled=dcw_enabled,
            mode=dcw_mode,
            scaler=dcw_scaler,
            high_scaler=dcw_high_scaler,
            wavelet=dcw_wavelet,
        )

        cover_steps = int(num_steps * audio_cover_strength)
        _switched_to_non_cover = False
        for step_idx in range(num_steps):
            current_timestep = t_schedule[step_idx].item()
            t_curr_tensor = current_timestep * torch.ones((bsz,), device=device, dtype=dtype)
            
            if step_idx >= cover_steps and not _switched_to_non_cover:
                _switched_to_non_cover = True
                encoder_hidden_states = encoder_hidden_states_non_cover
                encoder_attention_mask = encoder_attention_mask_non_cover
                context_latents = context_latents_non_cover
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            
            with torch.no_grad():        
                decoder_outputs = self.decoder(
                    hidden_states=xt,
                    timestep=t_curr_tensor,
                    timestep_r=t_curr_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                
            vt = decoder_outputs[0]
            past_key_values = decoder_outputs[1]

            # Velocity norm clamping — prevents outlier predictions
            if use_norm_clamp:
                vt_norm = torch.norm(vt, dim=(1, 2), keepdim=True)
                xt_norm = torch.norm(xt, dim=(1, 2), keepdim=True) + 1e-10
                scale = torch.clamp(velocity_norm_threshold * xt_norm / (vt_norm + 1e-10), max=1.0)
                vt = vt * scale

            # Velocity EMA smoothing — stabilises denoising trajectory
            if use_ema and prev_vt is not None:
                vt = (1.0 - velocity_ema_factor) * vt + velocity_ema_factor * prev_vt

            # Cache pre-step latent so DCW can reconstruct the predicted
            # clean sample `denoised = x - v * t` after the sampler update.
            # Also stash the raw velocity (pre-Heun-averaging) so the x0
            # reconstruction uses the single-evaluation v(t_curr), matching
            # the reference FLUX scheduler's `x0 = sample - sigma * v`.
            xt_before_step = xt
            vt_for_denoise = vt

            # On final step, directly compute x0 from noise
            if step_idx == num_steps - 1:
                xt = self.get_x0_from_noise(xt, vt, t_curr_tensor)
                if dcw_corrector.is_active:
                    t_unsq = current_timestep * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                    denoised = xt_before_step - vt_for_denoise * t_unsq
                    xt = dcw_corrector.apply(xt, denoised, current_timestep)
                prev_vt = vt
                break

            # Update x_t based on inference method
            if infer_method == "sde":
                # Stochastic Differential Equation: predict clean, then re-add noise
                pred_clean = self.get_x0_from_noise(xt, vt, t_curr_tensor)
                next_timestep = t_schedule[step_idx + 1].item()
                xt = self.renoise(pred_clean, next_timestep)
                t_after_step = next_timestep
            elif use_heun and infer_method == "ode":
                # Heun (second-order) ODE step via trapezoidal rule
                next_timestep = t_schedule[step_idx + 1].item()
                dt = current_timestep - next_timestep
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt_predicted = xt - vt * dt_tensor
                # Corrector: evaluate model at the predicted point
                t_next_tensor = next_timestep * torch.ones((bsz,), device=device, dtype=dtype)
                corrector_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
                with torch.no_grad():
                    decoder_outputs2 = self.decoder(
                        hidden_states=xt_predicted,
                        timestep=t_next_tensor,
                        timestep_r=t_next_tensor,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        context_latents=context_latents,
                        use_cache=False,
                        past_key_values=corrector_kv,
                    )
                vt2 = decoder_outputs2[0]
                if use_norm_clamp:
                    vt2_norm = torch.norm(vt2, dim=(1, 2), keepdim=True)
                    xt_pred_norm = torch.norm(xt_predicted, dim=(1, 2), keepdim=True) + 1e-10
                    scale2 = torch.clamp(velocity_norm_threshold * xt_pred_norm / (vt2_norm + 1e-10), max=1.0)
                    vt2 = vt2 * scale2
                if use_ema:
                    vt2 = (1.0 - velocity_ema_factor) * vt2 + velocity_ema_factor * vt
                vt_avg = 0.5 * (vt + vt2)
                xt = xt - vt_avg * dt_tensor
                vt = vt_avg
                t_after_step = next_timestep
            elif infer_method == "ode":
                # Ordinary Differential Equation: Euler method
                # dx/dt = -v, so x_{t+1} = x_t - v_t * dt
                next_timestep = t_schedule[step_idx + 1].item()
                dt = current_timestep - next_timestep
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * dt_tensor
                t_after_step = next_timestep

            # DCW correction — push x_next's frequency band(s) away from
            # the predicted clean sample.  Scaler decays with t_curr.
            if dcw_corrector.is_active:
                t_unsq = current_timestep * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                denoised = xt_before_step - vt_for_denoise * t_unsq
                xt = dcw_corrector.apply(xt, denoised, current_timestep)

            prev_vt = vt

            injection_cutoff = round(repaint_injection_ratio * num_steps)
            if repaint_mask is not None and clean_src_latents is not None and step_idx < injection_cutoff:
                xt = _repaint_step_injection(
                    xt, clean_src_latents, repaint_mask, t_after_step, noise,
                )
        
        x_gen = xt
        if repaint_mask is not None and clean_src_latents is not None and repaint_crossfade_frames > 0:
            x_gen = _repaint_boundary_blend(
                x_gen, clean_src_latents, repaint_mask, repaint_crossfade_frames,
            )
        end_time = time.time()
        time_costs["diffusion_time_cost"] = end_time - start_time
        time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / num_steps
        time_costs["total_time_cost"] = end_time - total_start_time
        return {
            "target_latents": x_gen,
            "time_costs": time_costs,
        }
