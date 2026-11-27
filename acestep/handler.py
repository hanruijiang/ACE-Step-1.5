"""
Business Logic Handler
Encapsulates all data processing and business logic for CLI inference
"""
import os
import sys

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import threading
from typing import Optional

import torch
import warnings

from acestep.gpu_config import get_global_gpu_config as _get_gpu_cfg
from acestep.core.generation.handler import (
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    DiffusionMixin,
    GenerateMusicDecodeMixin,
    GenerateMusicExecuteMixin,
    GenerateMusicMixin,
    GenerateMusicPayloadMixin,
    GenerateMusicRequestMixin,
    InitServiceMixin,
    IoAudioMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    MlxDitInitMixin,
    MlxVaeDecodeNativeMixin,
    MlxVaeEncodeNativeMixin,
    MlxVaeInitMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    ServiceGenerateMixin,
    TaskUtilsMixin,
    VaeDecodeChunksMixin,
    VaeDecodeMixin,
    VaeEncodeChunksMixin,
    VaeEncodeMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateOutputsMixin,
)


warnings.filterwarnings("ignore")


class AceStepHandler(
    DiffusionMixin,
    GenerateMusicMixin,
    GenerateMusicDecodeMixin,
    GenerateMusicPayloadMixin,
    GenerateMusicExecuteMixin,
    GenerateMusicRequestMixin,
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    IoAudioMixin,
    InitServiceMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    MlxDitInitMixin,
    MlxVaeDecodeNativeMixin,
    MlxVaeEncodeNativeMixin,
    MlxVaeInitMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    ServiceGenerateMixin,
    TaskUtilsMixin,
    VaeDecodeChunksMixin,
    VaeDecodeMixin,
    VaeEncodeChunksMixin,
    VaeEncodeMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateOutputsMixin,
):
    """ACE-Step Business Logic Handler"""
    
    def __init__(self, project_root: Optional[str] = None, checkpoint_dir: Optional[str] = None):
        """Initialize runtime model handles, feature flags, and generation state."""
        self.project_root = project_root
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32  # Will be set based on device in initialize_service

        # VAE for audio encoding/decoding
        self.vae = None
        
        # Text encoder and tokenizer
        self.text_encoder = None
        self.text_tokenizer = None
        
        # Silence latent for initialization
        self.silence_latent = None
        
        # Sample rate
        self.sample_rate = 48000
        
        # Batch size
        self.batch_size = 2
        
        self.offload_to_cpu = False
        self.offload_dit_to_cpu = False
        self.compiled = False
        self.current_offload_cost = 0.0
        self.disable_tqdm = os.environ.get("ACESTEP_DISABLE_TQDM", "").lower() in ("1", "true", "yes") or not getattr(sys.stderr, 'isatty', lambda: False)()
        self.debug_stats = os.environ.get("ACESTEP_DEBUG_STATS", "").lower() in ("1", "true", "yes")
        self._last_diffusion_per_step_sec: Optional[float] = None
        self._progress_estimates_lock = threading.Lock()
        self._progress_estimates = {"records": []}
        self._progress_estimates_path = os.path.join(
            self._get_project_root(),
            ".cache",
            "acestep",
            "progress_estimates.json",
        )
        self._load_progress_estimates()
        self.last_init_params = None
        
        # Quantization state - tracks if model is quantized (int8_weight_only, fp8_weight_only, or w8a8_dynamic)
        # Populated during initialize_service, remains None if quantization is disabled
        self.quantization = None
        
        # MLX DiT acceleration (macOS Apple Silicon only)
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False

        # MLX VAE acceleration (macOS Apple Silicon only)
        self.mlx_vae = None
        self.use_mlx_vae = False
        # MLX VAE decode chunk size — auto-detected from gpu_config,
        # overridable via ACESTEP_MLX_VAE_CHUNK env var.
        self.mlx_vae_chunk_size = _get_gpu_cfg().mlx_vae_chunk_size

