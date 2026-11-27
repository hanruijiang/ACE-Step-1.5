"""
GPU Configuration Module
Centralized GPU memory detection and adaptive configuration management

    Debug Mode:
        Set environment variable MAX_CUDA_VRAM to simulate different GPU memory sizes.
        Example: MAX_CUDA_VRAM=8 python acestep  # Simulates 8GB GPU

        For MPS testing, use MAX_MPS_VRAM to simulate MPS memory.
        Example: MAX_MPS_VRAM=16 python acestep  # Simulates 16GB MPS

    This is useful for testing GPU tier configurations on high-end hardware.
"""

import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict
from loguru import logger


# Environment variable for debugging/testing different GPU memory configurations
DEBUG_MAX_CUDA_VRAM_ENV = "MAX_CUDA_VRAM"
DEBUG_MAX_MPS_VRAM_ENV = "MAX_MPS_VRAM"
DEBUG_MAX_XPU_VRAM_ENV = "MAX_XPU_VRAM"
SAVE_MEMORY_ENV = "ACESTEP_SAVE_MEMORY"

# Tolerance for 16GB detection: reported VRAM like 15.5GB is effectively 16GB hardware
# Real-world 16GB GPUs often report 15.7-15.9GB due to system/driver reservations
VRAM_16GB_TOLERANCE_GB = 0.5
VRAM_16GB_MIN_GB = 16.0 - VRAM_16GB_TOLERANCE_GB  # treat as 16GB class if >= this

# Threshold below which auto_offload is enabled.
# 16GB GPUs cannot hold DiT + VAE + text_encoder + LM simultaneously without offloading.
VRAM_AUTO_OFFLOAD_THRESHOLD_GB = 20.0



def is_mps_platform() -> bool:
    """Check if running on macOS with MPS (Apple Silicon) available.

    This is the canonical check used across the codebase to apply
    Mac-specific configuration overrides (no compile, no quantization,
    PyTorch MPS, no offload, etc.).
    """
    if sys.platform != "darwin":
        return False
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Return whether CUDA runtime is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def is_mps_available() -> bool:
    """Return whether MPS runtime is available."""
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def is_xpu_available() -> bool:
    """Return whether XPU runtime is available."""
    try:
        import torch

        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def is_rocm_available() -> bool:
    """Return whether the active CUDA device is an AMD ROCm/HIP device.

    On ROCm, PyTorch exposes the GPU as a CUDA device but also sets
    ``torch.version.hip``.  This function returns ``True`` only when
    *both* conditions hold: a CUDA device is present **and** the build
    is a ROCm/HIP build.
    """
    try:
        import torch

        return (
            torch.cuda.is_available()
            and hasattr(torch.version, "hip")
            and torch.version.hip is not None
        )
    except Exception:
        return False


def cuda_supports_bfloat16(device_index: int | None = None) -> bool:
    """Return whether a CUDA device supports native bfloat16 kernels."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(device_index)
        return major >= 8
    except Exception:
        return False


# ===========================================================================
# Empirical VRAM measurements (GB) -- model weights only, bf16 precision
# These values should be calibrated using scripts/profile_vram.py
# ===========================================================================

# Base model weights (loaded once at startup)
MODEL_VRAM = {
    "dit_turbo": 4.7,  # DiT turbo model weights (bf16)
    "dit_base": 4.7,  # DiT base model weights (bf16)
    "dit_xl_turbo": 9.0,  # DiT XL (4B) turbo model weights (bf16)
    "dit_xl_base": 9.0,  # DiT XL (4B) base model weights (bf16)
    "vae": 0.33,  # VAE (AutoencoderOobleck) weights (fp16)
    "text_encoder": 1.2,  # Qwen3-Embedding-0.6B text encoder (bf16)
    "silence_latent": 0.01,  # Silence latent tensor
    "cuda_context": 0.5,  # CUDA context + driver overhead
}

# Conservative per-item diffusion activation estimates (GB).
DIT_INFERENCE_VRAM_PER_BATCH = {
    "turbo": 0.3,  # GB per batch item (no CFG)
    "base": 0.6,  # GB per batch item (with CFG, 2x forward)
    "xl_turbo": 0.5,  # GB per batch item, XL (4B) no CFG, larger activations
    "xl_base": 1.0,  # GB per batch item, XL (4B) with CFG, 2x forward
}

# Safety margin to keep free for OS/driver/fragmentation (GB)
VRAM_SAFETY_MARGIN_GB = 0.5


def _has_path_token(token: str, path: str) -> bool:
    """Check if *token* appears as a delimited word in *path*.

    Matches when *token* is bounded by start/end of string or a common
    path delimiter (``/``, ``\\``, ``.``, ``_``, ``-``).
    """
    return re.search(rf"(^|[\\/._-]){token}($|[\\/._-])", path) is not None


def get_dit_type_from_path(config_path: str) -> str:
    """Derive the DiT type key from a model checkpoint path.

    Returns the checkpoint model family.

    Examples::

        "acestep-v15-xl-turbo"  -> "xl_turbo"
        "acestep-v15-xl-base"   -> "xl_base"
        "acestep-v15-xl-sft"    -> "xl_base"   (sft shares base VRAM profile)
        "acestep-v15-turbo"     -> "turbo"
        "acestep-v15-base"      -> "base"
        "acestep-v15-sft"       -> "base"       (sft shares base VRAM profile)
    """
    path = (config_path or "").lower()
    is_xl = _has_path_token("xl", path)

    if _has_path_token("turbo", path):
        variant = "turbo"
    else:
        # Both "base" and "sft" use the base VRAM profile (CFG doubles forward)
        variant = "base"

    return f"xl_{variant}" if is_xl else variant


@dataclass
class GPUConfig:
    """GPU configuration based on available memory"""

    tier: str  # "tier1", "tier2", etc. or "unlimited"
    gpu_memory_gb: float

    # Duration limits (in seconds)
    max_duration_with_lm: int  # When LM is initialized
    max_duration_without_lm: int  # When LM is not initialized

    # Batch size limits
    max_batch_size_with_lm: int
    max_batch_size_without_lm: int

    # LM configuration
    init_lm_default: bool  # Whether to initialize LM by default
    available_lm_models: List[str]  # Available LM models for this tier
    recommended_lm_model: (
        str  # Recommended default LM model path (empty if LM not available)
    )

    # Offload defaults
    offload_to_cpu_default: bool  # Whether offload_to_cpu should be enabled by default
    offload_dit_to_cpu_default: (
        bool  # Whether offload_dit_to_cpu should be enabled by default
    )

    # Quantization / compile defaults
    quantization_default: bool  # Whether INT8 quantization should be enabled by default
    compile_model_default: bool  # Whether torch.compile should be enabled by default

    # LM memory allocation (GB) for each model size
    lm_memory_gb: Dict[str, float]  # e.g., {"0.6B": 3, "1.7B": 8, "4B": 12}

    # Save-memory mode: skip retaining generated latents in extra_outputs.
    # Controlled via ACESTEP_SAVE_MEMORY=1 environment variable.
    save_memory_mode: bool = False

# GPU tier configurations
# Separate memory budgets for 16–20 GB and 20–24 GB GPUs.
GPU_TIER_CONFIGS = {
    "tier1": {  # <= 4GB
        # Offload mode required.  DiT(4.46) barely fits with CUDA context(0.5).
        # VAE decode falls back to CPU.  Keep durations moderate.
        "max_duration_with_lm": 240,  # 4 minutes
        "max_duration_without_lm": 360,  # 6 minutes
        "max_batch_size_with_lm": 1,
        "max_batch_size_without_lm": 1,
        "init_lm_default": False,
        "available_lm_models": [],
        "recommended_lm_model": "",
        "offload_to_cpu_default": True,
        "offload_dit_to_cpu_default": True,
        "quantization_default": True,  # INT8 essential to fit DiT in ~4GB
        "compile_model_default": True,
        "lm_memory_gb": {},
    },
    "tier2": {  # 4-6GB
        # Offload mode.  DiT(4.46) + context(0.5) + activations ≈ 5.0GB.
        # ~1GB headroom.  Tiled VAE decode fits with chunk=256 (~0.8GB peak).
        # Duration barely affects peak VRAM (latent tensor is <2MB even at 10min).
        "max_duration_with_lm": 480,  # 8 minutes
        "max_duration_without_lm": 600,  # 10 minutes (max supported)
        "max_batch_size_with_lm": 1,
        "max_batch_size_without_lm": 1,
        "init_lm_default": False,
        "available_lm_models": [],
        "recommended_lm_model": "",
        "offload_to_cpu_default": True,
        "offload_dit_to_cpu_default": True,
        "quantization_default": True,
        "compile_model_default": True,
        "lm_memory_gb": {},
    },
    "tier3": {  # 6-8GB
        # Offload mode.  DiT(4.46) + context(0.5) ≈ 5.0GB.
        # ~1.5-3GB headroom allows LM 0.6B (1.2+0.6=1.8GB) and batch=2.
        # With CPU offload, DiT is offloaded before LM runs → the LM can use freed VRAM.
        "max_duration_with_lm": 480,  # 8 minutes
        "max_duration_without_lm": 600,  # 10 minutes (max supported)
        "max_batch_size_with_lm": 2,
        "max_batch_size_without_lm": 2,
        "init_lm_default": True,
        "available_lm_models": ["acestep-5Hz-lm-0.6B"],
        "recommended_lm_model": "acestep-5Hz-lm-0.6B",
        "offload_to_cpu_default": True,
        "offload_dit_to_cpu_default": True,
        "quantization_default": True,
        "compile_model_default": True,
        "lm_memory_gb": {"0.6B": 3},
    },
    "tier4": {  # 8-12GB
        # Can keep DiT + 0.6B LM simultaneously on GPU (4.46+1.2+0.6=6.26GB).
        # Offload VAE/TextEnc.  Plenty of room for inference activations.
        "max_duration_with_lm": 480,  # 8 minutes
        "max_duration_without_lm": 600,  # 10 minutes (max supported)
        "max_batch_size_with_lm": 2,
        "max_batch_size_without_lm": 4,
        "init_lm_default": True,
        "available_lm_models": ["acestep-5Hz-lm-0.6B"],
        "recommended_lm_model": "acestep-5Hz-lm-0.6B",
        "offload_to_cpu_default": True,
        "offload_dit_to_cpu_default": True,
        "quantization_default": True,
        "compile_model_default": True,
        "lm_memory_gb": {"0.6B": 3},
    },
    "tier5": {  # 12-16GB
        # DiT + 1.7B LM (4.46+3.45+0.44=8.35GB) fits comfortably.
        # VAE decode is batch-sequential so batch size doesn't affect VAE VRAM.
        "max_duration_with_lm": 480,  # 8 minutes
        "max_duration_without_lm": 600,  # 10 minutes (max supported)
        "max_batch_size_with_lm": 4,
        "max_batch_size_without_lm": 4,
        "init_lm_default": True,
        "available_lm_models": ["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B"],
        "recommended_lm_model": "acestep-5Hz-lm-1.7B",
        "offload_to_cpu_default": True,
        "offload_dit_to_cpu_default": False,  # 12-16GB can keep DiT on GPU
        "quantization_default": True,
        "compile_model_default": True,
        "lm_memory_gb": {"0.6B": 3, "1.7B": 8},
    },
    "tier6a": {  # 16-20GB (e.g., RTX 4060 Ti 16GB, RTX 3080 16GB)
        # On 16GB GPUs: DiT(INT8, ~2.4GB) + LM 1.7B(~7.6GB peak with offload) = ~10GB peak
        # Empirical batch tests (60s, turbo): noLM-4→13.3GB, LM-2→11.9GB, LM-4→~13.5GB
        # With CPU offload, LM is offloaded after inference → DiT batch has full 16GB budget.
        "max_duration_with_lm": 480,  # 8 minutes
        "max_duration_without_lm": 600,  # 10 minutes (max supported)
        "max_batch_size_with_lm": 4,
        "max_batch_size_without_lm": 8,
        "init_lm_default": True,
        "available_lm_models": ["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B"],
        "recommended_lm_model": "acestep-5Hz-lm-1.7B",
        "offload_to_cpu_default": True,  # Still offload VAE/TextEnc to save VRAM for LM
        "offload_dit_to_cpu_default": False,
        "quantization_default": True,
        "compile_model_default": True,
        "lm_memory_gb": {"0.6B": 3, "1.7B": 8},
    },
    "tier6b": {  # 20-24GB (e.g., RTX 3090, RTX 4090)
        # 20-24GB: no offload, no quantization. DiT(bf16, ~4.7GB) + LM 1.7B(~3.4GB) = ~8.1GB
        # Remaining ~12-16GB easily fits batch=8. VAE decode is batch-sequential.
        "max_duration_with_lm": 480,  # 8 minutes
        "max_duration_without_lm": 480,  # 8 minutes
        "max_batch_size_with_lm": 8,
        "max_batch_size_without_lm": 8,
        "init_lm_default": True,
        "available_lm_models": [
            "acestep-5Hz-lm-0.6B",
            "acestep-5Hz-lm-1.7B",
            "acestep-5Hz-lm-4B",
        ],
        "recommended_lm_model": "acestep-5Hz-lm-1.7B",
        "offload_to_cpu_default": False,  # 20-24GB can hold all models
        "offload_dit_to_cpu_default": False,
        "quantization_default": False,  # Enough VRAM, quantization optional
        "compile_model_default": True,
        "lm_memory_gb": {"0.6B": 3, "1.7B": 8, "4B": 12},
    },
    "unlimited": {  # >= 24GB
        "max_duration_with_lm": 600,  # 10 minutes (max supported)
        "max_duration_without_lm": 600,  # 10 minutes
        "max_batch_size_with_lm": 8,
        "max_batch_size_without_lm": 8,
        "init_lm_default": True,
        "available_lm_models": [
            "acestep-5Hz-lm-0.6B",
            "acestep-5Hz-lm-1.7B",
            "acestep-5Hz-lm-4B",
        ],
        "recommended_lm_model": "acestep-5Hz-lm-4B",
        "offload_to_cpu_default": False,
        "offload_dit_to_cpu_default": False,
        "quantization_default": False,  # Plenty of VRAM
        "compile_model_default": True,
        "lm_memory_gb": {"0.6B": 3, "1.7B": 8, "4B": 12},
    },
}

# 20–24 GB tier alias
GPU_TIER_CONFIGS["tier6"] = GPU_TIER_CONFIGS["tier6b"]


def get_gpu_memory_gb() -> float:
    """
    Get GPU memory in GB. Returns 0 if no GPU is available.

    Debug Mode:
        Set environment variable MAX_CUDA_VRAM to override the detected GPU memory.
        Example: MAX_CUDA_VRAM=8 python acestep  # Simulates 8GB GPU

        For MPS testing, set MAX_MPS_VRAM to override MPS memory detection.
        Example: MAX_MPS_VRAM=16 python acestep  # Simulates 16GB MPS

        This allows testing different GPU tier configurations on high-end hardware.
    """
    # Check for debug override first
    debug_vram = os.environ.get(DEBUG_MAX_CUDA_VRAM_ENV)
    if debug_vram is not None:
        try:
            simulated_gb = float(debug_vram)
            logger.warning(
                f"⚠️ DEBUG MODE: Simulating GPU memory as {simulated_gb:.1f}GB (set via {DEBUG_MAX_CUDA_VRAM_ENV} environment variable)"
            )
            # Also enforce a hard VRAM cap via PyTorch so that the allocator
            # cannot use more than the simulated amount.  This makes the
            # simulation realistic — without it, models still load into the
            # real (larger) GPU memory and nvitop shows much higher usage.
            try:
                import torch

                if torch.cuda.is_available():
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    total_gb = total_bytes / (1024**3)
                    if simulated_gb < total_gb:
                        # When simulating a smaller GPU on a larger one, the host
                        # GPU's CUDA context is typically much bigger (e.g. A100
                        # ~1.4GB vs GTX 1060 ~0.3GB).  Using the host context
                        # would over-penalise the allocator budget.
                        #
                        # Instead we use a *reference* context size that matches
                        # what the target-class GPU would actually have.  Consumer
                        # GPUs (≤24GB) typically have 0.3-0.5GB context overhead.
                        REFERENCE_CONTEXT_GB = MODEL_VRAM.get("cuda_context", 0.5)
                        allocator_budget_gb = max(
                            0.5, simulated_gb - REFERENCE_CONTEXT_GB
                        )
                        fraction = allocator_budget_gb / total_gb
                        # Clamp to [0.01, 1.0] to satisfy PyTorch constraints
                        fraction = max(0.01, min(1.0, fraction))
                        torch.cuda.set_per_process_memory_fraction(fraction)
                        logger.warning(
                            f"⚠️ DEBUG MODE: Set CUDA memory fraction to {fraction:.4f} "
                            f"(allocator_budget={allocator_budget_gb:.2f}GB, "
                            f"ref_context={REFERENCE_CONTEXT_GB:.2f}GB, target={simulated_gb:.1f}GB, "
                            f"total={total_gb:.1f}GB) to enforce hard VRAM cap"
                        )
            except Exception as e:
                logger.warning(f"⚠️ DEBUG MODE: Could not enforce CUDA memory cap: {e}")
            return simulated_gb
        except ValueError:
            logger.warning(
                f"Invalid {DEBUG_MAX_CUDA_VRAM_ENV} value: {debug_vram}, ignoring"
            )
    debug_mps_vram = os.environ.get(DEBUG_MAX_MPS_VRAM_ENV)
    if debug_mps_vram is not None:
        try:
            simulated_gb = float(debug_mps_vram)
            logger.warning(
                f"⚠️ DEBUG MODE: Simulating MPS memory as {simulated_gb:.1f}GB (set via {DEBUG_MAX_MPS_VRAM_ENV} environment variable)"
            )
            return simulated_gb
        except ValueError:
            logger.warning(
                f"Invalid {DEBUG_MAX_MPS_VRAM_ENV} value: {debug_mps_vram}, ignoring"
            )

    # XPU debug override
    debug_xpu_vram = os.environ.get(DEBUG_MAX_XPU_VRAM_ENV)
    if debug_xpu_vram is not None:
        try:
            simulated_gb = float(debug_xpu_vram)
            logger.warning(
                f"⚠️ DEBUG MODE: Simulating XPU memory as {simulated_gb:.1f}GB (set via {DEBUG_MAX_XPU_VRAM_ENV} environment variable)"
            )
            return simulated_gb
        except ValueError:
            logger.warning(
                f"Invalid {DEBUG_MAX_XPU_VRAM_ENV} value: {debug_xpu_vram}, ignoring"
            )

    try:
        import torch

        if torch.cuda.is_available():
            # Get total memory of the first GPU in GB
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)  # Convert bytes to GB
            device_name = torch.cuda.get_device_name(0)
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            if is_rocm:
                logger.info(
                    f"ROCm GPU detected: {device_name} ({memory_gb:.1f} GB, HIP {torch.version.hip})"
                )
            else:
                logger.info(f"CUDA GPU detected: {device_name} ({memory_gb:.1f} GB)")
            return memory_gb
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            # Get total memory of the first XPU in GB
            total_memory = torch.xpu.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)  # Convert bytes to GB
            device_name = getattr(
                torch.xpu.get_device_properties(0), "name", "Intel XPU"
            )
            logger.info(f"Intel XPU detected: {device_name} ({memory_gb:.1f} GB)")
            return memory_gb
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_module = getattr(torch, "mps", None)
            try:
                if mps_module is not None and hasattr(
                    mps_module, "recommended_max_memory"
                ):
                    total_memory = mps_module.recommended_max_memory()
                    memory_gb = total_memory / (1024**3)  # Convert bytes to GB
                    return memory_gb
                if mps_module is not None and hasattr(
                    mps_module, "get_device_properties"
                ):
                    props = mps_module.get_device_properties(0)
                    total_memory = getattr(props, "total_memory", None)
                    if total_memory:
                        memory_gb = total_memory / (1024**3)
                        return memory_gb
            except Exception as e:
                logger.warning(f"Failed to detect MPS memory: {e}")

            # Fallback: estimate from system unified memory (Apple Silicon shares CPU/GPU RAM)
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                total_system_bytes = int(result.stdout.strip())
                # MPS can use up to ~75% of unified memory for GPU workloads
                memory_gb = (total_system_bytes / (1024**3)) * 0.75
                return memory_gb
            except Exception:
                logger.warning(
                    f"MPS available but total memory not exposed. Set {DEBUG_MAX_MPS_VRAM_ENV} to enable tiering."
                )
                # Conservative fallback for M1/M2
                return 8.0
        else:
            # No GPU detected - provide diagnostic information
            _log_gpu_diagnostic_info(torch)
            return 0
    except Exception as e:
        logger.warning(f"Failed to detect GPU memory: {e}")
        return 0


def _log_gpu_diagnostic_info(torch_module):
    """
    Log diagnostic information when GPU is not detected to help users troubleshoot.

    Args:
        torch_module: The torch module to inspect for build information
    """
    logger.warning("=" * 80)
    logger.warning("⚠️ GPU NOT DETECTED - DIAGNOSTIC INFORMATION")
    logger.warning("=" * 80)

    # Check PyTorch build type
    is_rocm_build = (
        hasattr(torch_module.version, "hip") and torch_module.version.hip is not None
    )
    is_cuda_build = (
        hasattr(torch_module.version, "cuda") and torch_module.version.cuda is not None
    )

    if is_rocm_build:
        logger.warning("✓ PyTorch ROCm build detected")
        logger.warning(f"  HIP version: {torch_module.version.hip}")
        logger.warning("")
        logger.warning("❌ torch.cuda.is_available() returned False")
        logger.warning("")
        logger.warning("Common causes for AMD/ROCm GPUs:")
        logger.warning("  1. ROCm drivers not installed or not properly configured")
        logger.warning("  2. GPU not supported by installed ROCm version")
        logger.warning(
            "  3. Missing or incorrect HSA_OVERRIDE_GFX_VERSION environment variable"
        )
        logger.warning("  4. ROCm runtime libraries not in system path")
        logger.warning("")

        # Check for common environment variables
        hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        if hsa_override:
            logger.warning(f"  HSA_OVERRIDE_GFX_VERSION is set to: {hsa_override}")
        else:
            logger.warning("  ⚠️ HSA_OVERRIDE_GFX_VERSION is not set")
            logger.warning("     For RDNA3 GPUs (RX 7000 series, RX 9000 series):")
            logger.warning(
                "       - RX 7900 XT/XTX, RX 9070 XT: set HSA_OVERRIDE_GFX_VERSION=11.0.0"
            )
            logger.warning(
                "       - RX 7800 XT, RX 7700 XT: set HSA_OVERRIDE_GFX_VERSION=11.0.1"
            )
            logger.warning("       - RX 7600: set HSA_OVERRIDE_GFX_VERSION=11.0.2")

        logger.warning("")
        logger.warning("Troubleshooting steps:")
        logger.warning("  1. Verify ROCm installation:")
        logger.warning("     rocm-smi  # Should list your GPU")
        logger.warning("  2. Check PyTorch ROCm build:")
        logger.warning(
            "     python -c \"import torch; print(f'ROCm: {torch.version.hip}')\""
        )
        logger.warning("  3. Set HSA_OVERRIDE_GFX_VERSION for your GPU (see above)")
        logger.warning(
            "  4. On Windows: Install the ROCm-compatible PyTorch build and configure its environment"
        )
        logger.warning(
            "  5. See README.md and requirements-rocm-linux.txt for Linux setup"
        )
        logger.warning(
            "  6. See requirements-rocm.txt for Windows ROCm setup instructions"
        )

    elif is_cuda_build:
        logger.warning("✓ PyTorch CUDA build detected")
        logger.warning(f"  CUDA version: {torch_module.version.cuda}")
        logger.warning("")
        logger.warning("❌ torch.cuda.is_available() returned False")
        logger.warning("")
        logger.warning("Common causes for NVIDIA GPUs:")
        logger.warning("  1. NVIDIA drivers not installed")
        logger.warning("  2. CUDA runtime not installed or version mismatch")
        logger.warning("  3. GPU not supported by installed CUDA version")
        logger.warning("")
        logger.warning("Troubleshooting steps:")
        logger.warning("  1. Verify NVIDIA driver installation:")
        logger.warning("     nvidia-smi  # Should list your GPU")
        logger.warning("  2. Check CUDA version compatibility")
        logger.warning("  3. Reinstall PyTorch with CUDA support:")
        logger.warning("     Follow README.md and requirements.txt for the matching CUDA build.")

    else:
        logger.warning("⚠️ PyTorch build type: CPU-only")
        logger.warning("")
        logger.warning("You have installed a CPU-only version of PyTorch!")
        logger.warning("")
        logger.warning("For NVIDIA GPUs:")
        logger.warning("  See README.md and requirements.txt for CUDA installation.")
        logger.warning("")
        logger.warning("For AMD GPUs with ROCm:")
        logger.warning("  Windows: See requirements-rocm.txt for detailed instructions")
        logger.warning(
            "  Linux: See requirements-rocm-linux.txt and README.md."
        )
        logger.warning("")
        logger.warning("For more information, see README.md section 'AMD / ROCm GPUs'")

    logger.warning("=" * 80)


def get_gpu_tier(gpu_memory_gb: float) -> str:
    """
    Determine GPU tier based on available memory.

    Args:
        gpu_memory_gb: GPU memory in GB

    Returns:
        Tier string: "tier1", "tier2", "tier3", "tier4", "tier5", "tier6a", "tier6b", or "unlimited"
    """
    if gpu_memory_gb <= 0:
        # CPU mode - use tier1 limits
        return "tier1"
    elif gpu_memory_gb <= 4:
        return "tier1"
    elif gpu_memory_gb <= 6:
        return "tier2"
    elif gpu_memory_gb <= 8:
        return "tier3"
    elif gpu_memory_gb <= 12:
        return "tier4"
    elif gpu_memory_gb < VRAM_16GB_MIN_GB:
        return "tier5"
    elif gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB:
        # 16-20GB range: tier6a (constrained, needs offload)
        if gpu_memory_gb < 16.0:
            logger.info(
                f"Detected {gpu_memory_gb:.2f}GB VRAM — treating as 16GB class GPU"
            )
        return "tier6a"
    elif gpu_memory_gb <= 24:
        return "tier6b"
    else:
        return "unlimited"


def get_gpu_config(gpu_memory_gb: Optional[float] = None) -> GPUConfig:
    """
    Get GPU configuration based on detected or provided GPU memory.

    On macOS with MPS (Apple Silicon), several overrides are applied
    automatically regardless of the tier selected by memory size:

    - ``compile_model_default = False`` — ``torch.compile`` is not supported
      on MPS and would error or silently fall back to eager mode.
    - ``quantization_default = False`` — torchao INT8 quantization is
      incompatible with MPS / macOS.
    - ``offload_to_cpu_default = False`` — Apple Silicon uses unified memory;
      offloading to CPU provides no benefit and adds overhead.
    - ``offload_dit_to_cpu_default = False`` — same reason.

    Args:
        gpu_memory_gb: GPU memory in GB. If None, will be auto-detected.

    Returns:
        GPUConfig object with all configuration parameters
    """
    if gpu_memory_gb is None:
        gpu_memory_gb = get_gpu_memory_gb()

    tier = get_gpu_tier(gpu_memory_gb)
    config = GPU_TIER_CONFIGS[tier]

    # --- MPS (Apple Silicon) overrides ---
    _mps = is_mps_platform()
    if _mps:
        logger.info(
            f"macOS MPS detected ({gpu_memory_gb:.1f} GB unified memory, tier={tier}). "
            "Applying Apple Silicon optimizations: no compile, no quantization, "
            "PyTorch MPS, no CPU offload."
        )

    config = GPUConfig(
        tier=tier,
        gpu_memory_gb=gpu_memory_gb,
        max_duration_with_lm=config["max_duration_with_lm"],
        max_duration_without_lm=config["max_duration_without_lm"],
        max_batch_size_with_lm=config["max_batch_size_with_lm"],
        max_batch_size_without_lm=config["max_batch_size_without_lm"],
        init_lm_default=config["init_lm_default"],
        available_lm_models=config["available_lm_models"],
        recommended_lm_model=config.get("recommended_lm_model", ""),
        # MPS: unified memory — offloading to CPU is pointless overhead
        offload_to_cpu_default=False
        if _mps
        else config.get("offload_to_cpu_default", True),
        offload_dit_to_cpu_default=False
        if _mps
        else config.get("offload_dit_to_cpu_default", True),
        # MPS: torchao quantization is not supported
        quantization_default=False
        if _mps
        else config.get("quantization_default", True),
        # MPS: keep torch.compile disabled.
        compile_model_default=False
        if _mps
        else config.get("compile_model_default", True),
        lm_memory_gb=config["lm_memory_gb"],
    )
    return config


def get_effective_free_vram_gb(device_index: int = 0) -> float:
    """
    Get the effective free VRAM in GB, accounting for PyTorch allocator cache and
    per-process memory fraction.

    torch.cuda.mem_get_info() reports *device-level* free memory.  After models
    are loaded, the PyTorch caching allocator may have reserved nearly all VRAM
    from the OS perspective, making device_free_bytes appear near zero even though
    the allocator can freely reuse its cached (reserved-but-not-allocated) blocks
    for new tensors without going back to the OS.

    This function computes:
        effective_free = device_free_bytes + pytorch_cache_free_bytes

    where pytorch_cache_free_bytes = memory_reserved - memory_allocated.

    When the MAX_CUDA_VRAM debug cap is active it additionally clamps to the
    simulated allocator budget:
        effective_free = min(effective_free, allocator_budget - memory_allocated)

    Returns 0 if no GPU is available or on error.
    """
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            device_free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)

            # Memory reserved by the PyTorch caching allocator from the OS but not
            # actively used by live tensors.  This can be reused without a new OS
            # allocation, so it counts as "available" from our perspective.
            reserved_bytes = torch.cuda.memory_reserved(device_index)
            allocated_bytes = torch.cuda.memory_allocated(device_index)
            pytorch_cache_free_bytes = max(0, reserved_bytes - allocated_bytes)
            effective_free_bytes = device_free_bytes + pytorch_cache_free_bytes

            # Check if a per-process memory fraction has been set
            # We detect this by checking MAX_CUDA_VRAM env var (our simulation mechanism)
            debug_vram = os.environ.get(DEBUG_MAX_CUDA_VRAM_ENV)
            if debug_vram is not None:
                try:
                    simulated_gb = float(debug_vram)
                    total_gb = total_bytes / (1024**3)
                    if simulated_gb < total_gb:
                        # Per-process cap is active.
                        # Use the same reference context as set_per_process_memory_fraction.
                        ref_context_gb = MODEL_VRAM.get("cuda_context", 0.5)
                        allocator_budget_gb = max(0.5, simulated_gb - ref_context_gb)
                        allocator_budget_bytes = allocator_budget_gb * (1024**3)
                        # Free = what the allocator is allowed minus what it has allocated
                        process_free = allocator_budget_bytes - allocated_bytes
                        effective_free_bytes = min(effective_free_bytes, process_free)
                        return max(0.0, effective_free_bytes / (1024**3))
                except (ValueError, TypeError):
                    pass

            return max(0.0, effective_free_bytes / (1024**3))

        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            # Support for Intel XPU (IPEX)
            # Try mem_get_info first (available in newer IPEX versions)
            if hasattr(torch.xpu, "mem_get_info"):
                try:
                    device_free_bytes, _ = torch.xpu.mem_get_info(device_index)
                    return device_free_bytes / (1024**3)
                except Exception:
                    pass

            # Fallback for older IPEX or if mem_get_info fails: total - reserved
            try:
                total_bytes = torch.xpu.get_device_properties(device_index).total_memory
                reserved_bytes = torch.xpu.memory_reserved(device_index)
                return max(0.0, (total_bytes - reserved_bytes) / (1024**3))
            except Exception:
                return 0.0

        return 0.0
    except Exception:
        return 0.0


# Global GPU config instance (initialized lazily)
_global_gpu_config: Optional[GPUConfig] = None


def get_global_gpu_config() -> GPUConfig:
    """Get the global GPU configuration, initializing if necessary.

    Respects the ``ACESTEP_SAVE_MEMORY`` environment variable: when set to
    ``"1"`` or ``"true"``, ``save_memory_mode`` is enabled regardless of tier.
    """
    global _global_gpu_config
    if _global_gpu_config is None:
        _global_gpu_config = get_gpu_config()
        env_val = os.environ.get(SAVE_MEMORY_ENV, "").strip().lower()
        if env_val in ("1", "true", "yes"):
            _global_gpu_config.save_memory_mode = True
            logger.info("[gpu_config] Save-memory mode enabled via {}={}".format(SAVE_MEMORY_ENV, env_val))
    return _global_gpu_config


def set_global_gpu_config(config: GPUConfig):
    """Set the global GPU configuration."""
    global _global_gpu_config
    _global_gpu_config = config
