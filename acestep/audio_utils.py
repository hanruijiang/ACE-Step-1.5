"""
Audio saving and transcoding utility module

Independent audio file operations outside of handler, supporting:
- Save audio tensor/numpy to files (default FLAC format, fast)
- Export audio in supported file formats
"""


import json
import os
import subprocess
import hashlib
import tempfile
from pathlib import Path
from typing import Union, Optional
import torch
import numpy as np
import torchaudio
import soundfile as sf
from loguru import logger


SUPPORTED_AUDIO_FORMATS = ("wav", "wav32", "flac", "mp3", "opus", "aac")


def _decode_audio_ffmpeg(path):
    """Decode to a seekable float WAV so sample rate and channel count are preserved."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "decoded.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(path), "-map", "0:a:0", "-c:a", "pcm_f32le", str(target)],
            check=True, capture_output=True, timeout=120,
        )
        return sf.read(str(target), dtype="float32")


def read_audio_file(path):
    """Return float32 samples-first audio and native sample rate."""
    try:
        return sf.read(str(path), dtype="float32")
    except Exception as sf_error:
        try:
            return _decode_audio_ffmpeg(path)
        except Exception as ffmpeg_error:
            detail = getattr(ffmpeg_error, "stderr", None)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Cannot read {path!s}: soundfile ({sf_error}); FFmpeg ({detail or ffmpeg_error}). "
                "Ensure ffmpeg is installed and on PATH."
            ) from ffmpeg_error


def apply_fade(
    audio_data: Union[torch.Tensor, np.ndarray],
    fade_in_samples: int = 0,
    fade_out_samples: int = 0,
) -> Union[torch.Tensor, np.ndarray]:
    """Apply linear fade in and/or fade out to audio data.

    Args:
        audio_data: Audio data as torch.Tensor [channels, samples] or numpy.ndarray.
        fade_in_samples: Number of samples for fade in ramp (0 = no fade in).
        fade_out_samples: Number of samples for fade out ramp (0 = no fade out).

    Returns:
        Audio data with fades applied, in the same format as input.
    """
    if fade_in_samples <= 0 and fade_out_samples <= 0:
        return audio_data

    is_tensor = isinstance(audio_data, torch.Tensor)
    if is_tensor:
        audio = audio_data.clone()
        total_samples = audio.shape[-1]
    else:
        audio = audio_data.copy()
        total_samples = audio.shape[-1]

    if fade_in_samples > 0:
        actual_in = min(fade_in_samples, total_samples)
        if is_tensor:
            ramp = torch.linspace(0.0, 1.0, actual_in, dtype=audio.dtype, device=audio.device)
            audio[..., :actual_in] = audio[..., :actual_in] * ramp
        else:
            ramp = np.linspace(0.0, 1.0, actual_in, dtype=np.float32)
            audio[..., :actual_in] = audio[..., :actual_in] * ramp

    if fade_out_samples > 0:
        actual_out = min(fade_out_samples, total_samples)
        if is_tensor:
            ramp = torch.linspace(1.0, 0.0, actual_out, dtype=audio.dtype, device=audio.device)
            audio[..., total_samples - actual_out:] = audio[..., total_samples - actual_out:] * ramp
        else:
            ramp = np.linspace(1.0, 0.0, actual_out, dtype=np.float32)
            audio[..., total_samples - actual_out:] = audio[..., total_samples - actual_out:] * ramp

    return audio


def normalize_audio(audio_data: Union[torch.Tensor, np.ndarray], target_db: float = -1.0) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply peak normalization to audio data.
    
    Args:
        audio_data: Audio data as torch.Tensor or numpy.ndarray
        target_db: Target peak level in dB (default: -1.0)
        
    Returns:
        Normalized audio data in the same format as input
    """
    # Create a copy to avoid modifying original in-place
    if isinstance(audio_data, torch.Tensor):
        audio = audio_data.clone()
        is_tensor = True
    else:
        audio = audio_data.copy()
        is_tensor = False
        
    # Calculate current peak
    if is_tensor:
        peak = torch.max(torch.abs(audio))
    else:
        peak = np.max(np.abs(audio))
        
    # Handle silence/near-silence to avoid division by zero or extreme gain
    if peak < 1e-6:
        return audio_data
        
    # Convert target dB to linear amplitude
    target_amp = 10 ** (target_db / 20.0)
    
    # Calculate needed gain
    gain = target_amp / peak
    
    # Apply gain
    audio = audio * gain
    
    return audio



class AudioSaver:
    """Audio saving and transcoding utility class"""

    MP3_DEFAULT_BITRATE = "128k"
    MP3_ALLOWED_BITRATES = {"128k", "192k", "256k", "320k"}
    MP3_DEFAULT_SAMPLE_RATE = 48000
    MP3_ALLOWED_SAMPLE_RATES = {44100, 48000}
    
    def __init__(self, default_format: str = "flac"):
        """
        Initialize audio saver
        
        Args:
            default_format: Default save format ('flac', 'wav', 'mp3', 'wav32', 'opus', 'aac')
        """
        self.default_format = default_format.lower()
        if self.default_format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {default_format}")
    
    def _save_mp3(
        self,
        audio_tensor: torch.Tensor,
        output_path: Path,
        input_sample_rate: int,
        mp3_bitrate: Optional[str] = None,
        mp3_sample_rate: Optional[int] = None,
    ) -> None:
        """Save MP3 with explicit ffmpeg settings and 128k/48k defaults."""
        bitrate = str(mp3_bitrate or self.MP3_DEFAULT_BITRATE).strip().lower()
        if bitrate not in self.MP3_ALLOWED_BITRATES:
            bitrate = self.MP3_DEFAULT_BITRATE

        try:
            target_sample_rate = int(mp3_sample_rate or self.MP3_DEFAULT_SAMPLE_RATE)
        except Exception:
            target_sample_rate = self.MP3_DEFAULT_SAMPLE_RATE
        if target_sample_rate not in self.MP3_ALLOWED_SAMPLE_RATES:
            target_sample_rate = self.MP3_DEFAULT_SAMPLE_RATE

        tensor_to_save = audio_tensor
        if int(input_sample_rate) != int(target_sample_rate):
            tensor_to_save = torchaudio.functional.resample(audio_tensor, int(input_sample_rate), int(target_sample_rate))

        self._save_compressed(tensor_to_save, output_path, target_sample_rate, "libmp3lame", ["-ar", str(target_sample_rate), "-b:a", bitrate])

    def _save_compressed(self, audio_tensor, output_path, sample_rate, codec, options=None):
        """Encode using the FFmpeg executable, independent of TorchAudio backends."""
        with tempfile.TemporaryDirectory() as directory:
            temp_wav_path = Path(directory) / "input.wav"
            audio_np = audio_tensor.detach().cpu()
            if audio_np.dim() == 2:
                audio_np = audio_np.transpose(0, 1)
            sf.write(str(temp_wav_path), audio_np.contiguous().numpy(), int(sample_rate),
                     format="WAV", subtype="FLOAT")
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(temp_wav_path), "-codec:a", codec,
                *(options or []), str(output_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg executable not found; install FFmpeg and add it to PATH.") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("FFmpeg audio export timed out after 120 seconds.") from exc
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
                raise RuntimeError(f"FFmpeg audio export failed: {detail}") from exc
            logger.debug(f"[AudioSaver] Saved audio to {output_path} ({codec})")

    def save_audio(
        self,
        audio_data: Union[torch.Tensor, np.ndarray],
        output_path: Union[str, Path],
        sample_rate: int = 48000,
        format: Optional[str] = None,
        channels_first: bool = True,
        mp3_bitrate: Optional[str] = None,
        mp3_sample_rate: Optional[int] = None,
    ) -> str:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data, torch.Tensor [channels, samples] or numpy.ndarray
            output_path: Output file path (extension can be omitted)
            sample_rate: Sample rate
            format: Audio format ('flac', 'wav', 'mp3', 'wav32', 'opus', 'aac'), defaults to default_format
            channels_first: If True, tensor format is [channels, samples], else [samples, channels]
            mp3_bitrate: Optional MP3 bitrate override (128k/192k/256k/320k)
            mp3_sample_rate: Optional MP3 sample rate override (44100/48000)
        
        Returns:
            Actual saved file path
        """
        format = (format or self.default_format).lower()
        if format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {format}")
        
        # Ensure output path has correct extension
        output_path = Path(output_path)
        
        # Determine extension based on format
        ext = ".wav" if format == "wav32" else f".{format}"
        
        if not (format == "aac" and output_path.suffix.lower() == ".m4a"):
            output_path = output_path.with_suffix(ext)

        # Convert to torch tensor
        if isinstance(audio_data, np.ndarray):
            if channels_first:
                # numpy already [channels, samples]
                audio_tensor = torch.from_numpy(audio_data).float()
            else:
                # numpy [samples, channels] -> tensor [samples, channels] -> [channels, samples] (if transposed)
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.dim() == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]:
                     # Assume [samples, channels] if dim0 > dim1 (heuristic)
                     audio_tensor = audio_tensor.T
        else:
            # torch tensor
            audio_tensor = audio_data.cpu().float()
            if not channels_first and audio_tensor.dim() == 2:
                # [samples, channels] -> [channels, samples]
                if audio_tensor.shape[0] > audio_tensor.shape[1]:
                    audio_tensor = audio_tensor.T
        
        # Ensure memory is contiguous
        audio_tensor = audio_tensor.contiguous()
        
        if format == "mp3":
            self._save_mp3(audio_tensor, output_path, sample_rate,
                           mp3_bitrate=mp3_bitrate, mp3_sample_rate=mp3_sample_rate)
        elif format in ("opus", "aac"):
            codec = "libopus" if format == "opus" else "aac"
            self._save_compressed(audio_tensor, output_path, sample_rate, codec)
        else:
            audio_np = audio_tensor.detach().cpu()
            if audio_np.dim() == 2:
                audio_np = audio_np.transpose(0, 1)
            sf.write(str(output_path), audio_np.contiguous().numpy(), sample_rate,
                     format="WAV" if format == "wav32" else format.upper(),
                     subtype="FLOAT" if format == "wav32" else None)
        return str(output_path)

def generate_uuid_from_params(params_dict) -> str:
    """
    Generate deterministic UUID from generation parameters.
    Same parameters will always generate the same UUID.
    
    Args:
        params_dict: Dictionary of parameters
    
    Returns:
        UUID string
    """
    
    params_json = json.dumps(params_dict, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha256(params_json.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    uuid_str = f"{hash_hex[0:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str


# Global default instance
_default_saver = AudioSaver(default_format="flac")


def save_audio(
    audio_data: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    sample_rate: int = 48000,
    format: Optional[str] = None,
    channels_first: bool = True,
    mp3_bitrate: Optional[str] = None,
    mp3_sample_rate: Optional[int] = None,
) -> str:
    """
    Convenience function: save audio (using default configuration)
    
    Args:
        audio_data: Audio data
        output_path: Output path
        sample_rate: Sample rate
        format: Format (default flac)
        channels_first: Tensor format flag
        mp3_bitrate: Optional MP3 bitrate override (128k/192k/256k/320k)
        mp3_sample_rate: Optional MP3 sample rate override (44100/48000)

    Returns:
        Saved file path
    """
    return _default_saver.save_audio(
        audio_data,
        output_path,
        sample_rate,
        format,
        channels_first,
        mp3_bitrate,
        mp3_sample_rate,
    )
