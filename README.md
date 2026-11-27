# ACE-Step 1.5 — CLI inference

A CLI-only distribution of ACE-Step 1.5 for local music generation and audio editing. The original model weights and license are unchanged. No Web UI, HTTP API, or training entry points are included.

## Installation

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). The default dependency configuration targets NVIDIA CUDA on Windows/Linux and Apple Silicon on macOS. Install FFmpeg separately and put it on PATH for MP3, Opus, AAC output and audio input formats not handled by SoundFile.

```bash
uv sync
```

For optional NVIDIA nano-vLLM acceleration in a source checkout:

```bash
uv sync --extra vllm
uv run --extra vllm acestep --backend vllm --config config.toml
```

Python 3.11 and 3.12 use platform-specific Triton and Flash Attention wheels where available. The pinned Windows Python 3.12 combination currently has no verified Flash Attention wheel; nano-vLLM uses the SDPA/eager fallback if needed. For optional DiT quantization on supported CUDA builds, use `uv sync --extra quantization` (combine extras when needed).

For pip installation from the repository:

```bash
pip install -r requirements.txt
# Optional, NVIDIA CUDA only:
pip install -e "./acestep/third_parts/nano-vllm[cuda]"
python cli.py --backend pt
```

For AMD ROCm or Intel XPU, create a separate virtual environment, install the appropriate matching `torch` and `torchaudio` builds first, and then install the corresponding `requirements-rocm-linux.txt`, `requirements-rocm.txt`, or `requirements-xpu.txt`. Run `python cli.py --backend pt` from that environment, not the default CUDA-targeted `uv run`. The Windows ROCm file contains SDK and wheel installation instructions.

## Inference

```bash
# Interactive wizard
uv run python cli.py
# Create a TOML configuration
uv run python cli.py --configure
# Generate from a saved configuration
uv run python cli.py --config config.toml
# Equivalent installed command
uv run acestep --config config.toml
# Download models explicitly (otherwise the CLI can download them)
uv run acestep-download --help
```

The wizard supports text2music, cover, repaint, lego, extract and complete. It preserves local model selection, 5Hz LM thinking and sample formatting, metadata, batch generation, and audio output options. Lego, extract and complete require a compatible base model. Use `--backend pt`, `--backend vllm`, or `--backend mlx` to select an LM backend. The CLI reports the actual backend, including fallbacks. Supported output formats: WAV, WAV32, FLAC, MP3, Opus and AAC.

Set `checkpoint_dir` in TOML or `ACESTEP_CHECKPOINTS_DIR` to store checkpoints outside the checkout; otherwise use `project_root/checkpoints` (project root defaults to `ACESTEP_PROJECT_ROOT` or the working directory). See [CLI guide](docs/en/CLI.md) for the TOML workflow and prompt editing.

## Checks

```bash
uv run python -m compileall -q cli.py acestep
uv run python cli.py --help
uv run python -m unittest acestep.cli_smoke_test
```

Model generation requires the appropriate hardware, checkpoints, and backend dependencies.

## License

See [LICENSE](LICENSE) and the bundled [nano-vllm license](acestep/third_parts/nano-vllm/LICENSE).
