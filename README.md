# ACE-Step 1.5 — CLI inference

A CLI-only distribution of ACE-Step 1.5 for local music generation and audio editing. The original model weights and license are unchanged. No Web UI, HTTP API, or training entry points are included.

## Installation

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). The default dependency configuration targets NVIDIA CUDA on Windows/Linux and Apple Silicon on macOS. Install FFmpeg separately and put it on PATH for MP3, Opus, AAC output and audio input formats not handled by SoundFile.

```bash
uv sync
```

All inference uses PyTorch (CUDA, ROCm, MPS, XPU or CPU). For optional DiT quantization on supported Linux CUDA builds, use `uv sync --extra quantization` and `uv run --extra quantization acestep --config config.toml`. Install a compatible TorchAO build separately for other supported PyTorch environments.

For pip installation from the repository:

```bash
pip install -r requirements.txt
python cli.py
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

The wizard supports text2music, cover, repaint, lego, extract and complete. It preserves local model selection, 5Hz LM thinking and sample formatting, metadata, batch generation, and audio output options. Lego, extract and complete require a compatible base model. PyTorch is the sole inference backend. Supported output formats: WAV, WAV32, FLAC, MP3, Opus and AAC.

Set `checkpoint_dir` in TOML or `ACESTEP_CHECKPOINTS_DIR` to store checkpoints outside the checkout; otherwise use `project_root/checkpoints` (project root defaults to `ACESTEP_PROJECT_ROOT` or the working directory). See [CLI guide](docs/en/CLI.md) for the TOML workflow and prompt editing.

## Checks

```bash
uv run python -m compileall -q cli.py acestep
uv run python cli.py --help
uv run python -m unittest acestep.cli_smoke_test
```

Model generation requires the appropriate hardware, checkpoints, and backend dependencies.

## License

See [LICENSE](LICENSE).
