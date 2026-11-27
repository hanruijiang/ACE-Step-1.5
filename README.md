# ACE-Step 1.5 — CLI inference

A CLI-only distribution of ACE-Step 1.5 for local music generation and audio editing. The original model weights and license are unchanged. No Web UI, HTTP API, or training entry points are included.

## Installation

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). The default dependency configuration targets NVIDIA CUDA on Windows/Linux and Apple Silicon on macOS. Install FFmpeg separately if needed for compressed audio output.

```bash
uv sync
```

For AMD ROCm or Intel XPU, first install the appropriate PyTorch build for your hardware, then use the corresponding `requirements-rocm-linux.txt`, `requirements-rocm.txt`, or `requirements-xpu.txt` in your virtual environment instead of the default CUDA dependencies. Use the `pt` backend if nano-vllm is unavailable.

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

The wizard supports text2music, cover, repaint, lego, extract and complete. It preserves local model selection, 5Hz LM thinking and sample formatting, metadata, batch generation, and audio output options. Lego, extract and complete require a compatible base model. Use `--backend pt`, `--backend vllm`, or `--backend mlx` to select an LM backend.

Model checkpoints can be stored outside the checkout using `ACESTEP_CHECKPOINTS_DIR`. See [CLI guide](docs/en/CLI.md) for the TOML workflow and prompt editing.

## Checks

```bash
uv run python -m compileall -q cli.py acestep
uv run python cli.py --help
uv run python -m unittest acestep.cli_smoke_test
```

Model generation requires the appropriate hardware, checkpoints, and backend dependencies.

## License

See [LICENSE](LICENSE) and the bundled [nano-vllm license](acestep/third_parts/nano-vllm/LICENSE).
