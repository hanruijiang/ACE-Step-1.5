# ACE-Step 1.5 CLI Guide

## Quick Start

Generate via wizard (interactive):

```bash
python cli.py
```

Generate from a saved config:

```bash
python cli.py --config config.toml
```

Create or edit a config without generating:

```bash
python cli.py --configure
python cli.py --configure --config config.toml
```

---

## CLI Flags

- `-c` / `--config` — Path to a `.toml` configuration file to load.
- `--configure` — Run wizard to save configuration without generating.
- `--log-level` — Logging level for internal modules. One of `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default: `INFO`.

---

## Wizard Flow

1. Choose one of six tasks. TOML also supports `task_type = "cover-nofsq"`.
2. Select a DiT model (from locally available models, or download from Hugging Face).
3. Select an LM model (from locally available models, or auto-download).
4. Provide task-specific inputs (source audio, tracks, etc.).
5. For `text2music`: choose between Simple Mode (auto-generate caption/lyrics via LM) or manual input.
6. Provide caption / description.
7. Choose lyrics mode (instrumental / auto-generate / file / paste).
8. Set number of outputs.
9. Optionally configure advanced parameters (metadata, DiT settings, LM settings, output settings).
10. Review summary and confirm generation.
11. Save configuration to a `.toml` file.

On low-memory tiers, the wizard leaves LM thinking and CoT off by default. Simple Mode and automatic lyrics load an LM when requested.

---

## Configure Mode (`--configure`)

`--configure` opens the wizard to save a TOML file. Pass `--config` to start from an existing file.

---

## Configuration File (`.toml`)

The wizard saves chosen settings in TOML. Loading a file with `--config` rejects unknown keys.

Checkpoint lookup uses `checkpoint_dir`, then `ACESTEP_CHECKPOINTS_DIR`, then `project_root/checkpoints`. Relative `checkpoint_dir` values resolve against `project_root`. The project root defaults to `ACESTEP_PROJECT_ROOT` or the working directory.

- **LM**. `thinking` enables audio-token generation. Unspecified CoT options follow `thinking`. Set `use_cot_metas`, `use_cot_caption`, or `use_cot_language` explicitly to control each feature.
- **Repaint**. `repaint_mode` and `repaint_strength` set the default crossfades. Set `repaint_latent_crossfade_frames` or `repaint_wav_crossfade_sec` to override them. Use `0` for a hard boundary. Aggressive mode disables crossfades.
- **FlowEdit**. Use `flow_edit_morph = true` with a source file on `text2music`, `cover`, or `cover-nofsq`. Set `flow_edit_source_caption` and `flow_edit_source_lyrics` to describe the source. Set `flow_edit_n_min`, `flow_edit_n_max`, and `flow_edit_n_avg` to control sampling.
- **Model options**. `vae_checkpoint` accepts a registered VAE or an absolute path. Run `acestep-download --model scragvae` to fetch ScragVAE separately. `quantization` requires TorchAO.
- **Output**. `audio_format` accepts `wav`, `wav32`, `flac`, `mp3`, `opus`, and `aac`. FFmpeg must be on PATH for compressed output, with `libmp3lame` for MP3 and `libopus` for Opus. Set `mp3_bitrate` and `mp3_sample_rate` to control MP3 export.

Generation runs synchronously by default. `ACESTEP_GENERATION_TIMEOUT` sets how many seconds the CLI waits. GPU work may continue after the timeout.

---

## Prompt Editing (`instruction.txt`)

When LM audio-token generation is active, the wizard writes `instruction.txt` in the project root and pauses for edits. With `--config`, the CLI reads the file if present and runs without pausing.

Edits apply to every output in the batch. Keep metadata in YAML inside the `<think>` block. To clear lyrics, leave the `# Lyric` section empty.
