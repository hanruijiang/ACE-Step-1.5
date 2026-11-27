import argparse
import re
import ast
import os
import sys
import shutil
import toml
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

# Load environment variables from .env or .env.example (if available)
try:
    from dotenv import load_dotenv
    _project_root = str(Path(__file__).resolve().parent.parent)
    if not os.path.isfile(os.path.join(_project_root, 'pyproject.toml')):
        _project_root = os.getcwd()
    _env_path = os.path.join(_project_root, '.env')
    _env_example_path = os.path.join(_project_root, '.env.example')

    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"Loaded configuration from {_env_path}")
    elif os.path.exists(_env_example_path):
        load_dotenv(_env_example_path)
        print(f"Loaded configuration from {_env_example_path} (fallback)")
except ImportError:
    pass

# Clear proxy settings that may affect network behavior
for _proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(_proxy_var, None)

def _configure_logging(
    level: Optional[str] = None,
    suppress_audio_tokens: Optional[bool] = None,
) -> None:
    try:
        from loguru import logger
    except Exception:
        return

    if suppress_audio_tokens is None:
        suppress_audio_tokens = os.environ.get("ACE_STEP_SUPPRESS_AUDIO_TOKENS", "1") not in {"0", "false", "False"}
    if level is None:
        level = "INFO"
    level = str(level).upper()

    def _log_filter(record) -> bool:
        message = record.get("message", "")
        # Suppress duplicate DiT prompt logs (we print a single final prompt in cli.py)
        if (
            "DiT TEXT ENCODER INPUT" in message
            or "text_prompt:" in message
            or (message.strip() and set(message.strip()) == {"="})
        ):
            return False
        if not suppress_audio_tokens:
            return True
        return "<|audio_code_" not in message

    logger.remove()
    logger.add(sys.stderr, level=level, filter=_log_filter)


_configure_logging()

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music, create_sample, format_sample, DIRECT_CONDITIONING_TASKS
from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS, TASK_TYPES
from acestep.gpu_config import get_gpu_config, set_global_gpu_config, is_mps_platform
import torch


TRACK_CHOICES = [
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
]


def _get_project_root() -> str:
    from acestep.model_downloader import get_project_root
    return str(get_project_root())


def _parse_description_hints(description: str) -> tuple[Optional[str], bool]:
    import re

    if not description:
        return None, False

    description_lower = description.lower().strip()

    language_mapping = {
        'english': 'en', 'en': 'en',
        'chinese': 'zh', '中文': 'zh', 'zh': 'zh', 'mandarin': 'zh',
        'japanese': 'ja', '日本語': 'ja', 'ja': 'ja',
        'korean': 'ko', '한국어': 'ko', 'ko': 'ko',
        'spanish': 'es', 'español': 'es', 'es': 'es',
        'french': 'fr', 'français': 'fr', 'fr': 'fr',
        'german': 'de', 'deutsch': 'de', 'de': 'de',
        'italian': 'it', 'italiano': 'it', 'it': 'it',
        'portuguese': 'pt', 'português': 'pt', 'pt': 'pt',
        'russian': 'ru', 'русский': 'ru', 'ru': 'ru',
        'bengali': 'bn', 'bn': 'bn',
        'hindi': 'hi', 'hi': 'hi',
        'arabic': 'ar', 'ar': 'ar',
        'thai': 'th', 'th': 'th',
        'vietnamese': 'vi', 'vi': 'vi',
        'indonesian': 'id', 'id': 'id',
        'turkish': 'tr', 'tr': 'tr',
        'dutch': 'nl', 'nl': 'nl',
        'polish': 'pl', 'pl': 'pl',
    }

    detected_language = None
    for lang_name, lang_code in language_mapping.items():
        if len(lang_name) <= 2:
            pattern = r'(?:^|\s|[.,;:!?])' + re.escape(lang_name) + r'(?:$|\s|[.,;:!?])'
        else:
            pattern = r'\b' + re.escape(lang_name) + r'\b'
        if re.search(pattern, description_lower):
            detected_language = lang_code
            break

    is_instrumental = False
    if 'instrumental' in description_lower:
        is_instrumental = True
    elif 'pure music' in description_lower or 'pure instrument' in description_lower:
        is_instrumental = True
    elif description_lower.endswith(' solo') or description_lower == 'solo':
        is_instrumental = True

    return detected_language, is_instrumental


def _prompt_non_empty(prompt: str) -> str:
    value = input(prompt).strip()
    while not value:
        value = input(prompt).strip()
    return value


def _prompt_with_default(prompt: str, default: Optional[str] = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default not in (None, ""):
            return str(default)
        if not required:
            return ""
        print("This value is required. Please try again.")


def _prompt_bool(prompt: str, default: bool) -> bool:
    default_str = "y" if default else "n"
    while True:
        value = input(f"{prompt} (y/n) [default: {default_str}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "1", "true"}:
            return True
        if value in {"n", "no", "0", "false"}:
            return False
        print("Please enter 'y' or 'n'.")


def _prompt_choice_from_list(
    prompt: str,
    options: List[str],
    default: Optional[str] = None,
    allow_custom: bool = True,
    custom_validator=None,
    custom_error: Optional[str] = None,
) -> Optional[str]:
    if not options:
        return default
    print("\n" + prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    default_display = default if default not in (None, "") else "auto"
    while True:
        choice = input(f"Choose a model (number or name) [default: {default_display}]: ").strip()
        if not choice:
            return None if default_display == "auto" else default
        if choice.lower() == "auto":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print("Invalid selection. Please choose a valid number.")
            continue
        if allow_custom:
            if custom_validator and not custom_validator(choice):
                print(custom_error or "Invalid selection. Please try again.")
                continue
            if choice not in options:
                print("Unknown model. Using as-is.")
            return choice
        print("Please choose a valid option.")


def _edit_formatted_prompt_via_file(formatted_prompt: str, instruction_path: str) -> str:
    """Write the prompt, wait for edits, and read the result."""
    path = Path(instruction_path)
    path.write_text(formatted_prompt, encoding="utf-8")
    print(f"\nEdit {path}, then press Enter to continue.")
    input()
    return path.read_text(encoding="utf-8")


def _extract_caption_lyrics_from_formatted_prompt(formatted_prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract caption and lyrics from a formatted prompt."""
    matches = list(re.finditer(r"# Caption\n(.*?)\n+# Lyric\n(.*)", formatted_prompt, re.DOTALL))
    if not matches:
        return None, None

    caption = matches[-1].group(1).strip()
    lyrics = matches[-1].group(2)

    # Trim lyrics if chat-template markers appear after the user message.
    cut_markers = ["<|eot_id|>", "<|start_header_id|>", "<|assistant|>", "<|user|>", "<|system|>", "<|im_end|>", "<|im_start|>"]
    cut_at = len(lyrics)
    for marker in cut_markers:
        pos = lyrics.find(marker)
        if pos != -1:
            cut_at = min(cut_at, pos)
    lyrics = lyrics[:cut_at].rstrip()

    return caption or None, lyrics or None


def _extract_instruction_from_formatted_prompt(formatted_prompt: str) -> Optional[str]:
    """Extract instruction text from a formatted prompt."""
    match = re.search(r"# Instruction\n(.*?)\n\n", formatted_prompt, re.DOTALL)
    if not match:
        return None
    instruction = match.group(1).strip()
    return instruction or None


def _extract_cot_metadata_from_formatted_prompt(formatted_prompt: str) -> dict:
    """Read the final reasoning block as YAML metadata."""
    matches = re.findall(r"<think>\s*(.*?)\s*</think>", formatted_prompt, re.DOTALL)
    if not matches:
        return {}
    metadata = yaml.safe_load(matches[-1]) or {}
    if not isinstance(metadata, dict):
        raise ValueError("The prompt's reasoning block must contain YAML metadata.")
    for key in ("bpm", "duration"):
        if key in metadata:
            value = _parse_number(metadata[key])
            if value is None or value <= 0:
                metadata.pop(key)
            else:
                metadata[key] = int(value) if key == "bpm" else value
    for key in ("caption", "genres", "keyscale", "language", "vocal_language", "timesignature"):
        if metadata.get(key) is not None:
            metadata[key] = str(metadata[key]).strip()
    return metadata


def _parse_number(value) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group(0)) if match else None


def _parse_timesteps_input(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        if all(isinstance(t, (int, float)) for t in value):
            return [float(t) for t in value]
        return None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("[") or raw.startswith("("):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        if isinstance(parsed, list) and all(isinstance(t, (int, float)) for t in parsed):
            return [float(t) for t in parsed]
        return None
    try:
        return [float(t.strip()) for t in raw.split(",") if t.strip()]
    except Exception:
        return None


def _make_prompt_editor(instruction_path: str, preloaded_prompt: Optional[str] = None):
    """Edit one audio-token prompt and reuse it for the generation batch."""
    cached = None

    def edit(prompt):
        nonlocal cached
        if cached is not None:
            return cached

        edited = preloaded_prompt if preloaded_prompt is not None else _edit_formatted_prompt_via_file(
            prompt, instruction_path
        )
        metadata = _extract_cot_metadata_from_formatted_prompt(edited)
        original_metadata = _extract_cot_metadata_from_formatted_prompt(prompt)
        overrides = {key: value for key, value in metadata.items() if value != original_metadata.get(key)}
        edits = {"metadata": metadata, "overrides": overrides}
        if edited != prompt:
            print("Using edited audio-token prompt.")
            caption, lyrics = _extract_caption_lyrics_from_formatted_prompt(edited)
            original_caption, original_lyrics = _extract_caption_lyrics_from_formatted_prompt(prompt)
            if caption is not None and caption != original_caption:
                edits["caption"] = caption
            if lyrics is not None and lyrics != original_lyrics:
                edits["lyrics"] = lyrics
            instruction = _extract_instruction_from_formatted_prompt(edited)
            if instruction and instruction != _extract_instruction_from_formatted_prompt(prompt):
                edits["instruction"] = instruction
        cached = (edited, edits)
        return cached

    return edit


def _prompt_int(prompt: str, default: Optional[int] = None, min_value: Optional[int] = None,
                max_value: Optional[int] = None) -> Optional[int]:
    default_display = "auto" if default is None else default
    while True:
        value = input(f"{prompt} [{default_display}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue
        if min_value is not None and parsed < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and parsed > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return parsed


def _prompt_float(prompt: str, default: Optional[float] = None, min_value: Optional[float] = None,
                  max_value: Optional[float] = None) -> Optional[float]:
    default_display = "auto" if default is None else default
    while True:
        value = input(f"{prompt} [{default_display}]: ").strip()
        if not value:
            return default
        try:
            parsed = float(value)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if min_value is not None and parsed < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and parsed > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return parsed


def _prompt_existing_file(prompt: str, default: Optional[str] = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        path = input(f"{prompt}{suffix}: ").strip()
        if not path and default:
            path = default
        if os.path.isfile(path):
            return _expand_audio_path(path)
        print("Invalid file path. Please try again.")


def _expand_audio_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str or not isinstance(path_str, str):
        return path_str
    try:
        return Path(path_str).expanduser().resolve(strict=False).as_posix()
    except Exception:
        return Path(path_str).expanduser().absolute().as_posix()


def _resolve_device(device: str) -> str:
    if device == "auto":
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _default_instruction_for_task(task_type: str, tracks: Optional[List[str]] = None) -> str:
    if task_type == "lego":
        track = tracks[0] if tracks else "guitar"
        return TASK_INSTRUCTIONS["lego"].format(TRACK_NAME=track.upper())
    if task_type == "extract":
        track = tracks[0] if tracks else "vocals"
        return TASK_INSTRUCTIONS["extract"].format(TRACK_NAME=track.upper())
    if task_type == "complete":
        tracks_list = ", ".join(tracks) if tracks else "drums, bass, guitar"
        return TASK_INSTRUCTIONS["complete"].format(TRACK_CLASSES=tracks_list)
    return DEFAULT_DIT_INSTRUCTION


def _apply_optional_defaults(args, params_defaults: GenerationParams, config_defaults: GenerationConfig) -> None:
    optional_defaults = {
        "duration": params_defaults.duration,
        "bpm": params_defaults.bpm,
        "keyscale": params_defaults.keyscale,
        "timesignature": params_defaults.timesignature,
        "vocal_language": params_defaults.vocal_language,
        "inference_steps": params_defaults.inference_steps,
        "seed": params_defaults.seed,
        "guidance_scale": params_defaults.guidance_scale,
        "use_adg": params_defaults.use_adg,
        "cfg_interval_start": params_defaults.cfg_interval_start,
        "cfg_interval_end": params_defaults.cfg_interval_end,
        "shift": 3.0,
        "infer_method": params_defaults.infer_method,
        "timesteps": None,
        "repainting_start": params_defaults.repainting_start,
        "repainting_end": params_defaults.repainting_end,
        "audio_cover_strength": params_defaults.audio_cover_strength,
        "thinking": params_defaults.thinking,
        "lm_temperature": params_defaults.lm_temperature,
        "lm_cfg_scale": params_defaults.lm_cfg_scale,
        "lm_top_k": params_defaults.lm_top_k,
        "lm_top_p": params_defaults.lm_top_p,
        "lm_negative_prompt": params_defaults.lm_negative_prompt,
        "use_cot_metas": params_defaults.use_cot_metas,
        "use_cot_caption": params_defaults.use_cot_caption,
        "use_cot_lyrics": params_defaults.use_cot_lyrics,
        "use_cot_language": params_defaults.use_cot_language,
        "use_constrained_decoding": params_defaults.use_constrained_decoding,
        "batch_size": config_defaults.batch_size,
        "use_random_seed": config_defaults.use_random_seed,
        "seeds": config_defaults.seeds,
        "lm_batch_chunk_size": config_defaults.lm_batch_chunk_size,
        "constrained_decoding_debug": config_defaults.constrained_decoding_debug,
        "audio_format": config_defaults.audio_format,
        "sample_mode": False,
        "sample_query": "",
        "use_format": False,
    }

    for key, default_value in optional_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_value)


def _summarize_lyrics(lyrics: Optional[str]) -> str:
    if not lyrics:
        return "none"
    if isinstance(lyrics, str):
        stripped = lyrics.strip()
        if not stripped:
            return "none"
        if os.path.isfile(stripped):
            return f"file: {os.path.basename(stripped)}"
        if len(stripped) <= 60:
            return stripped.replace("\n", " ")
        return f"text ({len(stripped)} chars)"
    return "provided"


def _print_final_parameters(
    args,
    params: GenerationParams,
    config: GenerationConfig,
    params_defaults: GenerationParams,
    config_defaults: GenerationConfig,
    compact: bool,
    resolved_device: Optional[str] = None,
) -> None:
    if not compact:
        print("\n--- Final Parameters (Args) ---")
        for k in sorted(vars(args).keys()):
            print(f"{k}: {getattr(args, k)}")
        print("------------------------------")
        print("\n--- Final Parameters (GenerationParams) ---")
        for k in sorted(vars(params).keys()):
            print(f"{k}: {getattr(params, k)}")
        print("-------------------------------------------")
        print("\n--- Final Parameters (GenerationConfig) ---")
        for k in sorted(vars(config).keys()):
            print(f"{k}: {getattr(config, k)}")
        print("-------------------------------------------\n")
        return

    device_display = args.device
    if resolved_device and resolved_device != args.device:
        device_display = f"{args.device} -> {resolved_device}"

    print("\n--- Final Parameters (Summary) ---")
    print(f"task_type: {params.task_type}")
    print(f"caption: {params.caption or 'none'}")
    print(f"lyrics: {_summarize_lyrics(params.lyrics)}")
    print(f"duration: {params.duration}s")
    print(f"outputs: {config.batch_size}")
    if params.bpm not in (None, params_defaults.bpm):
        print(f"bpm: {params.bpm}")
    if params.keyscale not in (None, params_defaults.keyscale):
        print(f"keyscale: {params.keyscale}")
    if params.timesignature not in (None, params_defaults.timesignature):
        print(f"timesignature: {params.timesignature}")
    print(f"instrumental: {params.instrumental}")
    print(f"thinking: {params.thinking}")
    print(f"lm_model: {args.lm_model_path or 'auto'}")
    print(f"dit_model: {args.config_path or 'auto'}")
    print(f"device: {device_display}")
    print(f"audio_format: {config.audio_format}")
    print(f"save_dir: {args.save_dir}")
    if config.seeds:
        print(f"seeds: {config.seeds}")
    else:
        print(f"seed: {params.seed} (random={config.use_random_seed})")
    print("-------------------------------\n")


def _build_meta_dict(params: GenerationParams) -> Optional[dict]:
    meta = {}
    if params.bpm is not None:
        meta["bpm"] = params.bpm
    if params.timesignature:
        meta["timesignature"] = params.timesignature
    if params.keyscale:
        meta["keyscale"] = params.keyscale
    if params.duration is not None:
        meta["duration"] = params.duration
    return meta or None


def _print_dit_prompt(dit_handler: "AceStepHandler", params: GenerationParams) -> None:
    meta = _build_meta_dict(params)
    caption_input, lyrics_input = dit_handler.build_dit_inputs(
        task=params.task_type,
        instruction=params.instruction,
        caption=params.caption or "",
        lyrics=params.lyrics or "",
        metas=meta,
        vocal_language=params.vocal_language or "unknown",
    )
    print("\n--- Final DiT Prompt (Caption Branch) ---")
    print(caption_input)
    print("\n--- Final DiT Prompt (Lyrics Branch) ---")
    print(lyrics_input)
    print("----------------------------------------\n")


def run_wizard(args, configure_only: bool = False, default_config_path: Optional[str] = None,
               params_defaults: Optional[GenerationParams] = None,
               config_defaults: Optional[GenerationConfig] = None):
    """
    Runs an interactive wizard to set generation parameters.
    """
    print("Welcome to the ACE-Step Music Generation Wizard!")
    print("This will guide you through creating your music.")
    print("Press Ctrl+C at any time to exit.")
    print("Note: Required models will be auto-downloaded if missing.")
    print("-" * 30)

    try:
        # Task selection
        print("\n--- Task Type ---")
        print("1. text2music - generate music from text/lyrics.")
        print("2. cover     - transform existing audio into a new style.")
        print("3. repaint   - regenerate a specific time segment of audio.")
        print("4. lego      - generate a specific instrument track in context.")
        print("5. extract   - isolate a specific instrument track from a mix.")
        print("6. complete  - complete/extend partial tracks with new instruments.")
        task_map = {
            "1": "text2music",
            "2": "cover",
            "3": "repaint",
            "4": "lego",
            "5": "extract",
            "6": "complete",
        }
        current_task = args.task_type or "text2music"
        task_default = next((k for k, v in task_map.items() if v == current_task), "1")
        task_choice = input(f"Choose a task (1-6) [default: {task_default}]: ").strip()
        if not task_choice:
            task_choice = task_default
        args.task_type = task_map.get(task_choice, "text2music")
        if args.task_type in {"lego", "extract", "complete"}:
            print("Note: This task requires a base DiT model (acestep-v15-base). It will be auto-downloaded if missing.")

        # Model selection (DiT)
        dit_handler = AceStepHandler(project_root=args.project_root, checkpoint_dir=args.checkpoint_dir)
        available_dit_models = dit_handler.get_available_acestep_v15_models()
        base_only = args.task_type in {"lego", "extract", "complete"}
        if base_only and available_dit_models:
            available_dit_models = [m for m in available_dit_models if "base" in m.lower()]

        if base_only and args.config_path and "base" not in str(args.config_path).lower():
            args.config_path = None

        if base_only:
            if available_dit_models:
                if args.config_path in available_dit_models:
                    selected = args.config_path
                else:
                    selected = available_dit_models[0]
                args.config_path = selected
                print(f"\nNote: This task requires a base model. Using: {selected}")
            else:
                print("\nNote: This task requires a base model (e.g., 'acestep-v15-base'). It will be auto-downloaded if missing.")
        elif available_dit_models:
            selected = _prompt_choice_from_list(
                "--- Available DiT Models ---",
                available_dit_models,
                default=args.config_path,
                allow_custom=True,
            )
            if selected is not None:
                args.config_path = selected
        else:
            print("\nNote: No local DiT models found. The main model will be auto-downloaded during initialization.")

        # Model selection (LM)
        llm_handler = LLMHandler()
        llm_handler.checkpoint_dir = args.checkpoint_dir
        available_lm_models = llm_handler.get_available_5hz_lm_models()
        if available_lm_models:
            selected_lm = _prompt_choice_from_list(
                "--- Available LM Models ---",
                available_lm_models,
                default=args.lm_model_path,
                allow_custom=True,
            )
            if selected_lm is not None:
                args.lm_model_path = selected_lm
        else:
            print("\nNote: No local LM models found. If LM features are enabled, a default LM will be auto-downloaded.")

        # Task-specific inputs
        if args.task_type in {"cover", "repaint", "lego", "extract", "complete"}:
            args.src_audio = _prompt_existing_file("Enter path to source audio file", default=args.src_audio)

        if args.task_type == "repaint":
            args.repainting_start = _prompt_float(
                "Repaint start time in seconds", args.repainting_start
            )
            args.repainting_end = _prompt_float(
                "Repaint end time in seconds", args.repainting_end
            )

        if args.task_type in {"lego", "extract"}:
            print("\nAvailable tracks:")
            print(", ".join(TRACK_CHOICES))
            track_default = args.lego_track if args.task_type == "lego" else args.extract_track
            track = _prompt_with_default("Choose a track", track_default, required=True)
            if track not in TRACK_CHOICES:
                print("Unknown track. Using as-is.")
            if args.task_type == "lego":
                args.lego_track = track
            else:
                args.extract_track = track
            if not args.instruction or args.instruction == DEFAULT_DIT_INSTRUCTION:
                args.instruction = _default_instruction_for_task(args.task_type, [track])
            args.instruction = _prompt_with_default("Instruction", args.instruction, required=True)

        if args.task_type == "complete":
            print("\nAvailable tracks:")
            print(", ".join(TRACK_CHOICES))
            tracks_raw = _prompt_with_default("Choose tracks (comma-separated)", args.complete_tracks, required=True)
            tracks = [t.strip() for t in tracks_raw.split(",") if t.strip()]
            args.complete_tracks = ",".join(tracks)
            if not args.instruction or args.instruction == DEFAULT_DIT_INSTRUCTION:
                args.instruction = _default_instruction_for_task(args.task_type, tracks)
            args.instruction = _prompt_with_default("Instruction", args.instruction, required=True)

        if args.task_type in {"cover", "repaint", "lego", "complete"}:
            args.caption = _prompt_with_default(
                "Enter a music description (e.g., 'upbeat electronic dance music')",
                args.caption,
                required=True,
            )
        elif args.task_type == "text2music":
            args.sample_mode = _prompt_bool("Use Simple Mode (auto-generate caption/lyrics via LM)", args.sample_mode)
            if args.sample_mode:
                args.sample_query = _prompt_with_default(
                    "Describe the music you want (for auto-generation)",
                    args.sample_query,
                    required=False,
                )
            if not args.sample_mode:
                caption = _prompt_with_default(
                    "Enter a music description (optional if you provide lyrics)",
                    args.caption,
                    required=False,
                )
                if caption:
                    args.caption = caption

        # Lyrics
        if args.task_type in {"text2music", "cover", "repaint", "lego", "complete"} and not args.sample_mode:
            print("\n--- Lyrics Options ---")
            print("1. Instrumental (no lyrics).")
            print("2. Generate lyrics automatically.")
            print("3. Provide path to a .txt file.")
            print("4. Paste lyrics directly.")

            if args.instrumental or args.lyrics == "[Instrumental]":
                default_choice = "1"
            elif args.use_cot_lyrics:
                default_choice = "2"
            elif args.lyrics and isinstance(args.lyrics, str) and os.path.isfile(args.lyrics):
                default_choice = "3"
            elif args.lyrics:
                default_choice = "4"
            else:
                default_choice = "1"
            choice = input(f"Your choice (1-4) [default: {default_choice}]: ").strip()
            if not choice:
                choice = default_choice

            if choice == "1":  # Instrumental
                args.instrumental = True
                args.lyrics = "[Instrumental]"
                args.use_cot_lyrics = False
                print("Instrumental music will be generated.")
            elif choice == "2":  # Generate lyrics automatically
                args.use_cot_lyrics = True
                args.lyrics = ""
                args.instrumental = False
                print("Lyrics will be generated automatically.")
            elif choice == "3":
                args.instrumental = False
                args.use_cot_lyrics = False
                default_lyrics_path = args.lyrics if isinstance(args.lyrics, str) and os.path.isfile(args.lyrics) else None
                while True:
                    lyrics_path = _prompt_existing_file("Please enter the path to your .txt lyrics file", default_lyrics_path)
                    if lyrics_path.endswith('.txt'):
                        args.lyrics = lyrics_path
                        print(f"Lyrics will be loaded from: {lyrics_path}")
                        break
                    print("Invalid file path or not a .txt file. Please try again.")
            elif choice == "4":
                args.instrumental = False
                args.use_cot_lyrics = False
                default_lyrics = args.lyrics if isinstance(args.lyrics, str) and args.lyrics and not os.path.isfile(args.lyrics) else None
                args.lyrics = _prompt_with_default("Paste lyrics (single line or use \\n)", default_lyrics, required=True)

            if not args.instrumental:
                lang = _prompt_with_default(
                    "Vocal language (e.g., 'en', 'zh', 'unknown')",
                    args.vocal_language,
                    required=False
                ).lower()
                if lang:
                    args.vocal_language = lang

            if args.use_cot_lyrics:
                if not args.caption:
                    args.caption = _prompt_non_empty("Enter a music description for lyric generation: ")
                if not args.thinking:
                    print("INFO: Automatic lyric generation requires the LM handler. Enabling LM 'thinking'.")
                    args.thinking = True

        args.batch_size = _prompt_int(
            "Number of outputs (audio clips) to generate",
            args.batch_size if args.batch_size is not None else 2,
            min_value=1,
        )

        advanced = input("\nConfigure advanced parameters? (y/n) [default: n]: ").lower()
        if advanced == 'y':
            if args.task_type == "text2music" and not args.sample_mode:
                args.use_format = _prompt_bool("Use format_sample to enhance caption/lyrics", args.use_format)
            print("\n--- Optional Metadata ---")
            args.duration = _prompt_float("Duration in seconds (10-600)", args.duration, min_value=10, max_value=600)
            args.bpm = _prompt_int("BPM (30-300, empty for auto)", args.bpm, min_value=30, max_value=300)
            args.keyscale = _prompt_with_default("Keyscale (e.g., 'C Major', empty for auto)", args.keyscale)
            args.timesignature = _prompt_with_default("Time signature (e.g., '4/4', empty for auto)", args.timesignature)
            args.vocal_language = _prompt_with_default("Vocal language (e.g., 'en', 'zh', 'unknown')", args.vocal_language)

            print("\n--- Advanced DiT Settings ---")
            args.seed = _prompt_int("Random seed (-1 for random)", args.seed)
            args.inference_steps = _prompt_int("Inference steps", args.inference_steps, min_value=1)
            if args.config_path and 'base' in args.config_path:
                args.guidance_scale = _prompt_float("Guidance scale (for base models)", args.guidance_scale)
                args.use_adg = _prompt_bool("Enable Adaptive Dual Guidance (ADG)", args.use_adg)
                args.cfg_interval_start = _prompt_float("CFG interval start (0.0-1.0)", args.cfg_interval_start, 0.0, 1.0)
                args.cfg_interval_end = _prompt_float("CFG interval end (0.0-1.0)", args.cfg_interval_end, 0.0, 1.0)
            args.shift = _prompt_float("Timestep shift (1.0-5.0)", args.shift, 1.0, 5.0)
            args.infer_method = _prompt_with_default("Inference method (ode/sde)", args.infer_method)
            timesteps_input = _prompt_with_default(
                "Custom timesteps list (e.g., [0.97, 0.5, 0])",
                args.timesteps,
                required=False,
            )
            if timesteps_input:
                args.timesteps = timesteps_input

            if args.task_type == "cover":
                args.audio_cover_strength = _prompt_float(
                    "Audio cover strength (0.0-1.0)", args.audio_cover_strength, 0.0, 1.0
                )

            print("\n--- Advanced LM Settings ---")
            args.thinking = _prompt_bool("Enable LM 'thinking'", args.thinking)
            args.lm_temperature = _prompt_float("LM temperature (0.0-2.0)", args.lm_temperature, 0.0, 2.0)
            args.lm_cfg_scale = _prompt_float("LM CFG scale", args.lm_cfg_scale)
            args.lm_top_k = _prompt_int("LM top-k (0 disables)", args.lm_top_k, min_value=0)
            args.lm_top_p = _prompt_float("LM top-p (0.0-1.0)", args.lm_top_p, 0.0, 1.0)
            args.lm_negative_prompt = _prompt_with_default("LM negative prompt", args.lm_negative_prompt)
            args.use_cot_metas = _prompt_bool("Use CoT for metadata", args.use_cot_metas)
            args.use_cot_caption = _prompt_bool("Use CoT for caption refinement", args.use_cot_caption)
            args.use_cot_lyrics = _prompt_bool("Use CoT for lyrics generation", args.use_cot_lyrics)
            args.use_cot_language = _prompt_bool("Use CoT for language detection", args.use_cot_language)
            args.use_constrained_decoding = _prompt_bool("Use constrained decoding", args.use_constrained_decoding)

            print("\n--- Output Settings ---")
            args.save_dir = _prompt_with_default("Save directory", args.save_dir)
            args.audio_format = _prompt_with_default("Audio format (wav/wav32/flac/mp3/opus/aac)", args.audio_format)
            # Batch size already captured above.
            args.use_random_seed = _prompt_bool("Use random seed per batch", args.use_random_seed)
            seeds_input = _prompt_with_default(
                "Custom seeds (comma/space separated, leave empty for random)",
                "",
                required=False,
            )
            if seeds_input:
                seeds = [s for s in seeds_input.replace(",", " ").split() if s.strip()]
                try:
                    args.seeds = [int(s) for s in seeds]
                except ValueError:
                    print("Invalid seeds input. Ignoring custom seeds.")
            args.lm_batch_chunk_size = _prompt_int("LM batch chunk size", args.lm_batch_chunk_size, min_value=1)
            args.constrained_decoding_debug = _prompt_bool("Constrained decoding debug", args.constrained_decoding_debug)
        else:
            if params_defaults and config_defaults:
                _apply_optional_defaults(args, params_defaults, config_defaults)

        # Ensure LM thinking is enabled when lyric generation is requested.
        if args.use_cot_lyrics and not args.thinking:
            print("INFO: Automatic lyric generation requires the LM handler. Enabling LM 'thinking'.")
            args.thinking = True

        print("\n--- Summary ---")
        print(f"Task: {args.task_type}")
        if args.caption:
            print(f"Description: {args.caption}")
        if args.task_type in {"lego", "extract", "complete"}:
            print(f"Instruction: {args.instruction}")
        if args.src_audio:
            print(f"Source audio: {args.src_audio}")
        print(f"Duration: {args.duration}s")
        print(f"Outputs: {args.batch_size}")
        if args.instrumental:
            print("Lyrics: Instrumental")
        elif args.use_cot_lyrics:
            print(f"Lyrics: Auto-generated ({args.vocal_language})")
        elif args.lyrics and os.path.isfile(args.lyrics):
             print(f"Lyrics: Provided from file ({args.lyrics})")
        elif args.lyrics:
             print(f"Lyrics: Provided as text")

        print("-" * 30)
        if not configure_only:
            confirm = input("Start generation with these settings? (y/n) [default: y]: ").lower()
            if confirm == 'n':
                print("Generation cancelled.")
                sys.exit(0)

        from acestep.audio_utils import SUPPORTED_AUDIO_FORMATS
        args.audio_format = str(args.audio_format).lower()
        if args.audio_format not in SUPPORTED_AUDIO_FORMATS:
            print(f"Unsupported audio format: {args.audio_format}", file=sys.stderr)
            sys.exit(1)

        default_filename = default_config_path or "config.toml"
        config_filename = input(f"\nEnter filename to save configuration [{default_filename}]: ")
        if not config_filename:
            config_filename = default_filename
        if not config_filename.endswith(".toml"):
            config_filename += ".toml"

        try:
            config_to_save = {
                k: v for k, v in vars(args).items()
                if k not in {'config'} and not k.startswith('_')
            }
            with open(config_filename, 'w') as f:
                toml.dump(config_to_save, f)
            print(f"Configuration saved to {config_filename}")
            print(f"You can reuse it next time with: python cli.py -c {config_filename}")
        except Exception as e:
            print(f"Error saving configuration: {e}", file=sys.stderr)
            sys.exit(1)

    except (KeyboardInterrupt, EOFError):
        print("\nWizard cancelled. Exiting.")
        sys.exit(0)

    return args, not configure_only


_cpu_threads_configured = False


def _configure_cpu_threads_if_requested() -> None:
    """Apply the opt-in PyTorch CPU thread limit once per CLI process."""
    global _cpu_threads_configured
    if _cpu_threads_configured:
        return
    if os.environ.get("ACESTEP_CONFIGURE_THREADS", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    threads = max(1, (os.cpu_count() or 1) - 2)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    _cpu_threads_configured = True


def main():
    """
    Main function to run ACE-Step music generation from the command line.
    """

    _configure_cpu_threads_if_requested()
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    mps_available = is_mps_platform()
    # Mac (Apple Silicon) uses unified memory — offloading provides no benefit
    auto_offload = (not mps_available) and gpu_config.gpu_memory_gb > 0 and gpu_config.gpu_memory_gb < 16
    print(f"\n{'='*60}")
    print("GPU Configuration Detected:")
    print(f"{'='*60}")
    print(f"  GPU Memory: {gpu_config.gpu_memory_gb:.2f} GiB")
    print(f"  Configuration Tier: {gpu_config.tier}")
    print(f"  Max Duration (with LM): {gpu_config.max_duration_with_lm}s ({gpu_config.max_duration_with_lm // 60} min)")
    print(f"  Max Duration (without LM): {gpu_config.max_duration_without_lm}s ({gpu_config.max_duration_without_lm // 60} min)")
    print(f"  Max Batch Size (with LM): {gpu_config.max_batch_size_with_lm}")
    print(f"  Max Batch Size (without LM): {gpu_config.max_batch_size_without_lm}")
    print(f"  Default LM Init: {gpu_config.init_lm_default}")
    print(f"  Available LM Models: {gpu_config.available_lm_models or 'None'}")
    print(f"{'='*60}\n")

    if auto_offload:
        print("Auto-enabling CPU offload (GPU < 16GB)")
    elif gpu_config.gpu_memory_gb > 0:
        print("CPU offload disabled by default (GPU >= 16GB)")
    elif mps_available:
        print("MPS detected, running on Apple GPU")
    else:
        print("No GPU detected, running on CPU")

    params_defaults = GenerationParams()
    config_defaults = GenerationConfig()

    parser = argparse.ArgumentParser(
        description="ACE-Step 1.5: Music generation (wizard/config only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--config", type=str, help="Path to a TOML configuration file to load.")
    parser.add_argument("--configure", action="store_true", help="Run wizard to save configuration without generating.")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level for internal modules (TRACE/DEBUG/INFO/WARNING/ERROR/CRITICAL).",
    )
    cli_args = parser.parse_args()

    default_batch_size = 1 if not cli_args.config else config_defaults.batch_size

    defaults = {
        "project_root": _get_project_root(),
        "config_path": None,
        "checkpoint_dir": None,
        "lm_model_path": None,
        "device": "auto",
        "offload_to_cpu": auto_offload,
        "offload_dit_to_cpu": False,
        "save_dir": "output",
        "audio_format": config_defaults.audio_format,
        "caption": "",
        "prompt": "",
        "lyrics": None,
        "duration": params_defaults.duration,
        "instrumental": False,
        "bpm": params_defaults.bpm,
        "keyscale": params_defaults.keyscale,
        "timesignature": params_defaults.timesignature,
        "vocal_language": params_defaults.vocal_language,
        "task_type": params_defaults.task_type,
        "instruction": params_defaults.instruction,
        "reference_audio": params_defaults.reference_audio,
        "src_audio": params_defaults.src_audio,
        "repainting_start": params_defaults.repainting_start,
        "repainting_end": params_defaults.repainting_end,
        "audio_cover_strength": params_defaults.audio_cover_strength,
        "lego_track": "",
        "extract_track": "",
        "complete_tracks": "",
        "sample_mode": False,
        "sample_query": "",
        "use_format": False,
        "inference_steps": params_defaults.inference_steps,
        "seed": params_defaults.seed,
        "guidance_scale": params_defaults.guidance_scale,
        "use_adg": params_defaults.use_adg,
        # DCW defaults to the loaded model's configuration.
        "dcw_enabled": None,
        "dcw_mode": params_defaults.dcw_mode,
        "dcw_scaler": params_defaults.dcw_scaler,
        "dcw_high_scaler": params_defaults.dcw_high_scaler,
        "dcw_wavelet": params_defaults.dcw_wavelet,
        "shift": 3.0,
        "infer_method": params_defaults.infer_method,
        "timesteps": None,
        "thinking": gpu_config.init_lm_default,
        "lm_temperature": params_defaults.lm_temperature,
        "lm_cfg_scale": params_defaults.lm_cfg_scale,
        "lm_top_k": params_defaults.lm_top_k,
        "lm_top_p": params_defaults.lm_top_p,
        "use_cot_metas": params_defaults.use_cot_metas,
        "use_cot_caption": params_defaults.use_cot_caption,
        "use_cot_lyrics": params_defaults.use_cot_lyrics,
        "use_cot_language": params_defaults.use_cot_language,
        "use_constrained_decoding": params_defaults.use_constrained_decoding,
        "batch_size": default_batch_size,
        "seeds": None,
        "use_random_seed": config_defaults.use_random_seed,
        "lm_batch_chunk_size": config_defaults.lm_batch_chunk_size,
        "constrained_decoding_debug": config_defaults.constrained_decoding_debug,
        "audio_codes": "",
        "cfg_interval_start": params_defaults.cfg_interval_start,
        "cfg_interval_end": params_defaults.cfg_interval_end,
        "lm_negative_prompt": params_defaults.lm_negative_prompt,
        "repaint_mode": params_defaults.repaint_mode,
        "repaint_strength": params_defaults.repaint_strength,
        "repaint_latent_crossfade_frames": params_defaults.repaint_latent_crossfade_frames,
        "repaint_wav_crossfade_sec": params_defaults.repaint_wav_crossfade_sec,
        "chunk_mask_mode": params_defaults.chunk_mask_mode,
        "retake_seed": params_defaults.retake_seed,
        "retake_variance": params_defaults.retake_variance,
        "flow_edit_morph": params_defaults.flow_edit_morph,
        "flow_edit_source_caption": params_defaults.flow_edit_source_caption,
        "flow_edit_source_lyrics": params_defaults.flow_edit_source_lyrics,
        "flow_edit_n_min": params_defaults.flow_edit_n_min,
        "flow_edit_n_max": params_defaults.flow_edit_n_max,
        "flow_edit_n_avg": params_defaults.flow_edit_n_avg,
        "cover_noise_strength": params_defaults.cover_noise_strength,
        "global_caption": params_defaults.global_caption,
        "sampler_mode": params_defaults.sampler_mode,
        "velocity_norm_threshold": params_defaults.velocity_norm_threshold,
        "velocity_ema_factor": params_defaults.velocity_ema_factor,
        "enable_normalization": params_defaults.enable_normalization,
        "normalization_db": params_defaults.normalization_db,
        "fade_in_duration": params_defaults.fade_in_duration,
        "fade_out_duration": params_defaults.fade_out_duration,
        "latent_shift": params_defaults.latent_shift,
        "latent_rescale": params_defaults.latent_rescale,
        "quantization": None,
        "vae_checkpoint": None,
        "mp3_bitrate": config_defaults.mp3_bitrate,
        "mp3_sample_rate": config_defaults.mp3_sample_rate,
        "log_level": cli_args.log_level or "INFO",
    }

    args = argparse.Namespace(**defaults)
    args.config = None
    if cli_args.config:
        if not os.path.exists(cli_args.config):
            parser.error(f"Config file not found: {cli_args.config}")
        try:
            with open(cli_args.config, 'r') as f:
                config_from_file = toml.load(f)
            print(f"Configuration loaded from {cli_args.config}")
        except Exception as e:
            parser.error(f"Error loading TOML config file {cli_args.config}: {e}")
        for key, value in config_from_file.items():
            if key not in defaults:
                parser.error(f"Unknown configuration parameter: {key}")
            setattr(args, key, value)
        args.config = cli_args.config
        # Explicit seeds disable random seeding unless requested otherwise.
        args._use_random_seed_explicit = "use_random_seed" in config_from_file

    # Validate runtime settings.
    if cli_args.log_level is not None:
        args.log_level = cli_args.log_level
    try:
        _configure_logging(level=args.log_level)
    except (ValueError, TypeError) as exc:
        parser.error(f"Invalid log level: {exc}")

    args.project_root = os.path.abspath(os.path.expanduser(str(args.project_root or _get_project_root())))
    from acestep.model_downloader import get_checkpoints_dir
    custom_dir = args.checkpoint_dir
    if custom_dir:
        custom_dir = os.path.join(args.project_root, os.path.expanduser(str(custom_dir)))
    args.checkpoint_dir = str(get_checkpoints_dir(custom_dir, project_root=args.project_root))

    if cli_args.configure:
        args, _ = run_wizard(
            args,
            configure_only=True,
            default_config_path=cli_args.config,
            params_defaults=params_defaults,
            config_defaults=config_defaults,
        )
        print("Configuration complete. Exiting without generation.")
        sys.exit(0)

    if not cli_args.config:
        args, should_generate = run_wizard(
            args,
            configure_only=False,
            default_config_path=None,
            params_defaults=params_defaults,
            config_defaults=config_defaults,
        )
        if not should_generate:
            print("Configuration complete. Exiting without generation.")
            sys.exit(0)

    # --- Post-parsing Setup ---
    if args.use_cot_lyrics and not args.thinking:
        print("INFO: Automatic lyric generation requires the LM handler. Forcing --thinking=True.")
        args.thinking = True
    
    if not args.project_root:
        args.project_root = _get_project_root()
    else:
        args.project_root = os.path.abspath(os.path.expanduser(str(args.project_root)))

    if args.checkpoint_dir:
        args.checkpoint_dir = os.path.expanduser(str(args.checkpoint_dir))
        if not os.path.isabs(args.checkpoint_dir):
            args.checkpoint_dir = os.path.join(args.project_root, args.checkpoint_dir)

    if args.src_audio:
        args.src_audio = _expand_audio_path(args.src_audio)
    if args.reference_audio:
        args.reference_audio = _expand_audio_path(args.reference_audio)

    device = _resolve_device(args.device)

    # --- Argument Post-processing ---
    try:
        timesteps = _parse_timesteps_input(args.timesteps)
        if args.timesteps and timesteps is None:
            raise ValueError("Timesteps must be a list of numbers or a comma-separated string.")
    except ValueError as e:
        parser.error(f"Invalid format for timesteps. Expected a list of numbers (e.g., '[1.0, 0.5, 0.0]' or '0.97,0.5,0'). Error: {e}")

    if isinstance(args.seeds, int):
        args.seeds = [args.seeds]
    if args.seeds:
        args.batch_size = len(args.seeds)
        args.use_random_seed = False
        args.seed = -1
    elif (
        args.config
        and not getattr(args, "_use_random_seed_explicit", False)
        and args.seed is not None
        and args.seed != -1
        and args.use_random_seed
    ):
        # Honor an explicit seed.
        args.use_random_seed = False
        print(
            f"INFO: use_random_seed not set explicitly while seed={args.seed} was provided in "
            f"'{args.config}'; defaulting use_random_seed to False so the seed takes effect."
        )

    if args.instrumental and not args.lyrics:
        args.lyrics = "[Instrumental]"
    elif isinstance(args.lyrics, str) and args.lyrics.strip().lower() in {"[inst]", "[instrumental]"}:
        args.instrumental = True

    # --- Task-specific validation and instruction helpers ---
    if args.task_type not in TASK_TYPES:
        parser.error(f"Unsupported task_type: {args.task_type}")
    if args.prompt and not args.caption:
        args.caption = args.prompt
    if args.flow_edit_morph:
        if args.task_type not in {"text2music", "cover", "cover-nofsq"}:
            parser.error("FlowEdit is supported on text2music, cover and cover-nofsq only.")
        if not args.src_audio:
            parser.error("FlowEdit requires src_audio.")
        if args.sample_mode or args.sample_query or args.use_format or args.use_cot_lyrics:
            parser.error("FlowEdit requires explicit target caption/lyrics, not LM sample/format/lyrics generation.")
        if not (0 <= args.flow_edit_n_min <= args.flow_edit_n_max <= 1) or args.flow_edit_n_avg < 1:
            parser.error("FlowEdit requires 0 <= n_min <= n_max <= 1 and n_avg >= 1.")
    if args.repaint_mode not in {"conservative", "balanced", "aggressive"}:
        parser.error(f"Unsupported repaint_mode: {args.repaint_mode}")
    if not 0 <= args.retake_variance <= 1:
        parser.error("retake_variance must be between 0 and 1.")
    if args.quantization not in {None, "int8_weight_only", "fp8_weight_only", "w8a8_dynamic"}:
        parser.error(f"Unsupported quantization: {args.quantization}")
    if args.task_type in DIRECT_CONDITIONING_TASKS:
        if not args.src_audio:
            parser.error(f"--src_audio is required for task_type '{args.task_type}'.")

    if args.task_type in {"cover", "cover-nofsq", "repaint", "lego", "complete"}:
        if not args.caption:
            parser.error(f"--caption is required for task_type '{args.task_type}'.")

    if args.task_type == "text2music":
        if not args.caption and not args.lyrics:
            if not args.sample_mode and not args.sample_query:
                parser.error("--caption or --lyrics is required for text2music.")
        if args.use_cot_lyrics and not args.caption:
            parser.error("--use_cot_lyrics requires --caption for lyric generation.")
        if args.sample_mode or args.sample_query:
            args.sample_mode = True
    else:
        if args.sample_mode or args.sample_query:
            parser.error("--sample_mode/sample_query are only supported for task_type 'text2music'.")

    if args.sample_mode and args.use_cot_lyrics:
        print("INFO: sample_mode enabled. Disabling --use_cot_lyrics.")
        args.use_cot_lyrics = False

    # Auto-select instruction based on task_type if user didn't provide a custom instruction.
    if args.instruction == DEFAULT_DIT_INSTRUCTION and args.task_type in TASK_INSTRUCTIONS:
        if args.task_type in {"text2music", "cover", "repaint"}:
            args.instruction = TASK_INSTRUCTIONS[args.task_type]

    # Base-model-only task enforcement
    base_only_tasks = {"lego", "extract", "complete"}
    if args.task_type in base_only_tasks and args.config_path:
        if "base" not in str(args.config_path).lower():
            parser.error(f"task_type '{args.task_type}' requires a base model config (e.g., 'acestep-v15-base').")

    if args.task_type == "repaint":
        if args.repainting_end != -1 and args.repainting_end <= args.repainting_start:
            parser.error("--repainting_end must be greater than --repainting_start (or -1).")

    if args.task_type in {"lego", "extract", "complete"}:
        has_custom_instruction = bool(args.instruction and args.instruction.strip() and args.instruction.strip() != params_defaults.instruction)
        if not has_custom_instruction:
            if args.task_type == "lego":
                if not args.lego_track:
                    parser.error("--instruction or --lego_track is required for lego task.")
                args.instruction = _default_instruction_for_task("lego", [args.lego_track.strip()])
            elif args.task_type == "extract":
                if not args.extract_track:
                    parser.error("--instruction or --extract_track is required for extract task.")
                args.instruction = _default_instruction_for_task("extract", [args.extract_track.strip()])
            elif args.task_type == "complete":
                if not args.complete_tracks:
                    parser.error("--instruction or --complete_tracks is required for complete task.")
                tracks = [t.strip() for t in args.complete_tracks.split(",") if t.strip()]
                if not tracks:
                    parser.error("--complete_tracks must contain at least one track.")
                args.instruction = _default_instruction_for_task("complete", tracks)
    
    # Handle lyrics argument
    lyrics_arg = args.lyrics
    if isinstance(lyrics_arg, str) and lyrics_arg:
        lyrics_arg = os.path.expanduser(lyrics_arg)
        if not os.path.isabs(lyrics_arg):
            # Resolve relative lyrics path against config file location first, then project_root.
            resolved = None
            if args.config:
                config_dir = os.path.dirname(os.path.abspath(args.config))
                candidate = os.path.join(config_dir, lyrics_arg)
                if os.path.isfile(candidate):
                    resolved = candidate
            if resolved is None and args.project_root:
                candidate = os.path.join(os.path.abspath(args.project_root), lyrics_arg)
                if os.path.isfile(candidate):
                    resolved = candidate
            if resolved is not None:
                lyrics_arg = resolved

    if lyrics_arg is not None:
        if lyrics_arg == "generate":
            args.use_cot_lyrics = True
            args.lyrics = ""
            print("Lyrics generation enabled.")
        elif os.path.isfile(lyrics_arg):
            print(f"INFO: Attempting to load lyrics from file: {lyrics_arg}")
            try:
                with open(lyrics_arg, 'r', encoding='utf-8') as f:
                    args.lyrics = f.read()
                print(f"Lyrics loaded from file: {lyrics_arg}")
            except Exception as e:
                parser.error(f"Could not read lyrics file {lyrics_arg}. Error: {e}")
        # else: lyrics is a string, use as is.

    if args.flow_edit_morph and args.use_cot_lyrics:
        parser.error("FlowEdit requires explicit target lyrics; lyrics=generate is not supported.")
    from acestep.audio_utils import SUPPORTED_AUDIO_FORMATS
    args.audio_format = str(args.audio_format).lower()
    if args.audio_format not in SUPPORTED_AUDIO_FORMATS:
        parser.error(f"Unsupported audio format: {args.audio_format}")
    if args.audio_format in {"mp3", "opus", "aac"} and not shutil.which("ffmpeg"):
        parser.error("FFmpeg is required for compressed audio output; install it and add it to PATH.")

    # --- Handler Initialization ---
    print("Initializing ACE-Step handlers...")
    dit_handler = AceStepHandler(project_root=args.project_root, checkpoint_dir=args.checkpoint_dir)
    llm_handler = LLMHandler()
    llm_handler.checkpoint_dir = args.checkpoint_dir

    base_only_tasks = {"lego", "extract", "complete"}
    skip_lm_tasks = DIRECT_CONDITIONING_TASKS
    # Direct conditioning tasks do not need LM audio codes, but explicit
    # sample formatting or lyric generation can still use the LM beforehand.
    requires_lm = not args.flow_edit_morph and (
        args.sample_mode or bool(args.sample_query and str(args.sample_query).strip())
        or args.use_format or args.use_cot_lyrics
        or (args.task_type not in skip_lm_tasks and (
            args.thinking or args.use_cot_metas or args.use_cot_caption or args.use_cot_language
        ))
    )

    if args.config_path is None:
        available_models = dit_handler.get_available_acestep_v15_models()
        if args.task_type in base_only_tasks and available_models:
            available_models = [m for m in available_models if "base" in m.lower()]
        if not available_models:
            preferred = "acestep-v15-base" if args.task_type in base_only_tasks else "acestep-v15-turbo"
            print(f"No compatible DiT models found. Downloading {preferred}...")
            from acestep.model_downloader import ensure_dit_model, get_checkpoints_dir
            checkpoints_dir = get_checkpoints_dir(args.checkpoint_dir)
            success, msg = ensure_dit_model(preferred, checkpoints_dir)
            print(msg)
            if not success:
                parser.error(f"Failed to download DiT model: {msg}")
            available_models = [preferred]
        if available_models:
            if args.task_type in {"lego", "extract", "complete"}:
                preferred = "acestep-v15-base"
            else:
                preferred = "acestep-v15-turbo"
            args.config_path = preferred if preferred in available_models else available_models[0]
            print(f"Auto-selected config_path: {args.config_path}")
        else:
            parser.error("No available DiT models found. Please specify --config_path.")
    if args.task_type in {"lego", "extract", "complete"} and "base" not in str(args.config_path).lower():
        parser.error(f"task_type '{args.task_type}' requires a base model config (e.g., 'acestep-v15-base').")

    # Component prechecks and downloads are owned by initialize_service.
    from acestep.model_downloader import get_checkpoints_dir, check_model_exists, SUBMODEL_REGISTRY, DEFAULT_LM_MODEL
    checkpoints_dir = get_checkpoints_dir(args.checkpoint_dir)

    compile_model = os.environ.get("ACESTEP_COMPILE_MODEL", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }

    print(f"Initializing DiT handler with model: {args.config_path}")
    status, success = dit_handler.initialize_service(
        project_root=args.project_root,
        config_path=args.config_path,
        device=device,
        compile_model=compile_model,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=args.offload_dit_to_cpu,
        checkpoint_dir=args.checkpoint_dir,
        quantization=args.quantization,
        vae_checkpoint=args.vae_checkpoint,
    )
    if not success:
        parser.error(f"DiT initialization failed: {status}")
    if args.quantization and dit_handler.quantization != args.quantization:
        parser.error(f"Requested quantization {args.quantization} is unavailable on this device; install a compatible torchao or omit quantization.")

    if requires_lm:
        from acestep.model_downloader import ensure_lm_model
        if args.lm_model_path is None:
            available_lm_models = [name for name in llm_handler.get_available_5hz_lm_models() if check_model_exists(name, checkpoints_dir)]
            if available_lm_models:
                args.lm_model_path = available_lm_models[0]
                print(f"Using default LM model: {args.lm_model_path}")
            else:
                success, msg = ensure_lm_model(checkpoints_dir=checkpoints_dir)
                print(msg)
                if not success:
                    parser.error("No LM models available. Please specify --lm_model_path or disable --thinking.")
                available_lm_models = llm_handler.get_available_5hz_lm_models()
                if not available_lm_models:
                    parser.error("No LM models available after download. Please specify --lm_model_path or disable --thinking.")
                args.lm_model_path = available_lm_models[0]
                print(f"Using default LM model: {args.lm_model_path}")
        else:
            lm_model_path = str(args.lm_model_path)
            if os.path.isabs(lm_model_path) and os.path.exists(lm_model_path):
                pass
            elif check_model_exists(lm_model_path, checkpoints_dir):
                pass
            elif lm_model_path == DEFAULT_LM_MODEL or lm_model_path in SUBMODEL_REGISTRY:
                success, msg = ensure_lm_model(lm_model_path, checkpoints_dir=checkpoints_dir)
                print(msg)
                if not success:
                    parser.error(f"Failed to download LM model '{lm_model_path}': {msg}")
            else:
                parser.error(f"LM model '{lm_model_path}' not found locally and not in registry. Please provide a valid --lm_model_path.")

        print(f"Initializing LM handler with model: {args.lm_model_path}")
        status, success = llm_handler.initialize(
            checkpoint_dir=args.checkpoint_dir,
            lm_model_path=args.lm_model_path,
            device=device,
            offload_to_cpu=args.offload_to_cpu,
            dtype=None,
        )
        if not success:
            parser.error(f"LM initialization failed: {status}")
    else:
        if args.task_type in skip_lm_tasks:
            print(f"LM is not required for task_type '{args.task_type}'. Skipping LM handler initialization.")
        else:
            print("LM 'thinking' is disabled. Skipping LM handler initialization.")

    print("Handlers initialized.")

    format_has_duration = False

    # --- Sample Mode / Description-based Auto-Generation ---
    if args.sample_mode or (args.sample_query and str(args.sample_query).strip()):
        if not llm_handler.llm_initialized:
            parser.error("--sample_mode/sample_query requires the LM handler, but it's not initialized.")

        sample_query = args.sample_query if args.sample_query and str(args.sample_query).strip() else "NO USER INPUT"
        parsed_language, parsed_instrumental = _parse_description_hints(sample_query)

        if args.vocal_language and args.vocal_language not in ("en", "unknown", ""):
            sample_language = args.vocal_language
        else:
            sample_language = parsed_language

        print("\nINFO: Creating sample via 'create_sample'...")
        sample_result = create_sample(
            llm_handler=llm_handler,
            query=sample_query,
            instrumental=parsed_instrumental,
            vocal_language=sample_language,
            temperature=args.lm_temperature,
            top_k=args.lm_top_k,
            top_p=args.lm_top_p,
        )

        if sample_result.success:
            args.caption = sample_result.caption
            args.lyrics = sample_result.lyrics
            args.instrumental = bool(sample_result.instrumental)
            if args.bpm is None:
                args.bpm = sample_result.bpm
            if not args.keyscale:
                args.keyscale = sample_result.keyscale
            if not args.timesignature:
                args.timesignature = sample_result.timesignature
            if args.duration is None or args.duration <= 0:
                args.duration = sample_result.duration
            if args.vocal_language in ("unknown", "", None):
                args.vocal_language = sample_result.language
            args.sample_mode = True
            print("✓ Sample created. Using generated parameters.")
        else:
            parser.error(f"create_sample failed: {sample_result.error or sample_result.status_message}")

    # --- Format caption/lyrics if requested ---
    if args.use_format and (args.caption or args.lyrics):
        if not llm_handler.llm_initialized:
            parser.error("--use_format requires the LM handler, but it's not initialized.")

        user_metadata_for_format = {}
        if args.bpm is not None:
            user_metadata_for_format["bpm"] = args.bpm
        if args.duration is not None and float(args.duration) > 0:
            user_metadata_for_format["duration"] = float(args.duration)
        if args.keyscale:
            user_metadata_for_format["keyscale"] = args.keyscale
        if args.timesignature:
            user_metadata_for_format["timesignature"] = args.timesignature
        if args.vocal_language and args.vocal_language != "unknown":
            user_metadata_for_format["language"] = args.vocal_language

        print("\nINFO: Formatting caption/lyrics via 'format_sample'...")
        format_result = format_sample(
            llm_handler=llm_handler,
            caption=args.caption or "",
            lyrics=args.lyrics or "",
            user_metadata=user_metadata_for_format if user_metadata_for_format else None,
            temperature=args.lm_temperature,
            top_k=args.lm_top_k,
            top_p=args.lm_top_p,
        )

        if format_result.success:
            args.caption = format_result.caption or args.caption
            args.lyrics = format_result.lyrics or args.lyrics
            if format_result.duration:
                args.duration = format_result.duration
                format_has_duration = True
            if format_result.bpm:
                args.bpm = format_result.bpm
            if format_result.keyscale:
                args.keyscale = format_result.keyscale
            if format_result.timesignature:
                args.timesignature = format_result.timesignature
            print("✓ Format complete.")
        else:
            parser.error(f"format_sample failed: {format_result.error or format_result.status_message}")

    # --- Auto-generate Lyrics if Requested ---
    if args.use_cot_lyrics:
        if not llm_handler.llm_initialized:
             parser.error("--use_cot_lyrics requires the LM handler, but it's not initialized. Ensure --thinking is enabled.")

        print("\nINFO: Generating lyrics and metadata via 'create_sample'...")
        sample_result = create_sample(
            llm_handler=llm_handler,
            query=args.caption,
            instrumental=False,
            vocal_language=args.vocal_language if args.vocal_language != 'unknown' else None,
            temperature=args.lm_temperature,
            top_k=args.lm_top_k,
            top_p=args.lm_top_p,
        )

        if sample_result.success:
            print("✓ Automatic sample creation successful. Using generated parameters:")
            # Update args with values from create_sample, respecting user-provided values
            args.caption = sample_result.caption
            args.lyrics = sample_result.lyrics
            if args.bpm is None: args.bpm = sample_result.bpm
            if not args.keyscale: args.keyscale = sample_result.keyscale
            if not args.timesignature: args.timesignature = sample_result.timesignature
            if args.duration is None or args.duration <= 0: args.duration = sample_result.duration
            if args.vocal_language == 'unknown': args.vocal_language = sample_result.language

            print(f"  - Caption: {args.caption}")
            lyrics_preview = args.lyrics[:150].strip().replace("\n", " ")
            print(f"  - Lyrics: '{lyrics_preview}...'")
            print(f"  - Metadata: BPM={args.bpm}, Key='{args.keyscale}', Lang='{args.vocal_language}'")

            # Disable subsequent CoT steps to avoid redundancy and save time
            args.use_cot_metas = False
            args.use_cot_caption = False
        else:
            print(f"⚠️ WARNING: Automatic lyric generation via 'create_sample' failed: {sample_result.error}")
            print("         Proceeding with an instrumental track instead.")
            args.lyrics = "[Instrumental]"
            args.instrumental = True

        # Flag has served its purpose, disable it to avoid issues with GenerationParams
        args.use_cot_lyrics = False

    if args.sample_mode or format_has_duration:
        args.use_cot_metas = False

    # Optional audio-token prompt editing.
    prompt_editor = None
    if args.thinking and args.task_type not in skip_lm_tasks and not args.flow_edit_morph:
        instruction_path = os.path.join(args.project_root, "instruction.txt")
        if args.config:
            if os.path.isfile(instruction_path):
                with open(instruction_path, "r", encoding="utf-8") as f:
                    preloaded_prompt = f.read()
                if preloaded_prompt.strip():
                    prompt_editor = _make_prompt_editor(instruction_path, preloaded_prompt)
        else:
            prompt_editor = _make_prompt_editor(instruction_path)

    # --- Configure Generation ---
    params = GenerationParams(
        task_type=args.task_type,
        instruction=args.instruction,
        reference_audio=args.reference_audio,
        src_audio=args.src_audio,
        audio_codes=args.audio_codes,
        caption=args.caption,
        lyrics=args.lyrics,
        instrumental=args.instrumental,
        vocal_language=args.vocal_language,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        duration=args.duration,
        inference_steps=args.inference_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        use_adg=args.use_adg,
        dcw_enabled=args.dcw_enabled,
        dcw_mode=args.dcw_mode,
        dcw_scaler=args.dcw_scaler,
        dcw_high_scaler=args.dcw_high_scaler,
        dcw_wavelet=args.dcw_wavelet,
        cfg_interval_start=args.cfg_interval_start,
        cfg_interval_end=args.cfg_interval_end,
        shift=args.shift,
        infer_method=args.infer_method,
        timesteps=timesteps,
        repainting_start=args.repainting_start,
        repainting_end=args.repainting_end,
        audio_cover_strength=args.audio_cover_strength,
        thinking=args.thinking,
        lm_temperature=args.lm_temperature,
        lm_cfg_scale=args.lm_cfg_scale,
        lm_top_k=args.lm_top_k,
        lm_top_p=args.lm_top_p,
        lm_negative_prompt=args.lm_negative_prompt,
        use_cot_metas=args.use_cot_metas,
        use_cot_caption=args.use_cot_caption,
        use_cot_lyrics=args.use_cot_lyrics,
        use_cot_language=args.use_cot_language,
        use_constrained_decoding=args.use_constrained_decoding,
        repaint_mode=args.repaint_mode,
        repaint_strength=args.repaint_strength,
        repaint_latent_crossfade_frames=args.repaint_latent_crossfade_frames,
        repaint_wav_crossfade_sec=args.repaint_wav_crossfade_sec,
        chunk_mask_mode=args.chunk_mask_mode,
        retake_seed=args.retake_seed,
        retake_variance=args.retake_variance,
        flow_edit_morph=args.flow_edit_morph,
        flow_edit_source_caption=args.flow_edit_source_caption,
        flow_edit_source_lyrics=args.flow_edit_source_lyrics,
        flow_edit_n_min=args.flow_edit_n_min,
        flow_edit_n_max=args.flow_edit_n_max,
        flow_edit_n_avg=args.flow_edit_n_avg,
        cover_noise_strength=args.cover_noise_strength,
        global_caption=args.global_caption,
        sampler_mode=args.sampler_mode,
        velocity_norm_threshold=args.velocity_norm_threshold,
        velocity_ema_factor=args.velocity_ema_factor,
        enable_normalization=args.enable_normalization,
        normalization_db=args.normalization_db,
        fade_in_duration=args.fade_in_duration,
        fade_out_duration=args.fade_out_duration,
        latent_shift=args.latent_shift,
        latent_rescale=args.latent_rescale,
    )

    config = GenerationConfig(
        batch_size=args.batch_size,
        use_random_seed=args.use_random_seed,
        seeds=args.seeds,
        lm_batch_chunk_size=args.lm_batch_chunk_size,
        constrained_decoding_debug=args.constrained_decoding_debug,
        audio_format=args.audio_format,
        mp3_bitrate=args.mp3_bitrate,
        mp3_sample_rate=args.mp3_sample_rate,
    )

    # --- Generate Music ---
    log_level = getattr(args, "log_level", "INFO")
    log_level_upper = str(log_level).upper()
    compact_logs = log_level_upper != "DEBUG"
    _print_final_parameters(
        args,
        params,
        config,
        params_defaults,
        config_defaults,
        compact=compact_logs,
        resolved_device=device,
    )

    print("\n--- Starting Generation ---")
    print(f"Caption: \"{params.caption}\"")
    print(f"Duration: {params.duration}s | Outputs: {config.batch_size}")
    if config.seeds:
        print(f"Custom Seeds: {config.seeds}")
    print("---------------------------\n")

    before_dit = None
    if log_level_upper in {"INFO", "DEBUG"}:
        before_dit = lambda final_params: _print_dit_prompt(dit_handler, final_params)
    result = generate_music(
        dit_handler, llm_handler, params, config,
        save_dir=args.save_dir, prompt_editor=prompt_editor, before_dit=before_dit,
    )

    # --- Process Results ---
    if result.success:
        print(f"\n✅ Generation successful! {len(result.audios)} audio(s) saved in '{args.save_dir}/'")
        for i, audio in enumerate(result.audios):
            print(f"  [{i+1}] Path: {audio['path']} | Seed: {audio['params']['seed']}")
        
        time_costs = result.extra_outputs.get("time_costs", {})
        if time_costs:
            print("\n--- Performance ---")
            total_time = time_costs.get('pipeline_total_time', 0)
            print(f"Total time: {total_time:.2f}s")
            if args.thinking:
                lm1_time = time_costs.get('lm_phase1_time', 0)
                lm2_time = time_costs.get('lm_phase2_time', 0)
                print(f"  - LM time: {lm1_time + lm2_time:.2f}s")
            dit_time = time_costs.get('dit_total_time_cost', 0)
            print(f"  - DiT time: {dit_time:.2f}s")
            print("-------------------\n")

    else:
        print(f"\n❌ Generation failed: {result.error}", file=sys.stderr)
        print(f"   Status: {result.status_message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
