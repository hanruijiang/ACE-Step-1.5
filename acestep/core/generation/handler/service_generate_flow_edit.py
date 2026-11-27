"""Run FlowEdit with separate source and target conditioning."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from .service_generate_flow_edit_source import embed_source, tokenize_source


def dispatch_flow_edit_overlay(
    handler,
    *,
    payload: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
    seed_param: Any,
    flow_edit_ctx: Dict[str, Any],
    return_intermediate: bool = False,
) -> Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run paired FlowEdit conditioning and diffusion."""
    if not hasattr(handler.model, "flowedit_generate_audio"):
        raise RuntimeError(
            "The loaded model does not support FlowEdit (flowedit_generate_audio)."
        )
    real_src_latents = payload["src_latents"]
    bsz, seq, ch = real_src_latents.shape
    task_type = flow_edit_ctx.get("task_type") or "text2music"
    n_min = float(flow_edit_ctx.get("n_min", 0.0))
    n_max = float(flow_edit_ctx.get("n_max", 1.0))
    n_avg = int(flow_edit_ctx.get("n_avg", 1))
    src_caption = flow_edit_ctx.get("source_caption") or ""
    src_lyrics = flow_edit_ctx.get("source_lyrics") or ""
    if not src_caption and not src_lyrics:
        logger.warning(
            "[flow_edit_overlay] both source_caption and source_lyrics are "
            "empty — V_src ≈ V_tar so the overlay will be a near no-op.",
        )
    logger.info(
        "[flow_edit_overlay] dispatch — task={}, bsz={}, n_min={}, n_max={}, n_avg={}",
        task_type, bsz, n_min, n_max, n_avg,
    )

    src_text_ids, src_text_am, src_lyric_ids, src_lyric_am = tokenize_source(
        handler,
        source_caption=src_caption,
        source_lyrics=src_lyrics,
        vocal_languages=flow_edit_ctx.get("vocal_languages"),
        metas=flow_edit_ctx.get("metas"),
        instructions=flow_edit_ctx.get("instructions"),
        batch_size=bsz,
    )
    src_text_hs, src_lyric_hs = embed_source(handler, src_text_ids, src_lyric_ids)

    device, dtype = real_src_latents.device, real_src_latents.dtype
    src_text_am = src_text_am.to(device=device, dtype=dtype)
    src_lyric_am = src_lyric_am.to(device=device, dtype=dtype)

    # Text-to-music uses silence context; cover tasks use source-audio context.
    if task_type == "text2music":
        sil = handler.silence_latent.to(device=device, dtype=dtype)
        available = sil.shape[1]
        if seq <= available:
            sil_slice = sil[0, :seq, :]
        else:
            repeats = (seq + available - 1) // available
            sil_slice = sil[0].repeat(repeats, 1)[:seq, :]
        ctx_input = sil_slice.unsqueeze(0).expand(bsz, seq, ch).contiguous()
        is_covers_arg = torch.zeros(bsz, dtype=torch.long, device=device)
        # Tokenize the silence context instead of using LM hints.
        precomputed_lm_hints_arg = None
    else:
        ctx_input = real_src_latents
        is_covers_arg = payload["is_covers"]
        precomputed_lm_hints_arg = payload.get("precomputed_lm_hints_25Hz")

    with torch.inference_mode():
        with handler._load_model_context("model"):
            outputs = handler.model.flowedit_generate_audio(
                # Target = the user's caption/lyrics already in payload.
                target_text_hidden_states=payload["text_hidden_states"],
                target_text_attention_mask=payload["text_attention_mask"],
                target_lyric_hidden_states=payload["lyric_hidden_states"],
                target_lyric_attention_mask=payload["lyric_attention_mask"],
                # Source = the freshly encoded original-prompt side.
                text_hidden_states=src_text_hs,
                text_attention_mask=src_text_am,
                lyric_hidden_states=src_lyric_hs,
                lyric_attention_mask=src_lyric_am,
                # Both branches share the same audio context.
                refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=payload["refer_audio_order_mask"],
                src_latents=real_src_latents,        # for zt_src/zt_tar formation
                ctx_src_latents=ctx_input,           # silence (text2music) or real (cover)
                chunk_masks=payload["chunk_mask"],
                is_covers=is_covers_arg,             # 0 (text2music) or payload (cover)
                silence_latent=handler.silence_latent,
                seed=seed_param,
                retake_seed=generate_kwargs.get("retake_seed"),
                infer_steps=generate_kwargs.get("infer_steps"),
                timesteps=generate_kwargs.get("timesteps"),
                diffusion_guidance_scale=generate_kwargs.get("diffusion_guidance_scale", 1.0),
                cfg_interval_start=generate_kwargs.get("cfg_interval_start", 0.0),
                cfg_interval_end=generate_kwargs.get("cfg_interval_end", 1.0),
                shift=generate_kwargs.get("shift", 1.0),
                velocity_norm_threshold=generate_kwargs.get("velocity_norm_threshold", 0.0),
                velocity_ema_factor=generate_kwargs.get("velocity_ema_factor", 0.0),
                edit_n_min=n_min,
                edit_n_max=n_max,
                edit_n_avg=n_avg,
                precomputed_lm_hints_25Hz=precomputed_lm_hints_arg,
                sampler_mode=generate_kwargs.get("sampler_mode", "euler"),
                use_adg=generate_kwargs.get("use_adg", False),
                dcw_enabled=generate_kwargs.get("dcw_enabled", False),
            )

            if not return_intermediate:
                return outputs, None, None, None
            attn = torch.ones(bsz, seq, device=device, dtype=dtype)
            enc_hs, enc_am, ctx = handler.model.prepare_condition(
                text_hidden_states=payload["text_hidden_states"],
                text_attention_mask=payload["text_attention_mask"],
                lyric_hidden_states=payload["lyric_hidden_states"],
                lyric_attention_mask=payload["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=payload["refer_audio_order_mask"],
                hidden_states=ctx_input,
                attention_mask=attn,
                silence_latent=handler.silence_latent,
                src_latents=ctx_input,
                chunk_masks=payload["chunk_mask"],
                is_covers=is_covers_arg,
                precomputed_lm_hints_25Hz=precomputed_lm_hints_arg,
            )
    return outputs, enc_hs, enc_am, ctx
