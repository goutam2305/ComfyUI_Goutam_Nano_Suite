"""
Gemini Detail Zoomer (Re-Photography)
─────────────────────────────────────
Generates high-resolution close-ups from low-res crops using high-res references.
Acts as a virtual macro photographer for product details.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .gemini_nodes import (
    ASPECT_RATIOS_IMAGE,
    CATEGORY,
    IMAGE_GEN_MODELS,
    IMAGE_SIZES,
    SAFETY_LEVELS,
    _handle_api_error,
)
from .utils import (
    build_safety_settings,
    extract_images_from_response,
    get_gemini_client,
    make_blank_image_tensor,
    pil_to_tensor,
    retry_on_failure,
    tensor_batch_to_pil_list,
)

DEPTH_OF_FIELD_OPTIONS = [
    "f/1.8 (Creamy Bokeh)",
    "f/2.8 (Soft Background)",
    "f/8 (Sharp Everywhere)",
]

LIGHTING_ENHANCEMENT_OPTIONS = [
    "Preserve Original Lighting",
    "Add Rim Light",
    "Brighten Shadows",
    "Studio Softbox",
]


# ═══════════════════════════════════════════════════════════════════════════
# Gemini Detail Zoomer
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Detail_Zoomer:
    """
    Re-photographs a scene crop at high resolution using a product reference
    for texture accuracy. Simulates a macro lens.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_crop": ("IMAGE", {
                    "tooltip": "Low-res crop from the wide shot (sets composition).",
                }),
                "product_reference": ("IMAGE", {
                    "tooltip": "High-res original product photo (sets texture/material).",
                }),
                "detail_focus": ("STRING", {
                    "multiline": True,
                    "default": "Focus on the wood grain texture and fabric weave.",
                    "tooltip": "What specific details to emphasize.",
                }),
                "depth_of_field": (DEPTH_OF_FIELD_OPTIONS, {
                    "default": DEPTH_OF_FIELD_OPTIONS[0],
                    "tooltip": "Simulated aperture setting.",
                }),
                "lighting_enhancement": (LIGHTING_ENHANCEMENT_OPTIONS, {
                    "default": LIGHTING_ENHANCEMENT_OPTIONS[0],
                    "tooltip": "How to treat the lighting in the close-up.",
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Gemini model for generation.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "1:1",
                    "tooltip": "Output aspect ratio.",
                }),
                "safety": (SAFETY_LEVELS, {
                    "default": "block_none",
                    "tooltip": "Content safety filter level.",
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "1K",
                    "tooltip": (
                        "Output resolution. 2K/4K only supported by "
                        "gemini-3-pro-image-preview."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "control_after_generate": "randomize",
                    "tooltip": "Seed for reproducibility.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (falls back to env var).",
                    "password": True,
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the result.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("zoomed_image",)
    FUNCTION = "zoom_in"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Re-photographs a low-res crop at high resolution using a product "
        "reference for texture accuracy. Simulates a macro lens."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, model_name="", image_size="1K", **kwargs):
        if image_size != "1K" and "3-pro" not in model_name:
            return (
                f"❌ image_size '{image_size}' requires "
                f"gemini-3-pro-image-preview. "
                f"{model_name} only supports 1K."
            )
        return True

    def zoom_in(
        self,
        scene_crop: torch.Tensor,
        product_reference: torch.Tensor,
        detail_focus: str,
        depth_of_field: str,
        lighting_enhancement: str,
        model_name: str,
        aspect_ratio: str = "1:1",
        safety: str = "block_none",
        image_size: str = "1K",
        seed: int = 0,
        api_key: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Convert inputs to PIL ────────────────────────────────
            # Take first image from batches
            crop_pil = tensor_batch_to_pil_list(scene_crop)[0].convert("RGB")
            ref_pil = tensor_batch_to_pil_list(product_reference)[0].convert("RGB")
            
            print(f"[Detail Zoomer] Re-photographing crop with {model_name}...")

            # ── Construct System Prompt ──────────────────────────────
            system_prompt = (
                "You are an expert interior detail photographer.\n"
                "Task: Re-shoot the provided 'Scene Crop' image at 8K resolution.\n"
                "Reference: Use the 'Product Reference' image to apply the exact "
                "textures, materials, and finish to the objects in the scene. "
                "Do NOT hallucinate new designs; match the reference perfectly.\n"
                f"Camera Settings: Use a 100mm Macro lens with {depth_of_field}.\n"
                f"Lighting: {lighting_enhancement}.\n"
                f"Focus: {detail_focus}.\n"
                "Goal: Ultra-realistic material definition (wood pores, "
                "fabric weave, metal sheen). Eliminate all jpeg artifacts or "
                "blur from the scene crop."
            )

            # Add negative prompt if present
            if negative_prompt.strip():
                system_prompt += (
                    f"\n\nAvoid the following: {negative_prompt.strip()}."
                )

            # Payload: [prompt, scene_crop, product_reference]
            contents = [system_prompt, crop_pil, ref_pil]

            # ── Build config (validated by VALIDATE_INPUTS) ──────────
            image_config_kwargs = {"aspect_ratio": aspect_ratio}
            if "3-pro" in model_name:
                image_config_kwargs["image_size"] = image_size

            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=build_safety_settings(safety),
                image_config=types.ImageConfig(**image_config_kwargs),
            )
            if seed > 0:
                config.seed = seed

            # ── Call API ─────────────────────────────────────────────
            response = retry_on_failure(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=config,
            )

            result_images = extract_images_from_response(response)
            if result_images:
                return (pil_to_tensor(result_images[0]),)

            print(
                "[Detail Zoomer] No image in response — check prompt / safety."
            )
            return (make_blank_image_tensor(),)

        except Exception as exc:
            _handle_api_error(exc, "Gemini_Detail_Zoomer")
            return (make_blank_image_tensor(),)
