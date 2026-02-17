"""
Gemini Interior Architect
─────────────────────────
Professional interior photography node for high‑end furniture marketing.
Provides granular control over camera settings, lighting, and room layout
to produce magazine‑quality photorealistic renders.

Three control modules:
  A. Photography  — focal length, camera angle, framing
  B. Lighting     — style, colour temperature, practical lamps
  C. Architect    — room type, interior style, layout instructions
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
)
from .utils import (
    build_safety_settings,
    extract_images_from_response,
    get_gemini_client,
    make_blank_image_tensor,
    pil_to_tensor,
    retry_on_failure,
)

# ── Module A: Photography presets ─────────────────────────────────────────
FOCAL_LENGTHS = [
    "35mm (Wide)",
    "50mm (Standard)",
    "85mm (Portrait/Detail)",
    "24mm (Architectural Wide)",
]

CAMERA_ANGLES = [
    "Eye-Level",
    "Low Angle (Hero)",
    "High Angle (Overview)",
    "Top-Down (Plan View)",
]

FRAMING_OPTIONS = [
    "Wide Shot (Full Room)",
    "Medium Shot (Furniture Cluster)",
    "Close-Up (Detail)",
]

# ── Module B: Lighting presets ────────────────────────────────────────────
LIGHTING_STYLES = [
    "Natural Daylight (Soft)",
    "Golden Hour (Warm)",
    "Studio Flash (Crisp)",
    "Moody/Dark",
]

KELVIN_TEMPS = [
    "3000K (Warm)",
    "4000K (Neutral)",
    "6000K (Cool Daylight)",
]

PRACTICAL_LIGHTS = [
    "Turned OFF (Natural only)",
    "Turned ON (Cozy)",
]

# ── Module C: Architect presets ───────────────────────────────────────────
ROOM_TYPES = [
    "Living Room",
    "Bedroom",
    "Dining",
    "Office",
]

INTERIOR_STYLES = [
    "American Contemporary",
    "Modern Minimalist",
    "Traditional",
    "Industrial",
]


# ═══════════════════════════════════════════════════════════════════════════
# Gemini Interior Architect
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Interior_Architect:
    """
    Professional interior photography node for furniture marketing.

    Module A — Photography: focal length, camera angle, framing
    Module B — Lighting: style, colour temperature, practical lamps
    Module C — Architect: room type, interior style, layout instructions

    Sends a structured "Photographer Mode" system prompt + product images
    to Gemini for photorealistic scene generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Furniture input ───────────────────────────────────
                "furniture_batch": ("IMAGE", {
                    "tooltip": (
                        "Batch of product images [N,H,W,C]. "
                        "Use 'Batch Images' node to combine multiple products."
                    ),
                }),

                # ── Module A: Photography ─────────────────────────────
                "focal_length": (FOCAL_LENGTHS, {
                    "default": FOCAL_LENGTHS[0],
                    "tooltip": "Camera lens focal length.",
                }),
                "camera_angle": (CAMERA_ANGLES, {
                    "default": CAMERA_ANGLES[0],
                    "tooltip": "Camera shooting angle.",
                }),
                "framing": (FRAMING_OPTIONS, {
                    "default": FRAMING_OPTIONS[0],
                    "tooltip": "Shot framing / composition distance.",
                }),

                # ── Module B: Lighting ────────────────────────────────
                "lighting_style": (LIGHTING_STYLES, {
                    "default": LIGHTING_STYLES[0],
                    "tooltip": "Overall lighting mood.",
                }),
                "kelvin_temp": (KELVIN_TEMPS, {
                    "default": KELVIN_TEMPS[1],
                    "tooltip": "Colour temperature of the light source.",
                }),
                "practical_lights": (PRACTICAL_LIGHTS, {
                    "default": PRACTICAL_LIGHTS[0],
                    "tooltip": "Whether visible lamps / fixtures are on or off.",
                }),

                # ── Module C: Architect ───────────────────────────────
                "room_type": (ROOM_TYPES, {
                    "default": ROOM_TYPES[0],
                    "tooltip": "Type of room for the scene.",
                }),
                "interior_style": (INTERIOR_STYLES, {
                    "default": INTERIOR_STYLES[0],
                    "tooltip": "Architectural / interior design style.",
                }),
                "layout_instructions": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Arrange furniture naturally in the room."
                    ),
                    "tooltip": (
                        "Paste specific placement rules here "
                        "(e.g., 'Sofa in center, rug underneath, mirror above')."
                    ),
                }),

                # ── Generation settings ───────────────────────────────
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Gemini model for scene generation.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "16:9",
                    "tooltip": "Output aspect ratio. 16:9 for landscape rooms.",
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
                    "tooltip": "Seed for reproducibility (0 = random).",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (falls back to GEMINI_API_KEY env var).",
                    "password": True,
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the generated scene.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scene",)
    FUNCTION = "compose"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Professional interior photography node. Control camera lens, "
        "angle, framing, lighting style, colour temperature, and room "
        "layout to produce magazine-quality furniture scenes."
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

    def compose(
        self,
        furniture_batch: torch.Tensor,
        # Module A
        focal_length: str,
        camera_angle: str,
        framing: str,
        # Module B
        lighting_style: str,
        kelvin_temp: str,
        practical_lights: str,
        # Module C
        room_type: str,
        interior_style: str,
        layout_instructions: str,
        # Generation
        model_name: str,
        aspect_ratio: str = "16:9",
        safety: str = "block_none",
        image_size: str = "1K",
        seed: int = 0,
        api_key: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Split batch tensor into individual PIL images ─────────
            batch_size = furniture_batch.shape[0]
            pil_image_list = []
            for i in range(batch_size):
                img_np = (
                    furniture_batch[i].cpu().numpy() * 255.0
                ).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode="RGB")
                pil_image_list.append(pil_img)

            print(f"[Interior Architect] {batch_size} product image(s)")

            # ── Construct structured photographer prompt ──────────────
            layout = layout_instructions.strip() or "Arrange furniture naturally."
            master_prompt = (
                f"You are a professional interior photographer and set designer. "
                f"Create a high-end, photorealistic interior photograph.\n\n"
                f"SCENE: Create an {interior_style} {room_type}.\n\n"
                f"CAMERA: Shot with a {focal_length} lens at {camera_angle}. "
                f"Framing is {framing}.\n\n"
                f"LIGHTING: {lighting_style} lighting, approx {kelvin_temp}. "
                f"Practical lamps are {practical_lights}.\n\n"
                f"LAYOUT & COMPOSITION:\n{layout}\n\n"
                f"REFERENCE: Use the provided {batch_size} images as the EXACT "
                f"furniture pieces. Preserve their exact materials, colours, "
                f"textures, and proportions. Do NOT alter or substitute any item.\n\n"
                f"QUALITY: The final image must look like it belongs in "
                f"Architectural Digest or Elle Décor. Ensure realistic shadows, "
                f"reflections, depth of field consistent with the lens choice, "
                f"and natural colour grading."
            )
            if negative_prompt.strip():
                master_prompt = (
                    f"Avoid the following: {negative_prompt.strip()}. "
                    f"{master_prompt}"
                )

            # ── Build contents: [prompt, image1, image2, ...] ────────
            contents = [master_prompt, *pil_image_list]

            print(f"[Interior Architect] {interior_style} {room_type} | "
                  f"Camera: {focal_length}, {camera_angle}, {framing}")
            print(f"[Interior Architect] Light: {lighting_style}, {kelvin_temp}, "
                  f"Practicals {practical_lights}")
            print(f"[Interior Architect] Model: {model_name} | "
                  f"AR: {aspect_ratio} | Size: {image_size}")

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
                "[Interior Architect] No image in response — "
                "check prompt / safety."
            )
            return (make_blank_image_tensor(),)

        except Exception as exc:
            from .gemini_nodes import _handle_api_error
            _handle_api_error(exc, "Gemini_Interior_Architect")
            return (make_blank_image_tensor(),)
