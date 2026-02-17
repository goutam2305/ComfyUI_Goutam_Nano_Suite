"""
Gemini Multi-Furniture Composer
───────────────────────────────
Takes a batch of product / furniture images and arranges them into
a cohesive, photorealistic room scene using Gemini's multimodal
generation capabilities.

Each image in the batch is split into a separate PIL Image so the
model treats them as distinct objects in the scene.
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

# ── Preset options ────────────────────────────────────────────────────────
ROOM_TYPES = [
    "Living Room",
    "Dining Room",
    "Bedroom",
    "Home Office",
]

INTERIOR_STYLES = [
    "Modern Minimalist",
    "Japandi",
    "Industrial",
    "Mid-Century",
    "Luxury Contemporary",
    "Bohemian",
]

LIGHTING_OPTIONS = [
    "Natural Daylight",
    "Golden Hour (Warm)",
    "Cozy Evening (Interior Lights)",
    "Professional Studio",
]


# ═══════════════════════════════════════════════════════════════════════════
# Gemini Multi-Furniture Composer
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Multi_Furniture_Composer:
    """
    Takes a batch of furniture / product images and composes them into
    a cohesive, photorealistic room scene.

    • furniture_batch  = IMAGE batch [N, H, W, C] — each item is a product
    • Presets control the room type, interior style, and lighting
    • The model arranges all items naturally into the specified scene
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "furniture_batch": ("IMAGE", {
                    "tooltip": (
                        "Batch of product images [N,H,W,C]. "
                        "Use 'Batch Images' or similar node to combine multiple images."
                    ),
                }),
                "room_type": (ROOM_TYPES, {
                    "default": ROOM_TYPES[0],
                    "tooltip": "Type of room to compose the scene in.",
                }),
                "interior_style": (INTERIOR_STYLES, {
                    "default": INTERIOR_STYLES[0],
                    "tooltip": "Interior design style for the room.",
                }),
                "lighting": (LIGHTING_OPTIONS, {
                    "default": LIGHTING_OPTIONS[0],
                    "tooltip": "Lighting mood for the scene.",
                }),
                "composition_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Arrange these items naturally into a photorealistic scene."
                    ),
                    "tooltip": "Custom instructions for how to compose the scene.",
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Gemini model for scene generation.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "16:9",
                    "tooltip": "Output aspect ratio. 16:9 recommended for wide room shots.",
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
        "Compose multiple furniture / product images into a cohesive, "
        "photorealistic room scene. Feed a batch of product images and "
        "select room type, style, and lighting presets."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, model_name="", image_size="1K", **kwargs):
        if image_size != "1K" and "3-pro" not in model_name:
            return (f"❌ image_size '{image_size}' requires gemini-3-pro-image-preview. "
                    f"{model_name} only supports 1K.")
        return True

    def compose(
        self,
        furniture_batch: torch.Tensor,
        room_type: str,
        interior_style: str,
        lighting: str,
        composition_prompt: str,
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

            print(f"[Furniture Composer] Received {batch_size} product image(s)")

            # ── Construct master prompt ───────────────────────────────
            user_instructions = composition_prompt.strip()
            master_prompt = (
                f"You are an expert interior designer and 3D scene composer. "
                f"Create a high-resolution, photorealistic wide-angle photograph "
                f"of a {interior_style} {room_type}. "
                f"\n\nThe scene MUST prominently feature ALL {batch_size} "
                f"provided furniture/product items arranged naturally together "
                f"in the room. Each provided image is a separate piece of "
                f"furniture or decor — use the EXACT appearance, materials, "
                f"colors, and textures from each product image. "
                f"\n\nLighting: {lighting}. "
                f"\n\nComposition: {user_instructions} "
                f"\n\nRequirements:"
                f"\n- Place each item realistically on the floor, against walls, "
                f"or on surfaces as appropriate"
                f"\n- Maintain correct scale relationships between items"
                f"\n- Add complementary room elements (walls, flooring, windows, "
                f"plants, rugs) that match the {interior_style} aesthetic"
                f"\n- Ensure natural shadows and reflections"
                f"\n- The final image should look like a professional interior "
                f"design magazine photograph"
            )
            if negative_prompt.strip():
                master_prompt = (
                    f"Avoid the following: {negative_prompt.strip()}. "
                    f"{master_prompt}"
                )

            # ── Build contents: [prompt, image1, image2, ...] ────────
            contents = [master_prompt, *pil_image_list]

            print(f"[Furniture Composer] Style: {interior_style} | "
                  f"Room: {room_type} | Lighting: {lighting}")
            print(f"[Furniture Composer] Model: {model_name} | "
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
                "[Furniture Composer] No image in response — "
                "check prompt / safety."
            )
            return (make_blank_image_tensor(),)

        except Exception as exc:
            from .gemini_nodes import _handle_api_error
            _handle_api_error(exc, "Gemini_Multi_Furniture_Composer")
            return (make_blank_image_tensor(),)
