
import sys
import torch
import traceback
from typing import Tuple

from .utils import (
    get_gemini_client,
    retry_on_failure,
    pil_to_tensor,
    extract_images_from_response,
    make_blank_image_tensor,
    build_safety_settings,
    expand_prompt_magic,
)

# ---------------------------------------------------------------------------
# Constants (Copied from gemini_nodes.py to ensure standalone functionality)
# ---------------------------------------------------------------------------
IMAGE_GEN_MODELS = [
    "gemini-2.5-flash-image",          # Speed‑optimised (1024 px)
    "gemini-3-pro-image-preview",      # Quality / 4K capable
]

ASPECT_RATIOS_IMAGE = ["1:1", "16:9", "9:16", "4:3", "3:4"]
SAFETY_LEVELS = ["block_none", "block_few", "block_some", "block_most"]
IMAGE_SIZES = ["1K", "2K", "4K"]

CATEGORY = "Goutam_Nano_Suite"

# ---------------------------------------------------------------------------
#  Shared error handler
# ---------------------------------------------------------------------------
def _handle_api_error(exc: Exception, node_label: str):
    """Map common API errors to user‑friendly messages and re‑raise."""
    msg = str(exc)
    print(f"[{node_label}] Error: {msg}", file=sys.stderr)
    print(f"[{node_label}] Traceback:\n{traceback.format_exc()}", file=sys.stderr)

    low = msg.lower()
    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
        raise RuntimeError("❌ Invalid API key. Check your key in Google AI Studio.") from exc
    if "UNAUTHENTICATED" in msg:
        raise RuntimeError("❌ Authentication failed. Verify API key permissions.") from exc
    if "RESOURCE_EXHAUSTED" in msg or "quota" in low:
        raise RuntimeError("❌ Quota exceeded. Check usage limits in Google AI Studio.") from exc
    if "PERMISSION_DENIED" in msg:
        raise RuntimeError("❌ Permission denied. Your key may lack access to this model.") from exc
    if "rate limit" in low or "429" in msg:
        raise RuntimeError("❌ Rate limit exceeded. Wait a moment and retry.") from exc
    if "safety" in low or "blocked" in low or "SAFETY" in msg:
        raise RuntimeError(
            "⚠️ Content blocked by safety filters. "
            "Try setting Safety to 'block_none' or adjusting your prompt."
        ) from exc
    raise RuntimeError(f"Gemini API error: {msg}") from exc


class Goutam_TextToImage:

    """
    Dedicated Text-to-Image generation node.
    Derived from Gemini_Ultimate_ImgGen but strictly for text-based generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe the image you want to generate.",
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Gemini model for image generation.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio.",
                }),
                "safety": (SAFETY_LEVELS, {
                    "default": "block_none",
                    "tooltip": "Content safety filter level. 'block_none' allows all content.",
                }),
                "prompt_magic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto‑enhance prompt with quality boosters (8K, sharp focus, etc.).",
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "1K",
                    "tooltip": "Output resolution. 2K/4K only supported by gemini-3-pro-image-preview.",
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
                    "tooltip": "What to avoid in the generated image.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Dedicated Gemini Text-to-Image generator."

    @classmethod
    def VALIDATE_INPUTS(cls, model_name="", image_size="1K", **kwargs):
        if image_size != "1K" and "3-pro" not in model_name:
            return (f"❌ image_size '{image_size}' requires gemini-3-pro-image-preview. "
                    f"{model_name} only supports 1K.")
        return True

    def generate(
        self,
        prompt: str,
        model_name: str,
        aspect_ratio: str = "1:1",
        safety: str = "block_none",
        prompt_magic: bool = True,
        image_size: str = "1K",
        seed: int = 0,
        api_key: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Build prompt ─────────────────────────────────────────
            final_prompt = expand_prompt_magic(prompt, prompt_magic)
            if negative_prompt.strip():
                final_prompt = f"Avoid the following: {negative_prompt.strip()}. {final_prompt}"

            # ── Build contents ───────────────────────────────────────
            contents = [final_prompt]
            mode_label = "Text2Img"

            print(f"[Gemini Text2Img] Mode: {mode_label} | Model: {model_name} "
                  f"| AR: {aspect_ratio} | Size: {image_size} | Safety: {safety}")

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

            print("[Gemini Text2Img] No image in response — check prompt / safety.")
            return (make_blank_image_tensor(),)

        except Exception as exc:
            _handle_api_error(exc, "Goutam_TextToImage")

            return (make_blank_image_tensor(),)
