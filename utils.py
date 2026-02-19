"""
Utility helpers for the ComfyUI Gemini Nano Pro Suite.

Handles:
  - API client initialization with fallback to env‑var
  - Robust Tensor ↔ PIL conversions respecting ComfyUI's [B,H,W,C] format
  - Response image extraction from Gemini SDK parts
"""

import os
import time
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETRYABLE_KEYWORDS = ["rate limit", "timeout", "429", "503", "500", "connection", "timed out"]
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------
def get_gemini_client(api_key: str = "") -> genai.Client:
    """
    Create a `genai.Client`.
    Priority: widget value → GEMINI_API_KEY env‑var.
    """
    key = api_key.strip() if api_key else ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "No API key provided. Either enter it in the node widget "
            "or set the GEMINI_API_KEY environment variable."
        )
    if len(key) < 10:
        raise ValueError("API key appears invalid (too short). Please check your key.")
    return genai.Client(api_key=key)


def retry_on_failure(fn, *args, **kwargs):
    """Invoke *fn* with exponential‑backoff retry for transient errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if any(kw in msg for kw in RETRYABLE_KEYWORDS) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[GeminiNanoPro] Retryable error (attempt {attempt + 1}/{MAX_RETRIES}). "
                      f"Waiting {wait:.1f}s … {exc}")
                time.sleep(wait)
            else:
                raise
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tensor ↔ PIL conversions
# ---------------------------------------------------------------------------
def tensor_batch_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert a ComfyUI image tensor **[B, H, W, C]** (float 0‑1)
    into a list of PIL RGB images.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    images: List[Image.Image] = []
    for i in range(tensor.shape[0]):
        frame = tensor[i]  # [H, W, C]
        np_img = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        # Handle potential alpha channel
        if np_img.shape[-1] == 4:
            np_img = np_img[:, :, :3]
        elif np_img.shape[-1] == 1:
            np_img = np.concatenate([np_img] * 3, axis=-1)
        images.append(Image.fromarray(np_img, "RGB"))
    return images


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a single PIL Image → ComfyUI tensor **[1, H, W, C]** (float 0‑1, RGB).
    """
    img = pil_image.convert("RGB")
    np_arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_arr).unsqueeze(0)  # [1, H, W, C]


def pil_list_to_tensor_batch(images: List[Image.Image]) -> torch.Tensor:
    """
    Stack several PIL images into a single batch tensor **[N, H, W, C]**.
    All images are resized to match the first image's dimensions.
    """
    if not images:
        return torch.zeros((1, 64, 64, 3))

    ref = images[0].convert("RGB")
    ref_w, ref_h = ref.size
    tensors = [pil_to_tensor(ref)]

    for img in images[1:]:
        img_rgb = img.convert("RGB").resize((ref_w, ref_h), Image.LANCZOS)
        tensors.append(pil_to_tensor(img_rgb))

    return torch.cat(tensors, dim=0)


# ---------------------------------------------------------------------------
# Gemini response → PIL extraction
# ---------------------------------------------------------------------------
def extract_images_from_response(response) -> List[Image.Image]:
    """
    Walk the `response.candidates[*].content.parts` and collect any images
    returned as inline‑data blobs.
    """
    images: List[Image.Image] = []
    if not response or not getattr(response, "candidates", None):
        return images

    for candidate in response.candidates:
        if not getattr(candidate, "content", None):
            continue
        for part in candidate.content.parts:
            if getattr(part, "inline_data", None) is not None:
                try:
                    data = (
                        part.inline_data.data
                        if hasattr(part.inline_data, "data")
                        else part.inline_data.get("data", b"")
                    )
                    if data:
                        images.append(Image.open(BytesIO(data)).convert("RGB"))
                except Exception as exc:
                    print(f"[GeminiNanoPro] Failed to decode image part: {exc}")
    return images


def extract_text_from_response(response) -> str:
    """
    Collect all text parts from a Gemini response.
    """
    texts: List[str] = []
    if not response or not getattr(response, "candidates", None):
        return ""

    for candidate in response.candidates:
        if not getattr(candidate, "content", None):
            continue
        for part in candidate.content.parts:
            if getattr(part, "text", None) is not None:
                texts.append(part.text)
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Safety‑settings builder
# ---------------------------------------------------------------------------
HARM_CATEGORIES = [
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]

SAFETY_LEVEL_MAP = {
    "block_none": "BLOCK_NONE",
    "block_few": "BLOCK_ONLY_HIGH",
    "block_some": "BLOCK_MEDIUM_AND_ABOVE",
    "block_most": "BLOCK_LOW_AND_ABOVE",
}


def build_safety_settings(level: str = "block_some") -> list:
    """
    Convert a user‑friendly safety label into a list of SDK SafetySettings
    for all harm categories.

    Levels:
      block_none  → nothing blocked (user takes full responsibility)
      block_few   → only high‑probability harm blocked
      block_some  → medium + high blocked (default)
      block_most  → low + medium + high blocked
    """
    threshold_name = SAFETY_LEVEL_MAP.get(level, "BLOCK_MEDIUM_AND_ABOVE")
    settings = []
    for cat in HARM_CATEGORIES:
        settings.append(
            types.SafetySetting(
                category=cat,
                threshold=threshold_name,
            )
        )
    return settings


# ---------------------------------------------------------------------------
# Prompt Magic
# ---------------------------------------------------------------------------
PROMPT_MAGIC_SUFFIX = (
    " High quality, highly detailed, professional photography, "
    "sharp focus, 8K resolution, vivid colors, masterful composition."
)


def expand_prompt_magic(prompt: str, enabled: bool = True) -> str:
    """Append quality‑boosting tokens to the prompt when Prompt Magic is on."""
    if enabled and prompt.strip():
        return prompt.rstrip() + PROMPT_MAGIC_SUFFIX
    return prompt


def make_blank_image_tensor(width: int = 512, height: int = 512) -> torch.Tensor:
    """Return a solid‑black [1, H, W, 3] tensor (fallback on error)."""
    return torch.zeros((1, height, width, 3))
