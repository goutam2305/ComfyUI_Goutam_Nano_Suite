"""
Gemini Object Manipulator
─────────────────────────
Advanced object manipulation node using Gemini's multimodal capabilities.
Supports removing or moving objects via text instructions or precise masks.

Modes:
  • Remove Object (Mask): Cleanly remove masked object + reconstruct background.
  • Remove Object (Text): "Remove the red car" based on text description.
  • Move Object (Text): "Move the vase to the left" (mask ignored).
"""

from typing import Tuple, Optional

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

MANIPULATION_MODES = [
    "Remove Object",
    "Move Object",
]

# --- PRO SYSTEM TEMPLATES ---

REMOVE_TEMPLATE = """
You are an expert professional photo retoucher and digital artist.
Your goal is to REMOVE the object specified by the user (and defined by the mask if provided) while making the result look 100% authentic and unedited.

**EXECUTION GUIDELINES:**
1.  **Surgical Removal:** completely erase the specified object.
2.  **Background Reconstruction (Inpainting):** Fill the void by analyzing the surrounding textures, patterns, and depth. Continue lines and structures (e.g., table edges, floorboards, wall trim) seamlessly through the removed area.
3.  **Lighting & Shadows:**
    * Analyze the global scene lighting (direction, color temperature, hardness).
    * Remove the object's cast shadow on the floor/surface.
    * Ensure the new inpainted surface reacts to the light exactly like the surrounding area.
4.  **Final Polish:** Match the image grain, noise, and compression artifacts so the edit is undetectable.

**User Instruction:** {user_instruction}
"""

MOVE_TEMPLATE = """
You are an expert professional photo retoucher.
Your goal is to MOVE the object specified by the user to a new location within the scene.

**EXECUTION GUIDELINES:**
1.  **Selection:** Identify the object clearly. Lift it from its current position.
2.  **Inpainting (The Old Spot):** Fill the hole left behind by reconstructing the background texture, geometry, and lighting. Remove the old cast shadows.
3.  **Placement (The New Spot):**
    * Place the object in the new coordinates described.
    * **Perspective Match:** Adjust the object's perspective and scale to fit the depth of the new location.
    * **Lighting Integration:** Relight the object to match the light intensity and color at the new spot.
    * **Shadow Casting:** Generate a realistic cast shadow from the object onto the new surface, matching the direction and softness of other shadows in the room.
4.  **Compositing:** Blend the edges for a natural look.

**User Instruction:** {user_instruction}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Gemini Object Manipulator
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Object_Manipulator:
    """
    Remove or move objects in an image using Gemini.
    Optional mask input allows for precise "Remove Object" operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to edit.",
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Remove the object.",
                    "tooltip": (
                        "Description of the action. E.g., 'Remove the red car' "
                        "or 'Move the vase to the left'."
                    ),
                }),
                "mode": (MANIPULATION_MODES, {
                    "default": "Remove Object",
                    "tooltip": (
                        "Operation mode. 'Remove Object' supports masks; "
                        "'Move Object' relies on text instructions."
                    ),
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Gemini model for image editing.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "1:1",
                    "tooltip": "Target aspect ratio (usually match input).",
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
                "mask_input": ("MASK", {
                    "tooltip": (
                        "Optional mask for 'Remove Object' mode. "
                        "Connect to guide precise removal."
                    ),
                }),
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
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "manipulate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Advanced image editor. Remove objects (text or mask guided) or "
        "move objects (text guided). Uses Gemini's multimodal understanding."
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

    def manipulate(
        self,
        image: torch.Tensor,
        instruction: str,
        mode: str,
        model_name: str,
        aspect_ratio: str = "1:1",
        safety: str = "block_none",
        image_size: str = "1K",
        seed: int = 0,
        mask_input: Optional[torch.Tensor] = None,
        api_key: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Convert inputs to PIL ────────────────────────────────
            pil_image = tensor_batch_to_pil_list(image)[0].convert("RGB")
            
            # handle mask if present
            mask_pil = None
            if mask_input is not None:
                # Mask tensor is [B, H, W], take first item, convert to uint8 0-255
                mask_np = (mask_input[0].cpu().numpy() * 255.0).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np, mode='L')
                
                # Resize mask to match image if needed
                if mask_pil.size != pil_image.size:
                    mask_pil = mask_pil.resize(pil_image.size, Image.NEAREST)

            print(f"[Object Manipulator] Mode: {mode}")

            # ── Construct Prompt & Payload ───────────────────────────
            contents = []
            
            # Select the correct template based on the mode
            if mode == "Remove Object":
                system_prompt = REMOVE_TEMPLATE.format(user_instruction=instruction)
            elif mode == "Move Object":
                system_prompt = MOVE_TEMPLATE.format(user_instruction=instruction)
            else:
                system_prompt = instruction # Fallback

            # Scenario 1: Remove Object WITH Mask
            if mode == "Remove Object" and mask_pil is not None:
                # Payload: [prompt, image, mask]
                contents = [system_prompt, pil_image, mask_pil]
                print("[Object Manipulator] Using Mask-Guided Removal.")

            # Scenario 2: Remove Object WITHOUT Mask (Text Only)
            elif mode == "Remove Object":
                contents = [system_prompt, pil_image]
                print("[Object Manipulator] Using Text-Guided Removal.")

            # Scenario 3: Move Object
            else:  # "Move Object"
                if mask_pil is not None:
                    # Instruct to move highlighted object
                    move_prompt = (
                        f"{system_prompt}\n"
                        "Note: Use the provided mask to identify the object to move."
                    )
                    contents = [move_prompt, pil_image, mask_pil]
                    print("[Object Manipulator] Using Mask-Guided Move.")
                else:
                    contents = [system_prompt, pil_image]
                    print("[Object Manipulator] Using Text-Guided Move.")

            # Add negative prompt if present
            if negative_prompt.strip():
                # For first item which is text
                contents[0] = (
                    f"Avoid the following: {negative_prompt.strip()}. "
                    f"{contents[0]}"
                )

            print(f"[Object Manipulator] Model: {model_name} | "
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
                "[Object Manipulator] No image in response — "
                "check prompt / safety."
            )
            return (make_blank_image_tensor(),)

        except Exception as exc:
            _handle_api_error(exc, "Gemini_Object_Manipulator")
            return (make_blank_image_tensor(),)
