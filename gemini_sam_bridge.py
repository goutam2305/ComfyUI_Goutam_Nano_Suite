"""
Goutam SAM Bridge
─────────────────
Vision-Analysis node designed to automate the prompting for GroundingDINO and SAM.
Converts a user instruction (e.g., "All Furniture") into a specific list of objects
detected in the image, strictly formatted for GroundingDINO.
"""

from typing import Tuple

import torch

from .gemini_nodes import (
    CATEGORY,
    _handle_api_error,
)
from .utils import (
    get_gemini_client,
    retry_on_failure,
    tensor_batch_to_pil_list,
)


# ── Bridge Options ──────────────────────────────────────────────────────────
SEGMENTATION_TARGETS = [
    "All Furniture",
    "Decor Only",
    "Structural Elements (Walls/Floor)",
    "Lighting",
    "Soft Goods (Rugs/Curtains)",
]

# ── Available Vision Models ────────────────────────────────────────────────
VISION_MODELS = [
    "gemini-2.5-flash", 
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


# ═══════════════════════════════════════════════════════════════════════════
# Goutam SAM Bridge (Vision -> DINO Prompt)
# ═══════════════════════════════════════════════════════════════════════════
class Goutam_SAM_Bridge:
    """
    Vision-Analysis node.
    Analyzer role: "Director".
    Task: Identify objects matching the target category and return a dot-separated string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The room render or photo to analyze.",
                }),
                "segmentation_target": (SEGMENTATION_TARGETS, {
                    "default": SEGMENTATION_TARGETS[0],
                    "tooltip": "Category of objects to identify.",
                }),
                "model_name": (VISION_MODELS, {
                    "default": "gemini-2.5-flash",
                    "tooltip": "Vision model for analysis.",
                }),
            },
            "optional": {
                "custom_target": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override target (e.g., 'red pillows only').",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "Gemini API Key.",
                }),
                "prompt_hint": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Extra instructions for the director/analyzer.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dino_prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Analyzes an image to auto-generate a GroundingDINO prompt. "
        "Identifies specific objects (e.g., 'sofa . chair . table .') "
        "based on the selected target category."
    )

    def generate_prompt(
        self,
        image: torch.Tensor,
        segmentation_target: str,
        model_name: str,
        custom_target: str = "",
        api_key: str = "",
        prompt_hint: str = "",
    ) -> Tuple[str]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Convert input to PIL ─────────────────────────────────
            # Take first image from batch
            pil_img = tensor_batch_to_pil_list(image)[0].convert("RGB")
            
            # Construct target description
            target_description = custom_target.strip() if custom_target.strip() else segmentation_target

            print(f"[SAM Bridge] Analyzing image with {model_name} for '{target_description}'...")

            # ── Construct System Prompt (Director Role) ──────────────
            system_prompt = (
                "You are an object detection assistant for an architectural "
                "segmentation pipeline.\n"
                "Input: Analyze the provided interior image.\n"
                f"Goal: Identify all objects that belong to the category: {target_description}.\n"
                "Output Format: Return a clean, dot-separated string of object "
                "class names suitable for GroundingDINO detection.\n\n"
                "Rules:\n"
                "1. Be specific but concise (e.g., use 'armchair' instead of 'chair', "
                "'floor lamp' instead of 'lamp').\n"
                "2. Do NOT describe attributes like color unless necessary "
                "(use 'sofa' not 'brown leather sofa' unless asked for specific color).\n"
                "3. Output ONLY the string. No intro text, no markdown, no quotes.\n"
                "4. Example Output: 'sofa . armchair . coffee table . rug'\n"
            )

            if prompt_hint.strip():
                system_prompt += f"\nAdditional Instructions: {prompt_hint.strip()}"

            # Payload
            contents = [system_prompt, pil_img]

            # ── Call API ─────────────────────────────────────────────
            # We want text output configuration
            config = types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0.1, # Low temp for deterministic, factual output
            )

            # Call generate_content
            response = retry_on_failure(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=config,
            )

            # ── Parse Response ───────────────────────────────────────
            if response.text:
                result_text = response.text.strip()
                # Clean up any potential markdown or quotes
                result_text = result_text.replace("```", "").replace('"', "").replace("'", "")
                
                # GroundingDINO prompts typically separate by . 
                # Ensure it ends with a dot if not present, though splitting usually handles it.
                if not result_text.endswith("."):
                    result_text += " ."
                
                print(f"[SAM Bridge] Generated Prompt: {result_text}")
                return (result_text,)

            print("[SAM Bridge] No text in response.")
            return ("",)

        except Exception as exc:
            _handle_api_error(exc, "Goutam_SAM_Bridge")
            # Return empty string on error so workflow doesn't crash completely if configured to continue
            return ("",)
