import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

# Import the new Google GenAI SDK
from google import genai
from google.genai import types

# Import system prompts
from .system_prompts import VISUAL_ANALYSIS_PROMPT
from .utils import build_safety_settings, retry_on_failure

# ---------------------------------------------------------------------------
# Constants (Copied from gemini_nodes.py to avoid circular imports)
# ---------------------------------------------------------------------------
IMAGE_GEN_MODELS = [
    "gemini-2.5-flash-image",          # Speed-optimised (1024 px)
    "gemini-3-pro-image-preview",      # Quality / 4K capable
]

VISION_MODELS = [
    "gemini-3-pro-preview",            # Best Quality (Reasoning)
    "gemini-1.5-pro",                  # Stable / Production
    "gemini-2.5-flash",                # Fast
]

ASPECT_RATIOS_IMAGE = ["1:1", "16:9", "9:16", "4:3", "3:4"]
SAFETY_LEVELS = ["block_none", "block_few", "block_some", "block_most"]
IMAGE_SIZES = ["1K", "2K", "4K"]

class ZenModeArchVizAllInOne:
    """
    ZenMode ArchViz "All-In-One" Node
    Encapsulates the entire Architectural Visualization workflow:
    1. Visual Analysis (The "Eye") -> Extracts style from inspiration image.
    2. Scene Composition (The "Hand") -> Renders the final scene with furniture.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inspiration_image": ("IMAGE", {
                    "tooltip": "The target style/room image to analyze."
                }),
                "furniture_batch": ("IMAGE", {
                    "tooltip": "Batch of furniture images to place in the scene."
                }),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": "",
                    "tooltip": "gemini API Key (optional if env var is set)."
                }),
                "analysis_model": (VISION_MODELS, {
                    "default": VISION_MODELS[0],
                    "tooltip": "Model used for visual analysis (The Eye). Gemini 1.5 Pro is stable; 3.0 Pro is best."
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Model used for scene generation (The Hand)."
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio."
                }),
                "safety": (SAFETY_LEVELS, {
                    "default": "block_none",
                    "tooltip": "Safety filter strength."
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "1K",
                    "tooltip": "Output resolution (4K requires Pro model)."
                }),
                "creativity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Temperature: Higher = more creative/random, Lower = more strict."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "Random seed for reproducibility."
                }),
            },
            "optional": {
                "scale_reference": ("IMAGE", {
                    "tooltip": "Optional reference (e.g., human) to strictly calibrate furniture scale."
                }), 
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("rendered_image", "debug_log")
    FUNCTION = "process_scene"
    CATEGORY = "Goutam_Nano_Suite"
    DESCRIPTION = "Automated ArchViz pipeline: Analyzes style from inspiration + Places furniture batch (ZenMode)."

    def tensor2pil(self, image: torch.Tensor) -> Image.Image:
        """Safe Tensor to PIL conversion (Batch size 1)."""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def batch_tensor2pil_list(self, image_batch: torch.Tensor) -> List[Image.Image]:
        """Convert a batch tensor [B, H, W, C] to a list of PIL Images."""
        pil_list = []
        for i in range(image_batch.shape[0]):
            img_tensor = image_batch[i] # [H, W, C]
            pil_img = Image.fromarray(np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8))
            pil_list.append(pil_img)
        return pil_list

    def pil2tensor(self, image: Image.Image) -> torch.Tensor:
        """Safe PIL to Tensor conversion [1, H, W, C]."""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def process_scene(self, inspiration_image, furniture_batch, api_key, analysis_model, model_name, aspect_ratio, safety, image_size, creativity, seed, scale_reference=None):
        debug_log = []
        client = None
        
        # Memory cleanup helper
        def cleanup_images(images):
            if isinstance(images, list):
                for img in images:
                    if hasattr(img, 'close'): img.close()
                del images

        try:
            # 0. Setup Client
            if not api_key:
                # Try env var
                api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("API Key is missing.")
            
            client = genai.Client(api_key=api_key)
            
            # 1. Prepare Inputs
            style_ref_pil = self.tensor2pil(inspiration_image)
            furniture_pil_list = self.batch_tensor2pil_list(furniture_batch)
            
            # Helper for Scale Reference
            scale_ref_pil = None
            if scale_reference is not None:
                scale_ref_pil = self.tensor2pil(scale_reference)

            # ------------------------------------------------------------------
            # Phase 1: The Analysis (The "Eye")
            # ------------------------------------------------------------------
            debug_log.append(f"--- Phase 1: Visual Analysis ({analysis_model}) ---")
            
            analysis_response = client.models.generate_content(
                model=analysis_model,
                contents=[VISUAL_ANALYSIS_PROMPT, style_ref_pil],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2 # Low temp for analysis
                )
            )
            
            style_json_str = analysis_response.text
            debug_log.append(f"Raw Analysis:\n{style_json_str}")
            
            # ------------------------------------------------------------------
            # Phase 2: The Composition (The "Hand")
            # ------------------------------------------------------------------
            debug_log.append("\n--- Phase 2: Scene Composition ---")
            
            # Validate JSON
            try:
                style_data = json.loads(style_json_str)
                style_prompt_str = json.dumps(style_data, indent=2)
            except json.JSONDecodeError:
                debug_log.append("Warning: Failed to parse style JSON. Using raw text.")
                style_prompt_str = style_json_str

            # --- Construct Prompt Blocks ---

            # Block 1: Strict Negative Constraint
            strict_constraint = (
                "STRICT CONSTRAINT: You are FORBIDDEN from adding any furniture, objects, decor, plants, or clutter that is not explicitly provided in the 'furniture_batch' images.\n"
                "If the layout seems empty, LEAVE IT EMPTY.\n"
                "Do not generate coffee tables, rugs, lamps, or paintings unless they are in the input batch.\n"
                "Your goal is 'Virtual Staging', not 'Interior Design'. Only place what you are given."
            )

            # Block 2: Incorrect Scale Logic (The "Invisible" Logic)
            if scale_ref_pil is not None:
                scale_logic = (
                    "SCALE REFERENCE INSTRUCTION: I have provided an additional image labeled 'SCALE_REFERENCE'.\n"
                    "Analyze this image to understand the exact size of the furniture relative to the human/object shown.\n"
                    "Apply this size ratio to the furniture in the final render.\n"
                    "CRITICAL: DO NOT RENDER the content of the 'SCALE_REFERENCE' image. It is for measurement data only. "
                    "The final image must contain ONLY the room and the 'furniture_batch' items."
                )
                debug_log.append("Scale Logic: Reference Image Provided")
            else:
                scale_logic = (
                    "SCALING FALLBACK: Use standard architectural elements in the 'inspiration_image' "
                    "(Door Height = 2.0m, Ceiling = 2.4-3.0m) to mathematically deduce the correct size of the furniture. "
                    "Do not hallucinate arbitrary sizes."
                )
                debug_log.append("Scale Logic: Standard Fallback")

            # Master Prompt Assembly
            dynamic_prompt = (
                f"You are an expert 3D Artist. Generate a photorealistic architectural visualization.\n\n"
                f"Style & Atmosphere: Strictly follow this analysis:\n{style_prompt_str}\n\n"
                f"{strict_constraint}\n\n"
                f"{scale_logic}\n\n"
                f"Task: Seamlessly integrate the provided furniture assets into this exact environment.\n"
                f"Placement: Respect the spatial depth and perspective of the style reference.\n"
                f"Lighting: Match the lighting described in the style analysis."
            )
            
            # --- API Payload Structure ---
            # Order: [Prompt, Inspiration, (Scale Ref), Furniture Label, Furniture Images]
            
            phase2_contents = [dynamic_prompt, style_ref_pil]

            if scale_ref_pil is not None:
                phase2_contents.append("SCALE_REFERENCE_IMAGE:")
                phase2_contents.append(scale_ref_pil)
            
            phase2_contents.append("FURNITURE_ASSETS:")
            phase2_contents.extend(furniture_pil_list)
            
            # Build Config
            image_config_kwargs = {"aspect_ratio": aspect_ratio}
            if "3-pro" in model_name:
                image_config_kwargs["image_size"] = image_size

            full_config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                temperature=creativity,
                safety_settings=build_safety_settings(safety),
                image_config=types.ImageConfig(**image_config_kwargs)
            )
            if seed > 0:
                full_config.seed = seed

            debug_log.append(f"Model: {model_name} | AR: {aspect_ratio} | Size: {image_size} | Safety: {safety}")

            # Call API with retry
            generation_response = retry_on_failure(
                client.models.generate_content,
                model=model_name,
                contents=phase2_contents,
                config=full_config
            )

            # Output Conversion
            generated_image_pil = None
            if generation_response.candidates and generation_response.candidates[0].content.parts:
                for part in generation_response.candidates[0].content.parts:
                    if part.inline_data:
                         # Decode image
                        import io
                        if hasattr(part.inline_data, 'data'):
                             img_data = part.inline_data.data
                        else:
                             img_data = part.inline_data.get('data', b"")
                        
                        generated_image_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                        break
            
            if generated_image_pil:
                result_tensor = self.pil2tensor(generated_image_pil)
                debug_log.append("\nSuccess: Image generated.")
                
                # Cleanup
                images_to_clean = [generated_image_pil, style_ref_pil] + furniture_pil_list
                if scale_ref_pil:
                    images_to_clean.append(scale_ref_pil)
                cleanup_images(images_to_clean)

                return (result_tensor, "\n".join(debug_log))
            
            else:
                 raise RuntimeError("No image returned from API.")

        except Exception as e:
            error_msg = f"Error in ZenModeArchViz: {str(e)}"
            print(error_msg, file=sys.stderr)
            debug_log.append(f"\nERROR: {str(e)}")
            
            # Return blank black image
            blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank, "\n".join(debug_log))
