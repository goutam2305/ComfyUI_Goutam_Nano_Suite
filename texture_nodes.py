"""
Goutam Texture Nodes
────────────────────
High-end procedural texture generation using Gemini and Imagen models.
"""

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps

from .utils import (
    get_gemini_client,
    tensor_batch_to_pil_list,
    pil_to_tensor,
    make_blank_image_tensor,
    retry_on_failure,
)

# Core Constants
TEXTURE_MODELS = [
    "Imagen 3 (Standard)",
    "Imagen 3 Fast",
    "Imagen 3.0 v2 (Latest)",
    "Gemini 2.5 Flash (Fastest)", 
    "Gemini 3 Pro (Preview)",
]

TEXTURE_TYPES = [
    "Auto-Detect",
    "Concrete",
    "Wood", 
    "Fabric",
    "Metal", 
    "Stone", 
    "Plaster",
    "Leather"
]

RESOLUTIONS = [
    "1K (1024x1024)",
    "2K (2048x2048)",
    "4K (4096x4096)"
]

MODEL_MAPPING = {
    "Imagen 3 (Standard)": "imagen-3.0-generate-001",
    "Imagen 3 Fast": "imagen-3.0-fast-generate-001",
    "Imagen 3.0 v2 (Latest)": "imagen-3.0-generate-002",
    "Gemini 2.5 Flash (Fastest)": "gemini-2.5-flash-image", # Updated to -image
    "Gemini 3 Pro (Preview)": "gemini-3-pro-image-preview",
}

class GoutamSeamlessTexturePro:
    """
    Goutam Texture Synthesizer (Pro)
    Standalone, high-end node for generating seamless PBR-ready textures.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Source material photo to analyze."
                }),
                "model_version": (TEXTURE_MODELS, {
                    "default": TEXTURE_MODELS[0],
                    "tooltip": "Model backend for generation."
                }),
                "resolution": (RESOLUTIONS, {
                    "default": RESOLUTIONS[0],
                    "tooltip": "Output resolution. 4K may use upscaling if model limits are reached."
                }),
                "texture_type": (TEXTURE_TYPES, {
                    "default": "Auto-Detect",
                    "tooltip": "Material category hint."
                }),
                "tileability_strictness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Enforce seamless edges strength."
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Your Google Gemini API Key.",
                    "password": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("diffuse_map", "glossiness_rough_guess")
    FUNCTION = "process_texture"
    CATEGORY = "Goutam_Nano_Suite/Texture"
    DESCRIPTION = "Professional PBR Texture Synthesizer. Analyzes reference image and generates high-fidelity, seamless textures."

    def process_texture(
        self, 
        reference_image, 
        model_version, 
        resolution, 
        texture_type, 
        tileability_strictness, 
        api_key
    ):
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Step A: Resolution Mapping ─────────────────
            res_str = resolution.split(" ")[0]
            target_dim = 1024
            if res_str == "2K":
                target_dim = 2048
            elif res_str == "4K":
                target_dim = 4096
            
            # API Generation Limit Handling
            # Most models currently cap at 1024x1024 or 2048x2048 directly.
            # We'll request the highest native res (usually 1024 or 2048) and upscale if needed.
            native_dim = min(target_dim, 2048) # Assuming 2048 is safe max for some, 1024 for others.
            # However, Imagen 3 typically does 1024x1024. 
            # safe_gen_dim for standard API is often 1024.
            # gemini-3-pro-image-preview might support higher.
            # Let's try to request native_dim, and handle errors or just default to 1024 for generation 
            # then upscale.
            
            # For simplicity and robustness: Generate 1024x1024, then upscale.
            # BUT user asked to "generate at the maximum supported resolution".
            # We will try 2048 if target is > 1024, depending on model capabilities/heuristics.
            # Since we can't easily probe capabilities, we'll try standard logic:
            # Request 1024x1024 for generation (safest) then upscale.
            # OR logic:
            gen_width = 1024
            gen_height = 1024
            
            # ── Step B: Model Routing ──────────────────────
            model_id = MODEL_MAPPING.get(model_version, "imagen-3.0-generate-001")

            # ── Step C: Material Synthesis Pipeline ──────── 
            
            # 1. Vision Pass (Analysis)
            pil_ref = tensor_batch_to_pil_list(reference_image)[0]
            
            analysis_prompt_text = (
                "Analyze the surface material of this image. "
                "Describe the grain, roughness, and pattern in technical 3D texturing terms. "
                "Ignore lighting/shadows. Output ONLY the description."
            )
            
            if texture_type != "Auto-Detect":
                analysis_prompt_text = f"[Material Type Hint: {texture_type}] " + analysis_prompt_text

            print(f"[Texture Pro] Analyzing reference using Gemini 3 Pro...")
            
            try:
                # Use a reliable vision model for analysis
                vision_model = "gemini-3-pro-image-preview" 
                vision_resp = client.models.generate_content(
                    model=vision_model,
                    contents=[analysis_prompt_text, pil_ref]
                )
                material_description = vision_resp.text.strip()
                print(f"[Texture Pro] Analysis: {material_description[:100]}...")
            except Exception as e:
                print(f"[Texture Pro] Analysis failed: {e}")
                material_description = f"High quality {texture_type if texture_type != 'Auto-Detect' else 'material'} texture"

            # 2. Synthesis Pass (Generation)
            # Improved prompt to forcefully match the reference style
            master_prompt = (
                f"Create a seamless, tileable PBR texture map based on this description: {material_description}. "
                "The texture MUST MATCH the visual style, color, and ruggedness of the reference description. "
                "View: Orthographic 90-degree top-down. "
                "Lighting: Flat, even, shadowless (Delighted). "
                "Tiling: Perfectly seamless edges. "
                "Resolution: High-fidelity micro-details."
            )
            
            negative_prompt = "shadows, ambient occlusion, highlights, specular reflection, perspective, blur, vignetting, mismatched colors, wrong scale"
            
            # Additional context for strictness
            if tileability_strictness > 0.8:
                master_prompt += " STRICTLY SEAMLESS and REPEATABLE pattern."

            print(f"[Texture Pro] Generating texture with {model_id}...")
            
            # Attempt generation
            final_image_pil = None
            
            # Try/Catch for 4K fallback logic (simulated by upscaling logic below)
            # We define a function for generation to allow retries/fallbacks
            def generate_at_res(width, height):
                # Map resolution string to API format
                api_image_size = "1K" # Default
                if target_dim == 2048:
                    api_image_size = "2K" 
                elif target_dim == 4096:
                    api_image_size = "4K"

                image_config_args = {"aspect_ratio": "1:1"}
                
                # Check for Gemini 3 Pro which supports native resolution
                if "gemini-3-pro" in model_id and api_image_size in ["2K", "4K"]:
                     image_config_args["image_size"] = api_image_size
                     print(f"[Texture Pro] Requesting native {api_image_size} generation...")

                cfg = types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(**image_config_args)
                )
                
                # Add resolution hint to prompt as fallback
                full_prompt = f"{master_prompt}\n\nNegative prompt: {negative_prompt}"
                
                return client.models.generate_content(
                    model=model_id,
                    contents=full_prompt,
                    config=cfg
                )

            try:
                # Attempt generation. For now, we stick to standard square output.
                # ImageGen models 4K native support is rare. 
                # We'll request 1024x1024 (standard) for stability, or 2048 if model allows.
                # Let's strictly use 1024x1024 for base generation to ensure success, then upscale.
                # This aligns with "If 4K generation fails... upscale". 
                # Actually, let's try to be smart: if they asked for 2K/4K, maybe we try 2048?
                # But Imagen-3.0-generate-001 often defaults to 1024.
                # We will generate at 1024x1024 to be safe and fast, then upscale.
                # The prompt asks to "generate at maximum supported resolution".
                
                resp = generate_at_res(1024, 1024) 
                
                # Check response
                from .utils import extract_images_from_response
                imgs = extract_images_from_response(resp)
                if not imgs:
                    raise ValueError("No images generated.")
                final_image_pil = imgs[0]
                
            except Exception as e:
                print(f"[Texture Pro] Generation error: {e}")
                # Fallback logic could be here, but we are already using safe settings.
                # If model_id failed, maybe try a fallback model?
                # For now, just re-raise or return blank
                raise e

            # ── Upscaling ──────────────────────────────────
            # Convert to tensor [B, H, W, C]
            import torchvision.transforms.functional as TF
            tensor_img = TF.to_tensor(final_image_pil).unsqueeze(0).permute(0, 2, 3, 1) # [1, H, W, 3]

            current_h, current_w = tensor_img.shape[1:3]
            
            if current_h != target_dim or current_w != target_dim:
                print(f"[Texture Pro] Upscaling from {current_h}x{current_w} to {target_dim}x{target_dim}...")
                # Permute to [B, C, H, W] for interpolate
                tensor_img = tensor_img.permute(0, 3, 1, 2)
                
                tensor_img = F.interpolate(
                    tensor_img, 
                    size=(target_dim, target_dim), 
                    mode="bicubic", 
                    align_corners=False
                )
                
                # Back to [B, H, W, C]
                tensor_img = tensor_img.permute(0, 2, 3, 1)

            # ── Step D: Glossiness/Roughness Map Estimation ──
            # "Grayscale approximation of roughness (derived from the B&W channel)"
            # Roughness is often inverted diffuse (lighter = rougher, darker = smoother/shiny) 
            # or direct luminance depending on material.
            # User requirement: "derived from the B&W channel".
            
            # Calculate luminance
            # Luminance = 0.299*R + 0.587*G + 0.114*B
            # This gives us a B&W image.
            
            # Usually for PBR:
            # - Darker diffuse often implies smoother (glossier) in some workflows (like metal), 
            #   but for "Roughness Map": Black = Smooth, White = Rough.
            # - A simple guess: Invert the luminance. 
            #   High points (white pixels) -> Low roughness (smooth/shiny)? 
            #   Or Shadows -> Occlusion.
            #   Let's just return the standard Grayscale Luminance as requested "B&W channel".
            #   If users want "Glossiness" (Black=Rough, White=Smooth), vs Roughness (White=Rough).
            #   User returned name is `glossiness_rough_guess`.
            
            # Let's produce a standard Luma conversion.
            # Using standard torch rgb_to_grayscale logic or manual
            # [B, H, W, 3]
            # Calculate luminance
            # [B, H, W, 3] -> [B, H, W, 1]
            # simple formula: 0.299*R + 0.587*G + 0.114*B
            
            # Extract channels
            r = tensor_img[..., 0]
            g = tensor_img[..., 1]
            b = tensor_img[..., 2]
            
            luma = (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(-1)
            
            # Expand to 3 channels for IMAGE output standardization in Comfy usually [B, H, W, 3]
            # or keep as 1 channel? Comfy usually expects 3 channels for previews, but masks are 1.
            # INPUT types usually explicitly say "IMAGE" (RGB) or "MASK" (Grey).
            # Output is "IMAGE". So we replicate channels.
            gloss_map = luma.repeat(1, 1, 1, 3)

            return (tensor_img, gloss_map)

        except Exception as e:
            from .gemini_nodes import _handle_api_error
            _handle_api_error(e, "GoutamSeamlessTexturePro")
            return (make_blank_image_tensor(), make_blank_image_tensor())
