import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import time

from .utils import (
    get_gemini_client,
    tensor_batch_to_pil_list,
    pil_to_tensor,
    retry_on_failure,
    extract_images_from_response,
    make_blank_image_tensor,
    build_safety_settings,
)
from google.genai import types

CATEGORY = "Goutam_Nano_Suite"

IMAGE_GEN_MODELS = [
    "gemini-2.5-flash-image",          # Speed-opt
    "gemini-3-pro-image-preview",      # Quality / 4K capable
]

IMAGE_SIZES = ["1K", "2K", "4K"]

ASPECT_RATIOS_IMAGE = ["1:1", "16:9", "9:16", "4:3", "3:4"]

class Goutam_Nano_Suite_MultiView:
    """
    Generates perfectly flat, orthographic multi-angle views (Front, Back, Side, Top)
    from two 3/4 angle reference images using Google's Gemini Vision API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_front_3_4": ("IMAGE", {
                    "tooltip": "The 3/4 front reference image."
                }),
                "image_back_3_4": ("IMAGE", {
                    "tooltip": "The 3/4 back reference image."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A sci-fi armored helmet",
                    "tooltip": "Base description of the object."
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Model to use for generation."
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "1K",
                    "tooltip": "Output resolution. 2K/4K only supported by gemini-3-pro-image-preview."
                }),
                 "view_front": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Generate Front",
                    "label_off": "Skip Front"
                }),
                "view_back": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Generate Back",
                    "label_off": "Skip Back"
                }),
                "view_side": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Generate Side",
                    "label_off": "Skip Side"
                }),
                "view_top": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Generate Top",
                    "label_off": "Skip Top"
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "control_after_generate": "randomize",
                    "tooltip": "Seed for reproducibility."
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
                    "default": "blur, distortion, different proportions, background clutter",
                    "tooltip": "What to avoid in the generated images."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("multi_view_images",)
    FUNCTION = "generate_views"
    CATEGORY = CATEGORY
    DESCRIPTION = "Generates orthographic multi-angle views from 3/4 references."

    @classmethod
    def VALIDATE_INPUTS(cls, model_name="", image_size="1K", **kwargs):
        if image_size != "1K" and "3-pro" not in model_name:
            return (f"❌ image_size '{image_size}' requires gemini-3-pro-image-preview. "
                    f"{model_name} only supports 1K.")
        return True

    def generate_views(
        self,
        image_front_3_4: torch.Tensor,
        image_back_3_4: torch.Tensor,
        prompt: str,
        model_name: str,
        image_size: str,
        view_front: bool,
        view_back: bool,
        view_side: bool,
        view_top: bool,
        aspect_ratio: str = "1:1",
        seed: int = 0,
        api_key: str = "",
        negative_prompt: str = "blur, distortion, different proportions, background clutter",
    ) -> Tuple[torch.Tensor]:
        
        try:
            client = get_gemini_client(api_key)
            
            # 1. Process Inputs
            pil_front = tensor_batch_to_pil_list(image_front_3_4)[0]
            pil_back = tensor_batch_to_pil_list(image_back_3_4)[0]

            # Logic for models is now handled by dropdown + validation
            target_model = model_name
            
            # 2. Define Views to Generate
            views_to_generate = []
            if view_front: views_to_generate.append("front")
            if view_back: views_to_generate.append("back")
            if view_side: views_to_generate.append("side")
            if view_top: views_to_generate.append("top")

            if not views_to_generate:
                print("[MultiView] No views selected. Returning blank.")
                return (make_blank_image_tensor(),)

            generated_tensors = []

            # 3. Iterate and Generate
            for view_name in views_to_generate:
                print(f"[MultiView] Generating view: {view_name}...")

                # Anti-Gravity Prompt Injection
                positive_suffix = (
                    f", perfectly flat orthographic projection, exact {view_name} profile view, "
                    "floating in empty space, zero gravity, perfectly aligned to grid, "
                    "completely isolated on a flat neutral mid-grey background. "
                    "Studio flat lighting, shadowless, completely devoid of cast shadows, "
                    "no floor, no ground plane. Hyper-detailed photorealistic textures, "
                    "8k resolution, precise proportions, matching exact materials, "
                    "colors, and physical textures from the input images, non-perspective."
                )

                full_positive = f"{prompt} {positive_suffix}"

                negative_suffix = (
                    ", blueprint, sketch, illustration, vector, 2D graphic, stylized, "
                    "perspective, foreshortening, dynamic angle, ground plane, floor, "
                    "cast shadows, ambient occlusion shadows, depth of field, tilted camera, "
                    "fish-eye lens, asymmetrical."
                )
                
                full_negative = f"{negative_prompt} {negative_suffix}"

                # Construct Prompt for this user instruction
                # We tell the model to use the images as reference.
                system_instruction = (
                    "You are a professional technical artist taking 3/4, perspective reference images "
                    "and converting them into precise orthographic projection vectors."
                )
                
                # Payload contents
                # Order: Text prompt -> Images
                contents = [full_positive, pil_front, pil_back]

                # Config
                image_config_kwargs = {"aspect_ratio": aspect_ratio}
                if "3-pro" in model_name:
                    image_config_kwargs["image_size"] = image_size

                config = types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(**image_config_kwargs),
                    safety_settings=build_safety_settings("block_none"),
                    system_instruction=system_instruction 
                )
                if seed > 0:
                    config.seed = seed

                # Loop API Call
                # Note: 'gemini-2.5-flash-image' might not support system_instruction in some versions yet,
                # if so, we prepend it to prompt. But let's try using config first.
                try:
                    response = retry_on_failure(
                        client.models.generate_content,
                        model=target_model,
                        contents=contents,
                        config=config
                    )
                    
                    images = extract_images_from_response(response)
                    if images:
                        generated_tensors.append(pil_to_tensor(images[0]))
                    else:
                        print(f"[MultiView] Warning: No image returned for {view_name}")
                        generated_tensors.append(make_blank_image_tensor())
                        
                except Exception as e:
                    print(f"[MultiView] Error generating {view_name}: {e}")
                    generated_tensors.append(make_blank_image_tensor())

                # Small delay to respect rate limits if not batching strictly
                time.sleep(1.0) 

            # 4. Batch Results
            if not generated_tensors:
                return (make_blank_image_tensor(),)
                
            # Concatenate all output tensors along batch dimension [N, H, W, C]
            # Ensure they are all same size (pil_to_tensor handles this mostly, but if models vary size we might need resize)
            # The utils 'pil_list_to_tensor_batch' handles resizing.
            # But here we have a list of tensors [1, H, W, C].
            
            # Check dimensions match first tensor
            base_shape = generated_tensors[0].shape # [1, H, W, 3]
            h, w = base_shape[1], base_shape[2]
            
            final_tensors = []
            for t in generated_tensors:
                if t.shape[1:] != base_shape[1:]:
                    # Resize logic if needed, simple bilinear interpolation
                    # t is [1, H, W, C] -> permute to [1, C, H, W] for interpolate -> back
                    t_perm = t.permute(0, 3, 1, 2)
                    t_resized = torch.nn.functional.interpolate(t_perm, size=(h, w), mode='bilinear', align_corners=False)
                    t_final = t_resized.permute(0, 2, 3, 1)
                    final_tensors.append(t_final)
                else:
                    final_tensors.append(t)

            batch_output = torch.cat(final_tensors, dim=0)
            return (batch_output,)
            
        except Exception as e:
            print(f"[MultiView] Critical Error: {e}")
            return (make_blank_image_tensor(),)
