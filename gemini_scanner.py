"""
Gemini ArchViz Scanner
──────────────────────
Backend logic for the ArchViz Scanner node.
Handles interactive image cropping via frontend "Ghost Drive" of INT widgets.
"""

import json
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths
from .gemini_nodes import CATEGORY


class Gemini_ArchViz_Scanner:
    """
    Accepts an image and crop coordinates (controlled by JS frontend).
    Returns the cropped image, mask, and context image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "crop_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "crop_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "crop_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "aspect_preset": (["Free", "1:1", "4:3", "3:2", "16:9", "21:9", "9:16", "2:3", "3:4", "Custom"], {
                    "default": "Free",
                    "tooltip": "Lock crop box to a specific aspect ratio. 'Free' allows any shape."
                }),
                "custom_ratio": ("FLOAT", {
                    "default": 1.77, "min": 0.1, "max": 10.0, "step": 0.01,
                    "tooltip": "Width/Height ratio (only used when aspect_preset is 'Custom')."
                }),
                "upscale_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 1.0, 
                    "max": 4.0, 
                    "step": 0.1,
                    "tooltip": "Resize the cropped area by this factor."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("cropped_image", "crop_mask", "context_image")
    FUNCTION = "scan_and_crop"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # Enable UI returns
    DESCRIPTION = (
        "Interactive cropping node. The frontend 'Ghost Drives' the input widgets. "
        "Draw a box on the image to update crop coordinates."
    )

    def scan_and_crop(
        self,
        image: torch.Tensor,
        crop_top: int = 0,
        crop_left: int = 0,
        crop_bottom: int = 0,
        crop_right: int = 0,
        aspect_preset: str = "Free",
        custom_ratio: float = 1.77,
        upscale_factor: float = 1.0,
    ):
        
        # ─── Save Context Image to Temp (for UI display) ───────────────────
        preview_image = None
        try:
            img_tensor = image[0]
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            filename = f"archviz_scanner_{random.randint(0, 1000000)}.png"
            subfolder = "archviz_temp"
            
            # Get temp path
            temp_dir = folder_paths.get_temp_directory()
            full_output_folder = os.path.join(temp_dir, subfolder)
            os.makedirs(full_output_folder, exist_ok=True)
            
            img.save(os.path.join(full_output_folder, filename))
            
            preview_image = {
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp"
            }
        except Exception as e:
            print(f"[ArchViz Scanner] Error saving preview: {e}")

        B, H, W, C = image.shape

        # ─── Validate Crop Dimensions ──────────────────────────────────────
        # Frontend sends: crop_top=Y, crop_left=X, crop_bottom=HEIGHT, crop_right=WIDTH
        top = max(0, min(crop_top, H - 1))
        left = max(0, min(crop_left, W - 1))
        # Convert height/width to absolute bottom/right coordinates
        bottom = max(top + 1, min(top + crop_bottom, H)) if crop_bottom > 0 else H
        right = max(left + 1, min(left + crop_right, W)) if crop_right > 0 else W

        # Check for valid area
        if bottom <= top or right <= left:
            print("[ArchViz Scanner] Invalid crop dimensions. Returning full image.")
            # Fallback to full
            cropped = image
            mask = torch.ones((1, H, W), dtype=torch.float32, device=image.device)
            result = (image, mask, image)
            return {
                "ui": {"images": [preview_image]} if preview_image else {},
                "result": result
            }

        # ─── Perform Crop ──────────────────────────────────────────────────
        try:
            print(f"[ArchViz Scanner] Cropping: Top={top}, Left={left}, Bottom={bottom}, Right={right} (Input: {W}x{H})")
            
            cropped = image[:, top:bottom, left:right, :]

            # Upscale if needed
            if upscale_factor > 1.0:
                cropped_p = cropped.permute(0, 3, 1, 2)
                h_crop = bottom - top
                w_crop = right - left
                new_h = int(h_crop * upscale_factor)
                new_w = int(w_crop * upscale_factor)
                
                cropped_p = F.interpolate(
                    cropped_p, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                cropped = cropped_p.permute(0, 2, 3, 1)

            # Generate Mask (White at crop, Black background)
            mask = torch.zeros((1, H, W), dtype=torch.float32, device=image.device)
            mask[:, top:bottom, left:right] = 1.0

            result = (cropped, mask, image)
            
            return {
                "ui": {"images": [preview_image]} if preview_image else {},
                "result": result
            }

        except Exception as e:
            print(f"[ArchViz Scanner] Error processing crop: {e}")
            mask = torch.ones((1, H, W), dtype=torch.float32, device=image.device)
            return {
                "ui": {"images": [preview_image]} if preview_image else {},
                "result": (image, mask, image)
            }
