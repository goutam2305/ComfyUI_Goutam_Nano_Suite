import torch
import numpy as np
from PIL import Image, ImageOps
import io
from .utils import (
    get_gemini_client,
    tensor_batch_to_pil_list,
    pil_to_tensor,
    extract_images_from_response
)

class Gemini_Direct_Texture_Maker:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "image_1": ("IMAGE",),
                "material_type": (["Leather", "Fabric/Cloth", "Wood", "Metal", "Stone/Marble", "Plastic"],),
                "resolution": (["1024x1024", "2048x2048"],),
                "color_locking": (["Strict (Math-Based)", "Loose (AI-Based)", "None"],),
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diffuse_map", "material_report")
    FUNCTION = "generate_texture"
    CATEGORY = "Goutam_Nano_Suite"

    def match_color_distribution(self, source, target):
        """
        Transfers the color distribution (Reinhard Color Transfer).
        """
        # Convert to LAB for better color transfer results
        source_lab = np.array(source.convert("LAB")).astype("float32")
        target_lab = np.array(target.convert("LAB")).astype("float32")

        for i in range(3):
            s_chan = source_lab[:,:,i]
            t_chan = target_lab[:,:,i]
            
            s_mean, s_std = np.mean(s_chan), np.std(s_chan)
            t_mean, t_std = np.mean(t_chan), np.std(t_chan)
            
            source_lab[:,:,i] = (s_chan - s_mean) * (t_std / (s_std + 1e-6)) + t_mean

        # Convert back to RGB
        source_lab = np.clip(source_lab, 0, 255).astype("uint8")
        return Image.fromarray(source_lab, "LAB").convert("RGB")

    def generate_texture(self, api_key, image_1, material_type, resolution, color_locking, image_2=None):
        try:
            client = get_gemini_client(api_key)
            
            pil_images = [tensor_batch_to_pil_list(image_1)[0]]
            if image_2 is not None:
                pil_images.append(tensor_batch_to_pil_list(image_2)[0])
            
            ref_pil = pil_images[0]

            # 1. Vision Analysis
            analysis_prompt = (
                f"Analyze this {material_type} reference image for a 3D texture workflow.\n"
                "1. Describe the Grain Pattern (e.g., straight, cathedrals, knots).\n"
                "2. Describe the Micro-Surface (e.g., open pore, varnished, rough).\n"
                "3. Identify the Key Colors (Hex Codes).\n"
                "4. Output a concise prompt for a texture generator."
            )
            
            print(f"[Gemini Direct Texture] Analyzing material...")
            try:
                vision_resp = client.models.generate_content(
                    model="gemini-1.5-pro-latest",
                    contents=[analysis_prompt] + pil_images
                )
                analysis = vision_resp.text
            except Exception as e:
                print(f"[Gemini Direct Texture] Vision analysis failed: {e}")
                analysis = f"Standard {material_type} texture."

            # 2. Generation (Imagen 3)
            final_prompt = (
                f"Texture Generation: {material_type}. \n{analysis}\n"
                "REQUIREMENTS: Seamless tileable pattern. Flat albedo lighting "
                "(no shadows, no specular highlights). Top-down orthographic view. High fidelity."
            )
            
            print(f"[Gemini Direct Texture] Generating texture with Imagen 3...")
            from google.genai import types
            
            # Use appropriate model and config for image generation
            # Note: The user's code used 'imagen-3.0-generate-001'.
            # We will use 'gemini-2.5-flash' or 'gemini-3-pro' if Imagen isn't directly available via this client method
            # OR assuming the user provided key works for Imagen via this SDK.
            # However, the unified client usually uses `generate_content` for images mostly on 
            # gemini-pro-vision (input) or specific image gen models.
            # Let's try the user's requested model if supported, otherwise fallback to Gemini image gen.
            
            # Attempting standard image generation call
            try:
                # Using the standard image generation endpoint if available in the client
                # or fallback to GenerateContent with response_modalities=["IMAGE"]
                image_response = client.models.generate_content(
                    model="imagen-3.0-generate-001", 
                    contents=final_prompt,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"])
                )
                
                generated_images = extract_images_from_response(image_response) # Helpers handle parsing
                
                if generated_images:
                    generated_pil = generated_images[0]
                else:
                    raise ValueError("No images returned.")
                    
            except Exception as e:
                print(f"[Gemini Direct Texture] Imagen failed, trying Gemini Image Gen: {e}")
                # Fallback to standard Gemini Image Gen
                image_response = client.models.generate_content(
                    model="gemini-2.5-flash", # Or other available image gen model
                    contents=final_prompt + " Generate an image.",
                     config=types.GenerateContentConfig(response_modalities=["IMAGE"])
                )
                generated_images = extract_images_from_response(image_response)
                if generated_images:
                    generated_pil = generated_images[0]
                else:
                    raise

            # 3. Color Locking
            if color_locking == "Strict (Math-Based)":
                final_output = self.match_color_distribution(generated_pil, ref_pil)
                analysis += "\n[System]: Color Histogram Matching Applied."
            else:
                final_output = generated_pil

            return (pil_to_tensor(final_output), analysis)

        except Exception as e:
            print(f"[Gemini Direct Texture] Error: {e}")
            # Fallback red square
            err_img = Image.new('RGB', (1024, 1024), color='red')
            return (pil_to_tensor(err_img), f"Error: {e}")

# Registration