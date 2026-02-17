"""
ComfyUI Gemini Nano Pro Suite — V2 Ultimate Nodes

Three professional nodes for multimodal Gemini workflows:
  1. Gemini_Ultimate_ImgGen   – Unified Text‑to‑Image & Image‑to‑Image
  2. Gemini_Ultimate_VideoGen – Image → video (Veo 3.1 / 2.0) with polling
  3. Gemini_Ultimate_Vision   – Multi‑image analysis / captioning
"""

import io
import os
import sys
import time
import tempfile
import traceback
from typing import Optional, Tuple
from PIL import Image

import numpy as np
import torch

from .utils import (
    get_gemini_client,
    retry_on_failure,
    tensor_batch_to_pil_list,
    pil_to_tensor,
    pil_list_to_tensor_batch,
    extract_images_from_response,
    extract_text_from_response,
    make_blank_image_tensor,
    build_safety_settings,
    expand_prompt_magic,
)

# Lazy‑loaded only when the video node runs
cv2 = None

CATEGORY = "Goutam_Nano_Suite"

# ═══════════════════════════════════════════════════════════════════════════
# Model lists
# ═══════════════════════════════════════════════════════════════════════════
IMAGE_GEN_MODELS = [
    "gemini-2.5-flash-image",          # Speed‑optimised (1024 px)
    "gemini-3-pro-image-preview",      # Quality / 4K capable
]

VISION_MODELS = [
    "gemini-3-flash-preview",              # Multimodal – speed (preview)
    "gemini-3-pro-preview",                # Multimodal – deep reasoning (preview)
    "gemini-2.5-flash",                    # Multimodal – stable, fast
    "gemini-2.5-pro",                      # Multimodal – stable, high quality
]

VIDEO_MODELS = [
    "veo-3.1-generate-preview",        # Latest Veo – 4K, native audio
    "veo-2.0-generate-001",            # Stable Veo 2 – silent
]

# ── Dropdown options ────────────────────────────────────────────────────
ASPECT_RATIOS_IMAGE = ["1:1", "16:9", "9:16", "4:3", "3:4"]
ASPECT_RATIOS_VIDEO = ["16:9", "9:16", "1:1"]
SAFETY_LEVELS = ["block_none", "block_few", "block_some", "block_most"]
VIDEO_DURATIONS = ["4 seconds", "6 seconds", "8 seconds"]
VIDEO_FPS = ["24", "30"]
IMAGE_SIZES = ["1K", "2K", "4K"]


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


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — Ultimate Image Generator (Text2Img + Img2Img)
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Ultimate_ImgGen:
    """
    Unified image generation node.
    • Text‑to‑Image: leave image_input disconnected.
    • Image‑to‑Image: connect an image_input for variations / editing.
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
                "image_input": ("IMAGE", {
                    "tooltip": "Optional. Connect an image for Img2Img variation / editing mode.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Unified Gemini image generator. "
        "Text‑to‑Image when no image connected; Image‑to‑Image when image provided."
    )

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
        image_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Build prompt ─────────────────────────────────────────
            final_prompt = expand_prompt_magic(prompt, prompt_magic)
            if negative_prompt.strip():
                final_prompt = f"Avoid the following: {negative_prompt.strip()}. {final_prompt}"

            # ── Build contents ───────────────────────────────────────
            contents = []
            if image_input is not None:
                pil_images = tensor_batch_to_pil_list(image_input)
                contents.append(final_prompt)
                contents.extend(pil_images)
                mode_label = f"Img2Img ({len(pil_images)} image(s))"
            else:
                contents.append(final_prompt)
                mode_label = "Text2Img"

            print(f"[Ultimate ImgGen] Mode: {mode_label} | Model: {model_name} "
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

            print("[Ultimate ImgGen] No image in response — check prompt / safety.")
            return (make_blank_image_tensor(),)

        except Exception as exc:
            _handle_api_error(exc, "Gemini_Ultimate_ImgGen")
            return (make_blank_image_tensor(),)


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — Ultimate Inpaint (Image + Reference + Mask)
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Ultimate_Inpaint:
    """
    Inpainting node: paste content from a reference image into the
    masked region of a base image.
    • base_image   = background to edit
    • reference_image = content to place in the masked area
    • mask         = white = region to replace on base_image
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {
                    "tooltip": "The background image. The masked region will be replaced.",
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image whose content will fill the masked region.",
                }),
                "mask": ("MASK", {
                    "tooltip": "White = area to replace on base_image.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Instructions for blending (e.g. 'blend naturally, match lighting').",
                }),
                "model_name": (IMAGE_GEN_MODELS, {
                    "default": IMAGE_GEN_MODELS[0],
                    "tooltip": "Gemini model for inpainting.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_IMAGE, {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio.",
                }),
                "safety": (SAFETY_LEVELS, {
                    "default": "block_none",
                    "tooltip": "Content safety filter level.",
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
                    "tooltip": "What to avoid in the inpainted region.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Inpaint a reference image onto a base image using a mask. "
        "White mask = area to replace on the base image."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, model_name="", image_size="1K", **kwargs):
        if image_size != "1K" and "3-pro" not in model_name:
            return (f"❌ image_size '{image_size}' requires gemini-3-pro-image-preview. "
                    f"{model_name} only supports 1K.")
        return True

    def generate(
        self,
        base_image: torch.Tensor,
        reference_image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        model_name: str,
        aspect_ratio: str = "1:1",
        safety: str = "block_none",
        image_size: str = "1K",
        seed: int = 0,
        api_key: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            # ── Convert inputs to PIL ────────────────────────────────
            base_pil = tensor_batch_to_pil_list(base_image)[0].convert("RGB")
            ref_pil = tensor_batch_to_pil_list(reference_image)[0].convert("RGB")

            # Convert mask tensor [B, H, W] → PIL grayscale 'L'
            mask_np = (mask[0].cpu().numpy() * 255.0).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            if mask_pil.size != base_pil.size:
                mask_pil = mask_pil.resize(base_pil.size, Image.NEAREST)

            # ── Pre-composite: white-out the masked region ───────────
            composited = base_pil.copy()
            white_fill = Image.new("RGB", base_pil.size, (255, 255, 255))
            composited.paste(white_fill, mask=mask_pil)

            # ── Build prompt ─────────────────────────────────────────
            user_instruction = prompt.strip() if prompt.strip() else "blend naturally"
            inpaint_instruction = (
                f"You are an expert photo editor performing a seamless composite. "
                f"The first image has a blank white region that needs to be filled. "
                f"The second image is a reference — use the subject/content from it "
                f"to fill the white region in the first image. "
                f"\n\nCRITICAL REQUIREMENTS for natural integration:\n"
                f"- Match the lighting direction, intensity, and color temperature of the base image\n"
                f"- Match the skin tone and color grading of the surrounding areas\n"
                f"- Match the perspective, angle, and scale of the base image\n"
                f"- Blend edges seamlessly — no hard edges or visible boundaries\n"
                f"- Match shadows, highlights, and ambient light\n"
                f"- The result must look like an original photograph, not a collage\n"
                f"\nAdditional instructions: {user_instruction}. "
                f"Keep everything outside the white area EXACTLY unchanged."
            )
            if negative_prompt.strip():
                inpaint_instruction = (
                    f"Avoid the following: {negative_prompt.strip()}. "
                    f"{inpaint_instruction}"
                )

            contents = [inpaint_instruction, composited, ref_pil]

            print(f"[Ultimate Inpaint] Model: {model_name} | AR: {aspect_ratio} "
                  f"| Size: {image_size} | Safety: {safety}")

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

            print("[Ultimate Inpaint] No image in response — check prompt / safety.")
            return (make_blank_image_tensor(),)

        except Exception as exc:
            _handle_api_error(exc, "Gemini_Ultimate_Inpaint")
            return (make_blank_image_tensor(),)


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — Ultimate Video Generator (Veo 3.1 / 2.0)
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Ultimate_VideoGen:
    """
    Generate a short video clip from a starting image (and optionally an end
    image for keyframe interpolation) using Google Veo.
    Async polling handles the long generation time automatically.
    Returns all frames as a batch IMAGE tensor [N, H, W, 3].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe the desired motion / action in the video.",
                }),
                "duration": (VIDEO_DURATIONS, {
                    "default": "4 seconds",
                    "tooltip": "Video length (Veo supports 4, 6, or 8 seconds).",
                }),
                "fps": (VIDEO_FPS, {
                    "default": "24",
                    "tooltip": "Frames per second for the output video.",
                }),
                "aspect_ratio": (ASPECT_RATIOS_VIDEO, {
                    "default": "16:9",
                    "tooltip": "Video aspect ratio.",
                }),
                "model_name": (VIDEO_MODELS, {
                    "default": VIDEO_MODELS[0],
                    "tooltip": "Veo model for video generation.",
                }),
            },
            "optional": {
                "end_image": ("IMAGE", {
                    "tooltip": "Optional last frame. When connected, the model "
                               "interpolates between start and end images.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (falls back to GEMINI_API_KEY env var).",
                    "password": True,
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the video.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Generate a short video from an image + prompt via Google Veo. "
        "Connect an end_image to interpolate between two keyframes. "
        "Returns all frames as a batch tensor [N, H, W, C]."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, model_name="", duration="4 seconds", **kwargs):
        dur_seconds = int(duration.split()[0])
        if "veo-2.0" in model_name and dur_seconds != 8:
            return (f"❌ veo-2.0 only supports 8s duration. "
                    f"'{duration}' requires veo-3.1-generate-preview.")
        return True

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_cv2():
        global cv2
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2

    @staticmethod
    def _video_bytes_to_frame_tensors(video_bytes: bytes, target_fps: int = 24) -> torch.Tensor:
        """Decode video bytes via OpenCV → [N, H, W, 3] float tensor."""
        Gemini_Ultimate_VideoGen._ensure_cv2()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(video_bytes)
            tmp.flush()
            tmp.close()

            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                raise RuntimeError("Failed to open generated video file with OpenCV.")

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 24
            frame_interval = max(1, round(source_fps / target_fps))

            frames = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    np_frame = frame_rgb.astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(np_frame))
                frame_idx += 1
            cap.release()

            if not frames:
                raise RuntimeError("No frames extracted from generated video.")

            return torch.stack(frames, dim=0)  # [N, H, W, 3]
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ── main execution ───────────────────────────────────────────────────
    def generate(
        self,
        start_image: torch.Tensor,
        prompt: str,
        duration: str = "4 seconds",
        fps: str = "24",
        aspect_ratio: str = "16:9",
        model_name: str = "veo-3.1-generate-preview",
        end_image: Optional[torch.Tensor] = None,
        api_key: str = "",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        try:
            client = get_gemini_client(api_key)
            self._ensure_cv2()

            from google.genai import types

            # ── Step A: Tensor → PIL conversion ──────────────────────
            start_pil = tensor_batch_to_pil_list(start_image)[0]

            end_pil = None
            if end_image is not None:
                end_pil = tensor_batch_to_pil_list(end_image)[0]

                # Aspect ratio safety check
                if start_pil.size != end_pil.size:
                    print(f"[Ultimate VideoGen] ⚠️  WARNING: Start image size "
                          f"{start_pil.size} != End image size {end_pil.size}. "
                          f"This may cause unexpected results.")

            # Parse duration
            dur_seconds = int(duration.split()[0])
            target_fps = int(fps)

            mode_str = "Keyframe Interpolation" if end_pil else "Single Start Frame"
            print(f"[Ultimate VideoGen] Submitting: {model_name}, "
                  f"{dur_seconds}s, {aspect_ratio}, {target_fps} FPS, "
                  f"Mode: {mode_str}")

            # ── Step B: Build config ─────────────────────────────────
            config_kwargs = dict(
                aspect_ratio=aspect_ratio,
                duration_seconds=dur_seconds,
            )
            if negative_prompt.strip():
                config_kwargs["negative_prompt"] = negative_prompt.strip()

            # End frame goes into config as last_frame (per Gemini API docs)
            # last_frame requires types.Image, not raw PIL
            if end_pil is not None:
                buf = io.BytesIO()
                end_pil.save(buf, format="PNG")
                end_image_obj = types.Image(
                    image_bytes=buf.getvalue(),
                    mime_type="image/png",
                )
                config_kwargs["last_frame"] = end_image_obj
                print(f"[Ultimate VideoGen] 🎞️  End frame provided — "
                      f"using keyframe interpolation mode.")

            video_config = types.GenerateVideosConfig(**config_kwargs)

            # ── Step C: Submit async operation ───────────────────────
            # Create start image object manually to ensure correct serialization
            buf_start = io.BytesIO()
            start_pil.save(buf_start, format="PNG")
            start_image_obj = types.Image(
                image_bytes=buf_start.getvalue(),
                mime_type="image/png",
            )

            operation = client.models.generate_videos(
                model=model_name,
                prompt=prompt,
                image=start_image_obj,
                config=video_config,
            )

            # ── Poll until done ──────────────────────────────────────
            poll_count = 0
            max_polls = 120  # ~20 min safety cap
            while not operation.done:
                poll_count += 1
                if poll_count > max_polls:
                    raise RuntimeError(
                        "Video generation timed out after ~20 minutes."
                    )
                status = getattr(operation, 'state', 'PROCESSING')
                print(f"[Ultimate VideoGen] Waiting … poll #{poll_count} "
                      f"(state: {status})")
                time.sleep(10)
                operation = client.operations.get(operation)

            # ── Download video ───────────────────────────────────────
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)

            tmp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            try:
                generated_video.video.save(tmp_path.name)
                tmp_path.close()
                with open(tmp_path.name, "rb") as f:
                    video_bytes = f.read()
            finally:
                try:
                    os.unlink(tmp_path.name)
                except OSError:
                    pass

            # ── Extract frames ───────────────────────────────────────
            frame_tensor = self._video_bytes_to_frame_tensors(
                video_bytes, target_fps=target_fps
            )
            print(f"[Ultimate VideoGen] ✅ Extracted {frame_tensor.shape[0]} frames "
                  f"({frame_tensor.shape[1]}×{frame_tensor.shape[2]}) @ {target_fps} FPS")
            return (frame_tensor,)

        except Exception as exc:
            _handle_api_error(exc, "Gemini_Ultimate_VideoGen")
            return (make_blank_image_tensor(),)


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — Ultimate Vision (Multi‑Image Analysis)
# ═══════════════════════════════════════════════════════════════════════════
class Gemini_Ultimate_Vision:
    """
    Analyze one or more images to produce a detailed text description,
    caption, or image‑generation prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail for use as an image generation prompt.",
                    "tooltip": "Analysis instruction for the vision model.",
                }),
                "model_name": (VISION_MODELS, {
                    "default": VISION_MODELS[0],
                    "tooltip": "Gemini multimodal model for vision analysis.",
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 256,
                    "display": "number",
                    "tooltip": "Maximum output tokens for the text response.",
                }),
                "safety": (SAFETY_LEVELS, {
                    "default": "block_none",
                    "tooltip": "Content safety filter level.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Gemini API key (falls back to GEMINI_API_KEY env var).",
                    "password": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze"
    CATEGORY = CATEGORY
    DESCRIPTION = "Analyze image(s) with Gemini vision to generate text (captioning, prompt generation, analysis)."
    OUTPUT_NODE = True

    def analyze(
        self,
        images: torch.Tensor,
        prompt: str,
        model_name: str,
        max_tokens: int = 1024,
        safety: str = "block_none",
        api_key: str = "",
    ) -> Tuple[str]:
        try:
            client = get_gemini_client(api_key)
            from google.genai import types

            pil_images = tensor_batch_to_pil_list(images)
            num = len(pil_images)
            print(f"[Ultimate Vision] Analyzing {num} image(s) | Model: {model_name}")

            contents = [prompt] + pil_images

            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                safety_settings=build_safety_settings(safety),
            )

            response = retry_on_failure(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=config,
            )

            text = extract_text_from_response(response)
            if text:
                print(f"[Ultimate Vision] ✅ Generated {len(text)} characters")
                return (text,)

            print("[Ultimate Vision] No text in response.")
            return ("(No text returned by the model.)",)

        except Exception as exc:
            _handle_api_error(exc, "Gemini_Ultimate_Vision")
            return ("(Error — see console for details.)",)
