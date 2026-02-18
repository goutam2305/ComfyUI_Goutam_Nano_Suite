"""
ComfyUI Goutam Nano Suite — V2
──────────────────────────────────
Register the Ultimate nodes with ComfyUI.

Nodes:
  🖼️  Goutam_Nano_Suite Ultimate ImgGen     – Text‑to‑Image & Image‑to‑Image
  🎨  Goutam_Nano_Suite Ultimate Inpaint    – Reference‑guided inpainting with mask
  🔧  Goutam_Nano_Suite Object Manipulator  – Remove/Move objects (Mask supported)
  🔍  Goutam_Nano_Suite Detail Zoomer       – High‑res close‑ups (Re‑photography)
  📐  Goutam_Nano_Suite ArchViz Scanner     – Interactive cropping & upscaling
  🛋️  Goutam_Nano_Suite Multi‑Furniture     – Batch product → room scene
  📐  Goutam_Nano_Suite Interior Architect  – Pro photography + lighting + layout
  🎬  Goutam_Nano_Suite Ultimate VideoGen   – Image → video via Veo
  👁️  Goutam_Nano_Suite Ultimate Vision     – Multi‑image analysis / captioning
  💎  Goutam_Nano_Suite Texture Maker       – Texture generation
  🖼️  Goutam_Nano_Suite Text to Image       – Text-to-Image generation
"""

from .gemini_architect import Gemini_Interior_Architect
from .gemini_manipulator import Gemini_Object_Manipulator
from .gemini_multi_composer import Gemini_Multi_Furniture_Composer
from .gemini_scanner import Gemini_ArchViz_Scanner
from .gemini_nodes import (
    Gemini_Ultimate_ImgGen,
    Gemini_Ultimate_Inpaint,
    Gemini_Ultimate_VideoGen,
    Gemini_Ultimate_Vision,
)
from .gemini_zoomer import Gemini_Detail_Zoomer

from .gemini_material import Goutam_Direct_Texture_Maker
from .gemini_t2i import Goutam_TextToImage

NODE_CLASS_MAPPINGS = {
    "Gemini_Ultimate_ImgGen": Gemini_Ultimate_ImgGen,
    "Gemini_Ultimate_Inpaint": Gemini_Ultimate_Inpaint,
    "Gemini_Object_Manipulator": Gemini_Object_Manipulator,
    "Gemini_Detail_Zoomer": Gemini_Detail_Zoomer,
    "Gemini_ArchViz_Scanner": Gemini_ArchViz_Scanner,
    "Gemini_Multi_Furniture_Composer": Gemini_Multi_Furniture_Composer,
    "Gemini_Interior_Architect": Gemini_Interior_Architect,
    "Gemini_Ultimate_VideoGen": Gemini_Ultimate_VideoGen,
    "Gemini_Ultimate_Vision": Gemini_Ultimate_Vision,
    "Goutam_Direct_Texture_Maker": Goutam_Direct_Texture_Maker,
    "Goutam_TextToImage": Goutam_TextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini_Ultimate_ImgGen": "🖼️ Goutam_Nano_Suite Ultimate ImgGen",
    "Gemini_Ultimate_Inpaint": "🎨 Goutam_Nano_Suite Ultimate Inpaint",
    "Gemini_Object_Manipulator": "🔧 Goutam_Nano_Suite Object Manipulator",
    "Gemini_Detail_Zoomer": "🔍 Goutam_Nano_Suite Detail Zoomer",
    "Gemini_ArchViz_Scanner": "Goutam_Nano_Suite ArchViz Scanner 📐",
    "Gemini_Multi_Furniture_Composer": "🛋️ Goutam_Nano_Suite Multi-Furniture",
    "Gemini_Interior_Architect": "📐 Goutam_Nano_Suite Interior Architect",
    "Gemini_Ultimate_VideoGen": "🎬 Goutam_Nano_Suite Ultimate VideoGen",
    "Gemini_Ultimate_Vision": "👁️ Goutam_Nano_Suite Ultimate Vision",
    "Goutam_Direct_Texture_Maker": "💎 Goutam_Nano_Suite Direct Texture Maker",
    "Goutam_TextToImage": "🖼️ Goutam_Nano_Suite Text to Image",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
