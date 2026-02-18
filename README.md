# ComfyUI Goutam Nano Suite

A suite of professional AI nodes for ComfyUI, powered by Google Gemini.

## Installation

1.  **Copy the Folder**: Place this `ComfyUI_Goutam_Nano_Suite` folder into your `ComfyUI/custom_nodes/` directory.
2.  **Install Dependencies**: Double-click the `install.bat` file in this folder.
    *   This will automatically install the critical `google-genai` library that is required for these nodes to work.
    *   If you don't run this, the nodes will **NOT** show up in ComfyUI.
3.  **Restart ComfyUI**: You must restart ComfyUI for the nodes to appear after installation.

## API Key Setup

You need a Google Gemini API key to use these nodes.
1.  Get a free key from [Google AI Studio](https://aistudio.google.com/).
2.  **Option A (Recommended)**: Set a `GEMINI_API_KEY` environment variable in your system.
3.  **Option B**: Paste your key directly into the `api_key` widget on any of the nodes.

## Included Nodes

*   **🖼️ Ultimate ImgGen**: Unified Text-to-Image & Image-to-Image generator.
*   **🎨 Ultimate Inpaint**: Reference-guided inpainting with mask support.
*   **🎬 Ultimate VideoGen**: Image-to-Video generation using Google Veo.
*   **👁️ Ultimate Vision**: Multi-image analysis and captioning.
*   **📐 Interior Architect**: Specialized interior design photography and layout.
*   **🔧 Object Manipulator**: Remove or move objects with mask support.
*   **🔍 Detail Zoomer**: High-res close-ups and re-photography.
*   **🛋️ Multi-Furniture**: Composition tool for furniture scenes.
*   **💎 Direct Texture Maker**: AI Texture generation from reference or description.
*   **🖼️ Text to Image**: Dedicated Text-to-Image generator.
*   **🏰 ZenMode ArchViz All-In-One**: Visual Analysis & Scene Composition in one node.
