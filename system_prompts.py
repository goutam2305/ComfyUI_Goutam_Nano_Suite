VISUAL_ANALYSIS_PROMPT = """
Analyze the submitted image as an expert Architectural Photographer and Interior Stylist. 
Extract a comprehensive style profile in strict JSON format.

Output JSON Structure:
{
  "style_name": "string (e.g., 'Mid-Century Modern', 'Industrial Loft')",
  "atmosphere": "string (mood, emotion, vibe)",
  "lighting": {
    "type": "string (natural, artificial, mixed)",
    "direction": "string (e.g., 'soft diffuse from left window')",
    "color_temperature": "string (warm, cool, neutral)",
    "intensity": "string (bright, dim, dramatic)"
  },
  "color_palette": {
    "dominant": ["hex_code", "hex_code"],
    "accents": ["hex_code", "hex_code"],
    "background": "hex_code"
  },
  "materials": {
    "flooring": "string description",
    "walls": "string description",
    "furniture": "string description"
  },
  "composition": {
    "focal_points": ["string"],
    "depth": "string description (shallow, deep, layered)",
    "perspective": "string description (eye-level, wide-angle, etc.)"
  },
  "architectural_features": ["string", "string"]
}

Constraint:
- Output ONLY valid JSON.
- No markdown formatting.
- No conversational text.
"""
