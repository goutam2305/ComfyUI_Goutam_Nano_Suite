import torch
import json

class Gemini_Florence_Translator:
    """
    Parses Florence-2 bounding boxes and scales them for SAM 2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "florence_data": ("*",),
            },
            "optional": {
                  # Changed default to True to prevent crashes by default
                  "index_error_fix": ("BOOLEAN", {"default": True, "label_on": "Enable fallback (Whole Image)", "label_off": "Disable fallback"}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "translate"
    CATEGORY = "Goutam_Nano_Suite"
    DESCRIPTION = "Converts Florence-2 bounding boxes to SAM 2 format."

    def translate(self, image, florence_data, index_error_fix=True):
        print(f"\n[Gemini_Florence_Translator] ------------------------------------------------")
        print(f"[Gemini_Florence_Translator] Processing Florence Data...")
        
        # 1. Get image dimensions
        if image is None:
             print("[Gemini_Florence_Translator] ❌ No image provided!")
             # Return a safe dummy box 1024x1024 just in case
             return (torch.tensor([[0, 0, 1024, 1024]], dtype=torch.float32),)
             
        _, h, w, _ = image.shape
        print(f"[Gemini_Florence_Translator] Image Shape: {image.shape} (H={h}, W={w})")
        
        # 2. Extract bboxes
        bboxes = []
        labels = []
        
        if isinstance(florence_data, dict):
            if 'bboxes' in florence_data:
                 print("[Gemini_Florence_Translator] Found 'bboxes' key directly.")
                 bboxes = florence_data['bboxes']
            else:
                print("[Gemini_Florence_Translator] Searching nested values for 'bboxes'...")
                for key, value in florence_data.items():
                    if isinstance(value, dict) and 'bboxes' in value:
                        print(f"[Gemini_Florence_Translator] Found 'bboxes' in key: '{key}'")
                        bboxes.extend(value['bboxes'])
        else:
             print(f"[Gemini_Florence_Translator] ❌ Error: florence_data is not a dict.")

        # 3. Validation and Fallback
        if not bboxes:
             print("[Gemini_Florence_Translator] ⚠️ No bounding boxes found (list is empty).")
             if index_error_fix:
                 print(f"[Gemini_Florence_Translator] 🛡️ Fallback ACTIVE: Returning whole image bbox [0, 0, {w}, {h}].")
                 print("[Gemini_Florence_Translator] This prevents Sam2Segment from crashing.")
                 return (torch.tensor([[0, 0, w, h]], dtype=torch.float32),)
             else:
                 print("[Gemini_Florence_Translator] Fallback DISABLED. Returning empty tensor (may cause crash).")
                 return (torch.zeros((0, 4)),)

        print(f"[Gemini_Florence_Translator] Found {len(bboxes)} raw boxes.")

        # 4. Processing and Scaling
        final_boxes = []
        for i, box in enumerate(bboxes):
            if len(box) < 4: continue
            
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            
            # Normalization Heuristic
            if x2 <= 1.5 and y2 <= 1.5:
                 x1 = x1 * w
                 y1 = y1 * h
                 x2 = x2 * w
                 y2 = y2 * h
            
            final_boxes.append([x1, y1, x2, y2])
        
        if not final_boxes:
             print("[Gemini_Florence_Translator] ⚠️ Processed list is empty.")
             if index_error_fix:
                 return (torch.tensor([[0, 0, w, h]], dtype=torch.float32),)
             return (torch.zeros((0, 4)),)

        print(f"[Gemini_Florence_Translator] ✅ Returning {len(final_boxes)} valid boxes.")
        
        # 5. Return as Tensor [N, 4]
        return (torch.tensor(final_boxes, dtype=torch.float32),)
