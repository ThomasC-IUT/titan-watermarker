import math
import random
import textwrap
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from app.config import config

logger = logging.getLogger("fortress.visible")

class LiquidInterferenceGenerator:
    """
    Generates 'Titan v2: Liquid Interference' watermarks.
    Features:
    - Full-page coverage
    - Heterogeneous opacity (Noise-modulated waves)
    - Text Stencil (Negative space)
    """
    def __init__(self, width: int, height: int, roi: dict = None, thickness: float = 0.7):
        self.width = width
        self.height = height
        self.roi = roi # {"x": int, "y": int, "w": int, "h": int}
        self.thickness = thickness

    def generate_noise_map(self, scale: int = 40) -> np.ndarray:
        """
        Generates a smooth noise map for heterogeneous opacity.
        """
        mw = self.width // scale + 1
        mh = self.height // scale + 1
        low_res = np.random.rand(mh, mw).astype(np.float32)
        noise_map = cv2.resize(low_res, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return noise_map

    def generate_heterogeneous_waves(self, displacement: np.ndarray = None) -> np.ndarray:
        """
        Generates the full-page wave field with varying opacity.
        If displacement (0.0-1.0) is provided, it tordues the waves violently.
        """
        y_grid, x_grid = np.indices((self.height, self.width), dtype=np.float32)
        freq_y = 0.22 
        freq_x = 0.015
        amp_x = 15.0
        
        # Base horizontal warping
        y_warped = y_grid + amp_x * np.sin(x_grid * freq_x)
        
        # Radical Structural Modulation: 
        # The text 'force' the phase of the wave to shift.
        if displacement is not None:
            # We use a large shift (e.g. half a period or more)
            # freq_y is 0.22, so period is approx 28px. 
            # A shift of 15px is a massive 180 degree phase flip.
            y_warped += displacement * 30.0 
            
        raw_waves = np.sin(y_warped * freq_y)
        
        # Mapping to make color proportional to thickness
        # thickness 0.0 -> very thin lines (offset roughly -0.8)
        # thickness 0.7 -> default (offset 0.7)
        # thickness 1.0 -> solid (offset 1.5+)
        offset = (self.thickness * 2.3) - 0.8 
        waves = np.clip(raw_waves * 1.5 + offset, 0, 1)
        
        # We want the waves to cover the whole page (for continuity)
        opacity_map = self.generate_noise_map(scale=100)
        final_pattern = (waves * opacity_map * 255).astype(np.uint8)
        return final_pattern

    def calculate_dynamic_grid(self, w: int, h: int) -> tuple[int, int]:
        """
        Calculates the optimal number of columns and rows based on the target area dimensions.
        """
        aspect = w / h
        
        # For small areas, we use a single stamp or a 1x2 grid.
        if w < 600 or h < 400:
            if aspect > 1.2: return 2, 1
            if aspect < 0.8: return 1, 2
            return 1, 1

        if aspect > 1.5: return 3, 2
        if aspect < 0.7: return 2, 4
        return 2, 3

    def render_smart_stamp(self, text: str, max_w: int, max_h: int) -> Image.Image:
        """
        Fits text into the given box (max_w, max_h) by optimizing line wrapping and font size.
        """
        safe_w = int(max_w * 0.85)
        safe_h = int(max_h * 0.85)
        
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        best_size = 12 # Minimum font size
        best_lines = [text]
        
        # Search for largest font that fits
        # We start from a size that is likely too big and shrink.
        start_size = min(200, max_h // 2)
        for size in range(start_size, 10, -5):
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                font = ImageFont.load_default()
                break

            avg_char_w = size * 0.6
            chars_per_line = max(5, int(safe_w / avg_char_w))
            lines = textwrap.wrap(text, width=chars_per_line)
            
            max_line_w = 0
            total_h = 0
            line_height = size * 1.1
            
            for line in lines:
                bbox = font.getbbox(line)
                w = bbox[2] - bbox[0]
                max_line_w = max(max_line_w, w)
                total_h += line_height
            
            if max_line_w <= safe_w and total_h <= safe_h:
                best_size = size
                best_lines = lines
                break
        
        try:
            font = ImageFont.truetype(font_path, best_size)
        except:
            font = ImageFont.load_default()
            
        line_height_px = int(best_size * 1.1)
        total_h_px = len(best_lines) * line_height_px
        
        max_w_px = 0
        for line in best_lines:
             b = font.getbbox(line)
             max_w_px = max(max_w_px, b[2]-b[0])
             
        img = Image.new('L', (max_w_px + 20, total_h_px + 20), 0)
        d = ImageDraw.Draw(img)
        
        y = 10
        for line in best_lines:
             lb = font.getbbox(line)
             lw = lb[2] - lb[0]
             x = (max_w_px - lw) // 2 + 10
             d.text((x, y), line, font=font, fill=255)
             y += line_height_px
             
        return img

    def generate_scattered_stamps(self, text: str) -> np.ndarray:
        """
        Generates the mask with scattered, rotated text stamps, constrained to ROI if provided.
        """
        canvas = Image.new('L', (self.width, self.height), 0)
        
        if self.roi:
            rx, ry, rw, rh = self.roi['x'], self.roi['y'], self.roi['w'], self.roi['h']
            target_area_w, target_area_h = rw, rh
            offset_x, offset_y = rx, ry
        else:
            target_area_w, target_area_h = self.width, self.height
            offset_x, offset_y = 0, 0
            
        cols, rows = self.calculate_dynamic_grid(target_area_w, target_area_h)
        
        zone_w = target_area_w // cols
        zone_h = target_area_h // rows
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                count += 1
                x0 = offset_x + (c * zone_w)
                y0 = offset_y + (r * zone_h)
                
                stamp = self.render_smart_stamp(text, zone_w, zone_h)
                angle = random.randint(-25, 25)
                
                if count % 2 == 0:
                    angle = random.randint(15, 35) if random.random() > 0.5 else random.randint(-35, -15)
                
                rot_stamp = stamp.rotate(angle, expand=True, resample=Image.BICUBIC)
                sw, sh = rot_stamp.size
                
                slack_x = max(0, zone_w - sw)
                slack_y = max(0, zone_h - sh)
                
                off_x = slack_x // 2 + random.randint(-int(slack_x*0.1), int(slack_x*0.1)) if slack_x > 10 else 0
                off_y = slack_y // 2 + random.randint(-int(slack_y*0.1), int(slack_y*0.1)) if slack_y > 10 else 0
                
                paste_x = x0 + off_x
                paste_y = y0 + off_y
                
                # Use itself as mask to paste only white pixels
                canvas.paste(rot_stamp, (paste_x, paste_y), rot_stamp)

        return np.array(canvas)

    def generate_stencil_overlay(self, text: str) -> np.ndarray:
        """
        Generates the final overlay: Wavy Field minus Text within ROI.
        """
        # 1. Background Pattern (Global for consistency)
        waves = self.generate_heterogeneous_waves()
        
        # 2. Text Mask
        text_mask = self.generate_scattered_stamps(text)
        
        # 3. Apply Stencil Logic
        inverted_text = cv2.bitwise_not(text_mask)
        stencil = cv2.bitwise_and(waves, inverted_text)
        
        # 4. If ROI is used, we need to mask the WHOLE stencil outside the ROI
        if self.roi:
            rx, ry, rw, rh = self.roi['x'], self.roi['y'], self.roi['w'], self.roi['h']
            roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.rectangle(roi_mask, (rx, ry), (rx+rw, ry+rh), 255, -1)
            stencil = cv2.bitwise_and(stencil, roi_mask)
            
        return stencil

def apply_visible_watermark(image: np.ndarray, watermark_text: str, roi: dict = None, thickness: float = 0.7, intensity: float = 1.0) -> np.ndarray:
    """
    Applies the "Titan v2.2.1: Luminance-Anchored" Watermark.
    Uses relative shadowing and chroma matching to blend with the document.
    """
    h, w = image.shape[:2]
    
    # Process ROI if provided in percentage format
    if roi:
        if "x_perc" in roi:
            roi = {
                "x": int(roi["x_perc"] * w),
                "y": int(roi["y_perc"] * h),
                "w": int(roi["w_perc"] * w),
                "h": int(roi["h_perc"] * h)
            }
    
    # 1. Generate Generator
    gen = LiquidInterferenceGenerator(w, h, roi=roi, thickness=thickness)
    
    # 2. Text Stamps & Displacement Map
    text_stamps_raw = gen.generate_scattered_stamps(watermark_text).astype(np.float32) / 255.0
    
    # --- ANALOG STAMP EFFECT: Ink Texture ---
    # Create high-frequency noise for ink voids
    noise = np.random.uniform(0.85, 1.0, (h, w)).astype(np.float32)
    # Larger spots (salt and pepper style) for realistic voids
    voids = np.random.choice([0, 1], size=(h, w), p=[0.02, 0.98]).astype(np.float32)
    text_stamps = text_stamps_raw * noise * voids
    
    # --- ANALOG STAMP EFFECT: Blooming (Ink Bleed) ---
    # Create a 'bloom' mask that's slightly larger than the text
    k_bloom = max(3, int(min(h, w) * 0.003)) | 1
    kernel_bloom = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bloom, k_bloom))
    text_bloom_mask = cv2.dilate(text_stamps_raw, kernel_bloom, iterations=1)
    text_bloom_mask = cv2.GaussianBlur(text_bloom_mask, (k_bloom*2+1, k_bloom*2+1), 0)
    
    # 3. Structural Distorsion Map for Waves
    dist_k = max(5, int(min(h, w) * 0.005)) | 1
    dist_map = cv2.GaussianBlur(text_stamps_raw, (dist_k, dist_k), 0)
    full_waves_raw = gen.generate_heterogeneous_waves(displacement=dist_map).astype(np.float32) / 255.0

    # 4. Ghost Borders (AI Poisoning)
    kernel_ghost = np.ones((3,3), np.uint8)
    dilated_text = cv2.dilate(text_stamps_raw, kernel_ghost, iterations=1)
    ghost_border = (dilated_text - text_stamps_raw)
    
    # 5. Global Alpha Calculation
    tone_A = 0.92 * intensity
    tone_B = 0.60 * intensity
    
    # Waves background
    combined_alpha = (tone_B + (tone_A - tone_B) * full_waves_raw)
    
    # Apply Text (Black Ink) - textured
    combined_alpha = np.maximum(combined_alpha, text_stamps * intensity)
    
    # Apply Bloom (Organic Bleeding)
    # Bloom should be less intense than the ink itself
    combined_alpha = np.maximum(combined_alpha, text_bloom_mask * intensity * 0.4)
    
    # Ghost Poisoning
    combined_alpha = np.where(ghost_border > 0.1, combined_alpha * 1.05, combined_alpha)
    
    # 6. ROI & Feathering
    roi_mask = np.zeros((h, w), dtype=np.float32)
    if roi:
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        cv2.rectangle(roi_mask, (rx, ry), (rx+rw, ry+rh), 1.0, -1)
        ksize = int(min(h, w) * 0.08) | 1
        roi_mask = cv2.GaussianBlur(roi_mask, (ksize, ksize), 0)
    else:
        roi_mask[:] = 1.0

    final_alpha = np.clip(combined_alpha * roi_mask, 0, 1)
    
    # --- ANALOG STAMP EFFECT: Structural Bleeding (Paper Warp) ---
    image_f = image.astype(np.float32)
    
    # Generate a displacement map based on text edges
    gradient_x = cv2.Sobel(text_bloom_mask, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(text_bloom_mask, cv2.CV_32F, 0, 1, ksize=3)
    
    # Create warping grids
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    map_x = x_idx + gradient_x * 2.5 * intensity 
    map_y = y_idx + gradient_y * 2.5 * intensity
    
    # Remap the original image (The Paper Displacement)
    warped_image_f = cv2.remap(image_f, map_x, map_y, cv2.INTER_LINEAR)
    
    # --- LUMINANCE-ANCHORED BLENDING & CHROMA MATCHING ---
    # Sample document color for ink tinting
    if roi:
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        roi_sample = image[ry:ry+rh, rx:rx+rw] if rw > 0 and rh > 0 else image
        mean_color = np.mean(roi_sample, axis=(0, 1))
    else:
        mean_color = np.mean(image, axis=(0, 1))
    
    # Target Ink: 15% of the original color (Shadow) + a bit of document tint
    # This prevents digital [0,0,0] which looks artificial.
    ink_shadow_factor = 0.12 # Leave 12% of original light/texture 
    target_ink_color = mean_color * 0.05 # Tiny tint
    
    # Add high-frequency grain to the alpha to blend with paper noise
    grain = np.random.normal(0, 0.02, (h, w)).astype(np.float32)
    final_alpha = np.clip(final_alpha + grain * final_alpha, 0, 1)
    
    # 7. Final Blending (Multiply/Shadow approach)
    a_x = np.dstack([final_alpha] * 3)
    
    # Result = WarpedImage * (1 - Alpha * (1 - InkShadowFactor)) + TargetInkColor * Alpha
    # This ensures the ink 'belongs' to the document pixels.
    # On very light paper, the ink will be very dark grey.
    # On darker regions, it will feel like a deeper shadow.
    
    # Base darkened layer (Ink behavior)
    ink_layer = warped_image_f * ink_shadow_factor + target_ink_color
    
    # Blend: out = Original * (1 - Alpha) + InkLayer * Alpha
    out = warped_image_f * (1.0 - a_x) + ink_layer * a_x
    
    result = np.clip(out, 0, 255).astype(np.uint8)
    
    logger.info(f"Applied Titan Analog Watermark: {watermark_text}")
    return result
