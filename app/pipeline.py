import os
import logging
import magic
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2

from app.config import config
from app.layers.visible_watermark import apply_visible_watermark

logger = logging.getLogger("fortress.pipeline")

async def process_document(input_path: Path, output_path: Path, watermark_text: str, zones: list = None, thickness: float = 0.7, intensity: float = 1.0, grayscale: bool = False):
    """
    Main pipeline: Supports PDF and Images (JPG, PNG).
    Applies ONLY the Titan Visual Watermark.
    """
    
    # 1. Detect File Type
    file_type = "unknown"
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(input_path))
    except Exception as e:
        logger.warning(f"Magic failed, using extension. Error: {e}")
        ext = input_path.suffix.lower()
        if ext == ".pdf":
            file_type = "application/pdf"
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            file_type = "image/" + ext[1:]
    
    logger.info(f"Processing file: {input_path} (Type: {file_type})")
    
    try:
        # Check against mapped types
        if "pdf" in file_type:
            await _process_pdf(input_path, output_path, watermark_text, zones, thickness, intensity, grayscale)
        elif "image" in file_type:
            await _process_image(input_path, output_path, watermark_text, zones, thickness, intensity, grayscale)
        else:
             # Fallback based on extension
             ext = input_path.suffix.lower()
             if ext == ".pdf":
                 await _process_pdf(input_path, output_path, watermark_text, zones, thickness, intensity, grayscale)
             elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
                 await _process_image(input_path, output_path, watermark_text, zones, thickness, intensity, grayscale)
             else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
        logger.info(f"Successfully processed: {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e

async def _process_image(input_path: Path, output_path: Path, watermark_text: str, zones: list = None, thickness: float = 0.7, intensity: float = 1.0, grayscale: bool = False):
    """
    Process a single image file.
    """
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError("Could not load image file.")
        
    logger.info(f"Applying Titan Visual Watermark to Image (Grayscale: {grayscale})...")
    
    if not zones:
        # Global mode
        watermarked = apply_visible_watermark(img, watermark_text, thickness=thickness, intensity=intensity)
    else:
        # Selective mode: apply each zone
        # For an image, we assume all zones with page_index=0 or null apply.
        watermarked = img.copy()
        for zone in zones:
            # For robustness, handle zones without page_index or with 0
            if zone.get("page_index", 0) in [0, None]:
                watermarked = apply_visible_watermark(watermarked, watermark_text, roi=zone, thickness=thickness, intensity=intensity)
    
    if grayscale:
        logger.info("Converting final image output to grayscale")
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
        # Convert back to 3-channel for consistent file metadata/viewer compatibility
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_GRAY2BGR)
        
    cv2.imwrite(str(output_path), watermarked)

async def _process_pdf(input_path: Path, output_path: Path, watermark_text: str, zones: list = None, thickness: float = 0.7, intensity: float = 1.0, grayscale: bool = False):
    """
    Process a PDF file.
    """
    logger.info(f"Converting PDF to images (Grayscale: {grayscale})...")
    images = convert_from_path(str(input_path))
    
    processed_images = []
    
    for i, pil_image in enumerate(images):
        logger.info(f"Processing Page {i+1}/{len(images)}...")
        
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        # Determine ROI for this page
        if not zones:
            # Global
            watermarked = apply_visible_watermark(open_cv_image, watermark_text, thickness=thickness, intensity=intensity)
        else:
            # Selective: only apply zones matching current page
            page_zones = [z for z in zones if z.get("page_index") == i]
            if not page_zones:
                watermarked = open_cv_image # No watermark on this page
            else:
                watermarked = open_cv_image.copy()
                for zone in page_zones:
                    watermarked = apply_visible_watermark(watermarked, watermark_text, roi=zone, thickness=thickness, intensity=intensity)
        
        if grayscale:
            watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
            # Re-convert to BGR for consistent processing in the loop (PIL expects RGB eventually)
            watermarked = cv2.cvtColor(watermarked, cv2.COLOR_GRAY2BGR)

        watermarked_rgb = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(watermarked_rgb)
        processed_images.append(processed_pil)
        
    # 2. Save as PDF
    logger.info("Reconstructing PDF...")
    if processed_images:
        processed_images[0].save(
            str(output_path), "PDF", resolution=100.0, save_all=True, append_images=processed_images[1:]
        )
