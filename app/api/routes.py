"""
Document Fortress — API Routes
REST endpoints for visual document watermarking.
"""

import json
import io
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
import cv2
from PIL import Image

from app.config import config
from app.pipeline import process_document
from app.layers.visible_watermark import apply_visible_watermark

router = APIRouter(prefix="/api", tags=["fortress"])
logger = logging.getLogger("fortress.api")

@router.get("/health")
async def health_check():
    """Health check endpoint for Docker healthchecks and monitoring."""
    return {
        "status": "ok",
        "service": "Titan Visual Watermarker",
        "version": "2.2.0"
    }

@router.post("/watermark")
async def watermark_document(
    file: UploadFile = File(..., description="PDF or Image file to watermark"),
    message: str = Form(..., description="Text message to apply as watermark"),
    zones: str = Form(None, description="JSON string of zones: [{'page_index': 0, 'x': 10, 'y': 10, 'w': 100, 'h': 100}]"),
    thickness: float = Form(0.7, description="Wave thickness (0.0 to 1.0)"),
    intensity: float = Form(1.0, description="Global intensity/opacity (0.0 to 1.0)"),
    grayscale: str = Form("false", description="Convert output to Black & White ('true'/'false')")
):
    """
    Apply visual watermark to the uploaded document.
    """
    # Parse zones
    zone_list = None
    if zones:
        try:
            zone_list = json.loads(zones)
        except Exception as e:
            logger.warning(f"Failed to parse zones JSON: {e}")
    
    do_grayscale = grayscale.lower() == "true"
    logger.info(f"Watermark request: text='{message.strip()}', grayscale={do_grayscale}")

    # Create temporary files
    suffix = Path(file.filename).suffix.lower()
    content = await file.read()
    
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(content)
        tmp_in_path = Path(tmp_in.name)

    tmp_out_path = Path(str(tmp_in_path).replace(suffix, f"_watermarked{suffix}"))

    try:
        # Run Pipeline
        await process_document(
            input_path=tmp_in_path,
            output_path=tmp_out_path,
            watermark_text=message.strip(),
            zones=zone_list,
            thickness=thickness,
            intensity=intensity,
            grayscale=do_grayscale
        )

        def cleanup():
            if tmp_in_path.exists(): tmp_in_path.unlink()
            if tmp_out_path.exists(): tmp_out_path.unlink()

        return FileResponse(
            path=tmp_out_path,
            filename=f"watermarked_{file.filename}",
            background=cleanup
        )

    except Exception as e:
        if tmp_in_path.exists(): tmp_in_path.unlink()
        if tmp_out_path.exists(): tmp_out_path.unlink()
        logger.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preview")
async def preview_watermark(
    file: UploadFile = File(...),
    message: str = Form(...),
    page_index: int = Form(0),
    zones: str = Form(None, description="Optional JSON string of zones for the preview"),
    thickness: float = Form(0.7, description="Wave thickness (0.0 to 1.0)"),
    intensity: float = Form(1.0, description="Global intensity/opacity (0.0 to 1.0)"),
    grayscale: str = Form("false", description="Convert output to Black & White ('true'/'false')")
):
    """
    Generates a low-res preview of the watermarked document, supporting multiple zones.
    """
    suffix = Path(file.filename).suffix.lower()
    content = await file.read()
    
    try:
        # 1. Load Image/Page
        if suffix == ".pdf":
            # Use convert_from_bytes for better robustness with memory content
            images = convert_from_bytes(content, first_page=page_index+1, last_page=page_index+1)
            if not images:
                raise HTTPException(status_code=400, detail="Requested page not found")
            img = np.array(images[0])
            img = img[:, :, ::-1].copy() # RGB to BGR
        else:
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

        # 2. Parse Zones
        zone_list = []
        if zones:
            try:
                zone_list = json.loads(zones)
            except:
                logger.warning("Invalid zones JSON in preview")
        
        do_grayscale = grayscale.lower() == "true"

        # 3. Apply Watermarks
        if not zone_list:
            # Global preview
            watermarked = apply_visible_watermark(img, message.strip(), thickness=thickness, intensity=intensity)
        else:
            # Apply only zones matching this page index (if provided) or all zones if no page index
            watermarked = img.copy()
            for zone in zone_list:
                z_page = zone.get("page_index", 0)
                if z_page == page_index:
                    watermarked = apply_visible_watermark(watermarked, message.strip(), roi=zone, thickness=thickness, intensity=intensity)

        # 4. Optional Grayscale
        if do_grayscale:
            watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
            watermarked = cv2.cvtColor(watermarked, cv2.COLOR_GRAY2BGR) # Back to BGR for JPEG export

        # 5. Resize for preview speed (max dimension 1024)
        h, w = watermarked.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            watermarked = cv2.resize(watermarked, (int(w * scale), int(h * scale)))

        # 5. Encode as JPEG
        _, buffer = cv2.imencode(".jpg", watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Preview Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
