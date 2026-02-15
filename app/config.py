"""
Document Fortress — Configuration Module
All settings are driven by environment variables for Docker compatibility.
"""

import os
from pathlib import Path


class FortressConfig:
    """Central configuration for Document Fortress."""

    # --- Paths ---
    DATA_DIR: Path = Path(os.getenv("FORTRESS_DATA_DIR", "./data"))
    KEYS_DIR: Path = Path(os.getenv("FORTRESS_KEYS_DIR", "./data/keys"))
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # --- Crypto ---
    CERT_PATH: Path = KEYS_DIR / "fortress.crt"
    KEY_PATH: Path = KEYS_DIR / "fortress.key"

    # --- Visible Watermark (Titan Engine) ---
    # Opacity of the interference pattern (0.0 to 1.0)
    GUILLOCHE_OPACITY: float = float(os.getenv("FORTRESS_GUILLOCHE_OPACITY", "0.55"))
    
    # Complexity: Number of sine waves/spirograph nodes (3-10 recommended)
    GUILLOCHE_COMPLEXITY: int = int(os.getenv("FORTRESS_GUILLOCHE_COMPLEXITY", "5"))
    
    # Fusion Mode: 'DIFFERENCE', 'EXCLUSION', 'HARD_LIGHT', 'OVERLAY'
    # 'DIFFERENCE' is often the most secure against AI-based removal.
    FUSION_MODE: str = os.getenv("FORTRESS_FUSION_MODE", "DIFFERENCE")

    # Legacy Params (kept for text generation)
    OPACITY_MIN: float = float(os.getenv("FORTRESS_OPACITY_MIN", "0.12"))
    OPACITY_MAX: float = float(os.getenv("FORTRESS_OPACITY_MAX", "0.35"))
    FONT_SIZE_MIN: int = int(os.getenv("FORTRESS_FONT_SIZE_MIN", "28"))
    FONT_SIZE_MAX: int = int(os.getenv("FORTRESS_FONT_SIZE_MAX", "48"))
    MICROTEXT_FONT_SIZE: int = 6
    GRID_SPACING_CM: float = 3.0

    # --- Adversarial ---
    ADVERSARIAL_SIGMA: float = float(os.getenv("FORTRESS_ADVERSARIAL_SIGMA", "4.0"))
    PERLIN_SCALE: float = float(os.getenv("FORTRESS_PERLIN_SCALE", "100.0"))
    PERLIN_OCTAVES: int = int(os.getenv("FORTRESS_PERLIN_OCTAVES", "6"))

    # --- Invisible Watermark ---
    DCT_BLOCK_SIZE: int = 8
    DCT_ALPHA: float = float(os.getenv("FORTRESS_DCT_ALPHA", "0.08"))
    REED_SOLOMON_NSYM: int = 64  # Error correction symbols

    # --- PDF ---
    DPI: int = int(os.getenv("FORTRESS_DPI", "300"))
    PDF_QUALITY: int = 95  # JPEG quality for PDF reconstruction

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("FORTRESS_LOG_LEVEL", "info").upper()

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        for d in [cls.DATA_DIR, cls.KEYS_DIR, cls.UPLOADS_DIR, cls.PROCESSED_DIR]:
            d.mkdir(parents=True, exist_ok=True)


config = FortressConfig()
