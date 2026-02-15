# Project Titan: Technical Specifications

## 1. Overview
Titan is an adversarial visual watermarking engine designed to protect documents against automated AI-based removal (inpainting and generative erasure). Unlike traditional watermarks that sit on a separate layer, Titan integrates itself into the document's structure and luminance.

## 2. Core Features
### 2.1 Titan Wave Engine (Structural Modulation)
- **Fluid Interference Patches**: Generates noise-modulated sine waves that cover the document.
- **Phase Shift Distortion**: The text stencil violently shifts the phase of the waves. If the ink is removed, the structural distortion in the waves still "records" the watermark's presence.

### 2.2 Analog Ink Stamp Effect
- **Blooming (Bleed)**: Simulates physical ink absorption by dilating and blurring the text mask against the document's luminance.
- **Organic Voids**: Applies high-frequency noise and saturation irregularities to simulate uneven ink coverage.
- **Paper Warp (Displacement)**: Micro-distorts document pixels at the text edges to simulate physical wetting of paper fibers.

### 2.3 Luminance-Anchored Blending
- **Shadow-Based Inking**: Instead of solid black, the ink is a relative multiplier of the local document luminance.
- **Chroma Matching**: Samples the document's hue and samples it into the ink's tint to ensure color-space consistency.
- **Sensor Grain Matching**: Injects high-frequency ISO-like noise to match the source document's sensor signature.

## 3. Technical Requirements
- **Backend**: FastAPI (Python 3.12)
- **Processing**: OpenCV with NumPy and Pillow.
- **PDF Core**: `pdf2image` (Poppler) for high-fidelity conversion.
- **Infrastructure**: Docker & Docker Compose for isolation.

## 4. UI/UX
- Responsive Dark Mode interface built with Vanilla CSS.
- Selective ROI (Region of Interest) support with multi-zone management.
- Live-ish Preview for configuration validation.
