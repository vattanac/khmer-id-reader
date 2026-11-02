# 🇰🇭 Khmer National ID Card Reader

OCR-powered reader for Cambodian national ID cards. Upload a photo, adjust the card frame, and extract all fields including MRZ.

## Quick Start

```bash
docker run -p 8765:8765 ghcr.io/vattanac/khmer-id-reader:latest
```

Then open [http://localhost:8765](http://localhost:8765)

| Page | URL | Description |
|------|-----|-------------|
| Desktop | `http://localhost:8765/` | Upload UI with corner-drag canvas |
| Mobile | `http://localhost:8765/m` | Phone-friendly camera UI |

## Features

- **Frontend**: Desktop and mobile web UI with canvas-based 4-corner adjustment
- **Backend**: Python HTTP server with CRNN-CTC OCR engine
- Auto card edge detection with perspective warp
- Beam search with Khmer language model
- MRZ reading with check-digit verification
- Khmer and Latin field extraction
- Works offline — model weights bundled in the image

## Links

- [Model on HuggingFace](https://huggingface.co/vattanac/khmer-ocr-crnn-ctc)
- [GitHub](https://github.com/vattanac/khmer-id-reader)
