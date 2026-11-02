---
title: Khmer National ID Card Reader
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Khmer National ID Card Reader

Upload a photo of a Cambodian national ID card, adjust the corner frame, and extract all fields using OCR.

Features:
- Auto card detection with manual corner adjustment
- Khmer + Latin text recognition
- MRZ parsing with check-digit verification
- Name cross-verification

📦 `pip install khmer-id-reader`
