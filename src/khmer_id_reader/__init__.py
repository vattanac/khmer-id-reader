"""
khmer-id-reader — Free Cambodian national ID card OCR library.

Usage:
    from khmer_id_reader import read_card
    result = read_card("photo.jpg")
    print(result)
"""

__version__ = "0.1.0"

from .api import read_card, IDCardResult

__all__ = ["read_card", "IDCardResult"]
