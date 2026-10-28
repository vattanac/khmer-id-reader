"""Public API — the only file users need to know about."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .model_hub import ensure_models


@dataclass
class IDCardResult:
    """Structured result from reading one Cambodian national ID card."""
    name_khmer: str = ""
    name_latin: str = ""
    dob: str = ""
    sex: str = ""
    height: str = ""
    birthplace: str = ""
    address: str = ""
    id_number: str = ""
    validity: str = ""
    distinguishing_marks: str = ""
    nationality: str = ""
    mrz_verified: bool = False
    mrz_line1: str = ""
    mrz_line2: str = ""
    mrz_line3: str = ""
    name_verified: bool = False
    confidence: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self):
        lines = []
        for k, v in self.to_dict().items():
            if v and k not in ("confidence", "warnings"):
                lines.append(f"  {k}: {v}")
        return "IDCardResult(\n" + "\n".join(lines) + "\n)"


_FIELDS_JSON = Path(__file__).parent / "fields.json"

# Cached models so repeat calls don't reload
_loaded = {}


def _get_models(use_lm: bool):
    """Load models once and cache them."""
    if "model" not in _loaded:
        cache = ensure_models()
        from .ocr_engine import load_model, load_lm
        from .reader import load_latin_model

        model, itos, device = load_model(str(cache / "khmer_ocr_document.pt"))
        _loaded["model"] = model
        _loaded["itos"] = itos
        _loaded["device"] = device
        _loaded["load_lm"] = load_lm
        _loaded["cache"] = cache
        _loaded["latin"] = load_latin_model(load_model)

    if use_lm and "lm" not in _loaded:
        _loaded["lm"] = _loaded["load_lm"](
            str(_loaded["cache"] / "khmer_lm.pkl.gz"))

    return _loaded


def read_card(
    image_path: str,
    corners: Optional[list] = None,
    use_lm: bool = True,
    fields_path: Optional[str] = None,
) -> IDCardResult:
    """
    Read a Cambodian national ID card from a photo.

    Args:
        image_path:  Path to the card photo (JPG/PNG).
        corners:     Optional 4 corner points [(x,y), ...] for manual crop.
                     If None, auto-detection is attempted.
        use_lm:      Use language model for beam decoding (default True).
        fields_path: Custom fields.json path. If None, uses the bundled default.

    Returns:
        IDCardResult with all extracted fields.

    Example:
        >>> from khmer_id_reader import read_card
        >>> result = read_card("my_id.jpg")
        >>> print(result.name_latin)
        'SIM VATTANAC'
    """
    from .reader import warp_card, read_all

    m = _get_models(use_lm)

    # Flatten the card
    src = Path(image_path).expanduser()
    card, found = warp_card(src)
    if not found:
        pass  # uses whole photo as fallback

    # Load field boxes
    fp = Path(fields_path) if fields_path else _FIELDS_JSON
    boxes = json.loads(fp.read_text())

    # Read all fields
    res = read_all(
        card, boxes,
        m["model"], m["itos"], m["device"],
        lm=m.get("lm") if use_lm else None,
        latin=m["latin"],
    )

    fields = res.get("fields", {})
    mrz = res.get("mrz", {})
    raw = mrz.get("raw_lines", ["", "", ""])

    # Parse dob_sex_height into separate fields
    dsh = fields.get("dob_sex_height", "")

    return IDCardResult(
        name_khmer=fields.get("name_khmer", ""),
        name_latin=fields.get("name_latin", ""),
        dob=dsh,
        sex="",
        height="",
        birthplace=fields.get("place_of_birth", ""),
        address=res.get("address", ""),
        id_number=fields.get("id_number", ""),
        validity=fields.get("validity", ""),
        distinguishing_marks=fields.get("distinguishing_marks", ""),
        nationality=mrz.get("nationality", ""),
        mrz_verified=res.get("mrz_confidence", 0) >= 8,
        mrz_line1=raw[0] if len(raw) > 0 else "",
        mrz_line2=raw[1] if len(raw) > 1 else "",
        mrz_line3=raw[2] if len(raw) > 2 else "",
        name_verified=res.get("name_verified", False),
        confidence={"mrz_score": res.get("mrz_confidence", 0),
                    "seconds": res.get("seconds", 0)},
    )
