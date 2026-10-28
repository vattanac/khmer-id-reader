#!/usr/bin/env python3
"""
flatten_nid.py - warp a photographed ID card to the canonical 1200x757 crop.

The template boxes in extract_id.py / fields.json are all in percentages of a
card that fills the frame exactly. If the card does not fill the frame, every
box lands on the wrong pixels and the output is garbage that looks like a
model problem but is not. This script produces that canonical crop.

Corners are given as four "x,y" pairs in pixels of the EXIF-corrected image,
in the order they appear going clockwise from the card's own top-left (the
corner nearest the Khmer name line), regardless of how the photo is rotated.

    python3 flatten_nid.py --image my_NID.JPG --out card_flat.png \
        --corners 2452,378 2462,3322 570,3337 560,392

Leave --corners off and it reuses the saved set in flatten_nid.json, so a
re-flatten of the same photo is one command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

CARD_W, CARD_H = 1200, 757
STORE = Path(__file__).resolve().parent / "flatten_nid.json"


def load_upright(path: str) -> np.ndarray:
    """cv2.imread ignores EXIF orientation; PIL does not. Phone photos carry it."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def flatten(img: np.ndarray, corners: list[tuple[float, float]]) -> np.ndarray:
    src = np.float32(corners)
    dst = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                               (CARD_W, CARD_H))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="card_flat.png")
    ap.add_argument("--corners", nargs=4, default=None,
                    metavar="X,Y", help="four corners, clockwise from the card's top-left")
    args = ap.parse_args()

    saved = json.loads(STORE.read_text()) if STORE.exists() else {}
    key = Path(args.image).name

    if args.corners:
        corners = [tuple(float(v) for v in c.split(",")) for c in args.corners]
        saved[key] = corners
        STORE.write_text(json.dumps(saved, indent=2))
    elif key in saved:
        corners = [tuple(c) for c in saved[key]]
        print(f"reusing saved corners for {key}")
    else:
        raise SystemExit(f"no saved corners for {key}; pass --corners")

    img = load_upright(args.image)
    print(f"source {img.shape[1]}x{img.shape[0]} (EXIF applied)")
    out = flatten(img, corners)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}  {CARD_W}x{CARD_H}")
    print("read it with:  python3 read_id_card.py --image "
          f"{args.out} --flat --lm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
