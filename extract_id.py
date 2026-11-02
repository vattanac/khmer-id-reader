#!/usr/bin/env python3
"""
extract_id.py - photo of a Cambodian national ID in, fields out. No user steps.

    find the card      text-density detector (works on a same-colour surface),
                       with a contrast-quadrilateral fallback for dark backgrounds
    flatten            crop or perspective-warp to 1200x757 (ID-1 proportions)
    orient             try all four rotations, read the MRZ in each, keep the one
                       whose check digits verify: the card says which way up it is
    read               MRZ with constrained decoding and check-digit repair, then
                       the Khmer fields from the fixed template (fields.json)

Also usable from the command line:

    python3 extract_id.py --image photo.jpg [--lm] [--fields fields.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_id_card as rid                                  # noqa: E402

CARD_W, CARD_H = rid.CARD_W, rid.CARD_H

# default template for the current Cambodian ID design, in percent of the card
# The template boxes live in fields.json so there is ONE source of truth.
#
# They used to be duplicated here as a literal, and the copies drifted: every
# box fix went into fields.json, which score_cards.py reads, while the server
# fell back to this dict, so the browser app ran on the old geometry. The only
# visible symptom was a truncated Latin name, which the MRZ then papered over.
# The literal below is the fallback for when fields.json is missing.
_FALLBACK_FIELDS = {
    "id_number":            [65.5,  4.0, 97.0, 11.4],
    "name_khmer":           [22.4,  8.8, 66.0, 17.0],
    "name_latin":           [36.0, 17.0, 85.0, 21.0],
    "dob_sex_height":       [22.4, 21.9, 99.3, 28.0],
    "place_of_birth":       [22.4, 27.4, 84.0, 34.9],
    "address_line1":        [22.4, 34.3, 64.0, 41.8],
    "address_line2":        [22.4, 42.5, 68.0, 50.1],
    "validity":             [22.6, 51.6, 82.0, 55.75],
    "distinguishing_marks": [22.6, 55.75, 88.0, 62.9],
}


def _load_fields() -> dict:
    import json as _json
    from pathlib import Path as _Path
    f = _Path(__file__).resolve().parent / "fields.json"
    try:
        boxes = _json.loads(f.read_text(encoding="utf-8"))
        if isinstance(boxes, dict) and boxes:
            return boxes
    except (OSError, ValueError):
        pass
    return dict(_FALLBACK_FIELDS)


DEFAULT_FIELDS = _load_fields()
MRZ_TOP = 62.0


# ------------------------------------------------------------ detection ----

def find_card(pil):
    """Return (PIL card image in landscape, method). Never raises."""
    import cv2
    import numpy as np
    from PIL import ImageOps

    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.asarray(pil)
    h, w = rgb.shape[:2]
    s = 480.0 / max(h, w)
    small = cv2.resize(rgb, (int(w * s), int(h * s)))
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    H, W = g.shape

    # ---- method A: where the text is
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    gx = np.abs(cv2.Sobel(gb, cv2.CV_32F, 1, 0, 3))
    ink = (gx > 90).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.05) | 1, int(H * 0.05) | 1))
    blob = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blob)
    box = None
    if n > 1:
        x, y, bw, bh, _ = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
        if bw >= 0.25 * W and bh >= 0.25 * H:
            gb2 = cv2.GaussianBlur(g, (7, 7), 0)
            GX = np.abs(cv2.Sobel(gb2, cv2.CV_32F, 1, 0, 3))
            GY = np.abs(cv2.Sobel(gb2, cv2.CV_32F, 0, 1, 3))
            colp = GX[y:y + bh].mean(0)
            rowp = GY[:, x:x + bw].mean(1)

            def snap(prof, start, d, limit, margin):
                if d < 0:
                    lo, hi = max(0, start - limit), max(0, start - margin)
                else:
                    lo, hi = min(len(prof) - 1, start + margin), min(len(prof) - 1, start + limit)
                if hi <= lo:
                    return start
                seg = prof[lo:hi + 1]
                i = int(np.argmax(seg))
                if seg[i] > 2.5 * np.median(prof):
                    return lo + i
                return start - margin if d < 0 else start + margin

            L = snap(colp, x, -1, int(W * 0.18), int(W * 0.02))
            R = snap(colp, x + bw, +1, int(W * 0.18), int(W * 0.02))
            T = snap(rowp, y, -1, int(H * 0.18), int(H * 0.02))
            B = snap(rowp, y + bh, +1, int(H * 0.18), int(H * 0.02))
            asp = max(R - L, B - T) / max(1, min(R - L, B - T))
            if 1.2 <= asp <= 2.2:
                box = (L / s, T / s, R / s, B / s)
                method = "text"

    # ---- method B: contrast quadrilateral
    if box is None:
        grey = cv2.bilateralFilter(g, 9, 40, 40)
        edges = cv2.dilate(cv2.Canny(grey, 40, 140), np.ones((3, 3), np.uint8), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(ap) == 4 and cv2.contourArea(ap) > 0.2 * W * H:
                q = ap.reshape(4, 2).astype("float32") / s
                ssum, d = q.sum(1), np.diff(q, axis=1).ravel()
                src = np.array([q[np.argmin(ssum)], q[np.argmin(d)],
                                q[np.argmax(ssum)], q[np.argmax(d)]], dtype="float32")
                # keep landscape: map to CARD_W x CARD_H, orientation fixed later
                dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
                               dtype="float32")
                M = cv2.getPerspectiveTransform(src, dst)
                flat = cv2.warpPerspective(rgb, M, (CARD_W, CARD_H))
                from PIL import Image
                return Image.fromarray(flat), "contrast"

    if box is None:
        return pil, "none"

    # take the coarse box generously (8% each side): the refinement below needs
    # the whole card inside the crop, and background around it costs nothing
    global _last_box
    _last_box = box
    L, T, R, B = box
    mx, my = 0.08 * (R - L), 0.08 * (B - T)
    crop = pil.crop((int(max(0, L - mx)), int(max(0, T - my)),
                     int(min(w, R + mx)), int(min(h, B + my))))
    if crop.height > crop.width:                        # card lying vertically
        crop = crop.rotate(90, expand=True)             # direction fixed by MRZ later
    return crop, method


def locate_card_any(pil, detector: str = "auto") -> dict:
    """Card corners, from whichever detector finds something card-shaped.

    'contour' is the scanner-app approach and gives true corners, so a tilted
    card is straightened rather than boxed. 'text' is the density detector,
    which is what works when the card has almost no edge contrast against the
    surface it lies on. 'auto' prefers the contour when it finds a quadrilateral
    of the right shape and falls back to the text path."""
    if detector in ("contour", "auto"):
        try:
            q = contour_quad(pil)
        except Exception:
            q = None
        if q is not None:
            from PIL import ImageOps
            w, h = ImageOps.exif_transpose(pil).size
            return {"quad": q, "method": "contour", "width": w, "height": h}
        if detector == "contour":
            from PIL import ImageOps
            w, h = ImageOps.exif_transpose(pil).size
            return {"quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    "method": "none", "width": w, "height": h}
    return locate_card(pil)


def locate_card(pil) -> dict:
    """Where is the card in this photo? Returns the four corners in photo
    pixels (top-left, top-right, bottom-right, bottom-left of the *photo*
    region; orientation of the card itself is settled later by the MRZ)."""
    import numpy as np
    from PIL import ImageOps

    pil = ImageOps.exif_transpose(pil).convert("RGB")
    w, h = pil.size
    crop, method = find_card(pil)
    if method == "contrast":
        # find_card already warped; recover the quad by re-running its detector
        # is wasteful, so we return the full frame and let the user adjust
        quad = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        return {"quad": quad, "method": method, "width": w, "height": h}
    if method == "none":
        quad = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        return {"quad": quad, "method": method, "width": w, "height": h}

    # text path: crop is a generous box (possibly rotated 90 deg CCW); refine it
    # on the MRZ in either orientation, then map the box back to photo pixels
    L, T, R, B = _last_box
    mx, my = 0.08 * (R - L), 0.08 * (B - T)
    ox, oy = int(max(0, L - mx)), int(max(0, T - my))
    ex_, ey_ = int(min(w, R + mx)), int(min(h, B + my))
    cw, ch = ex_ - ox, ey_ - oy
    rotated = ch > cw                                   # find_card rotated it

    box = None
    for deg in (0, 180):
        cand = crop if deg == 0 else crop.rotate(180)
        geo = _mrz_geometry(cand)
        if geo is None:
            continue
        top, bottom, left, right = geo
        th = (bottom - top) / (MRZ_BOTTOM_FRAC - MRZ_TOP_FRAC)
        tw = (right - left) / (MRZ_RIGHT_FRAC - MRZ_LEFT_FRAC)
        if not (1.35 <= tw / max(1, th) <= 1.85):
            continue
        x0, y0 = left - MRZ_LEFT_FRAC * tw, top - MRZ_TOP_FRAC * th
        bx = (x0, y0, x0 + tw, y0 + th)
        if deg == 180:                                  # undo the 180 for mapping
            W_, H_ = cand.size
            bx = (W_ - bx[2], H_ - bx[3], W_ - bx[0], H_ - bx[1])
        box = bx
        break

    refined = box is not None
    if box is None:                                     # fall back to the coarse box
        box = (mx, my, cw - mx, ch - my) if not rotated else (my, mx, ch - my, cw - mx)

    x0, y0, x1, y1 = box
    if rotated:
        # crop was rotated 90 deg CCW with expand: new (x', y') <- old (x, y) by
        # x' = y, y' = (cw - 1) - x. Invert to get back to the unrotated crop.
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        pts = [((cw - 1) - py, px) for (px, py) in pts]
    else:
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    quad = [[float(np.clip(px + ox, 0, w - 1)), float(np.clip(py + oy, 0, h - 1))] for px, py in pts]
    # order tl, tr, br, bl in photo space
    quad.sort(key=lambda p: p[0] + p[1])
    tl = quad[0]; br = quad[-1]
    rest = sorted(quad[1:3], key=lambda p: p[0] - p[1])
    bl, tr = rest[0], rest[1]
    return {"quad": [tl, tr, br, bl], "method": method + ("+mrz" if refined else ""),
            "width": w, "height": h,
            "sharpness": (sh := sharpness(pil.crop((int(tl[0]), int(tl[1]), int(br[0]), int(br[1]))))),
            "sharpness_verdict": sharpness_verdict(sh)}


# sharpness of the card region, as variance of the Laplacian at a fixed size.
# Measured on the test card with increasing blur (blur_test.py):
#   >= 160  all four MRZ check-digit fields verify, Khmer fields exact
#   ~ 65    one MRZ field lost;  ~ 30  two lost;  ~ 10  Khmer starts to fail
SHARP_OK, SHARP_LOW = 120.0, 40.0


def sharpness(card) -> float:
    import cv2
    import numpy as np
    if card.width < 10 or card.height < 10:
        return 0.0
    if card.height > card.width:                    # measure the card landscape, like blur_test
        card = card.rotate(90, expand=True)
    g = np.asarray(card.convert("L").resize((CARD_W, CARD_H)))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def sharpness_verdict(s: float) -> str:
    return "ok" if s >= SHARP_OK else ("low" if s >= SHARP_LOW else "retake")


_last_box = None


# --------------------------------------------------------- MRZ anchoring ----

# where the MRZ sits on a correctly flattened card, measured once
MRZ_TOP_FRAC, MRZ_BOTTOM_FRAC = 0.666, 0.886     # top of line 1, bottom of line 3
MRZ_LEFT_FRAC, MRZ_RIGHT_FRAC = 0.041, 0.953     # stroke extent, identical on all 3 lines


def _mrz_geometry(card):
    """Find the three MRZ lines in a roughly-cropped card. Returns
    (top, bottom, left, right) in pixels of `card`, or None."""
    import numpy as np
    from PIL import ImageOps

    # measure at a fixed scale so every threshold below means the same thing
    scale = card.width / 1200.0
    work = card.resize((1200, max(1, int(round(card.height / scale)))))
    g = np.asarray(ImageOps.autocontrast(work.convert("L")))
    H, W = g.shape
    y0 = int(H * 0.40)
    dark = g[y0:] < 110
    rows = dark.sum(1)
    floor = max(3, int(0.03 * W))
    bands, start = [], None
    for i, r in enumerate(rows):
        if r > floor and start is None:
            start = i
        elif r <= floor and start is not None:
            if i - start >= 6:
                bands.append((start, i))
            start = None
    if start is not None and len(rows) - start >= 6:
        bands.append((start, len(rows)))

    # MRZ lines are the ones that run nearly the full width of the card.
    # Measure a band's extent only over columns that are dark through at least
    # a quarter of its height: glyph strokes are, the card's thin rim is not.
    cands = []
    for a, b in bands:
        colsum = dark[a:b].sum(0)
        cols = np.where(colsum >= 0.25 * (b - a))[0]
        if len(cols) < 20:
            continue
        lo, hi = np.percentile(cols, [1, 99])          # trim stray columns
        ext = (hi - lo) / W
        density = len(cols) / max(1.0, hi - lo)        # text ~0.4-0.6; a solid line ~1.0
        if ext >= 0.55 and 0.2 <= density <= 0.85 and (b - a) >= 0.02 * H:
            cands.append((a, b, int(lo), int(hi), ext))
    if len(cands) < 3:
        return None

    # The MRZ is three consecutive lines of equal height, equal width and equal
    # spacing. Score every consecutive triple on that and take the most regular
    # one; on a tie prefer the lowest, since the MRZ sits at the foot of the card.
    cands.sort()
    best, best_score = None, None
    for i in range(len(cands) - 2):
        t = cands[i:i + 3]
        hs = [c[1] - c[0] for c in t]
        gaps = [t[1][0] - t[0][0], t[2][0] - t[1][0]]
        exts = [c[4] for c in t]
        if max(hs) > 1.4 * min(hs) or max(gaps) > 1.4 * min(gaps):
            continue
        if max(exts) - min(exts) > 0.08:
            continue
        if min(gaps) < 1.05 * max(hs):                  # lines must not overlap
            continue
        score = (max(hs) / min(hs) - 1) + (max(gaps) / min(gaps) - 1) - 0.001 * t[0][0]
        if best is None or score < best_score:
            best, best_score = t, score
    if best is None:
        return None
    top, bottom = best[0][0] + y0, best[-1][1] + y0
    left = int(np.median([c[2] for c in best]))
    right = int(np.median([c[3] for c in best]))
    return (int(top * scale), int(bottom * scale), int(left * scale), int(right * scale))


def refine_by_mrz(card):
    """Re-crop `card` so the MRZ lands exactly where it does on the reference
    card; this fixes scale and offset in both axes at once. Falls back to a
    plain resize when the MRZ cannot be found."""
    from PIL import Image

    geo = _mrz_geometry(card)
    if geo is None:
        return card.resize((CARD_W, CARD_H))
    top, bottom, left, right = geo
    true_h = (bottom - top) / (MRZ_BOTTOM_FRAC - MRZ_TOP_FRAC)
    true_w = (right - left) / (MRZ_RIGHT_FRAC - MRZ_LEFT_FRAC)
    # sanity: the two estimates must agree with ID-1 proportions
    if not (1.35 <= true_w / max(1, true_h) <= 1.85):
        return card.resize((CARD_W, CARD_H))
    y_top = top - MRZ_TOP_FRAC * true_h
    x_left = left - MRZ_LEFT_FRAC * true_w
    box = (x_left, y_top, x_left + true_w, y_top + true_h)

    # pad with the card's own margin colour if the box runs outside the crop
    pad = int(max(0, -box[0], -box[1], box[2] - card.width, box[3] - card.height)) + 1
    if pad > 1:
        bg = card.resize((1, 1)).getpixel((0, 0))
        padded = Image.new("RGB", (card.width + 2 * pad, card.height + 2 * pad), bg)
        padded.paste(card, (pad, pad))
        card = padded
        box = tuple(v + pad for v in box)
    return card.crop(tuple(int(round(v)) for v in box)).resize((CARD_W, CARD_H))


# ---------------------------------------------------------- orientation ----

def _mrz_score(info: dict) -> int:
    s = 0
    for k in ("document_number", "date_of_birth", "date_of_expiry"):
        s += 2 if info[k][1] else 0
    s += 1 if info["sex"][1] else 0
    s += 3 if info["composite_ok"] else 0
    s += 1 if info["issuing_state"] == "KHM" else 0
    return s


def read_mrz(card, model, itos, device) -> tuple[dict, list]:
    bands = rid.mrz_bands(card, MRZ_TOP)
    raw = [rid.constrained_read(b, model, itos, device, rid.MRZ_CHARS) for b in bands]
    while len(raw) < 3:
        raw.append("")
    return rid.parse_td1(*raw[:3]), raw[:3]


def orient(card, model, itos, device):
    """Try 0 and 180 degrees (card is already landscape). Each candidate is
    first re-cropped on its MRZ; the right way up is the one whose MRZ then
    verifies. An upside-down card has no MRZ at the bottom, so refinement
    falls back to a plain resize and the check digits fail."""
    best = None
    for deg in (0, 180):
        cand = refine_by_mrz(card if deg == 0 else card.rotate(180))
        info, raw = read_mrz(cand, model, itos, device)
        sc = _mrz_score(info)
        if best is None or sc > best[0]:
            best = (sc, deg, cand, info, raw)
        if sc >= 10:                                    # everything verified; stop early
            break
    return best[2], best[1], best[3], best[4], best[0]


# -------------------------------------------------------------- extract ----

def extract(pil, model, itos, device, lm=None, fields: dict | None = None) -> dict:
    import time
    t0 = time.time()
    from khmer_ocr import read_line

    card, method = find_card(pil)
    card, deg, info, raw, score = orient(card, model, itos, device)

    res = rid.read_all(card, fields or DEFAULT_FIELDS, model, itos, device, lm, MRZ_TOP)
    # the Latin name is unreliable from this model; prefer the MRZ when verified
    res.update({
        "card_found_by": method,
        "rotation_applied": deg,
        "seconds": round(time.time() - t0, 2),
        "_card": card,                                  # PIL image, for the UI
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--lm", action="store_true")
    ap.add_argument("--fields", help="fields.json (default: built-in Cambodian template)")
    ap.add_argument("--save-card", help="write the flattened, oriented card here")
    args = ap.parse_args()

    files = ["khmer_ocr.py", "vocab.json", "khmer_ocr_document.pt"] + (["khmer_lm.pkl.gz"] if args.lm else [])
    paths = rid.fetch(files)
    sys.path.insert(0, str(rid.CACHE))
    from khmer_ocr import load_model, load_lm
    model, itos, device = load_model(str(paths["khmer_ocr_document.pt"]))
    lm = load_lm(str(paths["khmer_lm.pkl.gz"])) if args.lm else None

    from PIL import Image
    fields = json.loads(Path(args.fields).read_text()) if args.fields else None
    res = extract(Image.open(args.image), model, itos, device, lm, fields)
    card = res.pop("_card")
    if args.save_card:
        card.save(args.save_card)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Minimum source resolution. Measured: a card photographed at 543x368 gives
# about 5 px of stroke height in the MRZ, and both engines insert characters
# there - card 4 read 'IDKHMO0508365724' for 'IDKHM0508365724'. No model fixes
# missing pixels, so the honest answer is to ask for a closer photo.
MIN_SOURCE_W, MIN_SOURCE_H = 900, 560


def quality(card, source_size=None) -> dict:
    """Is this image good enough to read? Returns a verdict plus the numbers."""
    sw, sh = source_size or card.size
    s = sharpness(card)
    reasons = []
    if sw < MIN_SOURCE_W or sh < MIN_SOURCE_H:
        reasons.append(f"image is {sw}x{sh}; needs at least "
                       f"{MIN_SOURCE_W}x{MIN_SOURCE_H}. Move closer or use a larger photo.")
    v = sharpness_verdict(s)
    if v == "retake":
        reasons.append(f"too blurred (sharpness {s:.0f}, needs {SHARP_LOW:.0f}). Hold steady and retake.")
    elif v == "low":
        reasons.append(f"soft focus (sharpness {s:.0f}, ideal {SHARP_OK:.0f}). Fields may be misread.")
    return {"source_width": sw, "source_height": sh,
            "sharpness": round(s, 1), "sharpness_verdict": v,
            "ok": not any(r for r in reasons if "at least" in r or "too blurred" in r),
            "reasons": reasons}


# ------------------------------------------------- contour card detection ----

CARD_RATIO = CARD_W / CARD_H            # 1.585 for ID-1


def contour_quad(pil, debug: bool = False):
    """Find the card as the largest four-sided contour, the way a scanner app
    does: downscale, blur, edges, contours, approximate to a polygon.

    This is the complement of the text-density detector. That one was written
    because the first test card lay on ruled paper with almost no edge
    contrast, and it still wins there. This one wins whenever the card sits on
    a plainly different background, which is the common case - and it gives
    real corners rather than a box, so a tilted card is corrected properly.

    Returns the four corners in photo pixels (clockwise from the top-left of
    the photo), or None when nothing card-shaped is found."""
    import cv2
    import numpy as np
    from PIL import ImageOps

    pil = ImageOps.exif_transpose(pil).convert("RGB")
    W, H = pil.size
    k = 900.0 / max(W, H)
    small = np.asarray(pil.resize((max(1, int(W * k)), max(1, int(H * k)))))
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 9, 60, 60)        # smooth the guilloche, keep edges

    best, best_area = None, 0.0
    frame_area = small.shape[0] * small.shape[1]
    # several thresholds: one edge setting does not suit every lighting
    for lo, hi in ((30, 90), (50, 150), (75, 200)):
        edges = cv2.Canny(g, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.10 * frame_area or area <= best_area:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.04):
                ap = cv2.approxPolyDP(c, eps * peri, True)
                if len(ap) != 4 or not cv2.isContourConvex(ap):
                    continue
                q = ap.reshape(4, 2).astype("float32")
                # a card is a card only at the right shape: compare the two
                # side lengths against ID-1's 1.585, allowing for perspective
                d = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
                long_, short = max(d[0], d[1]), min(d[0], d[1])
                if short < 1 or not (1.25 < long_ / short < 2.0):
                    continue
                best, best_area = q, area
                break
    if best is None:
        return None
    # order: top-left, top-right, bottom-right, bottom-left of the photo
    s_ = best.sum(axis=1)
    d_ = np.diff(best, axis=1).ravel()
    quad = np.array([best[np.argmin(s_)], best[np.argmin(d_)],
                     best[np.argmax(s_)], best[np.argmax(d_)]], dtype="float32")
    quad = quad / k

    # A four-sided shape of roughly the right proportions is not yet a card.
    # On a screenshot of a document viewer the largest such shape is the white
    # page, and it passes every geometric test. Warp the candidate and require
    # the three MRZ lines to be where a card keeps them: that is the one thing
    # a page does not have.
    dst = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
    warp = cv2.warpPerspective(np.asarray(pil),
                               cv2.getPerspectiveTransform(quad.astype("float32"), dst),
                               (CARD_W, CARD_H))
    gg = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    band = gg[int(0.66 * CARD_H):int(0.90 * CARD_H), int(0.04 * CARD_W):int(0.96 * CARD_W)]
    thr = float(np.percentile(band, 85)) - 60
    row = (band < thr).sum(axis=1)
    need = band.shape[1] * 0.12
    lines, start = 0, None
    for y, v in enumerate(row):
        if v > need and start is None:
            start = y
        elif v <= need and start is not None:
            if y - start >= 4:
                lines += 1
            start = None
    if start is not None and len(row) - start >= 4:
        lines += 1
    if debug:
        print(f"contour_quad: mrz lines {lines}")
    if lines < 3:
        return None
    return quad.tolist()
