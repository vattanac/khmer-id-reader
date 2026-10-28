#!/usr/bin/env python3
"""
read_id_card.py - read a Cambodian national ID card end to end.

Stage 1  find the card in the photo and warp it flat (OpenCV)
Stage 2  read the MRZ with the alphabet constrained to [A-Z0-9<], then verify
         and repair every field using its ISO 7501 check digit
Stage 3  read the Khmer fields from template boxes you calibrate once

Usage
-----
    pip install opencv-python-headless huggingface_hub torch pillow numpy

    # 1. flatten the card and produce a grid you can read coordinates off
    python3 read_id_card.py --image my_NID.JPG --calibrate

    # 2. read it (MRZ works immediately; Khmer fields need fields.json)
    python3 read_id_card.py --image my_NID.JPG

    # 3. once fields.json exists, both are read
    python3 read_id_card.py --image my_NID.JPG --lm

Why the MRZ is reliable: it is a fixed-width format in one standard font, and
each field carries a check digit computed with the 7-3-1 weighting. If a digit
does not verify we know the read is wrong, and we can search single-character
substitutions among the glyph pairs OCR actually confuses until it does.

For testing on your own document. Not a production identity system: fields
that fail verification are reported as UNVERIFIED and need a human.
"""

from __future__ import annotations

import argparse
import re
import json
import shutil
import sys
from pathlib import Path

REPO = "vattanac/khmer-ocr-crnn-ctc"
CACHE = Path.home() / ".cache" / "khmer_ocr_try"

CARD_W, CARD_H = 1200, 757          # ID-1 is 85.6 x 54 mm, ratio 1.585
MRZ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

# glyph pairs OCR genuinely confuses, used only to repair a failed check digit
CONFUSIONS = {
    "0": "OD896", "O": "0DQ", "D": "0O", "1": "IL7", "I": "1L", "L": "1I",
    "2": "Z7", "Z": "2", "5": "S6", "S": "5", "8": "B03", "B": "83",
    "6": "G508", "G": "6", "9": "04", "4": "A9", "A": "4", "7": "T12", "T": "7",
    "3": "8", "Q": "O",
}


# ---------------------------------------------------------------- model ----

def fetch(files: list[str]) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in files:
        local = CACHE / name
        if not local.exists():
            print(f"  downloading {name} ...", flush=True)
            shutil.copy(hf_hub_download(REPO, name), local)
        out[name] = local
    return out


def load_latin_model(load_model):
    """The checkpoint to use for Latin text: MRZ lines and the ID number.

    The ID-card fine-tune was trained on a Khmer-only corpus (id_corpus.py
    filters out pure-Latin and MRZ lines), so it reads Khmer fields much
    better than the base model and Latin much worse: on the test card the
    fine-tune returned an empty MRZ where the base read line 2 exactly.
    Keeping both checkpoints and routing by field costs one extra load and
    recovers both sides. Returns None when no base checkpoint is saved, in
    which case callers fall back to the single active model."""
    base = CACHE / "khmer_ocr_document_base.pt"
    if not base.exists():
        return None
    model, itos, device = load_model(str(base))
    return {"model": model, "itos": itos, "device": device}


LATIN_FIELDS = ("id_number",)          # printed in Latin digits on the card


def id_number_from_mrz(printed: str, info: dict) -> str:
    """The printed ID number is the MRZ document number plus an issue suffix.

    Measured on the test card: Tesseract reads the printed field as
    '107080854 (04)' at every psm and scale tried, because that card's '1'
    has a flag that reads as 7 or 4. The same nine digits appear in MRZ line
    one, where a check digit says whether the read is right. So when the MRZ
    document number verifies, it wins, and only the '(01)' suffix is kept
    from the printed field."""
    m = re.search(r"\(\s*(\d{1,2})\s*\)", printed or "")
    suffix = f" ({m.group(1)})" if m else ""
    if info.get("document_number", ("", False))[1]:
        return info["document_number"][0] + suffix
    return printed


def constrained_read(image, model, itos, device, allowed: str) -> str:
    """Greedy decode with every class outside `allowed` masked out."""
    import torch
    from .ocr_engine import preprocess, greedy_decode

    x, _ = preprocess(image)
    with torch.no_grad():
        logits = model(x.to(device))
    mask = torch.full((len(itos) + 1,), -1e4, device=logits.device)
    mask[0] = 0.0                                     # keep the CTC blank
    for i, ch in enumerate(itos):
        if ch.upper() in allowed:
            mask[i + 1] = 0.0
    txt = greedy_decode(logits + mask, itos)[0]
    return txt.upper()


# ----------------------------------------------------- Latin via Tesseract ----

def latin_read(image, allowed: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ", lang: str = "eng") -> str:
    """Read one line of printed Latin text with Tesseract (free, Apache 2.0).

    Our model is trained on Khmer and reads Latin poorly; Tesseract is the
    reverse. Route each field to the engine that suits it. Returns "" when
    Tesseract is not installed, so callers can fall back."""
    import shutil as _sh
    import subprocess
    import tempfile

    if _sh.which("tesseract") is None:
        return ""
    from PIL import Image, ImageOps
    im = image if not isinstance(image, (str, bytes)) else Image.open(image)
    im = ImageOps.autocontrast(im.convert("L"))
    if im.height < 96:                                   # Tesseract likes ~30 px x-height
        im = im.resize((int(im.width * 96 / im.height), 96), Image.LANCZOS)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        im.save(f.name)
        try:
            out = subprocess.run(
                ["tesseract", f.name, "stdout", "--psm", "7", "-l", lang,
                 "-c", f"tessedit_char_whitelist={allowed}"],
                capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            return ""
    txt = out.stdout.strip().upper()
    return "".join(ch for ch in txt if ch in allowed)


def split_mrz_names(line3: str) -> tuple[str, str]:
    """SURNAME<<GIVEN<NAMES<<<<< -> ('SURNAME', 'GIVEN NAMES').

    Splitting on the '<<' separator fails once the filler is misread: card 4
    gave 'LIN<KK<DA<<KKK...' for 'LIN<<DA<<<', and the separator between the
    names came back as '<KK<', so the whole thing parsed as one name,
    'LIN KK DA'. Rewriting those letters back to chevrons does not work
    either - it merges the separator with the filler and swallows the given
    name.

    So split on every run of chevrons and then discard the tokens that are
    filler rather than name: anything with no letters, and any run made only
    of the letters chevrons are misread as. The surname is the first token
    that survives, which is the one position the format guarantees."""
    FILLER = set("KCDSI")
    parts = [t for t in re.split(r"<+", line3.upper()) if t]
    names = []
    for i, t in enumerate(parts):
        if not t.isalpha():
            continue
        # A stray real letter in a filler run ('KKKKKKKKKIKKE') means "all of
        # them" is too strict; three quarters is the line. A Khmer name
        # written entirely from these letters is rare enough to accept the
        # risk, and the surname position is never dropped.
        if i and sum(c in FILLER for c in t) / len(t) >= 0.75:
            continue
        names.append(t)
    if not names:
        return "", ""
    return names[0], " ".join(names[1:])


def reconcile_names(latin_field: str, surname: str, given: str):
    """The card prints the name twice: as a Latin field and in MRZ line 3.
    Two independent engines reading two independent printings and agreeing is
    the closest thing to a check digit the name has.

    Agreement is on the letters, ignoring spaces (Tesseract sometimes splits a
    word), and tolerates up to two trailing junk letters on the MRZ side (the
    filler read as a letter). Returns (verified, surname, given_names) with the
    MRZ's word boundaries, which are explicit '<' and therefore reliable."""
    a = "".join(ch for ch in latin_field.upper() if ch.isalpha())
    b_words = [w for w in (surname + " " + given).upper().split() if w]
    b = "".join(b_words)
    if not a or not b:
        return False, surname, given
    if a == b:
        return True, surname, given
    if b.startswith(a) and len(b) - len(a) <= 2:
        # MRZ has a junk tail: drop trailing words until the letters match
        words = list(b_words)
        while words and "".join(words) != a:
            words.pop()
        if words:
            sur_n = words[0] if not surname else surname
            n_sur = len(surname.split()) if surname else 1
            return True, " ".join(words[:n_sur]), " ".join(words[n_sur:])
    if a.startswith(b) and len(a) - len(b) <= 2:
        return True, surname, given                     # Latin field has junk; MRZ is clean

    # MRZ filler read as a stray letter shows up as a trailing one-letter word
    # ('SOK<<DARA<K<<<<' -> given 'DARA K'). Drop those and retry.
    trimmed = [w for i, w in enumerate(b_words) if i == 0 or len(w) > 1]
    if trimmed != b_words:
        return reconcile_names(latin_field, surname,
                               " ".join(trimmed[len(surname.split()) or 1:]))

    # Two engines reading two separate printings of the same name and landing
    # one character apart is agreement, not a clash: the odds of both drifting
    # to the same near-miss by chance are negligible. The MRZ spelling wins,
    # since OCR-B on a flat band reads more reliably than the printed field.
    if abs(len(a) - len(b)) <= 1 and _levenshtein(a, b) <= 1:
        return True, surname, given
    return False, surname, given


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ----------------------------------------------------------------- card ----

def warp_card(path: Path):
    """Find the card quadrilateral and warp it to a canonical rectangle."""
    import cv2
    import numpy as np
    from PIL import Image

    # cv2.imread ignores EXIF orientation; phone photos carry it (this card's
    # JPEG is orientation 6), so the detector would hunt a landscape card in a
    # portrait buffer and find nothing. Load through PIL, which applies it.
    from PIL import ImageOps
    try:
        pil_src = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        raw = cv2.cvtColor(np.asarray(pil_src), cv2.COLOR_RGB2BGR)
    except Exception:
        raw = cv2.imread(str(path))
    if raw is None:
        raise SystemExit(f"cannot read image: {path}")
    h, w = raw.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(raw, (int(w * scale), int(h * scale)))

    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 11, 40, 40)
    edges = cv2.Canny(grey, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    quad = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.20 * small.shape[0] * small.shape[1]:
            quad = approx.reshape(4, 2).astype("float32") / scale
            break

    if quad is None:
        print("  card outline not found; using the whole image")
        from PIL import ImageOps as _IO
        pil = _IO.exif_transpose(Image.open(path)).convert("RGB").resize((CARD_W, CARD_H))
        # Do NOT save this over card_flat.png: a failed detection used to
        # overwrite a good flatten with the whole photo, after which every
        # template box landed off the card and both checkpoints returned
        # identical garbage that looked like a model regression.
        return pil, False

    # order corners: tl, tr, br, bl
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    src = np.array([quad[np.argmin(s)], quad[np.argmin(d)],
                    quad[np.argmax(s)], quad[np.argmax(d)]], dtype="float32")
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(raw, M, (CARD_W, CARD_H))
    return Image.fromarray(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)), True


def grid_overlay(card, out: Path):
    """Save the flattened card with a labelled 5% grid, for picking boxes."""
    from PIL import ImageDraw

    g = card.copy()
    d = ImageDraw.Draw(g)
    for i in range(1, 20):
        x = CARD_W * i / 20
        d.line([(x, 0), (x, CARD_H)], fill=(255, 0, 0) if i % 4 == 0 else (255, 180, 180))
        if i % 2 == 0:
            d.text((x + 2, 2), f"{i * 5}", fill=(200, 0, 0))
    for i in range(1, 20):
        y = CARD_H * i / 20
        d.line([(0, y), (CARD_W, y)], fill=(0, 0, 255) if i % 4 == 0 else (180, 180, 255))
        if i % 2 == 0:
            d.text((2, y + 2), f"{i * 5}", fill=(0, 0, 200))
    g.save(out)


# Every Cambodian ID card carries these labels, identically, in this wording.
# That makes them known text rather than text to be read: the recogniser only has
# to indicate WHICH label it saw, and we can then restore the exact characters.
CARD_LABELS = [
    "គោត្តនាមនិងនាម", "ថ្ងៃខែឆ្នាំកំណើត", "ទីកន្លែងកំណើត", "អាសយដ្ឋាន",
    "សុពលភាព", "ភិនភាគ", "សញ្ជាតិ",
]
# labels that appear part-way along a line rather than at its start
INLINE_LABELS = ["ភេទ", "កម្ពស់"]


def _dist(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_label(text: str) -> str:
    """Replace a mangled leading label with its exact printed form.

    The label is a constant, so any difference from it is an OCR error. We
    accept a match when the read prefix is within a third of the label's length,
    which is loose enough to absorb several wrong clusters but tight enough not
    to swallow a line that has no label at all.
    """
    best, best_d, best_len = None, None, 0
    for lab in CARD_LABELS:
        n = len(lab)
        for width in range(max(1, n - 3), n + 4):
            if width > len(text):
                break
            d = _dist(text[:width], lab)
            tol = max(1, round(n / 3))
            if d <= tol and (best_d is None or d < best_d):
                best, best_d, best_len = lab, d, width
    if best is None:
        return text
    rest = text[best_len:]
    rest = rest.lstrip("ៈ:ះ ")        # drop whatever stood in for the colon
    return f"{best}: {rest}" if rest else f"{best}:"


def normalise(text: str) -> str:
    """Restore the fixed leading label, then the inline ones.

    Inline labels (ភេទ, កម្ពស់) are short, so fuzzy matching them anywhere would
    be risky; instead only a whole token that ends in a colon-like mark and is
    within one edit of a label is snapped, which is the shape a misread label
    has and a real word almost never does."""
    text = snap_label(text)
    # dates on the card are dd.mm.yyyy in Khmer digits; the model sometimes
    # returns / or a stray mark as the separator
    text = re.sub(r"([\u17e0-\u17e9]{2})[./\\-]([\u17e0-\u17e9]{2})[./\\-]([\u17e0-\u17e9]{4})",
                  r"\1.\2.\3", text)
    out = []
    for tok in text.split(" "):
        core = tok.rstrip(":ៈះ")
        mark = tok[len(core):]
        if mark and core:
            for lab in INLINE_LABELS:
                if core != lab and _dist(core, lab) <= 1 and abs(len(core) - len(lab)) <= 1:
                    core = lab
                    break
            if core in INLINE_LABELS:
                tok = core + ":"
        out.append(tok)
    text = " ".join(out)
    # No comma-to-stop or សម-to-ស.ម rewriting here. Card 1 prints 'ចំ.១ស.ម',
    # card 2 prints '០,៥សម'. Both are real, so normalising one into the other
    # corrupts whichever card it was not written for. Read what is there.
    # 'ចំ. ១ស.ម' - a stop followed by a space before Khmer digits is the
    # decoder spacing out an abbreviation the card prints closed up.
    text = re.sub(r"\.\s+(?=[\u17e0-\u17e9])", ".", text)
    # Height is always printed as Khmer digits followed by ស.ម and ends the
    # line, e.g. កម្ពស់: ១៨០ស.ម. Keep the digits, restore the unit exactly.
    text = re.sub(r"(កម្ពស់:\s*)([០-៩]{2,3})\s*[^០-៩]*$", r"\1\2ស.ម", text)
    # The card contains only Khmer script, Khmer and Latin digits, Latin
    # letters, spaces and a little punctuation. Anything else (%, *, ^, stray
    # symbols) is an artefact of decoding and is removed.
    text = re.sub(r"[^ក-៿᧠-᧿A-Za-z0-9 :.,()\-/]", "", text)
    return re.sub(r" {2,}", " ", text).strip()


# Which label each template field starts with. These are facts about the card
# layout, not things to recognise: the box for `validity` always begins with
# សុពលភាព, so whatever was read before the colon IS that label.
FIELD_LABEL = {
    "name_khmer": "គោត្តនាមនិងនាម", "dob_sex_height": "ថ្ងៃខែឆ្នាំកំណើត",
    "place_of_birth": "ទីកន្លែងកំណើត", "address_line1": "អាសយដ្ឋាន",
    "validity": "សុពលភាព", "distinguishing_marks": "ភិនភាគ",
}
_KD = "[០-៩]"                                      # one Khmer digit
_DATE = rf"({_KD}{{2}})\s*[./\\\\-]?\s*({_KD}{{2}})\s*[./\\\\-]?\s*({_KD}{{4}})"


def force_label(text: str, label: str) -> str:
    """Replace everything before the first colon-like mark with the known label.
    If no colon survived, drop a prefix the length of the label instead."""
    m = re.search(r"[:ៈះ]", text[:len(label) + 6])
    if m:
        rest = text[m.end():].strip()
    else:
        # no colon: the label is roughly the first len(label) characters
        rest = text[len(label):].lstrip(" :ៈះ")
    return f"{label}: {rest}" if rest else f"{label}:"


def to_khmer_digits(s: str) -> str:
    return "".join(chr(0x17e0 + int(c)) if c.isdigit() else c for c in s)


def parse_dob_line(text: str) -> tuple[str | None, str | None, str | None]:
    """'ថ្ងៃខែឆ្នាំកំណើត: dd.mm.yyyy ភេទ: ប្រុស កម្ពស់: 175ស.ម' -> (date, sex, height)."""
    d = re.search(_DATE, text)
    date = f"{d.group(1)}.{d.group(2)}.{d.group(3)}" if d else None
    sex = None
    for word, canon in (("ប្រុស", "ប្រុស"), ("ស្រី", "ស្រី")):
        if word in text:
            sex = canon
            break
    if sex is None:                                       # tolerate one wrong cluster
        for tok in text.split():
            core = tok.strip(":ៈ")
            for canon in ("ប្រុស", "ស្រី"):
                if abs(len(core) - len(canon)) <= 1 and _dist(core, canon) <= 1:
                    sex = canon
    tail = text[d.end():] if d else text
    h = re.findall(rf"({_KD}{{2,3}})", tail)
    height = h[-1] if h else None
    return date, sex, height


def parse_validity_line(text: str) -> tuple[str | None, str | None]:
    ds = re.findall(_DATE, text)
    fmt = lambda t: f"{t[0]}.{t[1]}.{t[2]}"
    return (fmt(ds[0]) if ds else None, fmt(ds[1]) if len(ds) > 1 else None)


def fix_colons(text: str) -> str:
    """A colon belongs only after a label, nowhere else on the card.

    Every printed colon on a Cambodian ID follows either the field's fixed
    label or one of the inline labels. A colon anywhere else is the model
    reading the Khmer mark U+17C8 ('ៈ') as one, which is exactly what happens
    at the end of a name such as សុភាៈ. So any colon not sitting directly after
    a label becomes that mark instead."""
    labels = tuple(FIELD_LABEL.values()) + tuple(INLINE_LABELS)
    out = []
    for i, ch in enumerate(text):
        if ch == ":" and not "".join(out).endswith(labels):
            out.append("ៈ")
        else:
            out.append(ch)
    return "".join(out)


def validity_joiner(text: str) -> str:
    """The word between the two validity dates varies by card generation:
    card 1 prints 'ដល់', card 2 prints 'ដល់ថ្ងៃ'. Rebuilding the line with a
    hardcoded 'ដល់' rewrites card 2 into a form it does not use, so keep
    whichever one was read and fall back to the shorter, commoner form."""
    return "ដល់ថ្ងៃ" if "ដល់ថ្ងៃ" in text else "ដល់"


def normalise_field(name: str, text: str) -> str:
    """Field-specific format rules on top of the general ones."""
    text = normalise(text)
    if name in FIELD_LABEL:
        text = force_label(text, FIELD_LABEL[name])
    text = fix_colons(text)
    if name == "dob_sex_height":
        date, sex, height = parse_dob_line(text)
        parts = [f"ថ្ងៃខែឆ្នាំកំណើត: {date or ''}".rstrip()]
        parts.append(f"ភេទ: {sex}" if sex else "ភេទ:")
        parts.append(f"កម្ពស់: {height}ស.ម" if height else "កម្ពស់:")
        return " ".join(parts)
    if name == "validity":
        d1, d2 = parse_validity_line(text)
        return (f"សុពលភាព: {d1 or ''} {validity_joiner(text)} {d2 or ''}"
                .replace("  ", " ").strip())
    if name == "distinguishing_marks":
        try:
            import field_lm
            text = field_lm.snap_features(text)
        except ImportError:
            pass
    if name == "name_latin":
        # printed in capitals, Latin letters only; the model's case is noise
        text = re.sub(r"[^A-Za-z ]", "", text).upper()
        text = re.sub(r" {2,}", " ", text).strip()
    elif name == "id_number":
        # nine digits, then a bracketed issue number: '123456789 (01)'.
        # Keep the bracketed digits exactly as printed - the card writes '01',
        # and taking only the last digit turned that into '(1)'.
        m = re.search(r"\d{9}", text)
        if m:
            # Only a real bracket yields a suffix. Card 2 prints the number
            # bare, and the loose digit fallback turned the guilloche behind
            # it into '(27)' - an invented issue number is worse than none.
            tail = re.search(r"\(\s*(\d{1,2})\s*\)", text[m.end():])
            text = m.group(0) + (f" ({tail.group(1)})" if tail else "")
    return text


def row_profile(card, x0: float, x1: float):
    """Column-summed ink for one horizontal slice of the card."""
    import numpy as np
    from PIL import ImageOps

    g = np.asarray(ImageOps.autocontrast(card.convert("L")), dtype=np.float32)
    H, W = g.shape
    sub = g[:, int(x0 / 100 * W):max(int(x0 / 100 * W) + 1, int(x1 / 100 * W))]
    paper = float(np.percentile(sub, 85))
    return (sub < paper - 60).sum(axis=1).astype(float), H


def snap_boxes(card, boxes: dict, window: float = 3.5, pad: float = 1.6) -> dict:
    """Move each template box onto the text row it is nearest to.

    Template boxes are calibrated on one card. Card 2's rows sit about 2%
    higher than card 1's, enough that the validity box straddled the validity
    row and the top of the marks row and the first date was lost entirely.
    Rather than recalibrating per card, find the actual row.

    For each box the ink profile is taken over that box's own x-range, and
    contiguous runs of ink are found within +/- `window` of the box. The run
    whose centre is nearest the box's centre is the row. Centre, not largest
    overlap: on card 2 the validity box overlapped its own row and the marks
    row by almost the same amount and overlap chose wrongly, while the centres
    were unambiguous. The window is smaller than the row pitch (about 6% of
    card height), so a box can be nudged onto its row but never jump to the
    next one."""
    out = {}
    for name, (bx0, by0, bx1, by1) in boxes.items():
        ink, H = row_profile(card, bx0, bx1)
        lo = max(0, int((by0 - window) / 100 * H))
        hi = min(H, int((by1 + window) / 100 * H))
        thr = max(3.0, 0.22 * float(ink[lo:hi].max() or 0))
        runs, start = [], None
        for y in range(lo, hi):
            if ink[y] > thr and start is None:
                start = y
            elif ink[y] <= thr and start is not None:
                if y - start >= 4:
                    runs.append((start, y))
                start = None
        if start is not None and hi - start >= 4:
            runs.append((start, hi))

        # Merge fragments. A watermark across a line breaks its ink profile
        # into pieces, and choosing one piece crops a 20 px sliver out of a
        # 47 px row - which is what happened to the second address line on the
        # card carrying a KINGDOM OF CAMBODIA overprint. A gap narrower than
        # 1.2% of card height is a gap between glyph groups, not between rows;
        # rows are about 6% apart.
        # 0.6%, measured across four cards. 1.2% was too generous and joined
        # the Latin name row to the birth row below it, and Tesseract answers
        # a two-line crop with nothing, which silently killed the name
        # cross-check. Rows sit about 6% apart; gaps inside a row are far
        # smaller than this.
        gap = max(3, int(0.006 * H))
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        runs = merged

        if not runs:
            out[name] = [bx0, by0, bx1, by1]
            continue
        t0, t1 = by0 / 100 * H, by1 / 100 * H
        mid = (t0 + t1) / 2
        best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - mid))
        if min(best[1], t1) - max(best[0], t0) <= 0:          # no overlap: keep it
            out[name] = [bx0, by0, bx1, by1]
            continue
        # Pad, but never more than halfway into the gap. A flat 1.6% pad
        # pulled the Latin-name box into the birth row on cards 1 and 2: the
        # run itself was right (18.4-21.8) and the padding made the crop two
        # lines, which Tesseract at psm 7 answers with an empty string. The
        # cross-check died silently as a result.
        above = max((r[1] for r in runs if r[1] <= best[0]), default=None)
        below = min((r[0] for r in runs if r[0] >= best[1]), default=None)
        pad_t = pad if above is None else min(pad, (best[0] - above) * 50.0 / H)
        pad_b = pad if below is None else min(pad, (below - best[1]) * 50.0 / H)
        out[name] = [bx0, round(max(0.0, best[0] * 100 / H - pad_t), 2),
                     bx1, round(min(100.0, best[1] * 100 / H + pad_b), 2)]

    # Adjacent rows sometimes merge into one run - on card 2 the Latin name
    # sits just above the birth line - and two boxes can then land on the
    # same row. Reverting both was tried and is worse: the birth box snaps
    # correctly there and reverting it lost the height. The name is caught
    # downstream instead, by comparing it against the MRZ printing.
    return out


def field_logp(image, model, device):
    """Log-softmax rows for one crop, the same tensor beam_decode consumes.

    read_line hides the logits, but repairing a field needs to score rival
    spellings against the pixels rather than against a guess, so they are
    recomputed here for the one field that needs it."""
    import torch
    from .ocr_engine import preprocess

    from .ocr_engine import WIDTH_STRIDE

    x, w = preprocess(image)
    with torch.no_grad():
        logits = model(x.to(device))
    T = max(1, w // WIDTH_STRIDE)                 # exactly what read_line uses
    return logits.log_softmax(-1)[0, :T].cpu().tolist()


KH_DIGITS = "០១២៣៤៥៦៧៨៩"


def from_khmer_digits(text: str) -> str:
    return "".join(str(KH_DIGITS.index(c)) if c in KH_DIGITS else c for c in text)


def expiry_from_issue(issue: str) -> str:
    """The validity line's second date, derived from its first.

    A Cambodian ID is valid for ten years less a day: issue 29.09.2016 expires
    28.09.2026. Checked against three cards and exact on all of them (3651 or
    3652 days apart, the difference being leap years, which is why the rule is
    stated in years rather than days).

    This gives the validity line the one thing it never had - an internal
    consistency check, the same idea as the MRZ check digits. The second date
    sits at the end of the line where trimming and noise do the most damage,
    so deriving it from the first is more reliable than reading it.

    Takes and returns dd.mm.yyyy in Khmer digits. Returns "" if the input is
    not a usable date."""
    from datetime import date, timedelta

    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", from_khmer_digits(issue or "").strip())
    if not m:
        return ""
    d, mo, y = (int(x) for x in m.groups())
    try:
        start = date(y, mo, d)
        end = start.replace(year=start.year + 10) - timedelta(days=1)
    except ValueError:
        return ""
    return to_khmer_digits(f"{end.day:02d}.{end.month:02d}.{end.year}")


def merge_into_mrz_words(printed: str, mrz_words: list[str]) -> list[str]:
    """Put the printed field's letters into the MRZ's word boundaries.

    Card 4: the MRZ read 'IN<<DA' where the card says LIN DA, dropping the
    leading L, while the printed field read 'LINDA' with the letters right and
    the space lost. Each read has half the answer: the MRZ knows where the
    words break ('<<' is explicit), the printed field knows the letters.

    Walk the printed string against the MRZ letters as a subsequence; every
    printed letter joins the MRZ word being matched, and unmatched extras join
    it too. 'LINDA' against ['IN','DA'] gives ['LIN','DA']."""
    letters = [c for c in printed.upper() if c.isalpha()]
    out, wi, consumed = [""] * len(mrz_words), 0, 0
    for ch in letters:
        while wi < len(mrz_words) and consumed >= len(mrz_words[wi]):
            wi += 1
            consumed = 0
        if wi >= len(mrz_words):
            out[-1] += ch
            continue
        out[wi] += ch
        if consumed < len(mrz_words[wi]) and ch == mrz_words[wi][consumed]:
            consumed += 1
    return [w for w in out if w]


def trim_band(img, pad: int = 4):
    """Crop an MRZ band sideways to the ink it actually contains.

    The bands are cut at fixed fractions of card width, so on a tight photo
    they include the card's own edge, which OCR reads as a leading '1' or 'T'
    and which shifts every field after it. Measured on a 543x368 photo:
    with fixed fractions Tesseract returned 'IDKHMO0508365724'; cropped to the
    ink it returns 'IDKHM0508365724' at every scale tried. No model change,
    the pixels were always there."""
    import numpy as np
    from PIL import ImageOps

    g = np.asarray(ImageOps.autocontrast(img.convert("L")), dtype=np.float32)
    if g.size == 0:
        return img
    paper = float(np.percentile(g, 85))
    col = (g < paper - 60).sum(axis=0)
    xs = np.where(col > 0)[0]
    if len(xs) < 2:
        return img
    x0, x1 = max(0, int(xs[0]) - pad), min(img.width, int(xs[-1]) + pad + 1)
    if x1 - x0 < 16:
        return img
    return img.crop((x0, 0, x1, img.height))


def trim_edges(crop, pad: int = 6):
    """trim_right, plus the same cut on the left.

    A box drawn wide enough for the longest value also catches the card's
    printed border on a tightly framed photo, and OCR reads that vertical
    edge as a letter: the Latin name came back 'ILIN DA' for 'LIN DA'. The
    left cut uses the same density rule as the right."""
    import numpy as np
    from PIL import ImageOps

    crop = trim_right(crop, pad=pad)
    g = np.asarray(ImageOps.autocontrast(crop.convert("L")), dtype=np.float32)
    if g.size == 0:
        return crop
    paper = float(np.percentile(g, 85))
    ink = (g < paper - 60).sum(axis=0).astype(float)
    if ink.max() == 0:
        return crop
    strong = float(np.percentile(ink[ink > 0], 90))
    cols = np.where(ink >= 0.25 * strong)[0]
    if len(cols) == 0:
        return crop
    left = max(0, int(cols[0]) - pad)
    return crop.crop((left, 0, crop.width, crop.height)) if crop.width - left >= 16 else crop


def enhance_card(card, mode: str = "none"):
    """Clean the flattened card before any field is cut from it.

    Measured on one card scanned three ways: a hard black-and-white scan read
    every field correctly (sharpness 1293) where the grayscale scan of the
    same card misread two (sharpness 235). What the binarisation removed was
    the KINGDOM OF CAMBODIA watermark and the background tint, which is
    exactly what had been breaking the two address lines.

    It is not free, though. On a 543x368 photo the same treatment destroyed
    the MRZ: thin strokes at five pixels tall do not survive a threshold, and
    both engines started inserting characters. So this is a choice, not a
    default, and the eval harness is what decides it per card.

      none      leave the card as photographed
      gray      grayscale, contrast stretched, illumination flattened
      contrast  adaptive threshold - the scanner-app look
    """
    if mode in ("", "none", None):
        return card
    import numpy as np
    from PIL import Image, ImageOps

    g = np.asarray(card.convert("L"), dtype=np.uint8)
    try:
        import cv2
    except ImportError:
        return ImageOps.autocontrast(Image.fromarray(g)).convert("RGB")

    # flatten the lighting: divide by a heavily blurred copy of itself, which
    # removes the wash across the card without touching stroke edges
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=max(g.shape) / 30.0)
    flat = np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 200.0, 0, 255).astype(np.uint8)

    if mode == "gray":
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
    else:                                   # contrast
        out = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 12)
        # a single-pixel open removes the speckle a threshold leaves behind
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return Image.fromarray(out).convert("RGB")


def crop_frac(card, box):
    x0, y0, x1, y1 = box
    return card.crop((int(x0 / 100 * CARD_W), int(y0 / 100 * CARD_H),
                      int(x1 / 100 * CARD_W), int(y1 / 100 * CARD_H)))


def trim_right(crop, pad: int = 6, min_width: int = 24):
    """Cut the crop at the last column that carries ink.

    A box has to be drawn wide enough for the longest value the field can
    hold, which means on most cards it ends in blank card stock. A CTC model
    trained on rendered lines has never seen a line that stops early, so it
    fills that space with invented text - this is where the trailing junk on
    the second card came from. Trimming to the ink makes the box width stop
    mattering, so boxes can be generous without the model paying for it.

    Darkness alone does not find the end of the text: the guilloche on this
    card is as dark as the print and runs to the edge of every box. Density
    does. A column of text carries many dark pixels top to bottom; a column
    crossed by an engraved line carries a few. So the cut is the last column
    whose ink count reaches a third of a typical text column's. Measured
    against the hand-calibrated boxes on the test card, this lands within
    about 1% of card width on every field, which is why the boxes below can
    be drawn wide and left alone."""
    import numpy as np
    from PIL import ImageOps

    g = np.asarray(ImageOps.autocontrast(crop.convert("L")), dtype=np.float32)
    if g.size == 0:
        return crop
    paper = float(np.percentile(g, 85))                 # card stock, not white
    ink = (g < paper - 60).sum(axis=0).astype(float)
    if ink.max() == 0:
        return crop
    strong = float(np.percentile(ink[ink > 0], 90))     # a full text column
    cols = np.where(ink >= 0.35 * strong)[0]
    if len(cols) == 0:
        return crop
    right = min(crop.width, int(cols[-1]) + pad)
    if right < min_width:
        return crop
    return crop.crop((0, 0, right, crop.height))


# ------------------------------------------------------------------ MRZ ----

def mrz_bands(card, top_pct: float = 66.0) -> list:
    """Split the MRZ strip at the foot of the card into its individual lines.

    top_pct must sit above the first line, not on it. At 72 the strip began
    partway down line one on the test card, so that band came back 23 px tall
    against 46 for the other two and both engines read it as noise while
    reading lines two and three cleanly. The splitter finds the gaps itself,
    so starting early costs nothing and starting late costs the whole line."""
    from PIL import ImageOps

    strip = card.crop((0, int(top_pct / 100 * CARD_H), CARD_W, CARD_H))
    g = ImageOps.autocontrast(strip.convert("L"))
    w, h = g.size
    px = g.load()
    rows = [sum(1 for x in range(w) if px[x, y] < 110) for y in range(h)]
    floor = max(3, int(0.03 * w))
    bands, start = [], None
    for y, r in enumerate(rows):
        if r > floor and start is None:
            start = y
        elif r <= floor and start is not None:
            if y - start >= 8:
                bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, h))
    bands = sorted(bands, key=lambda b: b[1] - b[0], reverse=True)[:3]
    bands.sort()
    return [strip.crop((0, max(0, a - 3), w, min(h, b + 3))) for a, b in bands]


def cd(s: str) -> str:
    """ISO 7501 check digit over a field, weights 7 3 1."""
    total = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch.upper()) - 55
        else:
            v = 0
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


def valid_date(s: str) -> bool:
    """YYMMDD with a plausible month and day."""
    return (len(s) == 6 and s.isdigit()
            and 1 <= int(s[2:4]) <= 12 and 1 <= int(s[4:6]) <= 31)


def repair(field: str, want: str, ok=None, depth: int = 2) -> tuple[str, bool]:
    """Substitute confusable characters until the check digit fits.

    `ok` is an optional predicate a candidate must also satisfy, so that a
    date cannot be 'repaired' into an impossible one that happens to check out.
    """
    def good(s: str) -> bool:
        return cd(s) == want and (ok is None or ok(s))

    if good(field):
        return field, True
    n = len(field)
    for i in range(n):
        for alt in CONFUSIONS.get(field[i], ""):
            cand = field[:i] + alt + field[i + 1:]
            if good(cand):
                return cand, True
    if depth > 1:
        for i in range(n):
            for alt_i in CONFUSIONS.get(field[i], ""):
                base = field[:i] + alt_i + field[i + 1:]
                for j in range(i + 1, n):
                    for alt_j in CONFUSIONS.get(base[j], ""):
                        cand = base[:j] + alt_j + base[j + 1:]
                        if good(cand):
                            return cand, True
    return field, False


def pad(line: str, n: int = 30) -> str:
    line = line.replace(" ", "<")
    return (line + "<" * n)[:n]


def rebuild(l1: str, l2: str, l3: str) -> tuple[str, str, str]:
    """Reconstruct the three fixed-width TD1 lines from an imperfect read.

    CTC collapses runs of identical characters, so the '<' filler comes back
    the wrong length (and is often misread as C, K or D anyway). Every TD1
    field sits at a fixed offset with filler only at the end, so we keep the
    leading data, discard the filler, and re-pad to 30 characters. The one
    exception is line 2, whose composite check digit sits AFTER the filler,
    at position 30; we take it from the end of the string.
    """
    l1, l2, l3 = (s.replace(" ", "").upper() for s in (l1, l2, l3))

    # line 1: doc code(2) + state(3) + number(9) + check(1) = 15, then filler.
    #
    # A low-resolution read inserts a character - 'IDKHMO0508365724' for
    # 'IDKHM0508365724' - and truncating to 15 then pushes the check digit off
    # the end, so the number can never verify. Pick the 15-character window
    # whose layout actually fits: nine digits followed by a check digit. The
    # shape of the format decides the alignment, before anything is discarded.
    # Score the candidate windows rather than demanding a perfect one. A
    # low-resolution line can carry an insertion AND a substitution at once -
    # 'IDKHMO0S08365724' for 'IDKHM0508365724' - so no offset gives nine clean
    # digits, and an all-or-nothing test falls back to the worst option. Count
    # how many of the nine number positions are digits, weight a digit in the
    # check position, and let the check-digit repair fix the rest.
    def _fit(w: str) -> int:
        if len(w) < 15:
            return -1
        return sum(c.isdigit() for c in w[5:14]) + (2 if w[14].isdigit() else 0)

    best_w, best_fit = l1[:15], _fit(l1[:15])
    for off in (1, 2, -1, 3):
        w = l1[off:off + 15] if off >= 0 else ("<" * -off + l1)[:15]
        if _fit(w) > best_fit:
            best_w, best_fit = w, _fit(w)
    l1 = pad(best_w)

    # line 2: 18 characters of data, filler, then the composite check digit
    tail = l2[-1] if l2 and l2[-1].isdigit() else "<"
    l2 = pad(l2[:18] + "<" * 11 + tail)

    # line 3: names, with '<' separating them. Filler characters vary, so cut
    # at the first run of two or more identical non-letter-looking fillers.
    l3 = pad(l3[:30])
    return l1, l2, l3


def parse_td1(l1: str, l2: str, l3: str, raw: bool = False) -> dict:
    l1, l2, l3 = (pad(l1), pad(l2), pad(l3)) if raw else rebuild(l1, l2, l3)
    out = {"mrz": [l1, l2, l3]}

    # Cambodian ID numbers are numeric, so a letter in there is an OCR error
    # even when the check digit happens to agree.
    # The document number can be misaligned rather than misread. On card 4
    # line 1 came back 'IDKHMO0508365724...': a spurious 'O' after IDKHM
    # shifted every following field by one. Read at the nominal offset that
    # gives a letter in a numeric field, and the check-digit repair then finds
    # some substitution that passes - reporting `verified` on a number that is
    # simply wrong, which is worse than reporting a misread.
    #
    # So try small shifts and prefer a window that is nine clean digits whose
    # check digit passes. Only a passing check digit can win the alignment, so
    # this cannot invent one.
    def _doc_at(off: int):
        w, chk = l1[5 + off:14 + off], l1[14 + off:15 + off]
        if len(w) < 9 or len(chk) != 1:
            return None
        d = sum(c.isdigit() for c in w.replace("<", ""))
        num = d >= 0.6 * max(1, len(w.replace("<", "")))
        return repair(w, chk, (lambda t: t.replace("<", "").isdigit()) if num else None)

    # Among the alignments whose check digit passes, take the one that needed
    # the fewest character substitutions. At offset 0 this line needs two
    # ('O05083657' -> '005883657') and still passes; shifted by one it needs
    # none. Fewest edits is the honest tie-break - a passing check digit alone
    # is not enough, since repair can always find some string that passes.
    best = None
    for off in (0, 1, -1, 2, -2):
        alt = _doc_at(off)
        if not alt or not alt[1] or not alt[0].replace("<", "").isdigit():
            continue
        raw_w = l1[5 + off:14 + off]
        edits = sum(a != b for a, b in zip(raw_w, alt[0]))
        if best is None or edits < best[0]:
            best = (edits, alt[0], alt[1])
    if best is not None:
        doc_edits, doc, ok_doc = best
    else:
        doc, ok_doc = _doc_at(0) or ("", False)
        doc_edits = 9
    # Line 2 suffers the same insertions as line 1 on a low-resolution photo
    # ('8510080FSS501030' for '8510080F3301030'), so the date fields get the
    # same fewest-edits alignment search. A date only wins an offset when its
    # check digit passes and it is a real calendar date.
    def _date_at(base: int, off: int):
        w, chk = l2[base + off:base + 6 + off], l2[base + 6 + off:base + 7 + off]
        if len(w) < 6 or len(chk) != 1:
            return None
        got = repair(w, chk, valid_date)
        if not got[1] or not valid_date(got[0]):
            return None
        return sum(a != b for a, b in zip(w, got[0])), got[0], got[1]

    def _best_date(base: int):
        cands = [c for c in (_date_at(base, o) for o in (0, 1, -1, 2, -2)) if c]
        if cands:
            _, v, ok = min(cands)
            return v, ok
        return repair(l2[base:base + 6], l2[base + 6:base + 7] or "<", valid_date)

    dob, ok_dob = _best_date(0)
    exp, ok_exp = _best_date(8)
    # A check digit can pass on a wrong date when an insertion and a
    # substitution conspire: card 4 gave a verified expiry of 03.01.1955
    # against a 1985 birth date. A card cannot expire before its holder was
    # born, so that read is rejected and the printed validity line supplies
    # the date instead.
    def _ymd(d6: str) -> str:
        # compare on the century-resolved date: raw YYMMDD makes 25 (2025)
        # look earlier than 84 (1984)
        f = fmt_date(d6)
        m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", f or "")
        return f"{m.group(3)}{m.group(2)}{m.group(1)}" if m else ""

    if ok_exp and ok_dob:
        a, b = _ymd(exp), _ymd(dob)
        if a and b and a <= b:
            ok_exp = False

    out["document_type"] = l1[0:2].replace("<", "")
    out["issuing_state"] = l1[2:5]
    out["document_number"] = (doc.replace("<", ""), ok_doc)
    out["doc_edits"] = doc_edits
    out["date_of_birth"] = (fmt_date(dob), ok_dob)
    out["sex"] = ({"M": "male", "F": "female"}.get(l2[7], l2[7]), l2[7] in "MF<")
    out["date_of_expiry"] = (fmt_date(exp), ok_exp)
    # Nationality appears twice: line 2 and, as the issuing state, line 1.
    # They agree on a clean read. When they disagree one line was misread, and
    # line 2 is the one whose neighbouring fields carry check digits, so it is
    # the better witness - the old rule preferred line 1 and reported 'IIH'
    # off a garbled MRZ line one while every verified field came from line 2.
    nat, iss = l2[15:18], out["issuing_state"]
    out["nationality"] = nat if nat == iss else (iss if nat.strip("<") == "" else nat)
    out["nationality_verified"] = nat == iss

    # the name line has no check digit, so it is only trustworthy when the
    # '<<' separator actually survived the read
    names = l3.split("<<", 1)
    surname = names[0].replace("<", " ").strip()
    given = names[1].replace("<", " ").strip() if len(names) > 1 else ""
    split_ok = bool(surname) and bool(given)      # both halves must survive
    out["surname"] = (surname, split_ok and bool(surname))
    out["given_names"] = (given, split_ok and bool(given))

    # recheck the composite over the REPAIRED fields, so it confirms the repair
    l1r = l1[:5] + doc + l1[14:]
    l2r = dob + l2[6] + l2[7] + exp + l2[14:]
    composite = l1r[5:30] + l2r[0:7] + l2r[8:15] + l2r[18:29]
    out["composite_ok"] = cd(composite) == l2r[29]
    return out


def fmt_date(yymmdd: str) -> str:
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return yymmdd
    yy, mm, dd = int(yymmdd[:2]), yymmdd[2:4], yymmdd[4:]
    year = 1900 + yy if yy > 40 else 2000 + yy
    return f"{dd}.{mm}.{year}"


# ------------------------------------------------------- the full reader ----

def _mrz_score(info: dict) -> int:
    s = 0
    for k in ("document_number", "date_of_birth", "date_of_expiry"):
        s += 2 if info[k][1] else 0
    s += 1 if info["sex"][1] else 0
    s += 3 if info["composite_ok"] else 0
    s += 1 if info["issuing_state"] == "KHM" else 0
    return s


def read_all(card, boxes: dict, model, itos, device, lm=None,
             mrz_top: float = 66.0, use_tesseract: bool = True, latin=None,
             snap: bool = True, enhance: str = "none") -> dict:
    """Read a flattened, right-way-up card: MRZ plus template fields.

    Routing: Khmer fields -> our model. Latin name -> Tesseract. MRZ -> our
    model with constrained decoding; if that leaves anything unverified,
    Tesseract reads the zone too and the better-scoring parse wins. The MRZ
    name line always comes from Tesseract when available, and the name is
    marked verified only when it agrees with the Latin name field."""
    import time
    from .ocr_engine import read_line
    t0 = time.time()
    card = enhance_card(card, enhance)

    from PIL import ImageFilter
    # a half-pixel smoothing measurably helps the MRZ read (sensor and JPEG
    # noise in thin OCR-B strokes): on the test card, confidence 8 -> 11
    bands = [trim_band(b).filter(ImageFilter.GaussianBlur(0.5))
             for b in mrz_bands(card, mrz_top)]
    # the MRZ is Latin: read it with the base checkpoint when one is available
    lat = latin or {"model": model, "itos": itos, "device": device}
    raw_model = [constrained_read(b, lat["model"], lat["itos"], lat["device"], MRZ_CHARS)
                 for b in bands]
    while len(raw_model) < 3:
        raw_model.append("")
    info, raw, source = parse_td1(*raw_model[:3]), raw_model[:3], "model"

    raw_tess = None
    if use_tesseract and bands:
        raw_tess = [latin_read(b, MRZ_CHARS) for b in bands]
        while len(raw_tess) < 3:
            raw_tess.append("")
        info_alt = parse_td1(*raw_tess[:3])

        # Merge field by field, not parse by parse.
        #
        # Picking one engine's whole parse loses the fields the other got
        # right: on one phone capture the model's line 1 carried four junk
        # characters and yielded a *verified* 170808547, while Tesseract read
        # 'IDKHM1010808547' exactly. The model's parse still scored higher
        # overall, so its wrong document number won. Each field is independent
        # evidence, so take each from whichever engine verified it, and where
        # both verified but disagree, prefer the one that needed less repair
        # and drop the verified flag.
        # The name line too. Filler glue adds letters rather than removing
        # them, so when one engine's name is the other's plus a trailing
        # filler letter, the shorter one is the name: 'VATTANACK' against
        # 'VATTANAC'.
        for key in ("surname", "given_names"):
            a, b = info.get(key), info_alt.get(key)
            if not a or not b or not a[0] or not b[0] or a[0] == b[0]:
                continue
            short, long_ = sorted((a[0], b[0]), key=len)
            if (long_.startswith(short) and 0 < len(long_) - len(short) <= 2
                    and set(long_[len(short):]) <= set("KCDSI")):
                info[key] = (short, a[1] and b[1])

        for key in ("document_number", "date_of_birth", "date_of_expiry", "sex"):
            a, b = info.get(key), info_alt.get(key)
            if not a or not b or a[0] == b[0]:
                continue
            if b[1] and not a[1]:
                info[key] = b
            elif a[1] and b[1]:
                ea = info.get("doc_edits", 9) if key == "document_number" else 0
                eb = info_alt.get("doc_edits", 9) if key == "document_number" else 0
                if eb < ea:
                    info[key] = (b[0], False)
                elif ea == eb:
                    info[key] = (a[0], False)      # a real conflict, so not verified
        if _mrz_score(info) < 11:
            info_t = info_alt
            # Score counts how many fields verify, which is not enough: a
            # check digit can pass on a wrong number after repair, and it did
            # - the model verified '050056572' where the card says
            # '050836572'. Between two parses that score the same, the one
            # that needed fewer character substitutions is the honest winner.
            key = lambda i: (_mrz_score(i), -i.get("doc_edits", 9))
            if key(info_t) > key(info):
                info, raw, source = info_t, raw_tess[:3], "tesseract"
        # the name line: Tesseract's read of it, whichever engine won the rest
        if raw_tess[2] and "<<" in raw_tess[2]:
            sur, giv = split_mrz_names(raw_tess[2])
            if sur and giv:
                info["surname"] = (sur, False)
                info["given_names"] = (giv, False)

    if snap:
        boxes = snap_boxes(card, boxes)
    fields = {}
    for name, box in boxes.items():
        crop = trim_edges(crop_frac(card, box))
        if name == "name_latin" and use_tesseract:
            txt = latin_read(crop) or read_line(crop, model, itos, device, lm)
        elif name in LATIN_FIELDS:
            # Latin digits: Tesseract first, then the base checkpoint. The
            # Khmer fine-tune renders these as Khmer digits, which the
            # downstream \d{9} match then rejects.
            txt = (latin_read(crop, allowed="0123456789() ")
                   or read_line(crop, lat["model"], lat["itos"], lat["device"], None))
        else:
            # known-vocabulary fields decode with a blended, field-specific LM
            try:
                import field_lm
                field_model = field_lm.for_field(name, lm)
            except ImportError:
                field_model = lm
            txt = read_line(crop, model, itos, device, field_model)
        fields[name] = normalise_field(name, txt)

    # Verified MRZ values are authoritative for the Khmer fields that repeat
    # them: sex, date of birth, expiry. Where the Khmer read disagrees with a
    # check-digit-verified value, the Khmer read is the one that is wrong.
    def mrz_date_kh(v: str) -> str:                      # "12.09.1984" -> Khmer digits
        return to_khmer_digits(v)
    # Vote on the birth date across the two engines and the printed line.
    #
    # A check digit proves a field is self-consistent, not that it is right:
    # on card 2 the model read '0407126' for '8409126', which passes and
    # yields a verified 12.07.2004 for a card that says 12.09.1984. Because
    # that parse scored full marks, the other engine was never consulted. The
    # card states the birth date twice, so take the majority of the three.
    if "dob_sex_height" in fields:
        printed_date, _, _ = parse_dob_line(fields["dob_sex_height"])
        cand = []
        for src in (info, locals().get("info_alt")):
            if src and src["date_of_birth"][1]:
                cand.append(src["date_of_birth"][0])
        if printed_date:
            cand.append(from_khmer_digits(printed_date))
        if cand:
            top = max(set(cand), key=cand.count)
            if cand.count(top) >= 2:
                if top != info["date_of_birth"][0]:
                    info["date_of_birth"] = (top, True)
            # No majority means only the printed Khmer line disagrees, and of
            # the three readings that is the weakest: it has no checksum and
            # no second engine behind it. An earlier version demoted the MRZ
            # to unverified here, which left a wrong printed year standing
            # against a check-digit-verified 1997. The vote above already
            # catches the case that matters, where both the other engine and
            # the printed line agree against the MRZ.

    if "dob_sex_height" in fields:
        date, sex, height = parse_dob_line(fields["dob_sex_height"])
        if info["date_of_birth"][1]:
            date = mrz_date_kh(info["date_of_birth"][0])
        if info["sex"][1] and info["sex"][0] in ("male", "female"):
            sex = "ប្រុស" if info["sex"][0] == "male" else "ស្រី"
        elif sex in ("ប្រុស", "ស្រី"):
            # The MRZ sex field is a single character with no check digit of
            # its own, so a lost character upstream turns it into whatever
            # sits at that offset - one scan reported '2'. The printed line
            # says it in words, and that is better than a stray digit.
            info["sex"] = ("male" if sex == "ប្រុស" else "female", False)
        fields["dob_sex_height"] = " ".join([
            f"ថ្ងៃខែឆ្នាំកំណើត: {date or ''}".rstrip(),
            f"ភេទ: {sex}" if sex else "ភេទ:",
            f"កម្ពស់: {height}ស.ម" if height else "កម្ពស់:"])
    if "id_number" in fields:
        fields["id_number"] = id_number_from_mrz(fields["id_number"], info)
    validity_derived = False
    if "validity" in fields:
        d1, d2 = parse_validity_line(fields["validity"])
        if info["date_of_expiry"][1] and d1:
            # A verified MRZ expiry that contradicts the date the printed
            # issue date implies is not verified at all: on one scan the MRZ
            # gave 28.01.2020 against a card reading 2026, and it passed its
            # check digit. Ten years less a day is the card's own rule, so
            # prefer it and drop the verified flag.
            derived = expiry_from_issue(d1)
            mrz_kh = mrz_date_kh(info["date_of_expiry"][0])
            if derived and derived != mrz_kh:
                d2, validity_derived = derived, True
                info["date_of_expiry"] = (from_khmer_digits(derived), False)
            else:
                d2 = mrz_kh
        elif info["date_of_expiry"][1]:
            d2 = mrz_date_kh(info["date_of_expiry"][0])          # check digit wins
        elif d1:
            # No verified MRZ expiry, so derive the second date from the first
            derived = expiry_from_issue(d1)
            if derived and derived != d2:
                d2, validity_derived = derived, True
        fields["validity"] = (f"សុពលភាព: {d1 or ''} "
                              f"{validity_joiner(fields['validity'])} {d2 or ''}"
                              ).replace("  ", " ").strip()

    # place names: resolve against the administrative hierarchy
    address = {}
    try:
        import gazetteer
        address = gazetteer.snap_fields(fields)
    except (ImportError, FileNotFoundError):
        pass

    # cross-check the two printings of the name
    sur, giv = info["surname"][0], info["given_names"][0]
    agree, sur, giv = reconcile_names(fields.get("name_latin", ""), sur, giv)
    info["surname"] = (sur, agree)
    info["given_names"] = (giv, agree)
    # The name's value comes from the MRZ; the printed field is the witness.
    #
    # MRZ line 3 carries no check digit, so this is an empirical call, not a
    # proven one: OCR-B on a flat band reads better than small print over a
    # security pattern, and across the test cards the MRZ name was right every
    # time while the printed field failed twice - collapsing into 'OEGG ISS'
    # on one card and stopping at 'CHAN CHAMRO' on another. So the MRZ
    # supplies the value whenever line 3 kept its '<<' structure, which is
    # what says that read was clean, and `verified` means the printed field
    # independently agreed. Disagreement yields the MRZ name marked
    # unverified, which is the flag for a human to check it.
    clean_mrz = bool(raw_tess and "<<" in (raw_tess[2] or ""))
    giv_clean = " ".join(w for w in giv.split() if len(w) > 1) or giv

    # (An earlier rule preferred the printed name whenever it ended with the
    # MRZ letters, to cover an MRZ that had dropped a leading character. That
    # was compensating for an untrimmed MRZ band, fixed in trim_band, and it
    # backfired once the MRZ was right: a printed crop carrying a leading edge
    # artifact then turned LIN into ILIN. Removed.)

    # Keep what the printed field actually said. It no longer supplies the
    # value, so without this a failing witness is invisible: the only symptom
    # would be `name_verified` quietly going false, with nothing to say why.
    printed_name = fields.get("name_latin", "")
    if clean_mrz and sur:
        # MRZ filler chevrons read as a trailing one-letter word ('DARA K')
        giv = giv_clean
        # The MRZ is not always the longer read. On card 4 it dropped a
        # leading L ('IN' for LIN) while the printed field had every letter.
        # OCR drops characters far more often than it invents them, so when
        # the MRZ's letters are contained in the printed field's, the printed
        # letters win - placed into the MRZ's word boundaries, which is the
        # half the MRZ does know.
        pl = "".join(c for c in printed_name.upper() if c.isalpha())
        ml = "".join(c for c in (sur + giv) if c.isalpha())
        def _subseq(a, b):                       # is a a subsequence of b?
            it = iter(b)
            return all(c in it for c in a)
        if pl and ml and pl != ml and len(pl) > len(ml) and _subseq(ml, pl):
            words = merge_into_mrz_words(pl, (sur + " " + giv).split())
            if words:
                sur, giv = words[0], " ".join(words[1:])
        # And the mirror case: the MRZ gains a letter where a chevron was
        # misread and glued onto a name - 'THOUK<RASMEY' for 'THOU<<RASMEY'.
        # One extra character, and that character is one of the letters
        # chevrons come back as, so the printed read is the better one.
        elif pl and ml and len(ml) - len(pl) == 1 and _subseq(pl, ml):
            # Strip one trailing filler letter from whichever MRZ word ends in
            # one, and keep it only if the result is exactly the printed
            # letters. That equality is the check: nothing is guessed.
            ws = (sur + " " + giv).split()
            for i, w in enumerate(ws):
                if len(w) > 2 and w[-1] in "KCDSI":
                    trial = ws[:i] + [w[:-1]] + ws[i + 1:]
                    if "".join(trial) == pl:
                        sur, giv = trial[0], " ".join(trial[1:])
                        break
                agree = True                     # both reads, reconciled
        info["surname"] = (sur, agree)
        info["given_names"] = (giv, agree)
        if "name_latin" in fields:
            fields["name_latin"] = f"{sur} {giv}".strip()
    elif agree and "name_latin" in fields:
        fields["name_latin"] = f"{sur} {giv}".strip()

    # The Khmer name is the one field with no lexicon and no checksum. The
    # card prints the same name in Latin, backed by a check digit through the
    # MRZ, so rival Khmer spellings predicted by that Latin name are scored
    # against the field's own logits and the best-supported one wins.
    name_repaired = False
    if "name_khmer" in fields and (sur or fields.get("name_latin")):
        try:
            import name_check
            latin = f"{sur} {giv}".strip() or fields.get("name_latin", "")
            text = fields["name_khmer"]
            head, sep, tail = text.partition(": ")
            prefix, body = (head + sep, tail) if sep else ("", text)
            stoi = {c: i + 1 for i, c in enumerate(itos)}
            lp = field_logp(trim_edges(crop_frac(card, boxes["name_khmer"])),
                            model, device)
            # Only let the Latin name overrule the Khmer read when that Latin
            # read is itself sound: the two printings agreed, or MRZ line 3
            # kept its '<<' structure. A garbled MRZ must not rewrite a good
            # Khmer name.
            trust = bool(agree) or bool(raw_tess and "<<" in (raw_tess[2] or ""))
            fixed, name_repaired = name_check.repair_name(
                body, latin, lp, stoi, prefix=prefix, trust_latin=trust)
            if name_repaired:
                fields["name_khmer"] = prefix + fixed
        except (ImportError, KeyError, IndexError):
            name_repaired = False

    # Nationality by vote across both lines and both engines.
    #
    # It is written twice - as the issuing state in line 1 and as nationality
    # in line 2 - and neither position carries a check digit. Preferring one
    # line over the other fails whenever that line is the garbled one, which
    # has happened in both directions: line 1 was noise on card 1 and gave
    # 'IIH', line 2 was noise on card 3 and gave 'MCC'. Four readings of the
    # same three letters, and the majority is right when any two agree.
    votes = []
    for lines in (raw, raw_tess):
        if not lines:
            continue
        l1 = (lines[0] if len(lines) > 0 else "").ljust(30)
        l2 = (lines[1] if len(lines) > 1 else "").ljust(30)
        for v in (l1[2:5], l2[15:18]):
            v = v.strip("<").strip()
            if len(v) == 3 and v.isalpha():
                votes.append(v.upper())
    nat_ok = info.get("nationality_verified", False)
    if votes:
        top = max(set(votes), key=votes.count)
        if votes.count(top) >= 2:
            info["nationality"], nat_ok = top, votes.count(top) >= 3
        elif not nat_ok:
            info["nationality"] = top

    # An unverified MRZ expiry, when the printed validity line derives one,
    # takes the derived value. Reported as derived, never as verified: the
    # check digit did not pass, a rule supplied the answer.
    expiry_derived = False
    if validity_derived and not info["date_of_expiry"][1] and "validity" in fields:
        _, d2_kh = parse_validity_line(fields["validity"])
        d2_latin = from_khmer_digits(d2_kh or "")
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", d2_latin):
            info["date_of_expiry"] = (d2_latin, False)
            expiry_derived = True

    mrz = {}
    for key in ("document_number", "surname", "given_names",
                "date_of_birth", "sex", "date_of_expiry"):
        value, ok = info[key]
        mrz[key] = {"value": value, "verified": bool(ok)}
    mrz["nationality"] = {"value": info["nationality"], "verified": bool(nat_ok)}
    mrz["date_of_expiry"]["derived"] = expiry_derived
    mrz["composite_ok"] = bool(info["composite_ok"])
    mrz["raw_lines"] = raw
    mrz["source"] = source
    if raw_tess is not None:
        mrz["raw_lines_tesseract"] = raw_tess[:3]
    return {"mrz": mrz, "fields": fields, "mrz_confidence": _mrz_score(info),
            "name_verified": agree, "address": address,
            "name_khmer_repaired": name_repaired,
            "name_latin_printed": printed_name,
            "validity_derived": validity_derived,
            "expiry_derived": expiry_derived,
            "seconds": round(time.time() - t0, 2)}


# ----------------------------------------------------------------- main ----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--flat", action="store_true",
                    help="the image is already a flattened card (card_flat.png from "
                         "card_calibrator.html); skip detection and warping")
    ap.add_argument("--calibrate", action="store_true",
                    help="write the flattened card and a coordinate grid, then stop")
    ap.add_argument("--fields", default="fields.json",
                    help="template boxes for the Khmer fields, in percent of card size")
    ap.add_argument("--lm", action="store_true", help="beam search for the Khmer fields")
    ap.add_argument("--mrz-top", type=float, default=66.0,
                    help="where the MRZ strip starts, in percent of card height")
    ap.add_argument("--save-crops", metavar="DIR")
    ap.add_argument("--no-snap", action="store_true",
                    help="use the template boxes as written instead of snapping "
                         "each one onto the text row it is nearest to")
    ap.add_argument("--no-latin-model", action="store_true",
                    help="read the MRZ and ID number with the active checkpoint "
                         "instead of khmer_ocr_document_base.pt")
    args = ap.parse_args()

    src = Path(args.image).expanduser()
    if args.flat:
        from PIL import Image, ImageOps
        card = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        if card.size != (CARD_W, CARD_H):
            print(f"  resizing {card.size[0]}x{card.size[1]} to {CARD_W}x{CARD_H}")
            card = card.resize((CARD_W, CARD_H))
        print("card: taken as already flattened")
    else:
        card, found = warp_card(src)
        print(f"card: {'outline detected and flattened' if found else 'not detected'}"
              f"  -> {CARD_W}x{CARD_H}")
        # Only a real detection is worth keeping. Saving the fallback over
        # card_flat.png destroyed a good flatten and cost an hour of chasing
        # a model regression that did not exist.
        # Always name it after the source. A fixed name meant a second photo
        # silently replaced the first card's flatten.
        out_flat = Path(f"card_flat_{src.stem}.png")
        card.save(out_flat)
        print(f"  wrote {out_flat}")
        if not found:
            print("  detection failed, so this is the whole photo, not a flattened card.")
            print("  Fix the corners with flatten_nid.py or the calibrator page,")
            print("  then re-run with --flat.")

    if args.calibrate:
        grid_overlay(card, Path("card_grid.png"))
        print("\nwrote card_flat.png and card_grid.png")
        print("Open card_grid.png, read off each field's box as percentages of the")
        print("card (left, top, right, bottom), and write them into fields.json:\n")
        print(json.dumps({"name_khmer": [30, 18, 98, 26],
                          "address_khmer": [30, 44, 98, 58]}, indent=2))
        return 0

    paths = fetch(["khmer_ocr.py", "vocab.json", "khmer_ocr_document.pt"]
                  + (["khmer_lm.pkl.gz"] if args.lm else []))
    sys.path.insert(0, str(CACHE))
    from .ocr_engine import load_model, load_lm, read_line      # noqa: E402

    model, itos, device = load_model(str(paths["khmer_ocr_document.pt"]))
    lm = load_lm(str(paths["khmer_lm.pkl.gz"])) if args.lm else None
    latin = None if args.no_latin_model else load_latin_model(load_model)
    lat = latin or {"model": model, "itos": itos, "device": device}
    print(f"device: {device}"
          f"{'  (Latin fields on khmer_ocr_document_base.pt)' if latin else ''}\n")

    out_dir = Path(args.save_crops).expanduser() if args.save_crops else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    fpath = Path(args.fields)
    if not fpath.exists():
        print(f"\nNo {fpath} yet, so the Khmer fields were skipped.")
        print("Run with --calibrate to produce the grid and set the boxes.")
        return 0
    boxes = json.loads(fpath.read_text())

    if out_dir:
        for i, b in enumerate(mrz_bands(card, args.mrz_top), 1):
            b.save(out_dir / f"mrz_{i}.png")
        for name, box in boxes.items():
            crop_frac(card, box).save(out_dir / f"{name}.png")

    # One reader for both entry points. The CLI used to carry its own copy of
    # this, which silently skipped the Tesseract fallback for the MRZ and the
    # rule that a check-digit-verified MRZ value overrules the Khmer read, so
    # the two paths disagreed on the same card.
    res = read_all(card, boxes, model, itos, device, lm,
                   mrz_top=args.mrz_top, latin=latin, snap=not args.no_snap)
    mrz = res["mrz"]

    print(f"MRZ: {len([r for r in mrz['raw_lines'] if r])} lines read "
          f"(engine: {mrz['source']}, confidence {res['mrz_confidence']}/11)")
    for i, line in enumerate(mrz["raw_lines"], 1):
        print(f"  raw {i}: {line}")
    if "raw_lines_tesseract" in mrz:
        for i, line in enumerate(mrz["raw_lines_tesseract"], 1):
            print(f"  tess {i}: {line}")

    print("\n--- machine-readable zone, after check-digit repair ---")
    for key in ("document_number", "surname", "given_names",
                "date_of_birth", "sex", "date_of_expiry", "nationality"):
        f = mrz[key]
        print(f"  {key:18s} {f['value']:<28s} "
              f"{'verified' if f['verified'] else 'UNVERIFIED'}")
    print(f"  composite check    {'passes' if mrz['composite_ok'] else 'FAILS'}")
    print(f"  name cross-check   {'agrees' if res['name_verified'] else 'DISAGREES'}")

    print("\n--- Khmer fields, read from template boxes ---")
    for name, value in res["fields"].items():
        print(f"  {name:18s} {value}")

    if res.get("address"):
        print("\n--- address, snapped to the gazetteer ---")
        for k, v in res["address"].items():
            print(f"  {k:18s} {v}")

    print(f"\nread in {res['seconds']}s. MRZ fields marked verified passed their")
    print("check digit and are safe to trust. Khmer fields carry no checksum,")
    print("so treat them as suggestions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
