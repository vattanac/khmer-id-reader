#!/usr/bin/env python3
"""
field_lm.py - field-specific language models for the ID card.

Some fields are not free text. The distinguishing-marks line (ភិនភាគ) is built
from a small vocabulary: a mark type, a size, a position, a facial feature, a
side. A character n-gram trained on phrases generated from that vocabulary,
blended with the general Khmer model at decode time, makes the beam search
prefer real feature names (ចិញ្ចើម) over near-misses (ចំព្វើ) while the image is
still being read, which is where the correction belongs.

The vocabulary below was supplied by Vattanac and is the list printed on
Cambodian ID cards. Extend FEATURES / MARKS / POSITIONS as more cards are seen.
"""

from __future__ import annotations

import itertools
import math
import random

# ---- facial vocabulary (Khmer) --------------------------------------------
# The vocabulary itself lives in phinpheak_vocab.json so it can be edited
# without touching code and reused by the training-corpus generator.
import json as _json
from pathlib import Path as _Path

_VOCAB_PATH = _Path(__file__).resolve().parent / "phinpheak_vocab.json"
_V = _json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))

FEATURES = _V["features"]
CHONG = _V["chong_compounds"]
MARKS = _V["marks"]
QUALIFIERS = [""] * 2 + _V["qualifiers"]          # weight the bare form
SIZES = _V["sizes"]
POSITIONS = _V["positions"]
SIDES = _V["sides"] + [""]
ADJ_COLOUR = _V["adjectives"]["colour"]
ADJ_SHAPE = _V["adjectives"]["shape"]
ADJ_COND = _V["adjectives"]["condition"]
ADJECTIVES = ADJ_COLOUR + ADJ_SHAPE + ADJ_COND
SENTENCES = _V["official_sentences"]
EXAMPLES = _V["feature_adjective_examples"]
LABEL = _V["label"] + " "


def mark_phrases(n: int = 6000, seed: int = 7) -> list[str]:
    """Generate plausible ភិនភាគ lines from the vocabulary."""
    rnd = random.Random(seed)
    out = set()
    while len(out) < n:
        parts = []
        if rnd.random() < 0.35:                          # official "មាន…" sentence form
            parts.append("មាន" + rnd.choice(MARKS) + rnd.choice(QUALIFIERS))
        else:
            parts.append(rnd.choice(MARKS) + rnd.choice(QUALIFIERS))
        if rnd.random() < 0.5:
            parts.append(rnd.choice(SIZES))
        pos = rnd.choice(POSITIONS)
        feat = rnd.choice(FEATURES)
        side = rnd.choice(SIDES)
        adj = rnd.choice(ADJECTIVES) if rnd.random() < 0.35 else ""
        # positions and features are usually written joined, no space
        joined = rnd.random() < 0.75
        tail = (pos + feat + side + adj) if joined else " ".join(x for x in (pos, feat, side, adj) if x)
        parts.append(tail)
        out.add(LABEL + " ".join(parts))
    for sent in SENTENCES:
        out.add(LABEL + sent)
        # the same sentence with every side, and with/without មាន
        for a, b in (("ឆ្វេង", "ស្តាំ"), ("ស្តាំ", "ឆ្វេង")):
            if a in sent:
                out.add(LABEL + sent.replace(a, b))
        out.add(LABEL + (sent[3:] if sent.startswith("មាន") else "មាន" + sent))
    # explicit combinations, each also as a full marks line
    for ex in EXAMPLES:
        out.add(LABEL + ex)
        out.add(LABEL + "ប្រជ្រុយដុះ លើ" + ex)
        out.add(LABEL + "ស្នាម ក្រោម" + ex)
    # the ចុង compounds, with every position prefix and side
    for c in CHONG:
        for pre in ("", "ក្រោយ", "ក្រោម", "លើ", "ក្រោយលើ", "ក្រោមខាង", "ខាង"):
            for side in SIDES:
                out.add(LABEL + "ប្រជ្រុយដុះ " + pre + c + side)
                out.add(LABEL + "ស្នាម " + pre + c + side)
    # every feature name on its own as well, so rare ones get counts
    for f in FEATURES:
        out.add(LABEL + "ប្រជ្រុយដុះ លើ" + f)
        out.add(LABEL + "ស្នាម ក្រោម" + f + "ឆ្វេង")
        out.add(LABEL + "ស្នាម ក្រោម" + f + "ស្តាំ")
    return sorted(out)


class CountLM:
    """Same table shape as khmer_ocr.CharLM (ctx[context][ch], tot[context])."""

    BACKOFF = 0.4

    def __init__(self, order: int = 5):
        self.order = order
        self.ctx: dict[str, dict[str, int]] = {}
        self.tot: dict[str, int] = {}

    def fit(self, lines: list[str]) -> "CountLM":
        for line in lines:
            s = " " + line + " "
            for i in range(len(s)):
                for k in range(0, self.order):
                    if i - k < 0:
                        break
                    context, ch = s[i - k:i], s[i]
                    d = self.ctx.setdefault(context, {})
                    d[ch] = d.get(ch, 0) + 1
                    self.tot[context] = self.tot.get(context, 0) + 1
        return self

    def logp(self, context: str, ch: str) -> float:
        context = context[-(self.order - 1):]
        penalty = 0.0
        while True:
            d = self.ctx.get(context)
            if d is not None:
                k = d.get(ch)
                if k:
                    return math.log(k / self.tot[context]) + penalty
            if not context:
                return math.log(1e-9) + penalty
            context = context[1:]
            penalty += math.log(self.BACKOFF)


class MixLM:
    """Linear interpolation of two character LMs, in probability space.
    Anything with a .logp(context, ch) works, so it plugs straight into
    khmer_ocr.beam_decode in place of the general model."""

    def __init__(self, general, field, weight: float = 0.6):
        self.g, self.f, self.w = general, field, weight
        # beam_decode reads .order to size the context it passes to logp
        self.order = max(getattr(general, "order", 5), getattr(field, "order", 5))

    def logp(self, context: str, ch: str) -> float:
        a = math.log(1 - self.w) + self.g.logp(context, ch)
        b = math.log(self.w) + self.f.logp(context, ch)
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))


_MARKS_LM = None


def marks_lm() -> CountLM:
    global _MARKS_LM
    if _MARKS_LM is None:
        _MARKS_LM = CountLM(5).fit(mark_phrases())
    return _MARKS_LM


def for_field(name: str, general):
    """The LM to decode a given field with: blended for known-vocabulary
    fields, the general one otherwise."""
    if general is None:
        return None
    if name == "distinguishing_marks":
        return MixLM(general, marks_lm(), 0.6)
    if name in ("place_of_birth", "address_line1", "address_line2"):
        try:
            import gazetteer
            return MixLM(general, gazetteer.address_lm(), 0.6)
        except (ImportError, FileNotFoundError):
            return general
    return general


# ---- post-hoc safety net: snap a clearly-misread feature name -------------

def _dist(a: str, b: str) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_features(text: str) -> str:
    """Replace substrings within one edit of a feature name (length >= 5) by
    the exact name. Tight on purpose: this is a safety net, not the mechanism."""
    terms = sorted((t for t in FEATURES + CHONG if len(t) >= 5), key=len, reverse=True)
    for t in terms:
        if t in text:
            continue
        n = len(t)
        for width in (n, n + 1):          # substitution, or one extra char; never shorter
            for i in range(0, max(0, len(text) - width) + 1):
                seg = text[i:i + width]
                if seg and _dist(seg, t) <= 1:
                    text = text[:i] + t + text[i + width:]
                    break
            else:
                continue
            break
    return text


if __name__ == "__main__":
    lm = marks_lm()
    print(len(mark_phrases()), "training phrases;", len(lm.ctx), "contexts")
    for cand in ("ចិញ្ចើម", "ចំព្វើ"):
        s = sum(lm.logp(("ចុង" + cand)[:i][-4:], ch) for i, ch in enumerate("ចុង" + cand) if i >= 3)
        print(f"  log P({cand!r} | ចុង) = {s:7.2f}")
    print(snap_features("ក្រោមចុងចិញ្ចើមឆ្វេង"), "|", snap_features("ក្រោមចុងចិញ្ចើនឆ្វេង"))
