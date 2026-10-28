#!/usr/bin/env python3
"""
name_check.py - repair the Khmer name using the Latin name printed beside it.

The card prints the holder's name twice: in Khmer, and in Latin letters that
also appear in MRZ line 3, where a check digit makes them trustworthy. The
Khmer field has no checksum of its own, and it is the last field that still
misreads: on one test card the model returned ហុក where the card says សុក.

ហុក romanises to HOK. The Latin printing says SOK. Those cannot both be the
name, and the Latin one is the one backed by a check digit.

So this does what the MRZ repair already does for the machine-readable zone:
propose corrections, then test them against something authoritative. Here the
proposals come from the Latin spelling and the test is the model's own CTC
likelihood for the field image, so a correction is accepted only when the
pixels genuinely support it.

    candidates   substitute Khmer consonants whose romanisation matches the
                 Latin letter the skeleton is missing
    score        CTC forward probability of each candidate given the logits
                 the recogniser already produced for that crop
    accept       the best-scoring candidate that matches the Latin skeleton

Which authority decides what matters here. The Latin printing decides WHICH
NAME it is; the CTC score only decides WHICH SPELLING of that name the pixels
best support. An earlier version had this backwards - it required the winning
candidate to out-score the original read, and on card 2 the model was
confidently wrong: it scored the misread at -4.5 and the correct name at -13.7, so the
repair never fired. The model is exactly the thing that already failed on this
field; deferring to it defeats the purpose.

`floor` is a sanity limit, not a contest with the original: a candidate that
scores absurdly badly means the Latin name belongs to a different line, and
then the Khmer read stands.

Nothing here invents a name. Every candidate is a spelling the image supports
and the Latin printing predicts.
"""

from __future__ import annotations

import math

# Khmer consonants to the Latin letters they are written with in names, from
# the mapping Vattanac supplied. Both series collapse to the same skeleton
# letter (គ and ក are both K, ធ and ថ both TH): the point is to tell T from
# PH, not to produce a correct romanisation. Cambodian romanisation is not
# standardised, so aspiration is collapsed again in ASPIRATION below - សុខ is
# written SOK, not SOKH.
# Loaded from khmer_latin_map.json so the mapping can be corrected without
# touching code, the same way phinpheak_vocab.json and cambodia_gazetteer.json
# hold their vocabularies. The literal below is the fallback.
_FALLBACK_CONSONANT = {
    "ក": "K", "ខ": "KH", "គ": "K", "ឃ": "KH", "ង": "NG",
    "ច": "CH", "ឆ": "CH", "ជ": "CH", "ឈ": "CH", "ញ": "NY",
    "ដ": "D", "ឋ": "TH", "ឌ": "D", "ឍ": "TH", "ណ": "N",
    "ត": "T", "ថ": "TH", "ទ": "T", "ធ": "TH", "ន": "N",
    "ប": "B", "ផ": "PH", "ព": "P", "ភ": "PH", "ម": "M",
    "យ": "Y", "រ": "R", "ល": "L", "វ": "V",
    "ស": "S", "ហ": "H", "ឡ": "L", "ឝ": "SH",
    # អ is a vowel carrier: it writes the vowel, not a consonant, so it
    # contributes nothing to the skeleton (សំណាង is S-M-N-NG, matching SAMNANG).
    "អ": "",
}

def _load_map() -> dict:
    import json as _json
    from pathlib import Path as _Path
    f = _Path(__file__).resolve().parent / "khmer_latin_map.json"
    try:
        d = _json.loads(f.read_text(encoding="utf-8"))
        c = d.get("consonants")
        if isinstance(c, dict) and c:
            return c
    except (OSError, ValueError):
        pass
    return dict(_FALLBACK_CONSONANT)


CONSONANT = _load_map()


# U+17C6 nikahit is written as an M in a romanised name: it is the M of
# SAMNANG in សំណាង, and without it the skeleton would never match.
NIKAHIT = "\u17c6"

# Khmer letters that carry no consonant sound for skeleton purposes
COENG = "្"                      # the subscript marker
DIACRITICS = set("ាិីឹឺុូួើឿ"
                 "ៀេែៃោៅំះៈ៉"
                 "៊់៌៍៎៏័៑៓")

# consonants grouped by the skeleton letter they produce, for proposing swaps
BY_LETTER: dict[str, list[str]] = {}
for _c, _r in CONSONANT.items():
    BY_LETTER.setdefault(_r, []).append(_c)


def skeleton(khmer: str) -> str:
    """The consonant letters of a Khmer word, in order.

    Subscript consonants (after U+17D2) are skipped: they are part of a
    cluster and rarely surface as separate letters in a romanised name."""
    out, i = [], 0
    while i < len(khmer):
        ch = khmer[i]
        if ch == COENG:
            i += 2                                  # skip the subscript too
            continue
        if ch == NIKAHIT:
            out.append("M")
        elif ch in CONSONANT:
            out.append(CONSONANT[ch])
        i += 1
    return "".join(out)


def latin_skeleton(word: str) -> str:
    """The consonant letters of a Latin name, vowels removed."""
    w = "".join(c for c in word.upper() if c.isalpha())
    out, i = [], 0
    while i < len(w):
        if w[i:i + 2] in ("KH", "CH", "TH", "PH", "NG", "NY", "SH"):
            out.append(w[i:i + 2]); i += 2
        elif w[i] in "AEIOU":
            i += 1
        else:
            out.append(w[i]); i += 1
    return "".join(out)


# Aspiration is not reliably written in Khmer romanisation: សុខ is romanised
# SOK, not SOKH. Collapsing it is needed for any comparison of first letters.
ASPIRATION = {"KH": "K", "TH": "T", "PH": "P", "NY": "N", "SH": "S"}


def _first(sk: str) -> str:
    """The first skeleton letter, with aspiration collapsed and B/P merged."""
    for two in ("KH", "CH", "TH", "PH", "NG", "NY", "SH"):
        if sk.startswith(two):
            head = ASPIRATION.get(two, two)
            break
    else:
        head = sk[:1]
    return {"B": "P"}.get(head, head)


def worth_repairing(khmer_word: str, latin_word: str) -> bool:
    """Whether a skeleton mismatch is real enough to act on.

    The skeleton is a crude romanisation and it disagrees with correct names
    more often than it should: it reads សុខ as S-KH against SOK's S-K, and it
    splits the ដ្ឋ cluster in ពិសិដ្ឋ so that it misses PISETH's TH. Acting on
    those would rewrite names that were already right.

    The first consonant is the part the skeleton gets reliably, and it is
    where a genuinely wrong read shows: ហុក starts H where SOK starts S. So a
    mismatch is only acted on when the first letters differ too."""
    ks, ls = skeleton(khmer_word), latin_skeleton(latin_word)
    if ks == ls:
        return False
    return _first(ks) != _first(ls)


def candidates(khmer: str, target: str, max_subs: int = 2) -> list[str]:
    """Spellings of `khmer` whose skeleton is `target`, by swapping consonants.

    Only base consonants are swapped, and only for another consonant that
    produces the needed skeleton letter, so vowels and diacritics - which the
    model reads well - are never touched."""
    want = latin_skeleton(target)
    positions = [i for i, ch in enumerate(khmer)
                 if ch in CONSONANT and (i == 0 or khmer[i - 1] != COENG)]
    if not positions:
        return []
    out, seen = [], {khmer}

    def walk(word: str, idx: int, subs: int):
        if skeleton(word) == want and word not in seen:
            seen.add(word); out.append(word)
        if subs >= max_subs or idx >= len(positions):
            return
        walk(word, idx + 1, subs)                   # leave this one alone
        p = positions[idx]
        for letter, chars in BY_LETTER.items():
            for c in chars:
                if c == word[p]:
                    continue
                walk(word[:p] + c + word[p + 1:], idx + 1, subs + 1)

    walk(khmer, 0, 0)
    exact = list(out)

    # Also swap just the first consonant.
    #
    # Requiring the whole skeleton to match is too strict: subscript
    # consonants are skipped, so ចិន្នា reduces to CH-N while CHINDA gives
    # CH-N-D, and the correct spelling was never even proposed. The first
    # consonant is the position the skeleton reads reliably, and it is the
    # one worth_repairing keys on, so offer every consonant that could write
    # the Latin name's first letter and let the image decide.
    head = _first(want)
    if head and positions:
        p0 = positions[0]
        for c, r in CONSONANT.items():
            if _first(r) == head and c != khmer[p0]:
                cand = khmer[:p0] + c + khmer[p0 + 1:]
                if cand not in seen:
                    seen.add(cand); out.append(cand)
    # Exact-skeleton candidates first, and repair_name uses only those when
    # any exist. The two tiers must not compete: on card 2 the fallback
    # offered ទិន (first consonant fixed, last left as read) against the
    # exact ទិត, and the model scored the wrong one higher. A candidate that
    # accounts for the whole Latin name is better evidence than one that
    # accounts for its first letter.
    return exact if exact else out


# ------------------------------------------------------------ CTC scoring ----

NEG = -1e30


def _logsumexp(*xs: float) -> float:
    m = max(xs)
    if m <= NEG:
        return NEG
    return m + math.log(sum(math.exp(x - m) for x in xs))


def ctc_logprob(logp, target_ids: list[int]) -> float:
    """Log P(target | logits) under CTC, blank at class 0.

    `logp` is a T x C list of log-softmax rows, exactly what beam_decode is
    fed. Standard forward algorithm over the blank-interleaved label sequence:
    this is the likelihood the recogniser assigns to that exact string, which
    is what decides whether a proposed correction is supported by the pixels."""
    if not target_ids:
        return NEG
    ext = [0]
    for t in target_ids:
        ext += [t, 0]
    S, T = len(ext), len(logp)
    if T < len(target_ids):
        return NEG
    a = [NEG] * S
    a[0] = logp[0][0]
    if S > 1:
        a[1] = logp[0][ext[1]]
    for t in range(1, T):
        prev, a = a, [NEG] * S
        row = logp[t]
        for s in range(S):
            best = prev[s]
            if s > 0:
                best = _logsumexp(best, prev[s - 1])
            if s > 1 and ext[s] != 0 and ext[s] != ext[s - 2]:
                best = _logsumexp(best, prev[s - 2])
            a[s] = best + row[ext[s]]
    return _logsumexp(a[S - 1], a[S - 2] if S > 1 else NEG)


def repair_name(khmer_name: str, latin_name: str, logp, stoi: dict,
                floor: float = 45.0, prefix: str = "",
                trust_latin: bool = True) -> tuple[str, bool]:
    """Correct the Khmer name toward the Latin one.

    Returns (name, changed). The Khmer name is left alone when its skeleton
    already matches the Latin one, when no candidate spelling exists, when a
    candidate cannot be written in the model's alphabet, or when the best
    candidate scores more than `floor` nats below the original read, which
    says the two names are not the same name at all.

    With `trust_latin` false it reverts to the older, weaker rule of only
    accepting a candidate that out-scores the original read."""
    kh_words = khmer_name.split()
    lat_words = [w for w in latin_name.upper().split() if w]
    if len(kh_words) != len(lat_words) or not kh_words:
        return khmer_name, False

    def ids(text: str):
        # the logits cover the whole crop, label included, so every candidate
        # is scored with the same prefix in front of it
        try:
            return [stoi[c] for c in prefix + text]
        except KeyError:
            return None

    base = ids(khmer_name)
    if base is None:
        return khmer_name, False
    original = ctc_logprob(logp, base)
    best_text, changed = khmer_name, False

    # words are corrected independently, then reassembled
    for i, (kw, lw) in enumerate(zip(kh_words, lat_words)):
        if not worth_repairing(kw, lw):
            continue
        scored = []
        for cand in candidates(kw, lw):
            trial = " ".join(kh_words[:i] + [cand] + kh_words[i + 1:])
            tid = ids(trial)
            if tid is not None:
                scored.append((ctc_logprob(logp, tid), trial))
        if not scored:
            continue
        top, trial = max(scored)
        if top < original - floor:
            continue                       # not the same name; leave it alone
        if trust_latin or top > ctc_logprob(logp, ids(best_text)):
            best_text, changed = trial, True
        kh_words = best_text.split()

    return best_text, changed


if __name__ == "__main__":
    print("skeleton ហុក  =", skeleton("ហុក"), " សុក =", skeleton("សុក"),
          " សំណាង =", skeleton("សំណាង"))
    print("latin SOK =", latin_skeleton("SOK"), " SAMNANG =", latin_skeleton("SAMNANG"))
    c = candidates("ហុក", "SOK")
    print(f"{len(c)} candidates for ហុក -> SOK; សុក present:", "សុក" in c)
    print("first few:", c[:8])
