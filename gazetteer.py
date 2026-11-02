#!/usr/bin/env python3
"""
gazetteer.py - Cambodian place names for the address fields of the ID card.

The card writes an address as a chain of administrative units, each glued to
its type word, province last and bare:

    ភូមិព្រៃធាតុ                       (address line 1: village)
    ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ         (address line 2 / place of birth)

Every name is drawn from a fixed hierarchy: 25 provinces, ~210 districts,
~1,650 communes, ~14,600 villages. So a read is corrected *hierarchically*:
province first (25 candidates), then district within that province, commune
within that district, village within that commune. Each level narrows the next
from thousands of options to a handful, which makes even a badly misread name
recoverable, and makes an impossible combination detectable.

Also builds a character LM over all canonical address strings, blended into
the beam search for the three address fields (see field_lm.for_field).

Data: cambodia_gazetteer.json (nested provinces > districts > communes >
villages, with Khmer type words). Supplied by Vattanac.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAZ_PATH = HERE / "cambodia_gazetteer.json"

TYPE_WORDS = ["រាជធានី", "ខេត្ត", "ក្រុង", "ស្រុក", "ខណ្ឌ", "សង្កាត់", "ឃុំ", "ភូមិ"]
_TYPE_RE = re.compile("^(" + "|".join(sorted(TYPE_WORDS, key=len, reverse=True)) + ")")
_GRAPHEME = re.compile(r"[ក-អឥ-ឳៜ](?:្[ក-ឳ])*[឴-៑៝៎]*|.")


def graphemes(s: str) -> list[str]:
    return _GRAPHEME.findall(s)


def gdist(a: str, b: str) -> int:
    """Levenshtein distance over Khmer grapheme clusters."""
    A, B = graphemes(a), graphemes(b)
    if not A:
        return len(B)
    prev = list(range(len(B) + 1))
    for i, ca in enumerate(A, 1):
        cur = [i]
        for j, cb in enumerate(B, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- data ----

_DATA = None


def load(path: Path = GAZ_PATH):
    global _DATA
    if _DATA is None:
        _DATA = json.loads(Path(path).read_text(encoding="utf-8"))
    return _DATA


def _best(name: str, candidates, key=lambda x: x["km"]):
    """Closest candidate by cluster distance; returns (item, distance) or (None, None)."""
    best, bd = None, None
    for c in candidates:
        d = gdist(name, key(c))
        if bd is None or d < bd:
            best, bd = c, d
    return best, bd


def _ok(name: str, d: int | None) -> bool:
    """Accept a match if within 40% of the name's cluster length (min 1)."""
    if d is None:
        return False
    return d <= max(1, round(0.4 * len(graphemes(name))))


# --------------------------------------------------------------- parsing ----

def parse(line: str) -> dict:
    """Split an address line into {type: name}. Tokens without a type word
    attach to the previous name, except the last one, which is the province."""
    out, prev = {}, None
    toks = [t for t in re.split(r"\s+", line.strip()) if t]
    for i, tok in enumerate(toks):
        m = _TYPE_RE.match(tok)
        if m:
            prev = m.group(1)
            out[prev] = tok[m.end():]
        elif i == len(toks) - 1 and prev not in ("ខេត្ត", "រាជធានី"):
            out["province"] = tok
        elif prev:
            out[prev] += " " + tok
    if "ខេត្ត" in out:
        out["province"] = out.pop("ខេត្ត")
    if "រាជធានី" in out:
        out["province"] = out.pop("រាជធានី")
    return out


# --------------------------------------------------------------- snapping ----

def snap_chain(commune_line: str, village_line: str | None = None) -> dict:
    """Resolve a commune/district/province line (and optionally a village line)
    against the hierarchy. Returns canonical strings plus per-level distances."""
    data = load()
    p = parse(commune_line)
    res = {"input": commune_line, "levels": {}}

    # ---- province
    prov = None
    if "province" in p:
        prov, d = _best(p["province"], data)
        if not _ok(p["province"], d):
            prov = None
        else:
            res["levels"]["province"] = d

    # ---- district (search within province, or across all if province unknown)
    dname = p.get("ស្រុក") or p.get("ក្រុង") or p.get("ខណ្ឌ")
    dist = None
    if dname:
        pool = prov["districts"] if prov else [dd for pp in data for dd in pp["districts"]]
        dist, d = _best(dname, pool)
        if _ok(dname, d):
            res["levels"]["district"] = d
            if prov is None:                            # recover the province from the district
                prov = next(pp for pp in data if dist in pp["districts"])
                res["levels"]["province"] = "from district"
        else:
            dist = None

    # ---- commune
    cname = p.get("ឃុំ") or p.get("សង្កាត់")
    comm = None
    if cname and dist:
        comm, d = _best(cname, dist["communes"])
        if _ok(cname, d):
            res["levels"]["commune"] = d
        else:
            comm = None
    elif cname and prov and not dist:                   # commune known, district lost
        pool = [(cc, dd) for dd in prov["districts"] for cc in dd["communes"]]
        (comm, dist), d = _best(cname, pool, key=lambda x: x[0]["km"]) if pool else ((None, None), None)
        if _ok(cname, d):
            res["levels"]["commune"] = d
            res["levels"]["district"] = "from commune"
        else:
            comm = dist = None

    # ---- village (only meaningful inside a resolved commune)
    vill = None
    if village_line and comm:
        v = parse(village_line).get("ភូមិ") or village_line.replace("ភូមិ", "").strip()
        vill, d = _best(v, comm["villages"])
        if _ok(v, d):
            res["levels"]["village"] = d
        else:
            vill = None

    parts = []
    if comm:
        parts.append(comm["type"] + comm["km"])
    if dist:
        parts.append(dist["type"] + dist["km"])
    if prov:
        parts.append(prov["km"])
    res["canonical"] = " ".join(parts) if parts else commune_line
    res["village"] = ("ភូមិ" + vill["km"]) if vill else village_line
    res["resolved"] = {"province": prov and prov["km"], "district": dist and dist["km"],
                       "commune": comm and comm["km"], "village": vill and vill["km"]}
    res["verified"] = bool(prov and dist and comm)
    return res


def snap_fields(fields: dict) -> dict:
    """Apply to the reader's field dict in place; returns a summary."""
    summary = {}
    if "address_line2" in fields:
        r = snap_chain(fields["address_line2"], fields.get("address_line1"))
        if r["verified"]:
            fields["address_line2"] = r["canonical"]
            if "address_line1" in fields and r["resolved"]["village"]:
                fields["address_line1"] = "អាសយដ្ឋាន: " + r["village"]
        summary["address"] = {"verified": r["verified"], **r["resolved"]}
    if "place_of_birth" in fields:
        line = re.sub(r"^ទីកន្លែងកំណើត:\s*", "", fields["place_of_birth"])
        r = snap_chain(line)
        if r["verified"]:
            fields["place_of_birth"] = "ទីកន្លែងកំណើត: " + r["canonical"]
        summary["place_of_birth"] = {"verified": r["verified"], **r["resolved"]}
    return summary


# ------------------------------------------------------------ address LM ----

def address_lines() -> list[str]:
    """Every canonical address string the card could carry."""
    out = []
    for p in load():
        for d in p["districts"]:
            for c in d["communes"]:
                chain = f"{c['type']}{c['km']} {d['type']}{d['km']} {p['km']}"
                out.append(chain)
                out.append("ទីកន្លែងកំណើត: " + chain)
                for v in c["villages"]:
                    out.append("អាសយដ្ឋាន: ភូមិ" + v["km"])
    return out


_ADDR_LM = None


def address_lm():
    global _ADDR_LM
    if _ADDR_LM is None:
        from field_lm import CountLM
        _ADDR_LM = CountLM(5).fit(address_lines())
    return _ADDR_LM


if __name__ == "__main__":
    import sys
    for line in (sys.argv[1:] or ["ឃុំតេជសារ ស្រុកកោះអណ្តែត ត", "ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ"]):
        r = snap_chain(line, "ភូមិព្រៃធាតុ")
        print(f"{line!r}\n   -> {r['canonical']!r}  village={r['village']!r}  verified={r['verified']}  {r['levels']}")
