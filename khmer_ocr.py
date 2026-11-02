#!/usr/bin/env python3
"""
khmer_ocr.py - inference for the Khmer OCR CRNN-CTC models.

Self-contained: model definition, greedy decoding, and CTC prefix beam search
with a character n-gram language model. No training or data-pipeline code.

Command line
------------
    python khmer_ocr.py --image line.png
    python khmer_ocr.py --image line.png --lm khmer_lm.pkl.gz
    python khmer_ocr.py --image line.png --ckpt khmer_ocr_handwriting.pt --lm khmer_lm.pkl.gz

The input must be a CROPPED TEXT LINE, not a whole page. Line segmentation is
not included in this release.

Python
------
    from khmer_ocr import load_model, load_lm, read_line

    model, itos, device = load_model("khmer_ocr_document.pt")
    lm = load_lm("khmer_lm.pkl.gz")                 # optional, more accurate
    print(read_line("line.png", model, itos, device, lm))

Decoding
--------
Greedy decoding is fastest. Beam search with the language model is roughly 15%
better in relative CER. The defaults (beam 8, alpha 0.35, beta 1.5) were chosen
by a sweep; pushing alpha towards 1.0 makes results WORSE, because the language
model starts to override correct readings.
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
import sys

HEIGHT = 64          # the height the models were trained at; do not change
MAX_WIDTH = 800      # long lines are capped at this width
WIDTH_STRIDE = 8     # the CNN divides width by 8, giving w // 8 timesteps


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(n_classes: int, height: int = HEIGHT):
    """
    CRNN used by both checkpoints.

    A 64 px tall line goes through a CNN that reduces height to 1 and width by
    8, so a 640 px line yields 80 timesteps for roughly 35 characters. CTC needs
    at least one frame per character (plus a blank between repeats), so that
    leaves about 2x headroom.
    """
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self, nclass: int):
            super().__init__()

            def blk(i, o, pool):
                layers = [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True)]
                if pool:
                    layers.append(nn.MaxPool2d(pool))
                return layers

            self.cnn = nn.Sequential(
                *blk(1, 64, (2, 2)),        # 32 x W/2
                *blk(64, 128, (2, 2)),      # 16 x W/4
                *blk(128, 256, None),
                *blk(256, 256, (2, 2)),     #  8 x W/8
                *blk(256, 512, None),
                *blk(512, 512, (2, 1)),     #  4 x W/8
                *blk(512, 512, (4, 1)),     #  1 x W/8
            )
            self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=0.1)
            self.head = nn.Linear(512, nclass)

        def forward(self, x):
            f = self.cnn(x)                      # B x C x 1 x T
            f = f.squeeze(2).permute(0, 2, 1)    # B x T x C
            f, _ = self.rnn(f)
            return self.head(f)                  # B x T x nclass

    return CRNN(n_classes)


def pick_device(want: str = "auto"):
    import torch
    if want != "auto":
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt: str = "khmer_ocr_document.pt", device: str = "auto"):
    """Returns (model, itos, device). itos maps class index - 1 to a character."""
    import torch
    dev = pick_device(device)
    state = torch.load(ckpt, map_location="cpu")
    itos = state["vocab"]["itos"]
    model = build_model(len(itos) + 1, HEIGHT).to(dev)
    model.load_state_dict(state["model"])
    model.eval()
    return model, itos, dev


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def greedy_decode(logits, itos) -> list[str]:
    """logits: B x T x C -> list of strings (collapse repeats, drop the blank)."""
    out = []
    for seq in logits.argmax(-1).cpu().tolist():
        prev, chars = 0, []
        for k in seq:
            if k != prev and k != 0:
                chars.append(itos[k - 1])
            prev = k
        out.append("".join(chars))
    return out


class CharLM:
    """
    Character n-gram with stupid backoff, used only to rescore a CTC beam.

    This is the inference half of the model: it can score text, not build a new
    one. Load a released .pkl.gz with CharLM.load().
    """

    BACKOFF = 0.4

    def __init__(self, order: int = 5):
        self.order = order
        self.ctx: dict[str, dict[str, int]] = {}
        self.tot: dict[str, int] = {}

    def logp(self, context: str, ch: str) -> float:
        """log P(ch | context), backing off one order at a time to unigram."""
        context = context[-(self.order - 1):]
        penalty = 0.0
        while True:
            d = self.ctx.get(context)
            if d is not None:
                k = d.get(ch)
                if k:
                    return math.log(k / self.tot[context]) + penalty
            if not context:
                return math.log(1e-9) + penalty          # unseen character
            context = context[1:]
            penalty += math.log(self.BACKOFF)

    @staticmethod
    def load(path: str) -> "CharLM":
        # NOTE: this is a pickle. Only load language-model files you trust.
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        lm = CharLM(d["order"])
        lm.ctx, lm.tot = d["ctx"], d["tot"]
        return lm


def load_lm(path: str | None) -> CharLM | None:
    return CharLM.load(path) if path else None


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(logp, itos, lm: CharLM | None, beam: int = 8,
                alpha: float = 0.35, beta: float = 1.5, prune: float = -9.0) -> str:
    """
    CTC prefix beam search for ONE line. logp is T x C log-probabilities.

    Each beam tracks the probability of its prefix ending in a blank separately
    from ending in a real character, because that distinction decides whether
    the next identical character extends a run or starts a repeat.
    """
    NEG = -math.inf
    beams = {(): (0.0, NEG)}                     # prefix -> (log p_blank, log p_non_blank)

    for t in range(len(logp)):
        row = logp[t]
        cand = [c for c in range(len(row)) if row[c] > prune]
        if not cand:
            cand = [max(range(len(row)), key=lambda c: row[c])]

        nxt: dict[tuple, list] = {}

        def add(prefix, pb, pnb):
            cur = nxt.get(prefix)
            if cur is None:
                nxt[prefix] = [pb, pnb]
            else:
                cur[0] = _logsumexp(cur[0], pb)
                cur[1] = _logsumexp(cur[1], pnb)

        for prefix, (pb, pnb) in beams.items():
            ptot = _logsumexp(pb, pnb)
            for c in cand:
                p = row[c]
                if c == 0:                                   # blank
                    add(prefix, ptot + p, NEG)
                    continue
                last = prefix[-1] if prefix else None
                if c == last:
                    add(prefix, NEG, pnb + p)                # repeat without a blank
                    ext = prefix + (c,)
                    s = pb + p
                else:
                    ext = prefix + (c,)
                    s = ptot + p
                if lm is not None:
                    ctx = "".join(itos[k - 1] for k in prefix[-(lm.order - 1):])
                    s += alpha * lm.logp("\x02" + ctx if not prefix else ctx,
                                         itos[c - 1]) + beta
                add(ext, NEG, s)

        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -_logsumexp(kv[1][0], kv[1][1]))[:beam])
        beams = {k: (v[0], v[1]) for k, v in beams.items()}

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return "".join(itos[k - 1] for k in best)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

def preprocess(image, height: int = HEIGHT, max_width: int = MAX_WIDTH):
    """PIL image or path -> (1,1,H,W) tensor in [-1,1], and its width."""
    import numpy as np
    import torch
    from PIL import Image

    im = Image.open(image) if isinstance(image, (str, bytes)) else image
    im = im.convert("L")
    w = max(8, int(round(im.width * height / im.height)))
    w = min(w, max_width)
    a = np.asarray(im.resize((w, height)), dtype="float32") / 255.0
    return torch.from_numpy((a - 0.5) / 0.5)[None, None], w


def read_line(image, model, itos, device, lm: CharLM | None = None,
              beam: int = 8, alpha: float = 0.35, beta: float = 1.5) -> str:
    """Recognise one cropped text line. Returns the recognised string."""
    import torch
    x, w = preprocess(image)
    with torch.no_grad():
        logits = model(x.to(device))
    if lm is None:
        return greedy_decode(logits, itos)[0]
    T = max(1, w // WIDTH_STRIDE)
    return beam_decode(logits.log_softmax(-1)[0, :T].cpu().tolist(),
                       itos, lm, beam, alpha, beta)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="cropped line image")
    ap.add_argument("--ckpt", default="khmer_ocr_document.pt",
                    help="khmer_ocr_document.pt or khmer_ocr_handwriting.pt")
    ap.add_argument("--lm", help="khmer_lm.pkl.gz; omit for greedy decoding")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.35, help="language-model weight")
    ap.add_argument("--beta", type=float, default=1.5, help="length bonus")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--quiet", action="store_true", help="print only the text")
    args = ap.parse_args()

    model, itos, dev = load_model(args.ckpt, args.device)
    lm = load_lm(args.lm)
    text = read_line(args.image, model, itos, dev, lm, args.beam, args.alpha, args.beta)

    if not args.quiet:
        how = (f"beam+LM (beam={args.beam}, alpha={args.alpha}, beta={args.beta})"
               if lm else "greedy")
        print(f"device {dev}, {len(itos)} classes, decoder = {how}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
