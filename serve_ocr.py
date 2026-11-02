#!/usr/bin/env python3
"""
serve_ocr.py - put the Khmer OCR model behind card_calibrator.html.

    python3 serve_ocr.py            # listens on http://127.0.0.1:8765
    python3 serve_ocr.py --lm       # beam search with the language model

Then open card_calibrator.html, flatten a card, mark (or load) the field boxes
and press "Read card". The page posts the flattened image and the boxes here;
this returns the MRZ fields with their check-digit status and the text of each
Khmer field.

Everything stays on this machine. The server binds to 127.0.0.1 only, so no
other computer can reach it, and nothing is written to disk.

Uses the standard library only (no Flask). The model comes from
read_id_card.py in the same folder, which downloads it from the Hub on first
use and caches it in ~/.cache/khmer_ocr_try.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_id_card as rid                                  # noqa: E402

MODEL = {}                                                  # filled at start-up


def load(use_lm: bool) -> None:
    files = ["khmer_ocr.py", "vocab.json", "khmer_ocr_document.pt"]
    if use_lm:
        files.append("khmer_lm.pkl.gz")
    paths = rid.fetch(files)
    sys.path.insert(0, str(rid.CACHE))
    from khmer_ocr import load_model, load_lm, read_line      # noqa: E402

    model, itos, device = load_model(str(paths["khmer_ocr_document.pt"]))
    lm = load_lm(str(paths["khmer_lm.pkl.gz"])) if use_lm else None
    latin = rid.load_latin_model(load_model)
    MODEL.update(model=model, itos=itos, device=device, lm=lm, read_line=read_line,
                 latin=latin)
    print(f"model ready on {device}, decoding: {'beam + LM' if lm else 'greedy'}"
          f"{', Latin on base checkpoint' if latin else ''}")


def read_card(card, boxes: dict, mrz_top: float, enhance: str = "none") -> dict:
    m = MODEL
    return rid.read_all(card, boxes, m["model"], m["itos"], m["device"], m["lm"], mrz_top,
                        latin=m.get("latin"), snap=True, enhance=enhance)


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        # the page is opened from file://, whose origin is "null"; allow it
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:                           # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        pages = {"/": "id_app.html", "/index.html": "id_app.html",
                 "/m": "id_mobile.html", "/mobile": "id_mobile.html"}
        if self.path in pages:
            page = HERE / pages[self.path]
            if page.exists():
                body = page.read_bytes()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        self._json(200, {"ok": True, "device": str(MODEL.get("device")),
                         "lm": MODEL.get("lm") is not None})

    def do_POST(self) -> None:
        if self.path == "/extract":                         # raw photo in, fields out
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                data = req["image"].split(",", 1)[1]
                from PIL import Image, ImageOps
                import extract_id
                photo = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(data))))
                fields = req.get("fields") or None
                m = MODEL
                res = extract_id.extract(photo, m["model"], m["itos"], m["device"], m["lm"], fields)
                card = res.pop("_card")
                buf = io.BytesIO()
                card.save(buf, "PNG")
                res["card_png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                self._json(200, res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        if self.path == "/detect":                          # where is the card?
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                data = req["image"].split(",", 1)[1]
                from PIL import Image, ImageOps
                import extract_id
                photo = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(data))))
                self._json(200, extract_id.locate_card_any(
                    photo, str(req.get("detector", "auto"))))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        if self.path != "/read":
            self._json(404, {"error": "POST /read, /detect or /extract"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            data = req["image"].split(",", 1)[1]           # strip data:image/png;base64,
            from PIL import Image
            import extract_id
            from PIL import ImageOps as _IO
            card = _IO.exif_transpose(Image.open(io.BytesIO(base64.b64decode(data)))).convert("RGB")
            # A phone can send the raw photo plus the four corners the user
            # dragged, and let OpenCV do the warp. Doing it in JavaScript meant
            # a per-pixel loop over 900,000 pixels on the handset, which was
            # the whole of the wait - the read itself takes about a second.
            quad = req.get("quad")
            if quad and len(quad) == 4:
                import cv2
                import numpy as np
                src = np.float32([[float(x), float(y)] for x, y in quad])
                # Push the corners out from the centre by a little. Dragging
                # by thumb lands just inside the card as often as not, and a
                # quad 1% small clips whatever sits at the end of a line - on
                # one capture it took the height off the birth row and the
                # first date off the validity row. The template boxes are cut
                # back to their own ink anyway, so a slightly generous card
                # costs nothing.
                centre = src.mean(axis=0)
                src = centre + (src - centre) * 1.02
                dst = np.float32([[0, 0], [rid.CARD_W, 0],
                                  [rid.CARD_W, rid.CARD_H], [0, rid.CARD_H]])
                warped = cv2.warpPerspective(
                    cv2.cvtColor(np.asarray(card), cv2.COLOR_RGB2BGR),
                    cv2.getPerspectiveTransform(src, dst), (rid.CARD_W, rid.CARD_H))
                card = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            elif card.size != (rid.CARD_W, rid.CARD_H):
                card = card.resize((rid.CARD_W, rid.CARD_H))
            boxes = req.get("fields") or extract_id.DEFAULT_FIELDS
            m = MODEL
            if req.get("auto_orient", True):
                # settle which way up it is, and lock the crop on the MRZ
                card, deg, _, _, score = extract_id.orient(card, m["model"], m["itos"], m["device"])
            else:
                deg, score = 0, None
            mrz_top = float(req.get("mrz_top", 62.0))

            # Version 1 reads whatever it is given, as before. Version 2 first
            # checks the image is good enough and refuses when it is not,
            # rather than returning confident-looking values read from too few
            # pixels.
            version = str(req.get("version", "v2")).lower()
            src = req.get("source_size") or None
            q = extract_id.quality(card, tuple(src) if src else None)
            if version == "v2" and not q["ok"]:
                self._json(200, {"quality": q, "blocked": True, "version": version,
                                 "message": " ".join(q["reasons"])})
                return

            res = read_card(card, boxes, mrz_top, str(req.get("enhance", "none")))
            res["quality"] = q
            res["version"] = version
            res["rotation_applied"] = deg
            buf = io.BytesIO()
            card.save(buf, "PNG")
            res["card_png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            self._json(200, res)
        except Exception as e:                              # report, don't die
            import traceback
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args) -> None:              # quieter log
        sys.stderr.write("  %s %s\n" % (self.command, self.path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lm", action="store_true", help="beam search with the language model")
    ap.add_argument("--lan", action="store_true",
                    help="also accept connections from the local network, so a phone "
                         "on the same Wi-Fi can use /m. Off by default: this serves "
                         "identity documents and should not be reachable by default.")
    ap.add_argument("--https", action="store_true",
                    help="wrap the socket in TLS with a self-signed certificate. A "
                         "phone browser only allows the live camera on a secure page, "
                         "so /m falls back to the native camera without this.")
    args = ap.parse_args()

    load(args.lm)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    srv = ThreadingHTTPServer((host, args.port), Handler)
    scheme = "http"
    if args.https:
        import ssl
        cert = HERE / "_localhost_cert.pem"
        if not cert.exists():
            import subprocess
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048",
                            "-keyout", str(cert), "-out", str(cert), "-days", "365",
                            "-nodes", "-subj", "/CN=khmer-id-reader"], check=True)
            print(f"wrote a self-signed certificate to {cert.name}")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    print(f"listening on {scheme}://127.0.0.1:{args.port}   (Ctrl-C to stop)")
    print("  /   the desktop upload app")
    print("  /m  the phone app")
    if args.lan:
        import socket as _s
        try:
            probe = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            ip = probe.getsockname()[0]
            probe.close()
            print(f"\non your phone, same Wi-Fi:  {scheme}://{ip}:{args.port}/m")
            if not args.https:
                print("  (plain http, so the phone uses its own camera app rather than "
                      "a live preview; add --https for the live frame)")
        except OSError:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
