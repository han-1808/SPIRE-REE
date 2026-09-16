"""
figures/flatten_png.py

Convert every PNG under figures/ (and optionally other directories) from RGBA to
plain 8-bit RGB and strip ancillary chunks.

Why this exists: matplotlib always writes PNGs with an alpha channel, and every
figure in this paper is fully opaque, so that channel carries no information.
pdfTeX still has to take its alpha code path for such files, and some pdfTeX
builds abort there with

    pdfTeX error: pdflatex: libpng: internal error

The conversion is lossless for the RGB pixels -- it is verified byte-for-byte
below -- and removes the failure mode entirely.

Run after regenerating any figure:

    python figures/flatten_png.py
"""

import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Chunks pdfTeX has no use for here; the image is always placed at an explicit
# width, so pHYs (physical pixel size) does not affect the layout either.
_KEEP_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}


def _strip_ancillary(path: Path) -> list[str]:
    data = path.read_bytes()
    out = bytearray(data[:8])
    pos, dropped = 8, []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        if ctype in _KEEP_CHUNKS:
            out += data[pos:pos + 12 + length]
        else:
            dropped.append(ctype.decode("ascii", "replace"))
        pos += 12 + length
        if ctype == b"IEND":
            break
    if dropped:
        path.write_bytes(bytes(out))
    return dropped


def flatten(path: Path) -> str:
    """Make `path` a plain RGB PNG. Returns a one-line status string."""
    before = np.asarray(Image.open(path).convert("RGB")).copy()
    im = Image.open(path)

    notes = []
    if im.mode != "RGB":
        if im.mode == "RGBA":
            alpha_min = im.getchannel("A").getextrema()[0]
            if alpha_min != 255:
                # Real transparency would be lost by a bare convert(), so
                # composite onto white explicitly and say so.
                im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
                notes.append("composited transparency onto white")
        im.convert("RGB").save(path, optimize=True)
        notes.append(f"{im.mode} -> RGB")

    dropped = _strip_ancillary(path)
    if dropped:
        notes.append("dropped " + ",".join(dropped))

    after = np.asarray(Image.open(path).convert("RGB"))
    if after.shape != before.shape or not np.array_equal(before, after):
        raise SystemExit(f"ABORT: {path} changed visually; restore it from git")

    return f"{path.name}: {'; '.join(notes) if notes else 'already plain RGB'}"


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or [Path(__file__).parent]
    files = sorted(
        p for t in targets for p in ([t] if t.is_file() else t.rglob("*.png"))
    )
    if not files:
        print("no PNG files found")
        return 1
    for f in files:
        print("  " + flatten(f))
    print(f"{len(files)} file(s) verified pixel-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
