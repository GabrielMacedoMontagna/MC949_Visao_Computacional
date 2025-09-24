#!/usr/bin/env python3
"""
make_gif_from_images.py — build an animated GIF from images in a folder.

Usage examples:
  python make_gif_from_images.py ./frames out.gif
  python make_gif_from_images.py ./frames out.gif --fps 8 --sort mtime --size 800x600
  python make_gif_from_images.py ./frames out.gif --formats tif,tiff,png,jpg
"""

import argparse
import glob
import os
import re
from typing import List, Tuple, Optional
from PIL import Image

def natural_key(s: str):
    # Sorts "frame2.png" before "frame10.png"
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be like 800x600")

def collect_paths(input_dir: str, formats: List[str]) -> List[str]:
    paths = []
    for ext in formats:
        # accept case-insensitive matches
        patterns = [f"*.{ext.lower()}", f"*.{ext.upper()}"]
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(input_dir, pat)))
    # De-dup while preserving order by using dict
    return list(dict.fromkeys(paths))

def load_frames(paths: List[str], target_size: Optional[Tuple[int, int]], downscale: Optional[float] = None) -> List[Image.Image]:
    frames = []
    for p in paths:
        img = Image.open(p)

        # If a TIFF has multiple pages, just take the first page.
        if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1 and img.format == "TIFF":
            try:
                img.seek(0)
            except Exception:
                pass

        img = img.convert("RGBA")  # robust against mixed modes

        # Resize logic: explicit target_size takes precedence. If not provided,
        # a per-image downscale factor can be applied (0 < downscale <= 1.0).
        if target_size:
            img = img.resize(target_size, resample=Image.LANCZOS)
        elif downscale is not None:
            try:
                if downscale <= 0:
                    raise ValueError
            except Exception:
                raise argparse.ArgumentTypeError("--downscale must be a float > 0 and typically <= 1.0")
            w, h = img.size
            new_w = max(1, int(round(w * downscale)))
            new_h = max(1, int(round(h * downscale)))
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        frames.append(img)
    return frames

def main():
    ap = argparse.ArgumentParser(description="Create an animated GIF from images in a folder.")
    ap.add_argument("input_dir", help="Directory containing frames")
    ap.add_argument("output_gif", help="Output GIF filename, e.g. out.gif")
    ap.add_argument("--formats", default="tif,tiff,png",
                    help="Comma-separated list of extensions to include (default: tif,tiff,png)")
    ap.add_argument("--sort", choices=["name", "mtime"], default="name", help="Sort by filename or modification time")
    ap.add_argument("--reverse", action="store_true", help="Reverse sort order")
    ap.add_argument("--fps", type=float, default=None, help="Frames per second (mutually exclusive with --duration)")
    ap.add_argument("--duration", type=float, default=None, help="Milliseconds per frame (overrides --fps if set)")
    ap.add_argument("--size", type=parse_size, default=None, help="Resize to WIDTHxHEIGHT, e.g. 800x600")
    ap.add_argument("--downscale", type=float, default=None, help="Downscale factor (0.0 < factor <= 1.0) applied per-image; ignored if --size is set")
    ap.add_argument("--loop", type=int, default=0, help="Loop count for GIF (0=infinite)")
    ap.add_argument("--optimize", action="store_true", help="Enable Pillow GIF optimization")
    args = ap.parse_args()

    formats = [f.strip(". ").lower() for f in args.formats.split(",") if f.strip()]
    if not formats:
        raise SystemExit("No formats specified.")

    paths = collect_paths(args.input_dir, formats)
    if not paths:
        raise SystemExit(f"No images found with extensions: {', '.join(formats)}")

    if args.sort == "name":
        paths.sort(key=natural_key, reverse=args.reverse)
    else:
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=args.reverse)

    # Determine duration per frame
    if args.duration is not None:
        duration_ms = args.duration
    elif args.fps is not None and args.fps > 0:
        duration_ms = 1000.0 / args.fps
    else:
        duration_ms = 100  # default 10 fps

    frames = load_frames(paths, args.size, args.downscale)

    # Convert frames to palette mode with adaptive palette to keep size small
    paletted = []
    transparency_needed = False
    for im in frames:
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        alpha = im.split()[-1]
        if alpha.getextrema() != (255, 255):
            transparency_needed = True
        p = im.convert("P", palette=Image.ADAPTIVE, colors=256)
        paletted.append(p)

    first = paletted[0]
    append = paletted[1:] if len(paletted) > 1 else []

    save_kwargs = dict(
        save_all=True,
        append_images=append,
        duration=duration_ms,
        loop=args.loop,
        optimize=args.optimize,
        disposal=2,  # restore to background to reduce trails
    )

    if transparency_needed:
        # Most robust: index 0 is commonly used for transparency
        save_kwargs["transparency"] = 0

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output_gif))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    first.save(args.output_gif, format="GIF", **save_kwargs)
    print(f"GIF saved: {args.output_gif}")
    print(f"Frames used: {len(paths)} | Duration per frame: {duration_ms:.2f} ms | Loop: {args.loop}")
    print(f"Included formats: {', '.join(formats)}")

if __name__ == "__main__":
    main()
