import argparse, os, glob, math
from pathlib import Path
import numpy as np
import tifffile
import imageio.v3 as iio

# Optional lens correction
try:
    import lensfunpy, exifread, cv2
    LENS_OK = True
except Exception:
    LENS_OK = False

def _get_tag(tf, name, default=None):
    try:
        return tf.pages[0].tags[name].value
    except KeyError:
        return default

def _black_level_map(tf, shape, channels):
    """
    Build a per-pixel black level map according to BlackLevel & BlackLevelRepeatDim.
    Handles scalar, per-channel, and small repeating tiles.
    """
    bl = _get_tag(tf, "BlackLevel", None)
    if bl is None:
        return np.zeros(shape[:2] + (channels,), dtype=np.float64)

    bl = np.array(bl, dtype=np.float64).ravel()
    h, w = shape[:2]
    if bl.size == 1:
        return np.broadcast_to(bl[0], (h, w, channels)).astype(np.float64)

    # BlackLevelRepeatDim defines a small tile (rows, cols)
    rdim = _get_tag(tf, "BlackLevelRepeatDim", None)
    if rdim is not None:
        rdim = np.array(rdim, dtype=int)
        # Tile has size rdim[0] x rdim[1] x C (or just scalar per-tile)
        tile = bl.reshape((rdim[0], rdim[1], -1))
        if tile.shape[-1] == 1:
            tile = np.repeat(tile, channels, axis=-1)
        # Tile to full image
        reps_y = math.ceil(h / rdim[0])
        reps_x = math.ceil(w / rdim[1])
        tiled = np.tile(tile, (reps_y, reps_x, 1))[:h, :w, :]
        return tiled.astype(np.float64)

    # If we got here, assume per-channel
    if bl.size == channels:
        return np.broadcast_to(bl, (h, w, channels)).astype(np.float64)

    # Fallback
    return np.zeros((h, w, channels), dtype=np.float64)

def _white_level_per_channel(tf, channels):
    wl = _get_tag(tf, "WhiteLevel", None)
    if wl is None:
        return np.ones((1, 1, channels), dtype=np.float64) * 65535.0
    wl = np.array(wl, dtype=np.float64).ravel()
    if wl.size == 1:
        wl = np.repeat(wl, channels)
    return wl.reshape((1, 1, channels))

def _wb_from_as_shot_neutral(tf):
    # AsShotNeutral is the camera RGB of a neutral patch; WB multipliers ~ 1 / neutral, normalized to G=1
    asn = _get_tag(tf, "AsShotNeutral", None)
    if asn is None:
        return None
    asn = np.array(asn, dtype=np.float64).ravel()[:3]
    m = 1.0 / asn
    return m / m[1]  # normalize green to 1

def _baseline_exposure(tf):
    be = _get_tag(tf, "BaselineExposure", 0.0)
    try:
        return float(be)
    except Exception:
        return 0.0

def _apply_lens_correction(img16, dng_path, subject_distance_m):
    if not LENS_OK:
        return img16, {"lens_corrected": False, "reason": "lensfun/cv2 not available"}

    # Extract EXIF (camera model, lens model, focal, fnumber)
    with open(dng_path, "rb") as f:
        tags = exifread.process_file(f, details=False, strict=True)
    model = str(tags.get("Image Model", "")).strip()
    fl = tags.get("EXIF FocalLength")
    fn = tags.get("EXIF FNumber")

    def _ratio(v):
        try:
            return float(v.values[0].num) / float(v.values[0].den)
        except Exception:
            try:
                return float(v.values[0])
            except Exception:
                return None
    focal = _ratio(fl) or 35.0
    fnum = _ratio(fn) or 5.6

    db = lensfunpy.Database()
    cams = db.find_cameras(None, model)
    if not cams:
        return img16, {"lens_corrected": False, "reason": "camera not found"}
    cam = cams[0]
    lenses = db.find_lenses(cam, None)
    if not lenses:
        return img16, {"lens_corrected": False, "reason": "lens not found"}
    lens = lenses[0]

    h, w = img16.shape[:2]
    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(float(focal), float(fnum), max(0.2, float(subject_distance_m)))
    coords = mod.apply_geometry_distortion()
    mapx = coords[:, :, 0].astype(np.float32)
    mapy = coords[:, :, 1].astype(np.float32)
    remapped = cv2.remap(img16, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return remapped, {"lens_corrected": True, "camera": cam.model, "lens": lens.model, "focal_mm": focal, "fnumber": fnum}

def process_linear_dng(path, out_path, wb_user=None, use_as_shot=True, scale=1.0, lens_correct=False, subject_distance_m=10.0):
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()  # expect HxWx3, uint16
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise RuntimeError("Not an RGB Linear DNG.")
        img = arr[..., :3].astype(np.float64)

        # 1) Normalize using black/white levels
        bl = _black_level_map(tf, img.shape, 3)
        wl = _white_level_per_channel(tf, 3)
        img = np.clip(img - bl, 0, None)
        img = img / np.maximum(wl - bl, 1.0)

        # 2) White balance
        if wb_user is not None:
            wb = np.array(wb_user, dtype=np.float64).reshape(3)
        elif use_as_shot:
            wb = _wb_from_as_shot_neutral(tf)
        else:
            wb = None
        if wb is not None:
            img *= wb.reshape((1, 1, 3))

        # 3) Baseline exposure (in EV)
        be = _baseline_exposure(tf)
        img *= (2.0 ** be)

        # 4) Clip to [0,1] (still linear)
        img = np.clip(img, 0.0, 1.0)

    # Optional downscale
    if scale != 1.0:
        try:
            import cv2
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        except Exception:
            from skimage.transform import resize
            h, w = img.shape[:2]
            img = resize(img, (int(h*scale), int(w*scale)), preserve_range=True, anti_aliasing=True)

    # Convert to 16-bit
    img16 = (img * 65535.0 + 0.5).astype(np.uint16)

    # Optional lens geometry correction
    lens_info = {"lens_corrected": False}
    if lens_correct:
        img16, lens_info = _apply_lens_correction(img16, path, subject_distance_m)

    # Write (linear, 16-bit)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".png":
        iio.imwrite(out_path, img16, extension=".png", compression=9)
    else:
        tifffile.imwrite(out_path, img16, photometric="rgb")

    return {"input": str(path), "output": str(out_path), "linear": True, "bit_depth": 16, **lens_info}

def main():
    ap = argparse.ArgumentParser(description="Convert Linear DNG (Photometric 34892) to 16-bit linear TIFF/PNG for panorama.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--format", choices=["png","tiff"], default="tiff")
    ap.add_argument("--wb", nargs=3, type=float, metavar=("R","G","B"),
                    help="Fixed WB multipliers to apply across all frames (e.g., 2.0 1.0 1.5).")
    ap.add_argument("--no-as-shot", action="store_true", help="Ignore AsShotNeutral; use --wb or leave unbalanced.")
    ap.add_argument("--scale", type=float, default=1.0, help="Uniform downscale factor (e.g., 0.5).")
    ap.add_argument("--lens-correct", action="store_true", help="Apply lens geometry correction (lensfunpy).")
    ap.add_argument("--subject-distance-m", type=float, default=10.0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.DNG"))) + \
            sorted(glob.glob(os.path.join(args.input_dir, "*.dng")))
    if not paths:
        raise SystemExit("No DNGs found.")

    ext = ".png" if args.format == "png" else ".tif"
    for i, p in enumerate(paths, 1):
        out = Path(args.output_dir) / (Path(p).stem + ext)
        meta = process_linear_dng(
            p, out,
            wb_user=args.wb,
            use_as_shot=not args.no_as-shot if hasattr(args, "no_as-shot") else not args.no_as_shot,
            scale=args.scale,
            lens_correct=args.lens_correct,
            subject_distance_m=args.subject_distance_m
        )
        print(f"[{i}/{len(paths)}] -> {meta['output']} (linear 16-bit)")

if __name__ == "__main__":
    main()
