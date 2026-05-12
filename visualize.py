"""
visualize.py - Play back a recorded trial. Fixed scales + exact numbers.

The same colormap range and arrow scale are used across ALL trials, so
visual differences between trials reflect actual differences in the data.
A numeric panel on the right shows exact values per frame.

Usage:
    python visualize.py                              # latest trial overall
    python visualize.py metal                        # latest trial for material
    python visualize.py metal 1                      # specific material + trial
    python visualize.py dataset/material=metal/trial_001   # explicit path
    python visualize.py --compare metal              # peak-frame depth from every trial
    python visualize.py --stats metal                # per-trial numeric summary

Controls:
    SPACE / p   - pause / play
    n / Right   - next frame
    b / Left    - previous frame
    [ / ]       - slower / faster playback
    q / ESC     - quit
"""

import os
import sys
import json
import glob
import cv2
import numpy as np

DATASET_ROOT = "dataset"

# Fixed display scales — CONSTANT across all trials so a value of X always
# renders the same way. Pick ranges generous enough to cover the observed data
# (depth ~0..0.7, defo magnitude ~0..1.3, shear magnitude ~0..0.7 in the
# current dataset) without saturating.
DEPTH_RANGE = (0.0, 0.8)        # depth values mapped to colormap
DEFO_RANGE = (0.0, 1.5)         # deformation magnitude colormap
SHEAR_RANGE = (0.0, 1.0)        # shear magnitude colormap
DEFO_ARROW_PX_PER_UNIT = 60.0   # 60 px per unit of deformation
SHEAR_ARROW_PX_PER_UNIT = 60.0


def find_trial(args):
    if not args:
        c = sorted(glob.glob(os.path.join(DATASET_ROOT, "material=*", "trial_*")))
        if not c:
            sys.exit(f"No trials found under {DATASET_ROOT}/")
        return c[-1]
    a0 = args[0]
    if os.path.isdir(a0):
        return a0
    mat_dir = os.path.join(DATASET_ROOT, f"material={a0}")
    if not os.path.isdir(mat_dir):
        sys.exit(f"Material '{a0}' not found at {mat_dir}")
    if len(args) >= 2:
        td = os.path.join(mat_dir, f"trial_{int(args[1]):03d}")
        if not os.path.isdir(td):
            sys.exit(f"Trial not found: {td}")
        return td
    trials = sorted(glob.glob(os.path.join(mat_dir, "trial_*")))
    if not trials:
        sys.exit(f"No trials in {mat_dir}")
    return trials[-1]


def load_trial(trial_dir):
    return {
        "raw":   np.load(os.path.join(trial_dir, "rawimg.npy")),
        "depth": np.load(os.path.join(trial_dir, "depth.npy")),
        "defo":  np.load(os.path.join(trial_dir, "deformation.npy")),
        "shear": np.load(os.path.join(trial_dir, "shear.npy")),
        "meta":  json.load(open(os.path.join(trial_dir, "meta.json"))),
        "name":  trial_dir,
    }


def normalize(arr, lo, hi):
    rng = hi - lo
    if rng < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / rng, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def put_arrows(image, vec_field, px_per_unit, color=(255, 255, 255)):
    image = image.copy()
    f = vec_field * px_per_unit
    h, w = f.shape[:2]
    grid = np.stack(
        np.meshgrid(range(0, w, max(1, w // 20)),
                    range(0, h, max(1, h // 20))), 2)
    end = (f[grid[:, :, 1], grid[:, :, 0], :] + grid).astype(np.int32)
    norm = np.linalg.norm(f[grid[:, :, 1], grid[:, :, 0], :], axis=2)
    nz = np.nonzero(norm > 1.0)
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image, tuple(grid[y, x]), tuple(end[y, x]),
                        color, 1, tipLength=0.3)
    return image


def label_tile(img, label, sub=None):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)
    if sub:
        cv2.putText(img, sub, (img.shape[1] - 240, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def numeric_panel(width, height, frame_idx, n_frames, raw, depth, defo, shear, meta):
    """Right-side panel of exact numerical values for the current frame."""
    p = np.zeros((height, width, 3), dtype=np.uint8)
    defo_mag = np.linalg.norm(defo, axis=2)
    shear_mag = np.linalg.norm(shear, axis=2)

    lines = [
        ("MATERIAL",    str(meta.get("material", "?"))),
        ("TRIAL",       str(meta.get("trial", "?"))),
        ("FRAME",       f"{frame_idx + 1} / {n_frames}"),
        ("",            ""),
        ("RAW IMAGE",   ""),
        ("  shape",     f"{raw.shape[0]}x{raw.shape[1]} u8"),
        ("  min/mean/max", f"{raw.min()}/{raw.mean():.1f}/{raw.max()}"),
        ("",            ""),
        ("DEPTH",       ""),
        ("  range",     f"{depth.min():.4f} .. {depth.max():.4f}"),
        ("  mean",      f"{depth.mean():.4f}"),
        ("  display",   f"[{DEPTH_RANGE[0]} .. {DEPTH_RANGE[1]}]"),
        ("",            ""),
        ("DEFORMATION", ""),
        ("  |v| max",   f"{defo_mag.max():.4f}"),
        ("  |v| mean",  f"{defo_mag.mean():.4f}"),
        ("  display",   f"[{DEFO_RANGE[0]} .. {DEFO_RANGE[1]}]"),
        ("",            ""),
        ("SHEAR",       ""),
        ("  |v| max",   f"{shear_mag.max():.4f}"),
        ("  |v| mean",  f"{shear_mag.mean():.4f}"),
        ("  display",   f"[{SHEAR_RANGE[0]} .. {SHEAR_RANGE[1]}]"),
    ]
    y = 24
    for k, v in lines:
        if k.startswith("  "):
            cv2.putText(p, k, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (180, 180, 180), 1)
            cv2.putText(p, v, (140, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 0), 1)
        elif k:
            cv2.putText(p, k, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 255, 255), 1)
            if v:
                cv2.putText(p, v, (140, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.50, (255, 255, 255), 1)
        y += 22
    return p


def make_panel(data, idx, fps):
    raw = data["raw"][idx]
    depth = data["depth"][idx]
    defo = data["defo"][idx]
    shear = data["shear"][idx]
    meta = data["meta"]

    raw3 = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    depth_img = cv2.applyColorMap(
        normalize(depth, *DEPTH_RANGE), cv2.COLORMAP_TURBO)

    defo_mag = np.linalg.norm(defo, axis=2)
    defo_heat = cv2.applyColorMap(
        normalize(defo_mag, *DEFO_RANGE), cv2.COLORMAP_VIRIDIS)
    defo_img = put_arrows(defo_heat, defo, DEFO_ARROW_PX_PER_UNIT)

    shear_mag = np.linalg.norm(shear, axis=2)
    shear_heat = cv2.applyColorMap(
        normalize(shear_mag, *SHEAR_RANGE), cv2.COLORMAP_PLASMA)
    shear_img = put_arrows(shear_heat, shear, SHEAR_ARROW_PX_PER_UNIT)

    label_tile(raw3, "raw", f"u8 [0..255]")
    label_tile(depth_img, "depth", f"max={depth.max():.3f}")
    label_tile(defo_img, "deformation", f"|v|max={defo_mag.max():.3f}")
    label_tile(shear_img, "shear", f"|v|max={shear_mag.max():.3f}")

    top = np.hstack([raw3, depth_img])
    bot = np.hstack([defo_img, shear_img])
    images = np.vstack([top, bot])

    panel_h = images.shape[0]
    panel = numeric_panel(360, panel_h, idx, len(data["raw"]),
                          raw, depth, defo, shear, meta)
    full = np.hstack([images, panel])

    bar = np.zeros((28, full.shape[1], 3), dtype=np.uint8)
    status = (f"{meta.get('material', '?')} | trial {meta.get('trial', '?')} | "
              f"frame {idx + 1}/{len(data['raw'])} | playback {fps:.0f} FPS  "
              f"|  scales: depth{DEPTH_RANGE} defo{DEFO_RANGE} shear{SHEAR_RANGE}")
    cv2.putText(bar, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1)
    return np.vstack([full, bar])


def cmd_compare(material):
    """Tile peak-deformation depth frame from every trial of a material."""
    mat_dir = os.path.join(DATASET_ROOT, f"material={material}")
    trials = sorted(glob.glob(os.path.join(mat_dir, "trial_*")))
    if not trials:
        sys.exit(f"No trials in {mat_dir}")
    print(f"Comparing {len(trials)} trials of '{material}' "
          f"(depth at peak-deformation frame, fixed scale {DEPTH_RANGE})")

    tiles = []
    for t in trials:
        depth = np.load(os.path.join(t, "depth.npy"))
        defo_mag = np.linalg.norm(np.load(os.path.join(t, "deformation.npy")), axis=-1)
        peak = int(defo_mag.reshape(len(defo_mag), -1).max(axis=1).argmax())
        d = depth[peak]
        img = cv2.applyColorMap(normalize(d, *DEPTH_RANGE), cv2.COLORMAP_TURBO)
        cv2.putText(img, f"{os.path.basename(t)}  pk f{peak}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"d_max={d.max():.3f}",
                    (5, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        tiles.append(img)

    cols = min(5, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        grid[(i // cols) * h:((i // cols) + 1) * h,
             (i % cols) * w:((i % cols) + 1) * w] = t
    cv2.imshow(f"compare {material}", grid)
    print("Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cmd_stats(material):
    """Print exact numeric summary for every trial of a material."""
    mat_dir = os.path.join(DATASET_ROOT, f"material={material}")
    trials = sorted(glob.glob(os.path.join(mat_dir, "trial_*")))
    if not trials:
        sys.exit(f"No trials in {mat_dir}")

    hdr = f"{'trial':<10} {'raw_mean':>9} {'raw_std':>8} " \
          f"{'depth_max':>10} {'depth_mean':>11} " \
          f"{'defo_max':>10} {'defo_mean':>10} " \
          f"{'shear_max':>10} {'shear_mean':>10}"
    print(hdr)
    print("-" * len(hdr))
    for t in trials:
        raw = np.load(os.path.join(t, "rawimg.npy"))
        depth = np.load(os.path.join(t, "depth.npy"))
        defo = np.linalg.norm(np.load(os.path.join(t, "deformation.npy")), axis=-1)
        shear = np.linalg.norm(np.load(os.path.join(t, "shear.npy")), axis=-1)
        print(f"{os.path.basename(t):<10} "
              f"{raw.mean():>9.2f} {raw.std():>8.2f} "
              f"{depth.max():>10.4f} {depth.mean():>11.4f} "
              f"{defo.max():>10.4f} {defo.mean():>10.4f} "
              f"{shear.max():>10.4f} {shear.mean():>10.4f}")


def main():
    args = sys.argv[1:]

    if args and args[0] == "--compare":
        if len(args) < 2:
            sys.exit("Usage: visualize.py --compare <material>")
        cmd_compare(args[1])
        return
    if args and args[0] == "--stats":
        if len(args) < 2:
            sys.exit("Usage: visualize.py --stats <material>")
        cmd_stats(args[1])
        return

    trial_dir = find_trial(args)
    print(f"Loading: {trial_dir}")
    data = load_trial(trial_dir)
    n = len(data["raw"])
    print(f"  {n} frames")
    print(f"  fixed display scales: depth={DEPTH_RANGE} defo={DEFO_RANGE} "
          f"shear={SHEAR_RANGE}  arrows=60 px/unit")
    print("  SPACE/p=pause  n/Right=next  b/Left=prev  [/]=slower/faster  q=quit")

    meta = data["meta"]
    fps = float(meta.get("frames", n)) / max(meta.get("duration_sec", 1.0), 1e-3)
    if fps < 1 or fps > 120:
        fps = 30.0
    paused = False
    idx = 0
    win = "trial"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(win, make_panel(data, idx, fps))
        delay = 1 if paused else max(1, int(1000.0 / fps))
        k = cv2.waitKey(delay) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k in (ord('p'), ord(' ')):
            paused = not paused
        elif k in (ord('n'), 83):
            idx = (idx + 1) % n
            paused = True
        elif k in (ord('b'), 81):
            idx = (idx - 1) % n
            paused = True
        elif k == ord(']'):
            fps = min(fps * 1.5, 120)
        elif k == ord('['):
            fps = max(fps / 1.5, 1)
        elif not paused:
            idx = (idx + 1) % n

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
