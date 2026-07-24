"""
Futsal frame prep for FutStats.

Turns a match video into a curated, label-ready set of still frames in ONE run:

  1. SAMPLE   — grab frames at a fixed rate (default 1 fps) into
                data/futsal_frames_raw/.
  2. CURATE   — drop near-duplicate frames (neighbouring frames where almost
                nothing moved) using the same mean-pixel-difference threshold as
                the original curate_frames.py. NOTE: on handheld/wide footage
                sampled at 1 fps there are usually no near-duplicates to remove
                (every frame differs by more than the threshold); the step still
                matters if you sample at a higher fps.
  3. FINALISE — copy survivors to data/futsal_frames/ (clean names, untouched
                images ready for upload to Auta) and report the count.
  4. PREVIEW  — save a 6-frame contact sheet spread across the match for a quick
                quality spot-check.
  5. CONTACT  — save high-res contact sheets of ALL survivors (grids) into
                data/futsal_frames_contact/ so you can eyeball them and pick the
                frames where the ball is clearly visible during manual curation.

Why no automatic ball flag? On this wide, elevated shot the ball is small and
every frame is full of white distractors (shoes, line intersections, the far
boundary line, sun glints), so a simple Hough/white-blob heuristic flags almost
every frame and gives no useful signal. Once you have a trained futsal model,
detection is the right tool — for now the contact sheets let you pick by eye.

Everything lives under data/ which is gitignored — nothing here gets committed.

Run it with:
    python src/extract_frames.py                                   # defaults
    python src/extract_frames.py videos/match1.mov data/futsal_frames --fps 1
    python src/extract_frames.py "videos/my clip.mov" data/my_frames --fps 2
"""

import os
import glob
import shutil
import argparse

import cv2
import numpy as np

# --- Defaults (used when no command-line arguments are given) --------------
DEFAULT_VIDEO = "videos/match1.mov"
DEFAULT_OUTPUT_DIR = "data/futsal_frames"      # final, curated frames
DEFAULT_FPS = 1.0                              # frames to sample per second

# --- Near-duplicate curation (same approach as curate_frames.py) -----------
# Average per-pixel brightness difference (0-255) below which two frames are
# treated as near-duplicates. Lower = stricter (keeps more); higher = drops more.
DIFF_THRESHOLD = 4.0
THUMB_SIZE = (320, 180)     # shrink before comparing: fast, ignores pixel noise

# --- Previews --------------------------------------------------------------
PREVIEW_SAMPLES = 6         # frames in the quick quality spot-check
PREVIEW_COLS = 3
PREVIEW_TILE_W = 520        # px width per tile in the 6-frame preview

CONTACT_COLS = 6            # grid width of the full contact sheets
CONTACT_ROWS = 6            # grid height -> 36 frames per sheet
CONTACT_TILE_W = 400        # px width per tile (bigger = ball easier to spot)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clear_frames(directory):
    """Delete this script's own frame_*.jpg output so runs don't mix old + new."""
    old = glob.glob(os.path.join(directory, "frame_*.jpg"))
    for f in old:
        os.remove(f)
    if old:
        print(f"      cleared {len(old)} old frame(s) from {directory}/")


def _thumbnail(path):
    """Load an image as a small grayscale float thumbnail (for diffing)."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, THUMB_SIZE)
    return img.astype("float32")


def _labelled_tile(path, tile_w, fps):
    """Resized frame with a small caption bar (filename + mm:ss into the match)."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    tile = cv2.resize(img, (tile_w, round(tile_w * h / w)))
    # frame_00137 -> 137 -> ~137s into the match at 1 fps
    num = int(os.path.splitext(os.path.basename(path))[0].split("_")[1])
    secs = int((num - 1) / fps)
    label = f"{os.path.basename(path)}  {secs // 60:02d}:{secs % 60:02d}"
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _grid(tiles, cols):
    """Tile a list of equally-sized images into a cols-wide grid (pad last row)."""
    rows = []
    for start in range(0, len(tiles), cols):
        row = tiles[start:start + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


# ---------------------------------------------------------------------------
# Step 1 — sample frames at a fixed rate
# ---------------------------------------------------------------------------
def sample_frames(video_path, raw_dir, target_fps):
    os.makedirs(raw_dir, exist_ok=True)
    _clear_frames(raw_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not video_fps or video_fps <= 0:
        cap.release()
        raise SystemExit("Video reports 0 fps — cannot compute a sampling step.")

    # We save every Nth frame. E.g. 30 fps / 1 target fps = every 30th frame.
    step = max(1, round(video_fps / target_fps))
    duration = total / video_fps if total else 0
    print(f"[1/5] Sampling {video_path}")
    print(f"      {w}x{h}, {total} frames @ {video_fps:.2f} fps "
          f"(~{duration:.0f}s / {duration / 60:.1f} min)")
    print(f"      keeping every {step}th frame  (~{target_fps} fps)")

    idx = saved = 0
    while True:
        # grab() advances the decoder cheaply; retrieve() only pays the full
        # decode+copy cost on the frames we actually keep. On hi-res video this
        # is much faster than read()-ing (decoding) every single frame.
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            saved += 1
            # The number encodes sample order == ~seconds into the match, so it
            # stays meaningful after curation removes some frames.
            cv2.imwrite(os.path.join(raw_dir, f"frame_{saved:05d}.jpg"), frame)
            if saved % 60 == 0:
                print(f"      ...saved {saved} frames")
        idx += 1

    cap.release()
    print(f"      -> {saved} raw frames in {raw_dir}/")
    if saved == 0:
        raise SystemExit("Sampling produced no frames — is the video readable?")
    return saved


# ---------------------------------------------------------------------------
# Step 2 — drop near-duplicate frames
# ---------------------------------------------------------------------------
def curate(raw_dir):
    """Return the ordered list of frame paths to KEEP."""
    frames = sorted(glob.glob(os.path.join(raw_dir, "frame_*.jpg")))
    if not frames:
        raise SystemExit(f"No frames found in {raw_dir}/.")

    kept_paths = [frames[0]]                 # always keep the first frame
    kept_thumb = _thumbnail(frames[0])
    for path in frames[1:]:
        current = _thumbnail(path)
        # Mean absolute difference vs the last KEPT frame (not just the previous
        # file), so a long static run collapses to one representative.
        if np.mean(np.abs(current - kept_thumb)) < DIFF_THRESHOLD:
            continue                         # too similar -> drop
        kept_paths.append(path)
        kept_thumb = current

    dropped = len(frames) - len(kept_paths)
    print(f"[2/5] Curated near-duplicates: {len(frames)} -> {len(kept_paths)} kept "
          f"({dropped} dropped)")
    return kept_paths


# ---------------------------------------------------------------------------
# Step 3 — copy survivors to the final dir (clean, untouched images)
# ---------------------------------------------------------------------------
def finalise(kept_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _clear_frames(output_dir)
    # Remove any stale report from an earlier (heuristic) run.
    stale = os.path.join(output_dir, "ball_report.csv")
    if os.path.exists(stale):
        os.remove(stale)

    for path in kept_paths:
        shutil.copy2(path, os.path.join(output_dir, os.path.basename(path)))
    print(f"[3/5] Wrote {len(kept_paths)} clean frames to {output_dir}/")


# ---------------------------------------------------------------------------
# Step 4 — 6-frame quality spot-check spread across the match
# ---------------------------------------------------------------------------
def quality_preview(kept_paths, output_dir, fps):
    n = min(PREVIEW_SAMPLES, len(kept_paths))
    idxs = ([round(i * (len(kept_paths) - 1) / (n - 1)) for i in range(n)]
            if n > 1 else [0])
    tiles = [_labelled_tile(kept_paths[i], PREVIEW_TILE_W, fps) for i in idxs]
    out = os.path.join(os.path.dirname(output_dir) or ".", "futsal_frames_preview.jpg")
    cv2.imwrite(out, _grid(tiles, PREVIEW_COLS))
    print(f"[4/5] Saved {n}-frame quality preview -> {out}")
    print("      sampled: " + ", ".join(os.path.basename(kept_paths[i]) for i in idxs))
    return out


# ---------------------------------------------------------------------------
# Step 5 — high-res contact sheets of ALL survivors (for visual ball-picking)
# ---------------------------------------------------------------------------
def contact_sheets(kept_paths, output_dir, fps):
    sheets_dir = os.path.join(os.path.dirname(output_dir) or ".",
                              "futsal_frames_contact")
    os.makedirs(sheets_dir, exist_ok=True)
    for old in glob.glob(os.path.join(sheets_dir, "contact_*.jpg")):
        os.remove(old)

    per_sheet = CONTACT_COLS * CONTACT_ROWS
    n_sheets = (len(kept_paths) + per_sheet - 1) // per_sheet
    for s in range(n_sheets):
        chunk = kept_paths[s * per_sheet:(s + 1) * per_sheet]
        tiles = [_labelled_tile(p, CONTACT_TILE_W, fps) for p in chunk]
        out = os.path.join(sheets_dir, f"contact_{s + 1:02d}.jpg")
        cv2.imwrite(out, _grid(tiles, CONTACT_COLS))
    print(f"[5/5] Saved {n_sheets} contact sheet(s) ({per_sheet}/sheet) -> {sheets_dir}/")
    return sheets_dir


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Futsal frame pipeline: sample -> curate -> preview + contact sheets.")
    p.add_argument("video", nargs="?", default=DEFAULT_VIDEO,
                   help=f"input video (default: {DEFAULT_VIDEO})")
    p.add_argument("output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR,
                   help=f"final curated frame dir (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help=f"frames sampled per second (default: {DEFAULT_FPS})")
    p.add_argument("--raw-dir", default=None,
                   help="raw sample dir (default: <output_dir>_raw)")
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = args.raw_dir or (args.output_dir.rstrip("/") + "_raw")

    sample_frames(args.video, raw_dir, args.fps)
    kept = curate(raw_dir)
    finalise(kept, args.output_dir)
    quality_preview(kept, args.output_dir, args.fps)
    contact_sheets(kept, args.output_dir, args.fps)

    print("\n=== DONE ===")
    print(f"final frames : {len(kept)}")
    print(f"location     : {args.output_dir}/  (gitignored — not committed)")
    print("next         : open the preview to check quality, then scan the contact")
    print("               sheets (or browse the frames full-res) to pick ball-visible")
    print("               frames before uploading to Auta.")


if __name__ == "__main__":
    main()
