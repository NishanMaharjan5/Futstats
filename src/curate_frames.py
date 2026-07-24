"""
Thin out near-duplicate frames from a folder of extracted frames.

Even when we sample every N seconds, some neighbouring frames barely change
(the camera and players hardly moved). Labeling near-identical images wastes
time and teaches the model nothing new. This script walks the frames in order
and deletes any that look almost the same as the previous *kept* frame.

"How similar is too similar?" We shrink both frames to a small grayscale
thumbnail and measure the average per-pixel brightness difference. A big
number = lots changed (keep it); a tiny number = almost identical (drop it).

Run it with (the folder argument is optional):
    python src/curate_frames.py                      # defaults below
    python src/curate_frames.py data/futsal_frames/
"""

import os
import glob
import argparse
import cv2
import numpy as np

# --- Defaults / configuration ---------------------------------------------
DEFAULT_FRAMES_DIR = "data/frames_for_labeling"

# Average pixel difference (0-255 scale) below which two frames are treated as
# near-duplicates. Lower = stricter (removes fewer); higher = removes more.
# ~3-6 is a reasonable range for a mostly-static wide camera shot.
DIFF_THRESHOLD = 4.0

# Size we shrink each frame to before comparing. Small = fast and ignores tiny
# pixel-level noise while still catching real motion.
THUMB_SIZE = (320, 180)


def thumbnail(path):
    """Load an image as a small grayscale thumbnail (a 2D array of brightness)."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, THUMB_SIZE)
    # float so the subtraction below can go negative before we take abs().
    return img.astype("float32")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate frames from a folder of extracted frames."
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        default=DEFAULT_FRAMES_DIR,
        help=f"folder of frame_*.jpg files to curate (default: {DEFAULT_FRAMES_DIR})",
    )
    return parser.parse_args()


def main(frames_dir):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frames:
        raise SystemExit(f"No frames found in {frames_dir}/. Run extract_frames.py first.")

    kept = thumbnail(frames[0])   # always keep the very first frame
    removed = 0

    # Compare each frame to the last one we decided to KEEP (not just the
    # immediately previous file), so a long run of similar frames collapses
    # down to a single representative.
    for path in frames[1:]:
        current = thumbnail(path)

        # Mean absolute difference: average brightness change per pixel.
        diff = np.mean(np.abs(current - kept))

        if diff < DIFF_THRESHOLD:
            os.remove(path)       # too similar -> drop it
            removed += 1
        else:
            kept = current        # different enough -> keep, and compare to this next

    remaining = len(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    print(f"Checked {len(frames)} frames.")
    print(f"Removed {removed} near-duplicate(s).")
    print(f"{remaining} varied frames remain in {frames_dir}/")


if __name__ == "__main__":
    args = parse_args()
    main(args.frames_dir)
