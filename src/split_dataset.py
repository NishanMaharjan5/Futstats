"""
Split curated frames into train/test/valid folders for Auta upload.

Auta needs the split decided BEFORE upload (it does no export-time splitting), so
we shuffle with a FIXED SEED (reproducible) and COPY — never move — the images
into data/futsal_split/{train,test,valid}/. Copying keeps data/futsal_frames/
intact, so you can re-run with a different exclude list or seed any time.

Run it with:
    python src/split_dataset.py                                   # all frames, 70/20/10, seed 42
    python src/split_dataset.py --exclude "40-55, 207, 300-312"   # drop poor/no-ball frames
    python src/split_dataset.py data/futsal_frames data/futsal_split --seed 42

--exclude accepts frame numbers, N-M ranges, or full filenames (comma/space
separated), matching a "rough list of ranges/filenames" review workflow, e.g.:
    --exclude "40-55"                 -> drops frame_00040.jpg ... frame_00055.jpg
    --exclude "207 frame_00300.jpg"   -> drops frame_00207.jpg and frame_00300.jpg
"""

import os
import glob
import random
import shutil
import argparse

DEFAULT_SRC = "data/futsal_frames"
DEFAULT_OUT = "data/futsal_split"
SPLITS = ("train", "test", "valid")
RATIOS = (0.70, 0.20, 0.10)     # train, test, valid  (must sum to 1.0)
DEFAULT_SEED = 42


def parse_exclude(tokens):
    """Turn '40-55, 207, frame_00300.jpg' into a set of basenames to drop."""
    drop = set()
    for tok in (tokens or "").replace(",", " ").split():
        if tok.endswith(".jpg"):
            drop.add(tok)
        elif "-" in tok:                       # inclusive numeric range "N-M"
            a, b = tok.split("-", 1)
            for n in range(int(a), int(b) + 1):
                drop.add(f"frame_{n:05d}.jpg")
        else:                                  # a single frame number
            drop.add(f"frame_{int(tok):05d}.jpg")
    return drop


def parse_args():
    p = argparse.ArgumentParser(
        description="Shuffle (fixed seed) and split frames into train/test/valid.")
    p.add_argument("src", nargs="?", default=DEFAULT_SRC,
                   help=f"folder of frame_*.jpg (default: {DEFAULT_SRC})")
    p.add_argument("out", nargs="?", default=DEFAULT_OUT,
                   help=f"output split folder (default: {DEFAULT_OUT})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"random seed (default: {DEFAULT_SEED})")
    p.add_argument("--exclude", default="",
                   help="frame numbers / N-M ranges / filenames to drop")
    return p.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "frame_*.jpg")))
    if not files:
        raise SystemExit(f"No frame_*.jpg found in {args.src}/.")

    drop = parse_exclude(args.exclude)
    kept = [f for f in files if os.path.basename(f) not in drop]
    n = len(kept)
    excluded = len(files) - n
    if n == 0:
        raise SystemExit("Every frame was excluded — nothing to split.")

    # Deterministic: sorted input order + fixed seed -> same split every run.
    random.seed(args.seed)
    random.shuffle(kept)

    # Cumulative fraction boundaries so the three buckets always sum to n.
    i_train = int(n * RATIOS[0])
    i_test = int(n * (RATIOS[0] + RATIOS[1]))
    buckets = {
        "train": kept[:i_train],
        "test": kept[i_train:i_test],
        "valid": kept[i_test:],
    }

    print(f"source   : {args.src}  ({len(files)} frames)")
    if excluded:
        print(f"excluded : {excluded} frame(s)")
    print(f"splitting: {n} frames  (seed={args.seed}, 70/20/10)\n")

    for name in SPLITS:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for stale in glob.glob(os.path.join(d, "*.jpg")):   # clean previous run
            os.remove(stale)
        for f in buckets[name]:
            shutil.copy2(f, os.path.join(d, os.path.basename(f)))
        pct = 100 * len(buckets[name]) / n
        print(f"  {name:<5} : {len(buckets[name]):>3} frames ({pct:4.1f}%) -> {d}/")

    print(f"\nTotal copied: {sum(len(b) for b in buckets.values())} "
          f"(source left intact for re-runs)")


if __name__ == "__main__":
    main()
