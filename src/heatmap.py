"""
Generate a player-position heatmap from a football match video.

What it does, step by step:
  1. Load the fine-tuned YOLOv8 model.
  2. Run detection frame-by-frame across the whole video.
  3. For every detected *player*, record where their feet are
     (horizontal centre of the box, bottom edge of the box).
  4. Bin all those (x, y) points into a 2D grid and draw it as a heatmap,
     so you can see which areas of the pitch were most occupied.

Run it with:
    python src/heatmap.py
"""

import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Configuration --------------------------------------------------------
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/clip1.mp4"
OUTPUT_PATH = "assets/heatmap.png"
CONF = 0.4
DEVICE = "mps"           # Apple Silicon GPU. Change to "cpu" or "cuda" elsewhere.

# The dataset ("Football-Player-Detection" by Yolo Fifa) has two classes.
# We VERIFIED the mapping is {0: 'ball', 1: 'player'}, so players are id 1.
# The assert below re-checks this at runtime so a wrong assumption fails loudly.
PLAYER_CLASS_ID = 1


def main():
    model = YOLO(MODEL_PATH)

    # Guard against a wrong class id: if the model ever changes, stop early
    # with a clear message instead of silently plotting the wrong class.
    names = model.names
    print(f"Model classes: {names}")
    assert names.get(PLAYER_CLASS_ID) == "player", (
        f"Expected class id {PLAYER_CLASS_ID} to be 'player' but got "
        f"'{names.get(PLAYER_CLASS_ID)}'. Update PLAYER_CLASS_ID."
    )

    # stream=True returns a *generator* that yields one result per frame.
    # This processes frames one at a time instead of loading the whole video
    # into memory (which is what triggers ultralytics' RAM warning).
    results = model.predict(
        source=VIDEO_PATH,
        stream=True,
        conf=CONF,
        device=DEVICE,
        verbose=False,       # keep the console quiet; we print our own summary
    )

    xs, ys = [], []          # feet positions (x, y) of every player detection
    frame_count = 0
    frame_h = frame_w = None

    for result in results:
        frame_count += 1

        # orig_shape is (height, width) of the frame. We grab it once so we
        # know the pixel bounds of the pitch for the histogram range.
        if frame_h is None:
            frame_h, frame_w = result.orig_shape

        boxes = result.boxes
        if boxes is None:
            continue

        # boxes.cls  -> class id for each detected box
        # boxes.xyxy -> [x1, y1, x2, y2] corner pixels for each box
        for cls_id, (x1, y1, x2, y2) in zip(boxes.cls.tolist(), boxes.xyxy.tolist()):
            if int(cls_id) != PLAYER_CLASS_ID:
                continue                     # skip the ball / anything else
            x_center = (x1 + x2) / 2         # middle of the box, horizontally
            y_feet = y2                      # bottom edge ~= where the feet are
            xs.append(x_center)
            ys.append(y_feet)

    print(f"Processed {frame_count} frames.")
    print(f"Collected {len(xs)} player detections.")

    # Fail loudly rather than saving a meaningless empty image.
    if not xs:
        raise SystemExit(
            "No player detections collected — check PLAYER_CLASS_ID against "
            f"model.names ({names}) and the confidence threshold ({CONF})."
        )

    # --- Draw the heatmap -------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # hist2d drops every point into a grid cell and colours the cell by how
    # many points landed in it. Cells where players stood often become "hot".
    counts, xedges, yedges, im = ax.hist2d(
        xs, ys,
        bins=[64, 36],                        # grid roughly matches a 16:9 pitch
        range=[[0, frame_w], [0, frame_h]],   # cover the full frame in pixels
        cmap="hot",
    )

    ax.set_title("Player Position Heatmap — clip1.mp4 (feet positions)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    # In image coordinates (0, 0) is the TOP-left corner and y grows downward.
    # Flip the y-axis so the plot is oriented the same way as the video frame.
    ax.invert_yaxis()
    ax.set_aspect("equal")                     # don't stretch the pitch

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of player detections")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved heatmap to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
