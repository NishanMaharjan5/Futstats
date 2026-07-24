"""
Possession % (v1) for the futsal demo clip.

For every frame that has a ball, we find the player nearest to the ball, read the
colour of that player's shirt, and credit possession to their team (RED vs
WHITE/BLUE). Tally over the whole clip -> possession split.

v1 LIMITATIONS (to be refined at the hackathon):
  - "Nearest player to the ball" is a crude proxy for possession; there is no
    tracking, no pass/turnover logic, no "is the player actually in control".
  - Per-frame independent decision (no temporal smoothing).
It's a prototype to show the idea end-to-end, not a broadcast-grade stat.

Colour method: an HSV "redness" heuristic (not K-means). Reasoning: the two kits
are RED vs WHITE/BLUE, so "how red is the shirt?" is a direct, interpretable
discriminator. K-means on shirt colours tends to cluster on lighting/shadow
rather than team, and then needs an extra step to decide which cluster is which
team. Redness needs no fitting and fails loudly (low colour = "ambiguous").

Run it with:
    python src/possession.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Config ---------------------------------------------------------------
MODEL_PATH = "models/futsal.pt"
VIDEO_PATH = "videos/futsal_demo.mov"
OUT = "assets/futsal_possession.png"
CONF = 0.25
DEVICE = "mps"
BALL_CLASS_ID, PLAYER_CLASS_ID = 0, 1

# Fraction of strongly-coloured shirt pixels needed to commit to a team.
COLOR_MIN_FRAC = 0.18
AMBIGUITY_STOP = 0.20            # if >20% of attempts are ambiguous, don't fake a number

# Dark slide palette (matches futsal_heatmap.py).
BG = "#0d1117"
TEXT = "#e6edf3"
RED_HEX = "#e5484d"
WHITEBLUE_HEX = "#6cb6ff"


def shirt_team(crop_bgr):
    """Return 'RED', 'WHITEBLUE', or None (ambiguous) for a shirt crop."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 3 or crop_bgr.shape[1] < 3:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    total = h.size

    red = ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)
    blue = (h >= 95) & (h <= 135) & (s >= 80) & (v >= 50)
    white = (s <= 40) & (v >= 150)

    red_frac = red.sum() / total
    wb_frac = (blue | white).sum() / total

    if red_frac < COLOR_MIN_FRAC and wb_frac < COLOR_MIN_FRAC:
        return None                          # shirt too dark/washed-out to tell
    if abs(red_frac - wb_frac) < 0.05:
        return None                          # too close to call
    return "RED" if red_frac > wb_frac else "WHITEBLUE"


def main():
    model = YOLO(MODEL_PATH)
    results = model.predict(source=VIDEO_PATH, stream=True, conf=CONF,
                            device=DEVICE, verbose=False)

    frames_with_ball = 0
    attempts = 0                 # frames where we had a ball AND a nearest player
    tally = {"RED": 0, "WHITEBLUE": 0}
    ambiguous = 0

    for r in results:
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        cls = b.cls.tolist()
        xyxy = b.xyxy.tolist()
        confs = b.conf.tolist()

        balls = [(xyxy[i], confs[i]) for i in range(len(cls)) if int(cls[i]) == BALL_CLASS_ID]
        players = [xyxy[i] for i in range(len(cls)) if int(cls[i]) == PLAYER_CLASS_ID]
        if not balls:
            continue
        frames_with_ball += 1
        if not players:
            continue

        # highest-confidence ball, then nearest player by box-centre distance
        (bx1, by1, bx2, by2), _ = max(balls, key=lambda t: t[1])
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        px1, py1, px2, py2 = min(
            players, key=lambda p: ((p[0] + p[2]) / 2 - bcx) ** 2 + ((p[1] + p[3]) / 2 - bcy) ** 2
        )

        attempts += 1
        # Shirt sample = a tight TORSO patch, not the whole upper half:
        # central 60% width (drop the box edges, which are mostly background)
        # and the 15%-50% height band (skip head/hair at the top and shorts at
        # the bottom). This removes non-shirt pixels that were making shadowed
        # players read as "ambiguous".
        H, W = r.orig_img.shape[:2]
        bw, bh = px2 - px1, py2 - py1
        x1 = max(0, int(px1 + 0.20 * bw))
        x2 = min(W, int(px2 - 0.20 * bw))
        y1 = max(0, int(py1 + 0.15 * bh))
        y2 = min(H, int(py1 + 0.50 * bh))
        team = shirt_team(r.orig_img[y1:y2, x1:x2])
        if team is None:
            ambiguous += 1
        else:
            tally[team] += 1

    # --- sanity report ------------------------------------------------------
    assigned = tally["RED"] + tally["WHITEBLUE"]
    amb_rate = ambiguous / attempts if attempts else 0.0
    print("=== SANITY REPORT ===")
    print(f"frames with a ball        : {frames_with_ball}")
    print(f"possession attempts       : {attempts}  (ball + a player present)")
    print(f"assignments made          : {assigned}")
    print(f"ambiguous classifications : {ambiguous}  ({amb_rate:.1%} of attempts)")

    if attempts == 0 or assigned == 0:
        raise SystemExit("No possession assignments could be made — cannot produce a chart.")

    if amb_rate > AMBIGUITY_STOP:
        print(f"\nSTOP: ambiguity {amb_rate:.1%} > {AMBIGUITY_STOP:.0%}. "
              "Not producing a possession number — present possession as 'prototyped'.")
        return

    red_pct = 100 * tally["RED"] / assigned
    wb_pct = 100 * tally["WHITEBLUE"] / assigned
    print(f"\nPossession — RED {red_pct:.1f}%  |  WHITE/BLUE {wb_pct:.1f}%")

    # --- bar chart (dark, team colours) ------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    bars = ax.bar(["RED", "WHITE/BLUE"], [red_pct, wb_pct],
                  color=[RED_HEX, WHITEBLUE_HEX], width=0.6, zorder=3)
    for bar, pct in zip(bars, [red_pct, wb_pct]):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.5, f"{pct:.1f}%",
                ha="center", color=TEXT, fontsize=18, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("Ball Possession  (v1 · nearest-player proxy)", color=TEXT,
                 fontsize=14, fontweight="bold", pad=16)
    ax.tick_params(colors=TEXT, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="y", color="#30363d", zorder=0)
    ax.text(0.5, -0.12, f"{assigned:,} assigned frames · {ambiguous:,} ambiguous ({amb_rate:.0%})",
            transform=ax.transAxes, ha="center", color="#8b98a5", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
