"""
FutStats analytics core — the shared foundation for every downstream stat.

Runs the trained futsal detector (models/futstats.pt) with ByteTrack over a match
video and produces ONE structured, per-frame record set that later scripts
(passes, distance, ratings, pass-network, timeline) load without re-running
inference.

Pipeline:
  1. TRACK    model.track(..., tracker="bytetrack.yaml") over the video.
              Device auto-detected: CUDA > MPS (Apple Silicon) > CPU.
  2. EXTRACT  per frame -> each tracked PLAYER (track_id, box, centre) and the
              BALL position (highest-confidence ball) if present.
  3. TEAMS    sample each player's shirt (central 60% width, 15-50% height) and
              K-means (k=2) the kit colours. Team is classified PER DETECTION,
              EVERY frame (BUG1 fix — not locked to one early guess); a track's
              displayed team is the majority vote of its per-frame labels. A HARD
              CAP (BUG2 fix) keeps at most `players_per_team` detections per team
              per frame — lowest-confidence overflow is forced to unassigned.
  4. POSSESS  per frame with a ball -> nearest player holds possession; a gap of
              up to N ball-less frames carries the last holder forward, then
              "no possession".
  5. SAVE     everything -> data/analytics_raw.pkl. Plain dict of lists-of-dicts
              (no pandas needed to load it): pd.DataFrame(data["detections"]).
  6. REPORT   frames, unique track_ids, team A/B counts, %ball, %possession, and a
              TRACK-STABILITY read (the number that matters most).
  7. ANNOTATE write a video with team-coloured boxes + track IDs for eyeballing.

Run it (needs ultralytics/torch/opencv/scikit-learn; pandas NOT required):
    python src/analytics_core.py                                    # full match1.mov (slow!)
    python src/analytics_core.py videos/match1.mov --max-frames 450 \
        --out data/analytics_raw_sample.pkl --video-out runs/analytics_core/sample.mp4

NOTE: there is no CUDA GPU on a Mac, so full inference over the 8-min 1080p clip
is SLOW (many minutes on MPS/CPU). Use --max-frames for a quick check first.
"""

import os
import time
import pickle
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

# --- defaults --------------------------------------------------------------
DEFAULT_VIDEO = "videos/match1.mov"
DEFAULT_MODEL = "models/futstats.pt"
DEFAULT_OUT = "data/analytics_raw_static_trimmed.pkl"
DEFAULT_VIDEO_OUT = "runs/analytics_core/annotated.mp4"

PLAYER_CLASS, BALL_CLASS = 0, 1

# high-visibility box colours (BGR) — deliberately NOT the kit colours
# (a black kit drawn on the pitch would be invisible)
TEAM_DRAW = {"A": (0, 200, 255), "B": (255, 130, 0), None: (160, 160, 160)}
BALL_DRAW = (0, 255, 0)
HOLD_DRAW = (0, 255, 255)


def fourcc(code):
    # OpenCV 5 moved VideoWriter_fourcc -> VideoWriter.fourcc
    fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return fn(*code)


def pick_device(pref):
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    m = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return m


def shirt_color(frame, x1, y1, x2, y2):
    """Mean BGR of the torso patch: central 60% width, 15-50% height of the box."""
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    sx1, sx2 = max(0, int(x1 + 0.20 * bw)), min(w, int(x2 - 0.20 * bw))
    sy1, sy2 = max(0, int(y1 + 0.15 * bh)), min(h, int(y1 + 0.50 * bh))
    if sx2 - sx1 < 2 or sy2 - sy1 < 2:
        return None
    patch = frame[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return None
    return patch.reshape(-1, 3).mean(axis=0)  # BGR


# ---------------------------------------------------------------------------
# 1-2) track + extract
# ---------------------------------------------------------------------------
def run_tracking(model, video, device, a):
    results = model.track(source=video, stream=True, tracker=a.tracker,
                          imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                          device=device, verbose=False)
    per_frame = []       # {frame, players:[...], ball:{...}|None}
    shirt_samples = []   # {frame, track_id, color(np.ndarray), conf}
    t0, n = time.time(), 0
    for res in results:
        frame = res.orig_img
        players, ball, best_ball = [], None, -1.0
        b = res.boxes
        if b is not None and len(b) > 0:
            cls = b.cls.tolist()
            conf = b.conf.tolist()
            xyxy = b.xyxy.tolist()
            ids = b.id.tolist() if b.id is not None else [None] * len(cls)
            for i in range(len(cls)):
                c = int(cls[i]); cf = float(conf[i])
                x1, y1, x2, y2 = xyxy[i]
                if c == PLAYER_CLASS and ids[i] is not None:
                    tid = int(ids[i])
                    col = shirt_color(frame, x1, y1, x2, y2)   # BUG1 FIX: sample EVERY detection, every frame
                    players.append({"track_id": tid, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": cf,
                                    "color": col})
                    if cf >= a.team_min_conf and col is not None:
                        shirt_samples.append({"frame": n, "track_id": tid, "color": col})
                elif c == BALL_CLASS and cf > best_ball:
                    best_ball = cf
                    ball = {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": cf}
        per_frame.append({"frame": n, "players": players, "ball": ball})
        n += 1
        if n % 200 == 0:
            print(f"  ...{n} frames  ({n / (time.time() - t0):.1f} fps)")
        if a.max_frames and n >= a.max_frames:
            break
    print(f"  tracked {n} frames in {time.time() - t0:.0f}s")
    return per_frame, shirt_samples


# ---------------------------------------------------------------------------
# 3) team assignment — PER-FRAME classification + majority-vote label + hard cap
# ---------------------------------------------------------------------------
def fit_team_kmeans(shirt_samples):
    """Learn the two kit clusters (solid black vs solid pink) from ALL confident
    shirt samples across the whole clip — NOT just early frames, so a bad opening
    can't poison the classifier. Returns (kmeans, dark_cluster_idx, team_colors)."""
    if len(shirt_samples) < 2:
        return None, None, {"A": None, "B": None}
    fit = np.array([s["color"] for s in shirt_samples], dtype=np.float32)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(fit)
    centers = km.cluster_centers_
    dark = int(np.argmin(centers.sum(axis=1)))         # darker cluster == team A (black kit)
    team_colors = {"A": [round(float(v), 1) for v in centers[dark]],
                   "B": [round(float(v), 1) for v in centers[1 - dark]]}
    return km, dark, team_colors


def classify_team(color, km, dark):
    """BUG1 FIX: classify ONE detection's shirt colour -> 'A' / 'B' (or None)."""
    if color is None or km is None:
        return None
    label = int(km.predict(np.asarray(color, dtype=np.float32).reshape(1, -1))[0])
    return "A" if label == dark else "B"


def classify_and_cap(per_frame, km, dark, a):
    """BUG1 FIX: classify EVERY detection EVERY frame (p['team_raw']); a track's
    displayed team (team_map) is the MAJORITY VOTE of its per-frame labels.
    BUG2 FIX: enforce a hard per-frame cap of `players_per_team` per team —
    lowest-confidence overflow is revoked to unassigned. Sets p['team_raw'],
    p['team'] (post-cap), p['capped']; returns (team_map, cap_stats)."""
    CAP = a.players_per_team
    votes = defaultdict(Counter)
    frames_capped = dropped = 0
    max_A = max_B = max_pre_A = max_pre_B = 0

    for rec in per_frame:
        players = rec["players"]
        # (BUG1) independent per-frame classification for every detection
        for p in players:
            p["team_raw"] = classify_team(p.get("color"), km, dark)
            p["team"] = p["team_raw"]        # provisional; the cap below may revoke it
            p["capped"] = False
        max_pre_A = max(max_pre_A, sum(1 for p in players if p["team_raw"] == "A"))
        max_pre_B = max(max_pre_B, sum(1 for p in players if p["team_raw"] == "B"))

        # (BUG2) HARD CAP — explicit, per team, per frame
        triggered = False
        for team in ("A", "B"):
            members = [p for p in players if p["team_raw"] == team]
            if len(members) > CAP:                                  # <-- the enforcing check
                triggered = True
                members.sort(key=lambda p: p["conf"], reverse=True)  # highest YOLO conf first
                for p in members[CAP:]:                              # everyone past top-CAP is revoked
                    p["team"] = None
                    p["capped"] = True
                    dropped += 1
        if triggered:
            frames_capped += 1

        max_A = max(max_A, sum(1 for p in players if p["team"] == "A"))
        max_B = max(max_B, sum(1 for p in players if p["team"] == "B"))
        for p in players:                                            # (BUG1) accumulate votes
            if p["team_raw"] is not None:
                votes[p["track_id"]][p["team_raw"]] += 1

    team_map = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    cap_stats = {"players_per_team": CAP, "frames_capped": frames_capped, "detections_dropped": dropped,
                 "max_A_per_frame_postcap": max_A, "max_B_per_frame_postcap": max_B,
                 "max_A_per_frame_precap": max_pre_A, "max_B_per_frame_precap": max_pre_B}
    return team_map, cap_stats


# ---------------------------------------------------------------------------
# 4) possession (nearest player to ball, carry-forward on gaps)
# ---------------------------------------------------------------------------
def compute_possession(per_frame, team_map, a):
    poss, last, since = [], None, 0
    for rec in per_frame:
        ball, players = rec["ball"], rec["players"]
        if ball is not None and players:
            bx, by = ball["cx"], ball["cy"]
            p = min(players, key=lambda pl: (pl["cx"] - bx) ** 2 + (pl["cy"] - by) ** 2)
            tid = p["track_id"]
            # per-frame team of the holder (post-cap); fall back to its majority team
            pteam = p.get("team") or team_map.get(tid)
            holder = {"frame": rec["frame"], "track_id": tid,
                      "team": pteam, "carried": False}
            last, since = holder, 0
        else:
            since += 1
            if last is not None and since <= a.possession_carry:
                holder = {"frame": rec["frame"], "track_id": last["track_id"],
                          "team": last["team"], "carried": True}
            else:
                holder = {"frame": rec["frame"], "track_id": None, "team": None, "carried": False}
        poss.append(holder)
    return poss


# ---------------------------------------------------------------------------
# 6) sanity + track-stability report
# ---------------------------------------------------------------------------
def sanity_report(per_frame, detections, ball_rows, poss, team_map, cap_stats, a):
    n = len(per_frame)
    ids = set(d["track_id"] for d in detections)
    lengths = Counter(d["track_id"] for d in detections)
    len_arr = np.array(sorted(lengths.values())) if lengths else np.array([0])
    fragments = int((len_arr < a.fragment_frames).sum())
    max_players = max((len(r["players"]) for r in per_frame), default=0)
    teamc = Counter(team_map.get(i) for i in ids)
    pct_ball = 100 * len(ball_rows) / n if n else 0.0
    pct_poss = 100 * sum(1 for p in poss if p["track_id"] is not None) / n if n else 0.0
    inflation = len(ids) / a.expected_players if a.expected_players else 0.0

    # stability verdict
    if inflation <= 2.0 and fragments <= max(1, 0.3 * len(ids)):
        verdict = "STABLE-ISH — IDs look reasonably persistent"
    elif inflation <= 4.0:
        verdict = "SOME CHURN — noticeable ID switching/fragmentation"
    else:
        verdict = "UNSTABLE — IDs are switching a lot (see suggestions)"

    rep = {
        "frames": n,
        "unique_track_ids": len(ids),
        "expected_players": a.expected_players,
        "id_inflation": round(inflation, 2),
        "team_A_ids": teamc.get("A", 0),
        "team_B_ids": teamc.get("B", 0),
        "unassigned_ids": teamc.get(None, 0),
        "frames_capped": cap_stats["frames_capped"],
        "max_teamA_per_frame_postcap": cap_stats["max_A_per_frame_postcap"],
        "max_teamB_per_frame_postcap": cap_stats["max_B_per_frame_postcap"],
        "max_teamA_per_frame_precap": cap_stats["max_A_per_frame_precap"],
        "max_teamB_per_frame_precap": cap_stats["max_B_per_frame_precap"],
        "pct_frames_with_ball": round(pct_ball, 1),
        "pct_frames_with_possession": round(pct_poss, 1),
        "max_players_in_a_frame": max_players,
        "median_track_len_frames": int(np.median(len_arr)),
        "fragment_tracks(<%d f)" % a.fragment_frames: fragments,
        "verdict": verdict,
    }

    print("\n" + "=" * 60 + "\n SANITY REPORT\n" + "=" * 60)
    print(f"  frames processed            : {rep['frames']}")
    print(f"  unique track_ids            : {rep['unique_track_ids']}  "
          f"(expected ~{a.expected_players} players -> inflation x{rep['id_inflation']})")
    print(f"  team A / B / unassigned ids  : {rep['team_A_ids']} / {rep['team_B_ids']} / {rep['unassigned_ids']}  "
          f"(unique track_ids per team, whole clip)")
    print(f"  PER-FRAME cap ({cap_stats['players_per_team']}/team): post-cap max A={cap_stats['max_A_per_frame_postcap']}, "
          f"B={cap_stats['max_B_per_frame_postcap']}   (pre-cap was A={cap_stats['max_A_per_frame_precap']}, "
          f"B={cap_stats['max_B_per_frame_precap']})")
    print(f"  frames that hit the cap      : {cap_stats['frames_capped']}  "
          f"({cap_stats['detections_dropped']} detections revoked)")
    print(f"  % frames with ball           : {rep['pct_frames_with_ball']}%")
    print(f"  % frames with possession     : {rep['pct_frames_with_possession']}%")
    print(f"  max players in one frame     : {rep['max_players_in_a_frame']}")
    print(f"  median track length (frames) : {rep['median_track_len_frames']}")
    print(f"  fragment tracks (<{a.fragment_frames} frames): {fragments}")
    print(f"\n  TRACK STABILITY: {verdict}")
    print("=" * 60)
    return rep


# ---------------------------------------------------------------------------
# 7) annotated video (team-coloured boxes + track IDs)
# ---------------------------------------------------------------------------
def annotate_video(video, per_frame, poss, team_map, out_path, meta, scale=1.0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(video)
    ow, oh = int(meta["width"] * scale), int(meta["height"] * scale)
    writer = cv2.VideoWriter(out_path, fourcc("mp4v"), meta["fps"], (ow, oh))
    hold_by_frame = {p["frame"]: p["track_id"] for p in poss if p["track_id"] is not None}
    for i, rec in enumerate(per_frame):
        ok, frame = cap.read()
        if not ok:
            break
        hid = hold_by_frame.get(i)
        for p in rec["players"]:
            team = team_map.get(p["track_id"])
            color = TEAM_DRAW.get(team, TEAM_DRAW[None])
            x1, y1, x2, y2 = map(int, (p["x1"], p["y1"], p["x2"], p["y2"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{team or '?'}#{p['track_id']}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if p["track_id"] == hid:
                cv2.circle(frame, (int(p["cx"]), int(p["cy"])), 7, HOLD_DRAW, -1)
        if rec["ball"] is not None:
            bx, by = int(rec["ball"]["cx"]), int(rec["ball"]["cy"])
            cv2.circle(frame, (bx, by), 9, BALL_DRAW, 2)
            cv2.putText(frame, "BALL", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, BALL_DRAW, 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {i}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if scale != 1.0:
            frame = cv2.resize(frame, (ow, oh))
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"  annotated video -> {out_path}")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FutStats analytics core: track + teams + possession.")
    p.add_argument("video", nargs="?", default=DEFAULT_VIDEO, help=f"match video (default: {DEFAULT_VIDEO})")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT, help="pickle of per-frame data")
    p.add_argument("--video-out", default=DEFAULT_VIDEO_OUT, help="annotated video path")
    p.add_argument("--video-scale", type=float, default=1.0,
                   help="scale factor for the annotated video (0.5 -> half res, smaller file)")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | 0 (cuda)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="ultralytics tracker config")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=None, help="cap frames (quick test)")
    p.add_argument("--team-fit-frames", type=int, default=100)
    p.add_argument("--team-min-conf", type=float, default=0.5)
    p.add_argument("--team-vote-n", type=int, default=10)
    p.add_argument("--possession-carry", type=int, default=10)
    p.add_argument("--expected-players", type=int, default=8)
    p.add_argument("--players-per-team", type=int, default=4,
                   help="BUG2 FIX: hard per-frame cap of detections per team (4v4 -> 4)")
    p.add_argument("--fragment-frames", type=int, default=15)
    p.add_argument("--no-video", action="store_true", help="skip the annotated video (faster)")
    return p.parse_args()


def main():
    a = parse_args()
    device = pick_device(a.device)
    print(f"[core] video={a.video}  model={a.model}  device={device}  imgsz={a.imgsz}")
    model = YOLO(a.model)
    print(f"[core] model classes: {model.names}")

    meta = video_meta(a.video)
    print(f"[core] {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps, {meta['total_frames']} frames total")

    print("[1-2/7] tracking + extracting...")
    per_frame, shirt_samples = run_tracking(model, a.video, device, a)

    print("[3/7] classifying teams PER FRAME + enforcing hard 4-per-team cap...")
    km, dark, team_colors = fit_team_kmeans(shirt_samples)
    team_map, cap_stats = classify_and_cap(per_frame, km, dark, a)
    print(f"  team colours (mean shirt BGR): A={team_colors['A']}  B={team_colors['B']}")
    print(f"  cap: {cap_stats['frames_capped']} frames exceeded {cap_stats['players_per_team']}/team "
          f"(pre-cap max A={cap_stats['max_A_per_frame_precap']}, B={cap_stats['max_B_per_frame_precap']} "
          f"-> post-cap A={cap_stats['max_A_per_frame_postcap']}, B={cap_stats['max_B_per_frame_postcap']})")

    # flatten detections. detection['team'] = PER-FRAME team (post-cap);
    # 'team_track' = the track's majority-vote displayed team. 'color' is NOT saved.
    detections = [{"frame": r["frame"], "track_id": p["track_id"], "cx": p["cx"], "cy": p["cy"],
                   "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"], "conf": p["conf"],
                   "team": p["team"], "team_raw": p["team_raw"], "capped": p["capped"],
                   "team_track": team_map.get(p["track_id"])}
                  for r in per_frame for p in r["players"]]
    ball_rows = [{"frame": r["frame"], **r["ball"]} for r in per_frame if r["ball"] is not None]

    print("[4/7] computing possession...")
    poss = compute_possession(per_frame, team_map, a)

    print("[6/7] sanity report...")
    rep = sanity_report(per_frame, detections, ball_rows, poss, team_map, cap_stats, a)

    meta.update({"video": a.video, "model": a.model, "device": str(device),
                 "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "tracker": a.tracker,
                 "n_frames_processed": len(per_frame), "team_colors_bgr": team_colors,
                 "possession_carry": a.possession_carry, "cap": cap_stats})
    data = {"meta": meta, "team_map": team_map, "detections": detections,
            "ball": ball_rows, "possession": poss, "sanity": rep}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(data, f)
    print(f"[5/7] saved per-frame data -> {a.out}  "
          f"({len(detections)} player-rows, {len(ball_rows)} ball-rows)")

    if a.no_video:
        print("[7/7] annotated video skipped (--no-video)")
    else:
        print("[7/7] writing annotated video...")
        annotate_video(a.video, per_frame, poss, team_map, a.video_out, meta, a.video_scale)
    print("\ndone.")


if __name__ == "__main__":
    main()
