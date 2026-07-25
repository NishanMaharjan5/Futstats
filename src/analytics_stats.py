"""
FutStats downstream stats — reads data/analytics_raw.pkl (produced by
analytics_core.py) and computes match analytics. NO re-inference.

Outputs (into --outdir, default runs/analytics_stats/):
  Team-level (robust to ID fragmentation — team is re-derived per track from kit):
    - possession %  (A vs B, from the per-frame possession holder)
    - heatmaps: heatmap_match.png, heatmap_teamA.png, heatmap_teamB.png
  Per-player (grouped by track_id AS-IS):
    - touches, passes, turnovers, distance(px), rating  -> player_stats.csv
  Pass networks (SPLIT PER TEAM, like the heatmaps):
    - pass_network_teamA.png, pass_network_teamB.png  + usability metrics
  Team-balance validation:
    - flags if either team's unique track_id count exceeds the expected
      players-per-team (default 4, i.e. 4v4). Diagnostic only — not auto-fixed.
  Everything also dumped to stats_summary.json.

KNOWN LIMITATION (printed prominently, not hidden in a comment):
  Tracking on a moving camera fragments identities — one physical player can
  appear as several track_ids. So per-player rows are PER TRACK, not per person.
  The unique-ID count + inflation ratio + team-balance flags are printed up top so
  the distortion is visible in the data itself. Team-level stats are unaffected.

Rating is a heuristic composite (documented in rating_players), not calibrated.

Run:
    python src/analytics_stats.py                              # data/analytics_raw.pkl
    python src/analytics_stats.py --in data/analytics_raw_static.pkl \
        --outdir runs/analytics_stats_static --players-per-team 4
"""

import os
import json
import pickle
import argparse
from collections import defaultdict, Counter

import cv2
import numpy as np

TEAM_DRAW = {"A": (0, 200, 255), "B": (255, 130, 0), None: (160, 160, 160)}  # BGR
EXPECTED_PLAYERS = 8
FRAGMENT_FRAMES = 15


# ---------------------------------------------------------------------------
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def norm(vals):
    """min-max normalise a dict of {key: value} to [0,1] (0.5 if all equal)."""
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    if hi <= lo:
        return {k: 0.5 for k in vals}
    return {k: (v - lo) / (hi - lo) for k, v in vals.items()}


# ---------------------------------------------------------------------------
# possession spells + transitions (basis for touches / passes / turnovers)
# ---------------------------------------------------------------------------
def build_spells(possession):
    """Maximal runs of the same holder track_id (None breaks a spell)."""
    spells, cur = [], None
    for p in possession:
        tid = p["track_id"]
        if tid is None:
            if cur:
                spells.append(cur); cur = None
            continue
        if cur and cur["track_id"] == tid:
            cur["end"] = p["frame"]
        else:
            if cur:
                spells.append(cur)
            cur = {"track_id": tid, "team": p["team"], "start": p["frame"], "end": p["frame"]}
    if cur:
        spells.append(cur)
    return spells


def classify_transitions(spells, pos_at):
    """Between consecutive spells: pass (same team), turnover (diff team), or
    retained (same track_id regained after a gap)."""
    passes, turnovers, retained, edges = [], [], [], Counter()
    pass_travel = []
    for a, b in zip(spells, spells[1:]):
        if a["track_id"] == b["track_id"]:
            retained.append((a, b))
            continue
        if a["team"] == b["team"] and a["team"] is not None:
            passes.append((a, b))
            edges[(a["track_id"], b["track_id"])] += 1
            pa, pb = pos_at(a["track_id"], a["end"]), pos_at(b["track_id"], b["start"])
            if pa and pb:
                pass_travel.append(float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])))
        else:
            turnovers.append((a, b))
    return passes, turnovers, retained, edges, pass_travel


# ---------------------------------------------------------------------------
# per-player stats
# ---------------------------------------------------------------------------
def per_track_positions(detections):
    by = defaultdict(dict)                       # track_id -> {frame: (cx, cy, y2)}
    for d in detections:
        by[d["track_id"]][d["frame"]] = (d["cx"], d["cy"], d["y2"])
    return by


def track_distance(frames_pos):
    """Sum of centre movement between CONSECUTIVE frames only (skip gaps)."""
    dist = 0.0
    prev_f = prev = None
    for f in sorted(frames_pos):
        cx, cy, _ = frames_pos[f]
        if prev is not None and f - prev_f == 1:
            dist += float(np.hypot(cx - prev[0], cy - prev[1]))
        prev, prev_f = (cx, cy), f
    return dist


def rating_players(touches, passes, turnovers, distance, track_ids):
    """Heuristic 1-10 rating (NOT calibrated). Weighted, min-max-normalised mix:
       +0.40 passes  +0.25 touches  +0.20 distance  -0.20 turnovers, mapped to ~4-9."""
    nt, np_, nd = norm(touches), norm(passes), norm(distance)
    ntov = norm(turnovers)
    out = {}
    for t in track_ids:
        score = (0.40 * np_.get(t, 0) + 0.25 * nt.get(t, 0)
                 + 0.20 * nd.get(t, 0) - 0.20 * ntov.get(t, 0))
        out[t] = round(float(np.clip(5.0 + 4.0 * score, 1.0, 10.0)), 1)
    return out


# ---------------------------------------------------------------------------
# renders (cv2 — dark theme, matches the dashboard placeholders)
# ---------------------------------------------------------------------------
def draw_pitch(w, h):
    img = np.full((h, w, 3), (26, 22, 18), np.uint8)
    line = (60, 70, 60)
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), line, 2)
    cv2.line(img, (w // 2, 30), (w // 2, h - 30), line, 2)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 8, line, 2)
    return img


def render_heatmap(points, w, h, out, title):
    field = np.zeros((h, w), np.float32)
    for x, y in points:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            field[yi, xi] += 1.0
    if field.max() > 0:
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=w * 0.02)
        field /= field.max()
    heat = cv2.applyColorMap((field * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = draw_pitch(w, h)
    a = (field * 0.85)[..., None]
    img = (img * (1 - a) + heat * a).astype(np.uint8)
    cv2.putText(img, title, (36, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(img, f"{len(points)} position samples", (36, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 222, 128), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)


def render_team_pass_network(team, edges, node_pos, team_map, touches, w, h, out):
    """One team's pass network: only that team's nodes + intra-team pass edges."""
    tcolor = TEAM_DRAW.get(team, TEAM_DRAW[None])
    tedges = {(u, v): wt for (u, v), wt in edges.items() if team_map.get(u) == team}
    tnodes = {t: p for t, p in node_pos.items()
              if team_map.get(t) == team and (touches.get(t, 0) > 0 or any(t in e for e in tedges))}
    img = draw_pitch(w, h)
    mx = max(tedges.values()) if tedges else 1
    for (u, v), wgt in tedges.items():
        if u in tnodes and v in tnodes:
            p1, p2 = tuple(map(int, tnodes[u])), tuple(map(int, tnodes[v]))
            cv2.line(img, p1, p2, (120, 180, 120), 1 + int(4 * wgt / mx), cv2.LINE_AA)
    for tid, (x, y) in tnodes.items():
        cv2.circle(img, (int(x), int(y)), 16, tcolor, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 16, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, str(tid), (int(x) - 10, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, f"TEAM {team} PASS NETWORK", (36, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(img, f"{len(tnodes)} nodes  {len(tedges)} pass-pairs", (36, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 222, 128), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)
    return len(tnodes), len(tedges)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FutStats downstream stats from analytics_raw.pkl")
    ap.add_argument("--in", dest="inp", default="data/analytics_raw.pkl")
    ap.add_argument("--outdir", default="runs/analytics_stats")
    ap.add_argument("--players-per-team", type=int, default=4,
                    help="expected players per team for the team-balance check (4v4=4)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = load(args.inp)
    meta, team_map = data["meta"], data["team_map"]
    detections, possession = data["detections"], data["possession"]
    w, h = meta["width"], meta["height"]
    n_frames = meta["n_frames_processed"]

    ids = sorted(set(d["track_id"] for d in detections))
    inflation = round(len(ids) / EXPECTED_PLAYERS, 2)

    # ---- prominent limitation banner ----
    print("=" * 66)
    print(" FutStats stats  |  IDENTITY-FRAGMENTATION NOTICE")
    print("=" * 66)
    print(f"  unique track_ids : {len(ids)}   (expected ~{EXPECTED_PLAYERS} players)")
    print(f"  ID inflation     : x{inflation}   <-- per-player rows are PER TRACK,")
    print(f"                     not per person; one player may span several rows.")
    print(f"  Team-level stats below are unaffected by this.")
    print("=" * 66)

    ppt = args.players_per_team

    # ---- PER-FRAME balance verification (proves the BUG2 hard cap from saved data) ----
    pf = defaultdict(lambda: {"A": 0, "B": 0})
    for d in detections:
        if d.get("team") in ("A", "B"):            # d['team'] is the PER-FRAME, post-cap label
            pf[d["frame"]][d["team"]] += 1
    maxA = max((c["A"] for c in pf.values()), default=0)
    maxB = max((c["B"] for c in pf.values()), default=0)
    violations = sum(1 for c in pf.values() if c["A"] > ppt or c["B"] > ppt)
    frames_at_cap = sum(1 for c in pf.values() if c["A"] == ppt or c["B"] == ppt)
    capinfo = meta.get("cap", {})
    print(f"\n[PER-FRAME BALANCE] verifying the {ppt}-per-team hard cap from saved detections")
    print(f"  max team A in ANY single frame : {maxA}   (must be <= {ppt})")
    print(f"  max team B in ANY single frame : {maxB}   (must be <= {ppt})")
    print(f"  frames VIOLATING the cap       : {violations}   (must be 0)")
    print(f"  frames the core had to cap     : {capinfo.get('frames_capped', '?')}  "
          f"(pre-cap max was A={capinfo.get('max_A_per_frame_precap','?')}, B={capinfo.get('max_B_per_frame_precap','?')})")
    print(f"  frames sitting exactly at {ppt}     : {frames_at_cap}")
    assert violations == 0, f"CAP VIOLATED in saved data (maxA={maxA}, maxB={maxB}) — enforcement failed"

    # ---- team-balance validation (expected 4v4) — DIAGNOSTIC, not auto-fixed ----
    team_ids = defaultdict(list)
    for t in ids:
        team_ids[team_map.get(t)].append(t)
    nA_ids, nB_ids = len(team_ids.get("A", [])), len(team_ids.get("B", []))
    nU_ids = len(team_ids.get(None, []))
    a_flag, b_flag = nA_ids > ppt, nB_ids > ppt
    print(f"\n[TEAM BALANCE] expected {ppt} players/team ({ppt}v{ppt} = {2*ppt} on court)")
    print(f"  team A unique track_ids : {nA_ids:>3}   {('>> FLAG: exceeds ' + str(ppt)) if a_flag else 'ok'}")
    print(f"  team B unique track_ids : {nB_ids:>3}   {('>> FLAG: exceeds ' + str(ppt)) if b_flag else 'ok'}")
    if nU_ids:
        print(f"  unassigned track_ids    : {nU_ids:>3}   (no confident shirt sample)")
    if a_flag or b_flag:
        print(f"  DIAGNOSTIC: team count(s) exceed {ppt} -> ID fragmentation and/or non-player")
        print(f"              detections. Surfaced for your judgement; NOT auto-corrected.")
    else:
        print(f"  team counts are within the {ppt}v{ppt} expectation.")

    # ---- team-level: possession ----
    pc = Counter(p["team"] for p in possession if p["track_id"] is not None)
    tot = pc.get("A", 0) + pc.get("B", 0)
    poss_a = round(100 * pc.get("A", 0) / tot, 1) if tot else 0.0
    poss_b = round(100 * pc.get("B", 0) / tot, 1) if tot else 0.0
    print(f"\n[TEAM] possession  A {poss_a}%  |  B {poss_b}%   ({tot}/{n_frames} frames assigned)")

    # ---- team-level: heatmaps (foot = bottom-centre of box) ----
    feet = defaultdict(list)
    allfeet = []
    for d in detections:
        pt = (d["cx"], d["y2"])
        allfeet.append(pt)
        if d["team"] in ("A", "B"):
            feet[d["team"]].append(pt)
    render_heatmap(allfeet, w, h, f"{args.outdir}/heatmap_match.png", "MATCH HEATMAP")
    render_heatmap(feet["A"], w, h, f"{args.outdir}/heatmap_teamA.png", "TEAM A HEATMAP")
    render_heatmap(feet["B"], w, h, f"{args.outdir}/heatmap_teamB.png", "TEAM B HEATMAP")
    print(f"[TEAM] heatmaps -> {args.outdir}/heatmap_(match|teamA|teamB).png")

    # ---- per-player ----
    tp = per_track_positions(detections)

    def pos_at(tid, frame):
        fp = tp.get(tid, {})
        return (fp[frame][0], fp[frame][1]) if frame in fp else None

    spells = build_spells(possession)
    passes, turnovers, retained, edges, pass_travel = classify_transitions(spells, pos_at)

    touches = Counter(s["track_id"] for s in spells)
    passes_by = Counter(a["track_id"] for a, _ in passes)
    tov_by = Counter(a["track_id"] for a, _ in turnovers)
    distance = {t: round(track_distance(tp[t]), 1) for t in ids}
    track_len = {t: len(tp[t]) for t in ids}
    ratings = rating_players(dict(touches), dict(passes_by), dict(tov_by), distance, ids)

    players = []
    for t in ids:
        players.append({
            "track_id": t, "team": team_map.get(t), "frames_present": track_len[t],
            "touches": touches.get(t, 0), "passes": passes_by.get(t, 0),
            "turnovers": tov_by.get(t, 0), "distance_px": distance[t], "rating": ratings[t],
        })
    players.sort(key=lambda r: r["rating"], reverse=True)

    with open(f"{args.outdir}/player_stats.csv", "w", newline="") as f:
        import csv
        wtr = csv.DictWriter(f, fieldnames=list(players[0].keys()))
        wtr.writeheader(); wtr.writerows(players)
    print(f"\n[PLAYER] per-track stats -> {args.outdir}/player_stats.csv  (top 6 by rating):")
    print(f"  {'id':>4} {'tm':>2} {'frm':>5} {'tch':>4} {'pas':>4} {'tov':>4} {'dist_px':>8} {'rat':>4}")
    for r in players[:6]:
        print(f"  {r['track_id']:>4} {str(r['team']):>2} {r['frames_present']:>5} "
              f"{r['touches']:>4} {r['passes']:>4} {r['turnovers']:>4} {r['distance_px']:>8.0f} {r['rating']:>4}")

    # ---- pass networks (SPLIT PER TEAM) + usability metrics ----
    node_pos = {}
    for t in ids:
        pts = [(v[0], v[1]) for v in tp[t].values()]
        if pts:
            node_pos[t] = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
    nnA, neA = render_team_pass_network("A", edges, node_pos, team_map, dict(touches), w, h,
                                        f"{args.outdir}/pass_network_teamA.png")
    nnB, neB = render_team_pass_network("B", edges, node_pos, team_map, dict(touches), w, h,
                                        f"{args.outdir}/pass_network_teamB.png")

    n_pass = len(passes)
    diag = float(np.hypot(w, h))
    colo_thresh = max(60.0, 0.035 * diag)
    frag_pass = sum(1 for a, b in passes
                    if track_len.get(a["track_id"], 0) < FRAGMENT_FRAMES
                    or track_len.get(b["track_id"], 0) < FRAGMENT_FRAMES)
    colo_pass = sum(1 for d in pass_travel if d < colo_thresh)
    frag_rate = round(100 * frag_pass / n_pass, 1) if n_pass else 0.0
    colo_rate = round(100 * colo_pass / len(pass_travel), 1) if pass_travel else 0.0
    med_travel = round(float(np.median(pass_travel)), 1) if pass_travel else 0.0

    if n_pass < 5:
        verdict = "TOO FEW PASSES to judge"
    elif frag_rate > 40 or colo_rate > 40:
        verdict = "LIKELY UNUSABLE — dominated by ID-fragmentation artifacts (consider cutting)"
    elif frag_rate > 20 or colo_rate > 25:
        verdict = "IMPERFECT but presentable with the ID caveat"
    else:
        verdict = "REASONABLE — artifacts are a minority"

    print(f"\n[PASS NETWORK] split per team (nodes = track_ids):")
    print(f"  team A : {nnA} nodes, {neA} pass-pairs -> pass_network_teamA.png")
    print(f"  team B : {nnB} nodes, {neB} pass-pairs -> pass_network_teamB.png")
    print(f"  totals : passes={n_pass}  turnovers={len(turnovers)}  retained(same-id regain)={len(retained)}")
    print(f"  fragment-involved passes : {frag_rate}%  (passer/receiver track <{FRAGMENT_FRAMES}f)")
    print(f"  co-located 'passes'      : {colo_rate}%  (ball travel <{colo_thresh:.0f}px ~ same player re-ID'd)")
    print(f"  median pass travel       : {med_travel}px")
    print(f"  USABILITY: {verdict}")

    # ---- summary json ----
    summary = {
        "source": args.inp,
        "frames": n_frames,
        "unique_track_ids": len(ids),
        "id_inflation": inflation,
        "team_balance": {
            "players_per_team_expected": ppt, "team_A_ids": nA_ids, "team_B_ids": nB_ids,
            "unassigned_ids": nU_ids, "flag_A_exceeds": a_flag, "flag_B_exceeds": b_flag,
        },
        "possession_pct": {"A": poss_a, "B": poss_b},
        "players": players,
        "pass_network": {
            "teamA_nodes": nnA, "teamA_pass_pairs": neA,
            "teamB_nodes": nnB, "teamB_pass_pairs": neB,
            "passes": n_pass, "turnovers": len(turnovers), "retained": len(retained),
            "fragment_pass_rate_pct": frag_rate, "colocated_pass_rate_pct": colo_rate,
            "median_pass_travel_px": med_travel, "usability": verdict,
        },
        "outputs_dir": args.outdir,
    }
    with open(f"{args.outdir}/stats_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] stats_summary.json -> {args.outdir}/stats_summary.json")
    print("done.")


if __name__ == "__main__":
    main()
