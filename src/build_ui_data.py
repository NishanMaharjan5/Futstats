"""
Wire real FutStats analytics into the futstats-ui dashboard.

Reads the canonical stats (runs/analytics_stats_static_trimmed/stats_summary.json
+ the rendered images) and the pkl meta, maps them into the EXACT results.json
schema that futstats-ui/js/app.js consumes, copies the heatmap/pass-network
images into futstats-ui/assets/ (renamed to the filenames the UI references), and
writes futstats-ui/data/results.json.

Serve from the repo root and open /futstats-ui/index.html so the video path
(../runs/...) resolves.

Run: python src/build_ui_data.py
"""
import os
import json
import shutil
import pickle
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default="runs/analytics_stats_static_trimmed")
    ap.add_argument("--pkl", default="data/analytics_raw_static_trimmed.pkl")
    ap.add_argument("--ui-dir", default="futstats-ui")
    ap.add_argument("--video", default="runs/analytics_core/static_trimmed_annotated.mp4")
    # job-mode overrides (defaults below preserve the original static behaviour):
    ap.add_argument("--assets-out", default=None, help="dir to copy images into (default <ui-dir>/assets)")
    ap.add_argument("--out-json", default=None, help="results.json path (default <ui-dir>/data/results.json)")
    ap.add_argument("--asset-url-prefix", default="assets/", help="URL prefix for image asset paths in results.json")
    ap.add_argument("--video-url", default="", help="explicit video URL for results.json (job mode)")
    ap.add_argument("--top-per-team", type=int, default=4,
                    help="players shown per team (most active by touches; 0 = all)")
    ap.add_argument("--team-a-name", default="Black")
    ap.add_argument("--team-b-name", default="Pink/White")
    ap.add_argument("--venue", default="Static-camera futsal clip")
    ap.add_argument("--date", default="2026-07-25")
    # model_info facts (from training/dataset — editable):
    ap.add_argument("--map50", type=float, default=92.0)
    ap.add_argument("--frames-labeled", type=int, default=498)
    ap.add_argument("--annotations", type=int, default=3738)
    a = ap.parse_args()

    ss = json.load(open(f"{a.stats_dir}/stats_summary.json"))
    meta = pickle.load(open(a.pkl, "rb"))["meta"]
    duration_min = round(meta["n_frames_processed"] / meta["fps"] / 60, 1)

    # --- players: top-N per team by touches (tie-break rating, then frames) ---
    def pick(team):
        ps = [p for p in ss["players"] if p["team"] == team]
        ps.sort(key=lambda p: (p["touches"], p["rating"], p["frames_present"]), reverse=True)
        return ps if a.top_per_team == 0 else ps[: a.top_per_team]

    kept = pick("A") + pick("B")
    for i, p in enumerate(sorted(kept, key=lambda p: (p["touches"], p["rating"]), reverse=True), 1):
        p["activity_rank"] = i  # 1 = most active (mutates shared dicts in `kept`)
    players = [{
        "id": str(p["track_id"]), "team": p["team"],
        "touches": p["touches"], "passes": p["passes"], "turnovers": p["turnovers"],
        "activity_rank": p["activity_rank"], "rating": p["rating"],
    } for p in kept]

    # --- copy the real images into the assets dir (renamed to UI filenames) ---
    assets_dir = a.assets_out or f"{a.ui_dir}/assets"
    os.makedirs(assets_dir, exist_ok=True)
    prefix = a.asset_url_prefix
    img_map = {
        "heatmap_match.png": "heatmap_match.png",
        "heatmap_teamA.png": "heatmap_team_a.png",
        "heatmap_teamB.png": "heatmap_team_b.png",
        "pass_network_teamA.png": "pass_network_team_a.png",
        "pass_network_teamB.png": "pass_network_team_b.png",
    }
    urls = {}  # dst filename -> URL used in results.json
    for src, dst in img_map.items():
        s = f"{a.stats_dir}/{src}"
        if os.path.exists(s):
            shutil.copy2(s, f"{assets_dir}/{dst}")
            urls[dst] = prefix + dst
    copied = list(urls)

    # --- video: explicit --video-url (job mode) else referenced (../) to avoid a copy ---
    video_rel = a.video_url if a.video_url else (("../" + a.video) if os.path.exists(a.video) else "")

    results = {
        "match_info": {
            "team_a_name": a.team_a_name, "team_b_name": a.team_b_name,
            "date": a.date, "venue": a.venue, "duration_min": duration_min,
        },
        "possession": {
            "team_a_pct": ss["possession_pct"]["A"],
            "team_b_pct": ss["possession_pct"]["B"],
        },
        "team_balance_check": {  # not surfaced in UI; kept for schema parity
            "team_a_players_detected": ss["team_balance"]["team_A_ids"],
            "team_b_players_detected": ss["team_balance"]["team_B_ids"],
            "note": "unique track_ids per team over clip; per-frame hard-capped at 4v4 "
                    "(ID fragmentation). Internal QA — not surfaced.",
        },
        "players": players,
        "assets": {
            "video": video_rel,
            "heatmap_match": urls.get("heatmap_match.png", ""),
            "heatmap_team_a": urls.get("heatmap_team_a.png", ""),
            "heatmap_team_b": urls.get("heatmap_team_b.png", ""),
            "pass_network_team_a": urls.get("pass_network_team_a.png", ""),
            "pass_network_team_b": urls.get("pass_network_team_b.png", ""),
            # manually-placed illustration photos (present only in the static ui/assets)
            "radar_before": (prefix + "radar_expected.jpg"
                             if os.path.exists(f"{assets_dir}/radar_expected.jpg") else ""),
            "radar_after": (prefix + "radar_actual.jpg"
                            if os.path.exists(f"{assets_dir}/radar_actual.jpg") else ""),
        },
        "model_info": {
            "map50": a.map50, "frames_labeled": a.frames_labeled, "annotations": a.annotations,
            "dataset_note": "Filmed at a local futsal venue and hand-labeled frame-by-frame "
                            "for player and ball positions.",
        },
    }

    out = a.out_json or f"{a.ui_dir}/data/results.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"wrote {out}")
    print(f"  match     : {a.team_a_name} vs {a.team_b_name}  ({duration_min} min)")
    print(f"  possession: A {results['possession']['team_a_pct']} / B {results['possession']['team_b_pct']}")
    print(f"  players   : {len(players)}  ({a.top_per_team}/team by touches; full set in player_stats.csv)")
    print(f"  images -> {assets_dir}/: {copied}")
    print(f"  video     : {video_rel or '(none)'}")


if __name__ == "__main__":
    main()
