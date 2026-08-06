# FutStats

**PLAY. ANALYZE. IMPROVE.**

Upload a futsal match video and get a FotMob-style analytics dashboard — possession, heatmaps, pass networks, per-player stats, and an annotated video — powered by a custom-trained YOLOv8 model.

> 🏆 **Built in a 48-hour hackathon.** Scope, model training, analytics pipeline, and web app were all put together in one weekend — so this is a working proof-of-concept, with its limitations documented openly rather than hidden.

---

## Features

- **Upload → analyze** — drop in an `.mp4` / `.mov` / `.avi` clip and the full pipeline runs end-to-end.
- **Possession %** — derived from the per-frame nearest-player-to-ball.
- **Heatmaps** — full-match plus per-team positional density.
- **Pass networks** — split per team (node = involvement area, edge = pass frequency).
- **Per-player table** — touches, passes, turnovers, activity, and a composite rating.
- **Annotated video** — team-coloured boxes and track-ID labels, transcoded for in-browser playback.
- **FotMob-style dashboard** — clean tabbed UI (Overview, Heatmaps, Pass Networks, Players, Video, What We Tried, About) with light/dark themes.

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Detection & tracking | [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) + ByteTrack |
| Computer vision | OpenCV |
| Team classification | scikit-learn (KMeans on shirt colour) |
| Backend / serving | FastAPI + Uvicorn (single process for API **and** UI) |
| Video transcode | ffmpeg (FMP4 → H.264) |
| Frontend | Vanilla HTML / CSS / JS |

---

## How it works

A single upload kicks off a four-stage pipeline. The API runs each stage as a subprocess, so a crash in any stage fails the job loudly instead of hanging.

1. **Detect & track** — `src/analytics_core.py` runs YOLOv8 + ByteTrack over the video, classifies each player's team per-frame via KMeans kit colour (hard-capped at 4v4), and assigns possession → `raw.pkl` + an annotated video.
2. **Compute stats** — `src/analytics_stats.py` turns that into possession %, heatmaps, pass networks, and per-player touches/passes/turnovers/rating → `stats/` + `stats_summary.json`.
3. **Transcode** — ffmpeg converts OpenCV's FMP4 output to browser-playable H.264 (`assets/annotated.mp4`).
4. **Build dashboard data** — `src/build_ui_data.py` maps everything into the dashboard's `results.json` schema and copies the image assets.

The model is a custom **YOLOv8s** trained on a hand-labeled **Nepali futsal dataset** (~498 frames, ~3,738 annotations), reaching **mAP@50 ≈ 0.92**, with classes `{0: player, 1: ball}`. Weights live at `models/futstats.pt`; training is in `notebooks/futstats_training.ipynb` (Colab).

---

## Getting started

```bash
git clone https://github.com/NishanMaharjan5/Futstats.git
cd Futstats
pip install -r requirements.txt
```

**Requirements:**
- Python with `ultralytics` / `torch` / `opencv` (device auto-detected: CUDA > MPS (Apple Silicon) > CPU).
- `ffmpeg` available on your `PATH`.
- Model weights at `models/futstats.pt` (included).

---

## Running

### Web app (upload flow)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/** and click **"⬆ Analyze a video"**. The page polls job status and shows the current stage live.

> ⏱️ A full real run takes **~15–25+ minutes** depending on clip length and hardware — the UI says so up front.

Clear out old job directories when they pile up:

```bash
python src/api/cleanup_jobs.py --all --yes
```

### Offline pipeline (CLI)

Run the stages directly if you'd rather not go through the web app:

```bash
python src/analytics_core.py  path/to/match.mp4 --out data/raw.pkl --video-out data/annotated.mp4
python src/analytics_stats.py --in data/raw.pkl --outdir runs/stats --players-per-team 4
python src/build_ui_data.py   --stats-dir runs/stats --pkl data/raw.pkl
```

---

## Limitations — what we tried

Kept visible on purpose (there's a whole "What We Tried" tab in the app):

- **ID fragmentation (moving camera).** Tracking on a non-fixed camera splits one physical player into several `track_id`s, so **per-player rows are per-track, not per-person**. Team-level stats (possession, heatmaps) re-derive team from kit colour and are unaffected.
- **2D radar / top-down projection — abandoned.** A static homography can't survive camera pan/tilt, so player dots drift off their real pitch positions over a match.
- **No real-world distance / speed.** There's no camera calibration, so "activity" is on-screen pixel movement, not metres or km/h — and it's labeled as such.
- **Shot / goal detection — not built.** Documented as a next step, not part of this release.

---

## Project structure

```
FutStats/
├── src/
│   ├── analytics_core.py     # detect + track + teams + possession
│   ├── analytics_stats.py    # possession, heatmaps, pass nets, player stats
│   ├── build_ui_data.py      # stats -> dashboard results.json + assets
│   └── api/
│       ├── main.py           # FastAPI app: upload -> job pipeline -> serve UI
│       └── cleanup_jobs.py   # clear old job directories
├── futstats-ui/              # FotMob-style dashboard (HTML/CSS/JS) + upload UI
├── models/futstats.pt        # trained YOLOv8 weights
├── notebooks/                # futstats_training.ipynb (Colab)
├── data/ · videos/ · runs/   # working dirs (gitignored)
└── requirements.txt
```

---

## Author

**Nishan Maharjan** 
