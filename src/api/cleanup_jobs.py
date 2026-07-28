"""
Clear out old FutStats API job directories (runs/api_jobs/<job_id>/).
Full runs fill disk fast during testing, so this is the manual cleanup path.

Dry-run by default (lists what WOULD be deleted) — add --yes to actually delete.

Examples:
    python src/api/cleanup_jobs.py                     # dry-run: jobs older than 24h
    python src/api/cleanup_jobs.py --older-than-hours 6 --yes
    python src/api/cleanup_jobs.py --all --yes         # nuke every job dir
"""
import time
import shutil
import argparse
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parents[2] / "runs" / "api_jobs"


def main():
    ap = argparse.ArgumentParser(description="Delete old FutStats API job dirs.")
    ap.add_argument("--older-than-hours", type=float, default=24)
    ap.add_argument("--all", action="store_true", help="delete ALL job dirs regardless of age")
    ap.add_argument("--yes", action="store_true", help="actually delete (otherwise dry-run)")
    a = ap.parse_args()

    if not JOBS_DIR.exists():
        print(f"no job dir yet: {JOBS_DIR}")
        return

    now, cutoff = time.time(), time.time() - a.older_than_hours * 3600
    n, freed = 0, 0
    for d in sorted(JOBS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if a.all or d.stat().st_mtime < cutoff:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            age_h = (now - d.stat().st_mtime) / 3600
            n += 1
            freed += size
            print(f"  {'DELETE' if a.yes else 'would delete'}  {d.name}  "
                  f"({age_h:.1f}h old, {size / 1e6:.0f} MB)")
            if a.yes:
                shutil.rmtree(d, ignore_errors=True)

    verb = "freed" if a.yes else "would free"
    print(f"\n{n} job dir(s), {freed / 1e6:.0f} MB {verb}.")
    if not a.yes and n:
        print("(dry-run — re-run with --yes to actually delete)")


if __name__ == "__main__":
    main()
