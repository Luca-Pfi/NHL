"""Download boxscore + play-by-play JSON for every NHL game in the chosen seasons.

Source: the public (undocumented) NHL web API at https://api-web.nhle.com/v1
Raw files are cached gzipped under data/raw/<season>/ so re-runs only fetch what is missing.

Usage:  python src/fetch_games.py [--seasons 20212022 20222023 ...] [--workers 8]
"""
import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://api-web.nhle.com/v1"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]
# Fields we never use and that bloat the cache (video links, translations, etc.)
DROP_PLAY_DETAIL = {"highlightClipSharingUrl", "highlightClipSharingUrlFr", "highlightClip",
                    "highlightClipFr", "discreteClip", "discreteClipFr"}

session = requests.Session()
session.headers["User-Agent"] = "nhl-depth-analysis/1.0 (research)"


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def fetch_game(game_id: int, season: str):
    """Fetch boxscore + pbp for one game. Returns 'ok', 'missing' or 'not_final'."""
    d = RAW / season
    box_p, pbp_p = d / f"{game_id}_box.json.gz", d / f"{game_id}_pbp.json.gz"
    if box_p.exists() and pbp_p.exists():
        return game_id, "ok"
    box = get_json(f"{BASE}/gamecenter/{game_id}/boxscore")
    if box is None:
        return game_id, "missing"
    if box.get("gameState") not in ("OFF", "FINAL"):
        return game_id, "not_final"
    pbp = get_json(f"{BASE}/gamecenter/{game_id}/play-by-play")
    if pbp is None:
        return game_id, "missing"
    for k in ("tvBroadcasts", "summary"):
        pbp.pop(k, None)
    for play in pbp.get("plays", []):
        det = play.get("details")
        if det:
            for k in DROP_PLAY_DETAIL:
                det.pop(k, None)
        play.pop("pptReplayUrl", None)
    box.pop("tvBroadcasts", None)
    save(box_p, box)
    save(pbp_p, pbp)
    return game_id, "ok"


def regular_season_ids(season):
    start = int(season[:4])
    # 32 teams x 82 games / 2 = 1312 games; ask for a few extra to be safe (404 -> skipped)
    return [int(f"{start}02{n:04d}") for n in range(1, 1313)]


def playoff_ids(season):
    start = int(season[:4])
    ids = []
    for rnd, n_series in ((1, 8), (2, 4), (3, 2), (4, 1)):
        for s in range(1, n_series + 1):
            for g in range(1, 8):
                ids.append(int(f"{start}030{rnd}{s}{g}"))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    for season in args.seasons:
        ids = regular_season_ids(season) + playoff_ids(season)
        t0, counts = time.time(), {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_game, gid, season) for gid in ids]
            for i, fut in enumerate(as_completed(futs), 1):
                _, status = fut.result()
                counts[status] = counts.get(status, 0) + 1
                if i % 250 == 0:
                    print(f"  {season}: {i}/{len(ids)} {counts} ({time.time()-t0:.0f}s)", flush=True)
        print(f"{season} done: {counts} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
