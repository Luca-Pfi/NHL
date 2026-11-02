"""Turn the raw NHL boxscore + play-by-play JSON into tidy analysis tables.

Outputs (data/processed/):
  games.csv            one row per game
  player_games.csv.gz  one row per skater per game (boxscore + play-by-play derived stats)
  shots.csv.gz         every shot attempt with location / context (input to the xG model)
  goals.csv.gz         every non-shootout goal with game state (input to the win-probability model)
  wp_states.csv.gz     sampled game states for training the win-probability model

Usage: python src/build_dataset.py
"""
import gzip
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SHOT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def toi_to_sec(s):
    if not s:
        return 0
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def strength(sit_code, team_is_home):
    """Return (state, own_skaters, opp_skaters, own_goalie_in, opp_goalie_in).

    situationCode = <away goalie in><away skaters><home skaters><home goalie in>.
    A pulled goalie (extra attacker) does not create a power play, so we compare
    skaters net of the extra attacker.
    """
    if not sit_code or len(sit_code) != 4:
        return "ES", 5, 5, 1, 1
    ag, ask, hsk, hg = (int(c) for c in sit_code)
    own_sk, opp_sk, own_g, opp_g = (hsk, ask, hg, ag) if team_is_home else (ask, hsk, ag, hg)
    if own_sk <= 1 and opp_sk <= 1:
        return "PS", own_sk, opp_sk, own_g, opp_g  # penalty shot
    own_eff = own_sk - (1 - own_g)
    opp_eff = opp_sk - (1 - opp_g)
    if own_eff > opp_eff:
        state = "PP"
    elif own_eff < opp_eff:
        state = "SH"
    else:
        state = "ES"
    return state, own_sk, opp_sk, own_g, opp_g


def elapsed(play):
    per = play["periodDescriptor"]["number"]
    m, s = play["timeInPeriod"].split(":")
    # Regular-season OT is 5 min but elapsed time is still monotone with 20-min period blocks.
    return (per - 1) * 1200 + int(m) * 60 + int(s)


def process_game(args):
    season, gid = args
    d = RAW / season
    try:
        box = load(d / f"{gid}_box.json.gz")
        pbp = load(d / f"{gid}_pbp.json.gz")
    except Exception as e:  # corrupted / partial download
        return {"error": (season, gid, repr(e))}

    home_id, away_id = box["homeTeam"]["id"], box["awayTeam"]["id"]
    abbrev = {home_id: box["homeTeam"]["abbrev"], away_id: box["awayTeam"]["abbrev"]}
    game_type = int(box["gameType"])
    last_period_type = (box.get("gameOutcome") or {}).get("lastPeriodType", "REG")

    # ---------------- skaters from the boxscore ----------------
    players = {}
    pos_group = {}
    for side, tid in (("homeTeam", home_id), ("awayTeam", away_id)):
        pbg = box["playerByGameStats"][side]
        for grp, label in (("forwards", "F"), ("defense", "D")):
            for p in pbg.get(grp, []):
                pid = p["playerId"]
                pos_group[pid] = label
                players[pid] = dict(
                    game_id=gid, season=int(season), game_type=game_type, player_id=pid,
                    name=p["name"]["default"], team=abbrev[tid], team_id=tid, pos=p["position"],
                    pos_group=label, toi_sec=toi_to_sec(p.get("toi")), shifts=p.get("shifts", 0),
                    plus_minus=p.get("plusMinus", 0), pim=p.get("pim", 0), hits=p.get("hits", 0),
                    blocks=p.get("blockedShots", 0), giveaways=p.get("giveaways", 0),
                    takeaways=p.get("takeaways", 0),
                )
        for p in pbg.get("goalies", []):
            pos_group[p["playerId"]] = "G"

    # per-player counters derived from play-by-play
    cnt = defaultdict(lambda: defaultdict(float))
    shots, goals, wp_states = [], [], []
    score = {home_id: 0, away_id: 0}
    prev = None  # previous non-stoppage event (for rebound / rush flags)
    last_state_t = -999

    plays = sorted(pbp.get("plays", []), key=lambda x: x.get("sortOrder", 0))

    # manpower timeline: seconds of home power play / home penalty kill / even strength
    strength_sec = {"PP": 0, "SH": 0, "ES": 0, "PS": 0}
    reg = [p for p in plays if p["periodDescriptor"].get("periodType") != "SO"]
    for a, b in zip(reg, reg[1:]):
        if a["periodDescriptor"]["number"] != b["periodDescriptor"]["number"]:
            continue
        dt = elapsed(b) - elapsed(a)
        if dt > 0:
            strength_sec[strength(a.get("situationCode"), True)[0]] += dt

    for play in plays:
        ptype = play["periodDescriptor"].get("periodType", "REG")
        if ptype == "SO":
            continue  # shootouts are excluded from all stats
        t = elapsed(play)
        typ = play["typeDescKey"]
        det = play.get("details") or {}
        owner = det.get("eventOwnerTeamId")

        # sample a game state every ~30s of game time for the win-probability model
        if t - last_state_t >= 30 and ptype != "SO":
            wp_states.append((gid, season, game_type, t, play["periodDescriptor"]["number"], ptype,
                              score[home_id] - score[away_id]))
            last_state_t = t

        if typ in SHOT_TYPES and owner in score:
            is_home = owner == home_id
            st, own_sk, opp_sk, own_g, opp_g = strength(play.get("situationCode"), is_home)
            shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
            x, y = det.get("xCoord"), det.get("yCoord")
            dist = ang = None
            if x is not None and y is not None:
                side = play.get("homeTeamDefendingSide")
                if side in ("left", "right"):
                    home_attacks_pos = side == "left"
                    attacks_pos = home_attacks_pos if is_home else not home_attacks_pos
                    net_x = 89 if attacks_pos else -89
                else:
                    net_x = 89 if x >= 0 else -89
                # (blocked-shot coordinates are where the block happened; they feed Corsi only, not xG)
                dist = math.hypot(net_x - x, y)
                ang = math.degrees(math.atan2(abs(y), abs(net_x - x))) if net_x != x else 90.0
                toward = x * (1 if net_x > 0 else -1)  # distance from centre ice toward the target net
            else:
                toward = None
            rebound = rush = 0
            if prev is not None and prev["period"] == play["periodDescriptor"]["number"]:
                dt = t - prev["t"]
                if prev["typ"] in ("shot-on-goal", "missed-shot", "blocked-shot") and prev["owner"] == owner and dt <= 3:
                    rebound = 1
                # rush: previous event within 4 s happened outside the shooter's offensive zone
                if dt <= 4 and prev.get("x") is not None and toward is not None:
                    prev_toward = prev["x"] * (1 if toward == x else -1)
                    rush = int(prev_toward < 25)
            shots.append(dict(
                game_id=gid, season=int(season), game_type=game_type, period=play["periodDescriptor"]["number"],
                period_type=ptype, t=t, team_id=owner, is_home=int(is_home), shooter_id=shooter,
                event=typ, shot_type=det.get("shotType"), x=x, y=y, distance=dist, angle=ang,
                strength=st, own_skaters=own_sk, opp_skaters=opp_sk, opp_goalie_in=opp_g,
                rebound=rebound, rush=rush, score_diff=score[owner] - score[away_id if is_home else home_id],
                prev_type=prev["typ"] if prev else None,
                prev_dt=(t - prev["t"]) if prev and prev["period"] == play["periodDescriptor"]["number"] else None,
                prev_same_team=int(prev["owner"] == owner) if prev else None,
                prev_x=prev.get("x") if prev else None, prev_y=prev.get("y") if prev else None,
            ))
            if shooter:
                c = cnt[shooter]
                c["iCF"] += 1
                if typ != "blocked-shot":
                    c["iFF"] += 1
                if typ in ("shot-on-goal", "goal"):
                    c["iSOG"] += 1

        if typ == "goal" and owner in score:
            is_home = owner == home_id
            opp = away_id if is_home else home_id
            st, own_sk, opp_sk, own_g, opp_g = strength(play.get("situationCode"), is_home)
            sc_before, opp_before = score[owner], score[opp]
            goals.append(dict(
                game_id=gid, season=int(season), game_type=game_type, period=play["periodDescriptor"]["number"],
                period_type=ptype, t=t, team_id=owner, opp_id=opp, is_home=int(is_home),
                scorer=det.get("scoringPlayerId"), a1=det.get("assist1PlayerId"), a2=det.get("assist2PlayerId"),
                strength=st, empty_net=int(opp_g == 0), own_goalie_pulled=int(own_g == 0),
                team_score_before=sc_before, opp_score_before=opp_before,
            ))
            score[owner] += 1
            for role, key in (("G", "scoringPlayerId"), ("A1", "assist1PlayerId"), ("A2", "assist2PlayerId")):
                pid = det.get(key)
                if pid:
                    c = cnt[pid]
                    c[role] += 1
                    c[f"{role}_{st}"] += 1
                    if role == "G" and opp_g == 0:
                        c["G_EN"] += 1
                    if role == "G" and st == "ES" and own_sk == 5 and opp_sk == 5 and own_g and opp_g:
                        c["G_5v5"] += 1
                    if role != "A2" and st == "ES" and own_sk == 5 and opp_sk == 5 and own_g and opp_g:
                        c[f"{role}_5v5_primary"] += 1

        elif typ == "penalty":
            dur = det.get("duration", 0) or 0
            if det.get("committedByPlayerId") and det.get("typeCode") in ("MIN", "MAJ", "MAT", "PS"):
                cnt[det["committedByPlayerId"]]["pen_taken"] += 1
            if det.get("drawnByPlayerId") and det.get("typeCode") in ("MIN", "MAJ", "MAT", "PS"):
                cnt[det["drawnByPlayerId"]]["pen_drawn"] += 1
            del dur

        elif typ == "faceoff":
            w, l, z = det.get("winningPlayerId"), det.get("losingPlayerId"), det.get("zoneCode")
            # zoneCode is from the perspective of the event owner (the winning team)
            if w:
                cnt[w]["FOW"] += 1
                cnt[w][f"FO_{ {'O': 'OZ', 'D': 'DZ', 'N': 'NZ'}.get(z, 'NZ') }"] += 1
            if l:
                cnt[l]["FOL"] += 1
                cnt[l][f"FO_{ {'O': 'DZ', 'D': 'OZ', 'N': 'NZ'}.get(z, 'NZ') }"] += 1
                if z == "O":
                    cnt[l]["FOW_DZ_lost"] += 1
            if w and z == "D":
                cnt[w]["FOW_DZ"] += 1

        if typ not in ("stoppage", "period-start", "period-end", "game-end", "delayed-penalty"):
            prev = dict(typ=typ, owner=owner, t=t, period=play["periodDescriptor"]["number"],
                        x=det.get("xCoord"), y=det.get("yCoord"))

    # final score excluding the shootout
    home_goals, away_goals = score[home_id], score[away_id]
    box_home, box_away = box["homeTeam"]["score"], box["awayTeam"]["score"]
    winner = home_id if box_home > box_away else away_id

    game = dict(game_id=gid, season=int(season), game_type=game_type, date=box["gameDate"],
                home_id=home_id, away_id=away_id, home=abbrev[home_id], away=abbrev[away_id],
                home_goals=home_goals, away_goals=away_goals, home_score_final=box_home,
                away_score_final=box_away, winner_id=winner, last_period_type=last_period_type,
                home_pp_sec=strength_sec["PP"], home_pk_sec=strength_sec["SH"], es_sec=strength_sec["ES"])

    rows = []
    for pid, row in players.items():
        c = cnt.get(pid, {})
        for k in ("G", "A1", "A2", "G_ES", "G_PP", "G_SH", "G_PS", "G_EN", "G_5v5", "A1_ES", "A1_PP", "A1_SH",
                  "A2_ES", "A2_PP", "A2_SH", "G_5v5_primary", "A1_5v5_primary", "iCF", "iFF", "iSOG",
                  "pen_taken", "pen_drawn", "FOW", "FOL", "FO_OZ", "FO_DZ", "FO_NZ", "FOW_DZ"):
            row[k] = c.get(k, 0)
        rows.append(row)

    # fix up the wp_states with the outcome
    home_win = int(winner == home_id)
    wp = [dict(game_id=g, season=int(s), game_type=gt, t=tt, period=pr, period_type=pt,
               home_diff=hd, home_win=home_win, went_to_so=int(last_period_type == "SO"))
          for (g, s, gt, tt, pr, pt, hd) in wp_states]
    return dict(game=game, players=rows, shots=shots, goals=goals, wp=wp)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sdir in sorted(RAW.iterdir()):
        if not sdir.is_dir():
            continue
        ids = sorted({int(p.name.split("_")[0]) for p in sdir.glob("*_box.json.gz")})
        jobs += [(sdir.name, gid) for gid in ids if (sdir / f"{gid}_pbp.json.gz").exists()]
    print(f"processing {len(jobs)} games")

    games, players, shots, goals, wp, errors = [], [], [], [], [], []
    with ProcessPoolExecutor() as ex:
        for res in ex.map(process_game, jobs, chunksize=32):
            if "error" in res:
                errors.append(res["error"])
                continue
            games.append(res["game"])
            players += res["players"]
            shots += res["shots"]
            goals += res["goals"]
            wp += res["wp"]
    if errors:
        print(f"WARNING: {len(errors)} games could not be read (re-run fetch_games.py):", errors[:5])

    games = pd.DataFrame(games).sort_values("game_id")
    players = pd.DataFrame(players)
    shots = pd.DataFrame(shots)
    goals = pd.DataFrame(goals)
    wp = pd.DataFrame(wp)

    games.to_csv(OUT / "games.csv", index=False)
    players.to_csv(OUT / "player_games.csv.gz", index=False)
    shots.to_csv(OUT / "shots.csv.gz", index=False)
    goals.to_csv(OUT / "goals.csv.gz", index=False)
    wp.to_csv(OUT / "wp_states.csv.gz", index=False)
    print(f"games={len(games)} player_games={len(players)} shots={len(shots)} goals={len(goals)} wp_states={len(wp)}")


if __name__ == "__main__":
    main()
