"""Assign every forward to the Top 6 or Bottom 6 and aggregate everything to team-game and
team-season level.

Role definition (primary): within each team-season (regular season and playoffs separately) every
forward has an average time on ice (TOI). In each game the dressed forwards are ranked by that
season-average TOI: ranks 1-6 = Top 6, ranks 7+ = Bottom 6 (normally 6 players, 5 when a team dresses
11 F / 7 D). Using the *season role* rather than that night's TOI avoids game-script bias (e.g. a
4th liner who gets more ice time because the team is killing a lot of penalties).
Sensitivity definition: rank by that game's TOI (`group_game`).

Outputs written to data/processed/:
  player_games_enriched.csv.gz, goals_enriched.csv.gz, team_games.csv.gz, team_seasons.csv
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models import WinProb, fit_xg, goal_wpa

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)  # wide team-game frame is built column by column

GROUPS = ["T6", "B6", "D"]
SUM_COLS = ["toi_sec", "shifts", "G", "A1", "A2", "PTS", "P1", "G_ES", "G_PP", "G_SH", "G_EN", "G_5v5",
            "P1_5v5", "PPP", "iCF", "iFF", "iSOG", "ixG", "plus_minus", "hits", "blocks", "giveaways",
            "takeaways", "pim", "pen_taken", "pen_drawn", "FOW", "FOL", "FO_OZ", "FO_DZ", "FO_NZ",
            "wpa_scored", "GWG", "go_ahead", "tying", "n"]


def load_processed():
    return dict(
        games=pd.read_csv(P / "games.csv"),
        pg=pd.read_csv(P / "player_games.csv.gz"),
        shots=pd.read_csv(P / "shots.csv.gz", low_memory=False),
        goals=pd.read_csv(P / "goals.csv.gz"),
        wp=pd.read_csv(P / "wp_states.csv.gz"),
    )


def assign_roles(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.copy()
    pg["group"] = "D"
    pg["group_game"] = "D"
    f = pg["pos_group"] == "F"
    key = ["season", "game_type", "team", "player_id"]
    pg.loc[f, "role_toi"] = pg[f].groupby(key)["toi_sec"].transform("mean")
    pg.loc[f, "role_rank"] = pg[f].groupby(["game_id", "team"])["role_toi"].rank(ascending=False, method="first")
    pg.loc[f, "game_rank"] = pg[f].groupby(["game_id", "team"])["toi_sec"].rank(ascending=False, method="first")
    pg.loc[f, "group"] = np.where(pg.loc[f, "role_rank"] <= 6, "T6", "B6")
    pg.loc[f, "group_game"] = np.where(pg.loc[f, "game_rank"] <= 6, "T6", "B6")
    return pg


def enrich_goals(goals, games, wpm):
    g = goals.merge(games[["game_id", "winner_id", "home_goals", "away_goals", "last_period_type", "home_id"]],
                    on="game_id")
    g["wpa"] = goal_wpa(g, wpm).values
    loser_goals = np.where(g["winner_id"] == g["home_id"], g["away_goals"], g["home_goals"])
    g["gwg"] = ((g["team_id"] == g["winner_id"]) & (g["team_score_before"] == loser_goals)
                & (g["last_period_type"] != "SO")).astype(int)
    g["go_ahead"] = (g["team_score_before"] == g["opp_score_before"]).astype(int)
    g["tying"] = (g["team_score_before"] == g["opp_score_before"] - 1).astype(int)
    g["pm_eligible"] = g["strength"].isin(["ES", "SH"]).astype(int)
    # goal category for the even-strength cleaning (scorer's perspective)
    g["cat"] = np.select(
        [g["strength"] == "PS", g["strength"] == "PP", g["empty_net"] == 1, g["strength"] == "SH",
         g["own_goalie_pulled"] == 1],
        ["PS", "PP", "EN", "SH", "PULL"], default="ES")
    return g.drop(columns=["winner_id", "home_goals", "away_goals", "last_period_type", "home_id"])


def attribute_goals_to_lines(goals, pg):
    """Which forward group was on the ice for each goal?  The NHL web API has no on-ice lists, so we
    use the forwards credited with the goal/assists (they were certainly on the ice). If no forward
    is credited (D-to-D goal), split by the two groups' share of forward TOI in that game."""
    grp = pg.set_index(["game_id", "player_id"])["group"]
    out = goals[["game_id", "team_id"]].copy()
    w6 = np.zeros(len(goals))
    nf = np.zeros(len(goals))
    for col in ("scorer", "a1", "a2"):
        idx = pd.MultiIndex.from_arrays([goals["game_id"], goals[col].fillna(-1).astype(int)])
        gg = grp.reindex(idx).values
        w6 += (gg == "T6")
        nf += np.isin(gg, ["T6", "B6"])
    toi = pg[pg["pos_group"] == "F"].groupby(["game_id", "team_id", "group"])["toi_sec"].sum().unstack(fill_value=0)
    share = (toi["T6"] / (toi["T6"] + toi["B6"])).rename("t6_share")
    s = goals.merge(share.reset_index(), on=["game_id", "team_id"], how="left")["t6_share"].fillna(0.6).values
    out["line_T6"] = np.where(nf > 0, w6 / np.maximum(nf, 1), s)
    out["line_B6"] = 1 - out["line_T6"]
    return out[["line_T6", "line_B6"]]


def build():
    d = load_processed()
    games, pg, shots, goals, wp = d["games"], d["pg"], d["shots"], d["goals"], d["wp"]

    # ---------------- models ----------------
    shots, xg_metrics = fit_xg(shots)
    wpm = WinProb().fit(wp)
    goals = enrich_goals(goals, games, wpm)
    model_info = dict(xg=xg_metrics.to_dict(orient="records"), wp=wpm.metrics)

    # ---------------- player-game enrichment ----------------
    pg = assign_roles(pg)
    ixg = shots.groupby(["game_id", "shooter_id"])["xg"].sum().rename("ixG")
    pg = pg.merge(ixg, left_on=["game_id", "player_id"], right_index=True, how="left")
    pg["ixG"] = pg["ixG"].fillna(0)
    sc = goals.groupby(["game_id", "scorer"]).agg(wpa_scored=("wpa", "sum"), GWG=("gwg", "sum"),
                                                  go_ahead=("go_ahead", "sum"), tying=("tying", "sum"))
    pg = pg.merge(sc, left_on=["game_id", "player_id"], right_index=True, how="left")
    for c in ("wpa_scored", "GWG", "go_ahead", "tying"):
        pg[c] = pg[c].fillna(0)
    pg["PTS"] = pg["G"] + pg["A1"] + pg["A2"]
    pg["P1"] = pg["G"] + pg["A1"]
    pg["P1_5v5"] = pg["G_5v5"] + pg["A1_5v5_primary"]
    pg["PPP"] = pg["G_PP"] + pg["A1_PP"] + pg["A2_PP"]
    pg["n"] = 1

    # on-ice line attribution of goals (for estimated on-ice GF and WPA by line group)
    goals[["line_T6", "line_B6"]] = attribute_goals_to_lines(goals, pg).values

    # ---------------- team-game table ----------------
    tg = team_game_table(games, pg, goals, shots, group_col="group")
    tg_alt = team_game_table(games, pg, goals, shots, group_col="group_game")

    # k = forwards on the ice per +/- eligible goal (sum of forward +/- = k * (PM-eligible GF - GA))
    x = tg["pmGF"] - tg["pmGA"]
    yv = tg["T6_plus_minus"] + tg["B6_plus_minus"]
    k = float((x * yv).sum() / (x * x).sum())
    model_info["k_forwards_per_goal"] = k
    for t in (tg, tg_alt):
        for gname in ("T6", "B6"):
            t[f"{gname}_onGD"] = t[f"{gname}_plus_minus"] / k
            t[f"{gname}_onGA"] = t[f"{gname}_onGF"] - t[f"{gname}_onGD"]
    model_info["even_strength_cleaning"] = clean_even_strength(tg)
    clean_even_strength(tg_alt)

    ts = team_season_table(tg)

    pg.to_csv(P / "player_games_enriched.csv.gz", index=False)
    goals.to_csv(P / "goals_enriched.csv.gz", index=False)
    tg.to_csv(P / "team_games.csv.gz", index=False)
    tg_alt.to_csv(P / "team_games_gameTOI_roles.csv.gz", index=False)
    ts.to_csv(P / "team_seasons.csv", index=False)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "model_info.json").write_text(json.dumps(model_info, indent=2))
    print(json.dumps(model_info, indent=2))
    return pg, goals, tg, ts


def clean_even_strength(tg):
    """Estimate each line group's *even-strength* on-ice goals for/against and ice time.

    The web API has no shift data, so we combine three things we do know:
      * each group's total TOI and summed +/-  (boxscore)
      * the team's PP / PK / ES seconds        (play-by-play manpower timeline)
      * who scored/assisted every goal          (play-by-play)
    Step 1  deployment: regress group TOI on team ES/PP/PK seconds -> forwards of that group on the ice per
            second of each manpower state.  Group ES TOI = TOI - c_pp*PP - c_pk*PK.
    Step 2  +/- decomposition: regress summed forward +/- on clean-ES goal diff and the special cases
            (SH, empty-net, extra-attacker goals for/against) -> forwards credited per goal of each kind.
    Step 3  per group: remove the special-case credits (attributed by scorer, or by deployment share when the
            scorer is on the other team) and divide by forwards-per-ES-goal -> clean ES on-ice goal diff.
    Modifies tg in place; returns the fitted coefficients.
    """
    rs = tg[tg["game_type"] == 2]
    dep = {}
    for gname in ("T6", "B6"):
        X = rs[["es_sec", "pp_sec", "pk_sec"]]
        b = np.linalg.lstsq(X.values, rs[f"{gname}_toi_sec"].values, rcond=None)[0]
        dep[gname] = dict(zip(["es", "pp", "pk"], b))
    share_pp_T6 = dep["T6"]["pp"] / (dep["T6"]["pp"] + dep["B6"]["pp"])
    share_pull_T6 = rs["T6_onGF_PULL"].sum() / rs["GF_PULL"].sum()  # who is out with the goalie pulled
    share_en_T6 = rs["T6_onGF_EN"].sum() / rs["GF_EN"].sum()        # who is out protecting a late lead

    cols = ["dES", "GF_SH", "GA_SH", "GF_EN", "GA_EN", "GF_PULL", "GA_PULL"]
    Z = rs.assign(dES=rs["GF_ES"] - rs["GA_ES"])[cols]
    y = rs["T6_plus_minus"] + rs["B6_plus_minus"]
    kc = dict(zip(cols, np.linalg.lstsq(Z.values, y.values, rcond=None)[0]))

    shares = {"T6": dict(pp=share_pp_T6, pull=share_pull_T6, en=share_en_T6),
              "B6": dict(pp=1 - share_pp_T6, pull=1 - share_pull_T6, en=1 - share_en_T6)}
    es_toi = {}
    for gname in ("T6", "B6"):
        d, sh = dep[gname], shares[gname]
        es_toi[gname] = (tg[f"{gname}_toi_sec"] - d["pp"] * tg["pp_sec"] - d["pk"] * tg["pk_sec"]).clip(lower=0)
        tg[f"{gname}_es_toi_sec"] = es_toi[gname]
        adj = (kc["GF_SH"] * tg[f"{gname}_onGF_SH"] + kc["GA_SH"] * tg["GA_SH"] * sh["pp"]
               + kc["GF_EN"] * tg[f"{gname}_onGF_EN"] + kc["GA_EN"] * tg["GA_EN"] * sh["pull"]
               + kc["GF_PULL"] * tg[f"{gname}_onGF_PULL"] + kc["GA_PULL"] * tg["GA_PULL"] * sh["en"])
        tg[f"{gname}_esGD"] = (tg[f"{gname}_plus_minus"] - adj) / kc["dES"]
        tg[f"{gname}_esGF"] = tg[f"{gname}_onGF_ES"]
        tg[f"{gname}_esGA"] = tg[f"{gname}_esGF"] - tg[f"{gname}_esGD"]
    tot = es_toi["T6"] + es_toi["B6"]
    for gname in ("T6", "B6"):
        # seconds of team even-strength time this group's line was on the ice
        tg[f"{gname}_es_team_sec"] = tg["es_sec"] * (es_toi[gname] / tot.replace(0, np.nan)).fillna(0.5)
    tg["st_net"] = tg["GF_PP"] - tg["GA_PP"] + tg["GF_SH"] - tg["GA_SH"]
    tg["late_net"] = tg["GF_EN"] - tg["GA_EN"] + tg["GF_PULL"] - tg["GA_PULL"]
    return dict(deployment={g: {k: round(v, 3) for k, v in d.items()} for g, d in dep.items()},
                share_of_forward_time_top6={s: round(dep["T6"][s] / (dep["T6"][s] + dep["B6"][s]), 3)
                                            for s in ("es", "pp", "pk")},
                pm_credit_per_goal={k: round(v, 3) for k, v in kc.items()},
                share_pull_T6=round(share_pull_T6, 3), share_en_T6=round(share_en_T6, 3))


def team_game_table(games, pg, goals, shots, group_col):
    agg = pg.groupby(["game_id", "team_id", group_col])[SUM_COLS].sum().unstack(group_col, fill_value=0)
    agg.columns = [f"{g}_{c}" for c, g in agg.columns]
    agg = agg.reset_index()

    # goal-level team totals
    cats = ["ES", "SH", "EN", "PULL", "PP", "PS"]
    extra = {f"GF_{c}": (goals["cat"] == c).astype(int) for c in cats}
    for c in ("ES", "SH", "EN", "PULL"):
        for gname in ("T6", "B6"):
            extra[f"{gname}_onGF_{c}"] = goals[f"line_{gname}"] * (goals["cat"] == c)
    gl = goals.assign(
        pmGF=goals["pm_eligible"], ppGF=(goals["strength"] == "PP").astype(int),
        shGF=(goals["strength"] == "SH").astype(int), enGF=goals["empty_net"],
        T6_onGF=goals["line_T6"] * goals["pm_eligible"], B6_onGF=goals["line_B6"] * goals["pm_eligible"],
        T6_onWPA=goals["line_T6"] * goals["wpa"], B6_onWPA=goals["line_B6"] * goals["wpa"],
        wpa_total=goals["wpa"], **extra)
    sum_cols = ["pmGF", "ppGF", "shGF", "enGF", "T6_onGF", "B6_onGF", "T6_onWPA", "B6_onWPA", "wpa_total"] + list(extra)
    gf = gl.groupby(["game_id", "team_id"])[sum_cols].sum().reset_index()

    # shot-level team totals (for goaltending / xG controls)
    sh = shots.assign(sog=shots["event"].isin(["shot-on-goal", "goal"]).astype(int),
                      goal=(shots["event"] == "goal").astype(int))
    sh_g = sh[sh["opp_goalie_in"] == 1]
    sf = sh.groupby(["game_id", "team_id"]).agg(CF=("event", "size"), xGF=("xg", "sum"), SF=("sog", "sum")).reset_index()
    sv = sh_g.groupby(["game_id", "team_id"]).agg(SF_g=("sog", "sum"), GF_g=("goal", "sum"), xGF_g=("xg", "sum")).reset_index()

    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = games[["game_id", "season", "game_type", "date", "last_period_type", "winner_id"]].copy()
        r["team_id"], r["opp_id"] = games[f"{side}_id"], games[f"{opp}_id"]
        r["team"], r["opp"] = games[side], games[opp]
        r["is_home"] = int(side == "home")
        r["GF"], r["GA"] = games[f"{side}_goals"], games[f"{opp}_goals"]
        r["es_sec"] = games["es_sec"]
        r["pp_sec"] = games["home_pp_sec"] if side == "home" else games["home_pk_sec"]
        r["pk_sec"] = games["home_pk_sec"] if side == "home" else games["home_pp_sec"]
        rows.append(r)
    tg = pd.concat(rows, ignore_index=True)
    tg["win"] = (tg["winner_id"] == tg["team_id"]).astype(int)
    tg["extra_time"] = tg["last_period_type"].isin(["OT", "SO"]).astype(int)
    tg["pts"] = np.where(tg["win"] == 1, 2, np.where(tg["extra_time"] == 1, 1, 0))
    tg = tg.merge(agg, on=["game_id", "team_id"], how="left")
    tg = tg.merge(gf, on=["game_id", "team_id"], how="left")
    tg = tg.merge(sf, on=["game_id", "team_id"], how="left").merge(sv, on=["game_id", "team_id"], how="left")
    opp_map = {"team_id": "opp_id", "pmGF": "pmGA", "ppGF": "ppGA", "shGF": "shGA", "enGF": "enGA"}
    opp_map.update({f"GF_{c}": f"GA_{c}" for c in cats})
    opp_cols = gf[["game_id"] + list(opp_map)].rename(columns=opp_map)
    tg = tg.merge(opp_cols, on=["game_id", "opp_id"], how="left")
    opp_sh = sf.rename(columns={"team_id": "opp_id", "CF": "CA", "xGF": "xGA", "SF": "SA"})
    opp_sv = sv.rename(columns={"team_id": "opp_id", "SF_g": "SA_g", "GF_g": "GA_g", "xGF_g": "xGA_g"})
    tg = tg.merge(opp_sh, on=["game_id", "opp_id"], how="left").merge(opp_sv, on=["game_id", "opp_id"], how="left")
    num = tg.select_dtypes("number").columns
    tg[num] = tg[num].fillna(0)

    # rest days (regular season & playoffs, by team-season)
    tg["date"] = pd.to_datetime(tg["date"])
    tg = tg.sort_values(["team_id", "season", "date"])
    tg["rest_days"] = tg.groupby(["team_id", "season"])["date"].diff().dt.days
    tg["b2b"] = (tg["rest_days"] == 1).astype(int)
    return tg.sort_values(["game_id", "is_home"]).reset_index(drop=True)


def team_season_table(tg):
    rs = tg[tg["game_type"] == 2]
    num = [c for c in rs.columns if c not in ("game_id", "season", "team_id", "opp_id", "winner_id", "date",
                                               "last_period_type", "team", "opp", "game_type", "rest_days")]
    ts = rs.groupby(["season", "team"])[num].sum(numeric_only=True)
    ts["GP"] = rs.groupby(["season", "team"]).size()
    ts = ts.reset_index()
    ts["pts_pct"] = ts["pts"] / (2 * ts["GP"])
    ts["win_pct"] = ts["win"] / ts["GP"]
    po = tg[tg["game_type"] == 3].groupby(["season", "team"]).agg(po_games=("win", "size"), po_wins=("win", "sum"))
    ts = ts.merge(po.reset_index(), on=["season", "team"], how="left").fillna({"po_games": 0, "po_wins": 0})
    ts["made_playoffs"] = (ts["po_games"] > 0).astype(int)
    ts["sv_pct"] = 1 - ts["GA_g"] / ts["SA_g"]
    ts["gsax_pg"] = (ts["xGA_g"] - ts["GA_g"]) / ts["GP"]  # goals saved above expected per game
    ts["GD_pg"] = (ts["GF"] - ts["GA"]) / ts["GP"]
    ts["ppGF_pg"] = ts["ppGF"] / ts["GP"]
    ts["ppGA_pg"] = ts["ppGA"] / ts["GP"]
    for g in ("T6", "B6", "D"):
        hrs = ts[f"{g}_toi_sec"] / 3600
        ts[f"{g}_pm_pg"] = ts[f"{g}_plus_minus"] / ts["GP"]
        ts[f"{g}_pm60"] = ts[f"{g}_plus_minus"] / hrs
        ts[f"{g}_G_pg"] = ts[f"{g}_G"] / ts["GP"]
        for c in ("G", "PTS", "P1", "iCF", "ixG", "iSOG", "hits", "blocks", "takeaways", "giveaways",
                  "pen_taken", "pen_drawn", "G_5v5", "P1_5v5"):
            ts[f"{g}_{c}60"] = ts[f"{g}_{c}"] / hrs
        ts[f"{g}_pendiff60"] = (ts[f"{g}_pen_drawn"] - ts[f"{g}_pen_taken"]) / hrs
        ts[f"{g}_takegive60"] = (ts[f"{g}_takeaways"] - ts[f"{g}_giveaways"]) / hrs
        ts[f"{g}_fo_pct"] = ts[f"{g}_FOW"] / (ts[f"{g}_FOW"] + ts[f"{g}_FOL"]).replace(0, np.nan)
        ts[f"{g}_dz_share"] = ts[f"{g}_FO_DZ"] / (ts[f"{g}_FO_DZ"] + ts[f"{g}_FO_OZ"]).replace(0, np.nan)
        ts[f"{g}_gax60"] = (ts[f"{g}_G"] - ts[f"{g}_ixG"]) / hrs
    fwd_toi = ts["T6_toi_sec"] + ts["B6_toi_sec"]
    ts["B6_toi_share"] = ts["B6_toi_sec"] / fwd_toi
    for g in ("T6", "B6"):
        team_hrs = ts[f"{g}_toi_sec"] / fwd_toi * ts["GP"]  # team-games of ice time this line group covered
        ts[f"{g}_onGF60"] = ts[f"{g}_onGF"] / team_hrs
        ts[f"{g}_onGA60"] = ts[f"{g}_onGA"] / team_hrs
        ts[f"{g}_onGD60"] = ts[f"{g}_onGD"] / team_hrs
        ts[f"{g}_onGD_pg"] = ts[f"{g}_onGD"] / ts["GP"]
        # clean even strength (see clean_even_strength): per 60 of team ES time with this group on the ice
        es_hrs = ts[f"{g}_es_team_sec"] / 3600
        ts[f"{g}_esGF60"] = ts[f"{g}_esGF"] / es_hrs
        ts[f"{g}_esGA60"] = ts[f"{g}_esGA"] / es_hrs
        ts[f"{g}_esGD60"] = ts[f"{g}_esGD"] / es_hrs
        ts[f"{g}_esGD_pg"] = ts[f"{g}_esGD"] / ts["GP"]
        ts[f"{g}_es_share"] = ts[f"{g}_es_team_sec"] / ts["es_sec"]
    ts["st_net_pg"] = ts["st_net"] / ts["GP"]
    ts["late_net_pg"] = ts["late_net"] / ts["GP"]
    ts["es_clean_GD_pg"] = (ts["GF_ES"] - ts["GA_ES"]) / ts["GP"]
    return ts


if __name__ == "__main__":
    build()
