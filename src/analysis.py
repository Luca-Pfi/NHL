"""Top 6 vs Bottom 6 forwards: contribution to winning.

Reads the tables built by features.py and writes
  figures/*.png       charts
  results/*.csv       the table behind every chart
  results/summary.json every headline number quoted in REPORT.md

Usage: python src/analysis.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import style as S
from style import C_B6, C_T6, INK, INK2, MUTED, NEUTRAL, plt

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RES = ROOT / "results"
SUMMARY = {}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def table(df, name):
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / f"{name}.csv", index=False)


def pct(x, d=1):
    return round(100 * float(x), d)


def season_label(s):
    s = str(s)
    return f"{s[:4]}-{s[6:]}"


# --------------------------------------------------------------------------------------------
def load():
    tg = pd.read_csv(P / "team_games.csv.gz", parse_dates=["date"])
    tg_alt = pd.read_csv(P / "team_games_gameTOI_roles.csv.gz", parse_dates=["date"])
    ts = pd.read_csv(P / "team_seasons.csv")
    pg = pd.read_csv(P / "player_games_enriched.csv.gz")
    goals = pd.read_csv(P / "goals_enriched.csv.gz")
    info = json.loads((RES / "model_info.json").read_text())
    return tg, tg_alt, ts, pg, goals, info


# --------------------------------------------------------------------------------------------
# 1. Who produces what?  Share of each stat that comes from the Top 6 (forwards only)
# --------------------------------------------------------------------------------------------
SHARE_METRICS = [
    ("toi_sec", "Ice time", "Usage"),
    ("G", "Goals", "Scoring"), ("G_5v5", "5-on-5 goals", "Scoring"), ("G_PP", "Power-play goals", "Scoring"),
    ("G_SH", "Shorthanded goals", "Scoring"), ("PTS", "Points", "Scoring"),
    ("GWG", "Game-winning goals", "Clutch"), ("go_ahead", "Go-ahead goals", "Clutch"),
    ("wpa_scored", "Win probability added by goals", "Clutch"),
    ("iCF", "Shot attempts", "Shooting"), ("ixG", "Expected goals (xG)", "Shooting"),
    ("hits", "Hits", "Physical / defense"), ("blocks", "Blocked shots", "Physical / defense"),
    ("takeaways", "Takeaways", "Physical / defense"), ("giveaways", "Giveaways", "Physical / defense"),
    ("pen_drawn", "Penalties drawn", "Discipline"), ("pen_taken", "Penalties taken", "Discipline"),
    ("FO_DZ", "Defensive-zone faceoffs taken", "Deployment"), ("FO_OZ", "Offensive-zone faceoffs taken", "Deployment"),
]


def production_shares(tg, info):
    rs = tg[tg.game_type == 2]
    rows = []
    dep = info["even_strength_cleaning"]["share_of_forward_time_top6"]
    for col, label, cat in SHARE_METRICS:
        t6, b6 = rs[f"T6_{col}"].sum(), rs[f"B6_{col}"].sum()
        d = rs[f"D_{col}"].sum()
        rows.append(dict(metric=label, category=cat, top6=t6, bottom6=b6, defense=d,
                         top6_share_of_forwards=t6 / (t6 + b6), bottom6_share_of_forwards=b6 / (t6 + b6),
                         bottom6_share_of_team=b6 / (t6 + b6 + d)))
        if col == "toi_sec":  # estimated split of ice time by manpower state (see features.clean_even_strength)
            for key, lab in (("es", "Even-strength ice time (est.)"), ("pp", "Power-play ice time (est.)"),
                             ("pk", "Penalty-kill ice time (est.)")):
                rows.append(dict(metric=lab, category="Usage", top6=np.nan, bottom6=np.nan, defense=np.nan,
                                 top6_share_of_forwards=dep[key], bottom6_share_of_forwards=1 - dep[key],
                                 bottom6_share_of_team=np.nan))
        if col == "G_5v5":  # clean even-strength on-ice goals for / against (line attribution)
            for c2, lab in (("esGF", "On-ice ES goals for (est.)"), ("esGA", "On-ice ES goals against (est.)")):
                a6, b = rs[f"T6_{c2}"].sum(), rs[f"B6_{c2}"].sum()
                rows.append(dict(metric=lab, category="Scoring", top6=a6, bottom6=b, defense=np.nan,
                                 top6_share_of_forwards=a6 / (a6 + b), bottom6_share_of_forwards=b / (a6 + b),
                                 bottom6_share_of_team=np.nan))
    df = pd.DataFrame(rows)
    table(df, "01_production_shares")
    toi_share = df.loc[df.metric == "Ice time", "bottom6_share_of_forwards"].item()
    SUMMARY["shares"] = {r.metric: pct(r.bottom6_share_of_forwards) for r in df.itertuples()}
    SUMMARY["shares_of_team_goals"] = {
        "top6": pct(rs.T6_G.sum() / rs.GF.sum()), "bottom6": pct(rs.B6_G.sum() / rs.GF.sum()),
        "defense": pct(rs.D_G.sum() / rs.GF.sum())}

    fig, ax = plt.subplots(figsize=(10, 9.6))
    d = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d.top6_share_of_forwards, color=C_T6, height=0.72, edgecolor=S.SURFACE, linewidth=2)
    ax.barh(y, d.bottom6_share_of_forwards, left=d.top6_share_of_forwards, color=C_B6, height=0.72,
            edgecolor=S.SURFACE, linewidth=2)
    for i, r in d.iterrows():
        ax.text(0.015, i, f"{r.top6_share_of_forwards*100:.0f}%", va="center", color="white", fontsize=9.5, weight="bold")
        ax.text(0.985, i, f"{r.bottom6_share_of_forwards*100:.0f}%", va="center", ha="right", color="white",
                fontsize=9.5, weight="bold")
    ax.axvline(1 - toi_share, color=INK, lw=1.2)
    ax.text(1 - toi_share, len(d) - 0.35, f"  Ice-time split ({(1-toi_share)*100:.0f} / {toi_share*100:.0f})",
            color=INK, fontsize=9, va="bottom")
    ax.set_yticks(y, d.metric)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(d) + 0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Who produces what: Top 6 vs Bottom 6 share of every forward stat", pad=30)
    S.subtitle(ax, "Regular seasons 2021-22 to 2025-26. A bar segment wider than the ice-time line = that group "
                   "over-produces for its minutes.", y=1.01)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_T6), plt.Rectangle((0, 0), 1, 1, color=C_B6)],
              labels=["Top 6 forwards", "Bottom 6 forwards"], loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    save(fig, "01_production_shares")
    return df


# --------------------------------------------------------------------------------------------
# 2. Per-minute rates
# --------------------------------------------------------------------------------------------
RATE_METRICS = [("G", "Goals"), ("PTS", "Points"), ("ixG", "Expected goals"), ("iCF", "Shot attempts"),
                ("G_5v5", "5-on-5 goals"), ("pen_drawn", "Penalties drawn"), ("takeaways", "Takeaways"),
                ("giveaways", "Giveaways"), ("pen_taken", "Penalties taken"), ("blocks", "Blocked shots"),
                ("hits", "Hits")]


def per60_rates(tg, pg):
    rs = tg[tg.game_type == 2]
    rows = []
    for col, label in RATE_METRICS:
        r6 = rs[f"T6_{col}"].sum() / (rs.T6_toi_sec.sum() / 3600)
        rb = rs[f"B6_{col}"].sum() / (rs.B6_toi_sec.sum() / 3600)
        rows.append(dict(metric=label, top6_per60=r6, bottom6_per60=rb, bottom6_as_pct_of_top6=rb / r6))
    df = pd.DataFrame(rows)
    # usage / shooting context
    f = pg[(pg.game_type == 2) & (pg.pos_group == "F")]
    ctx = f.groupby("group").agg(toi=("toi_sec", "mean"), shifts=("shifts", "mean"), G=("G", "sum"),
                                 sog=("iSOG", "sum"), xg=("ixG", "sum"), toi_tot=("toi_sec", "sum"))
    ctx["avg_toi_min"] = ctx.toi / 60
    ctx["avg_shift_sec"] = ctx.toi / ctx.shifts
    ctx["shooting_pct"] = ctx.G / ctx.sog
    ctx["goals_minus_xg_per60"] = (ctx.G - ctx.xg) / (ctx.toi_tot / 3600)
    SUMMARY["usage"] = {g: dict(avg_toi_min=round(ctx.loc[g, "avg_toi_min"], 2),
                                avg_shift_sec=round(ctx.loc[g, "avg_shift_sec"], 1),
                                shooting_pct=pct(ctx.loc[g, "shooting_pct"]),
                                goals_minus_xg_per60=round(ctx.loc[g, "goals_minus_xg_per60"], 3))
                        for g in ("T6", "B6")}
    SUMMARY["per60"] = {r.metric: dict(top6=round(r.top6_per60, 3), bottom6=round(r.bottom6_per60, 3),
                                       ratio=pct(r.bottom6_as_pct_of_top6)) for r in df.itertuples()}
    table(df, "02_per60_rates")
    table(ctx.reset_index(), "02_usage_context")

    d = df.sort_values("bottom6_as_pct_of_top6").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(d))
    ax.barh(y, d.bottom6_as_pct_of_top6, color=C_B6, height=0.62, edgecolor=S.SURFACE, linewidth=2)
    ax.axvline(1, color=INK, lw=1.2)
    ax.text(1.005, len(d) - 0.45, "Same rate as Top 6", color=INK, fontsize=9, va="bottom")
    for i, r in d.iterrows():
        ax.text(max(r.bottom6_as_pct_of_top6, 1.0) + 0.015, i,
                f"{r.bottom6_as_pct_of_top6*100:.0f}%   ({r.bottom6_per60:.2f} vs {r.top6_per60:.2f} /60)",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, d.metric)
    ax.set_xlim(0, max(1.6, d.bottom6_as_pct_of_top6.max() + 0.55))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Minute for minute: Bottom 6 rate as a % of the Top 6 rate", pad=30)
    S.subtitle(ax, "Per 60 minutes of ice time (all situations), regular seasons 2021-26. Labels: Bottom 6 vs Top 6 "
                   "per-60 rate.", y=1.01)
    save(fig, "02_per60_rates")
    return df


def even_strength_on_ice(ts):
    """League-wide on-ice even-strength goals for / against per 60 of each group's line time."""
    rows = []
    for g in ("T6", "B6"):
        hrs = ts[f"{g}_es_team_sec"].sum() / 3600
        gf, ga = ts[f"{g}_esGF"].sum() / hrs, ts[f"{g}_esGA"].sum() / hrs
        rows.append(dict(group=g, GF60=gf, GA60=ga, GD60=gf - ga, GF_pct=gf / (gf + ga),
                         share_of_es_time=ts[f"{g}_es_team_sec"].sum() / ts.es_sec.sum(),
                         teams_with_positive_GD=(ts[f"{g}_esGD60"] > 0).mean()))
    df = pd.DataFrame(rows)
    table(df, "02b_even_strength_on_ice")
    SUMMARY["even_strength_on_ice"] = {r.group: dict(GF60=round(r.GF60, 3), GA60=round(r.GA60, 3), GD60=round(r.GD60, 3),
                                                     GF_pct=pct(r.GF_pct), share_of_es_time=pct(r.share_of_es_time),
                                                     teams_with_positive_GD=pct(r.teams_with_positive_GD))
                                       for r in df.itertuples()}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    metrics = [("GF60", "Goals for /60"), ("GA60", "Goals against /60"), ("GD60", "Goal differential /60")]
    w = 0.34
    for j, (g, col) in enumerate((("T6", C_T6), ("B6", C_B6))):
        r = df.set_index("group").loc[g]
        xs = np.arange(len(metrics)) + (j - 0.5) * w
        vals = [r[m] for m, _ in metrics]
        ax.bar(xs, vals, width=w, color=col, edgecolor=S.SURFACE, linewidth=2, label=S.LABEL[g])
        for x_, v in zip(xs, vals):
            ax.text(x_, v + (0.04 if v >= 0 else -0.04), f"{v:+.2f}" if abs(v) < 1 else f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10, weight="bold", color=INK)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(metrics)), [l for _, l in metrics])
    ax.legend(loc="upper right")
    ax.set_ylabel("Goals per 60 min of team ES time")
    ax.set_title("At even strength the Top 6 win their minutes; the Bottom 6, on average, lose theirs", pad=26)
    S.subtitle(ax, f"On-ice even-strength goals while each group's line is out (estimated from +/- and goal credits). "
                   f"{len(ts)} team-seasons 2021-26.", y=1.01)
    save(fig, "02b_even_strength_on_ice")
    return df


# --------------------------------------------------------------------------------------------
# 3. Game level: how goals from each group translate into wins
# --------------------------------------------------------------------------------------------
def game_level(tg):
    rs = tg[tg.game_type == 2].copy()
    rs["t6g"] = rs.T6_G.clip(upper=5)
    rs["b6g"] = rs.B6_G.clip(upper=3)
    heat = rs.pivot_table(index="b6g", columns="t6g", values="win", aggfunc="mean")
    n = rs.pivot_table(index="b6g", columns="t6g", values="win", aggfunc="size")
    out = heat.stack().rename("win_pct").reset_index().merge(n.stack().rename("games").reset_index())
    table(out, "03_winpct_by_goals_heatmap")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", S.BLUE_RAMP)
    im = ax.imshow(heat.values, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v, c = heat.values[i, j], n.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i + 0.08, f"{v*100:.0f}%", ha="center", va="center", fontsize=11, weight="bold",
                    color="white" if v > 0.55 else INK)
            ax.text(j, i - 0.25, f"n={int(c):,}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK2)
    ax.set_xticks(range(heat.shape[1]), [str(int(c)) if c < 5 else "5+" for c in heat.columns])
    ax.set_yticks(range(heat.shape[0]), [str(int(c)) if c < 3 else "3+" for c in heat.index])
    ax.set_xlabel("Goals scored by the Top 6 forwards")
    ax.set_ylabel("Goals scored by the Bottom 6")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("Win % by who scores", pad=26)
    S.subtitle(ax, f"{len(rs):,} team-games, regular seasons 2021-26 (wins include OT/SO).", y=1.01)
    save(fig, "03_winpct_heatmap")

    # depth-goal lift
    rs["t6c"] = rs.T6_G.clip(upper=4)
    lift = rs.groupby(["t6c", rs.B6_G > 0]).win.agg(["mean", "size"]).unstack()
    lift.columns = [f"{a}_{'b6_scored' if b else 'b6_blanked'}" for a, b in lift.columns]
    lift = lift.reset_index()
    lift["lift_pts"] = lift["mean_b6_scored"] - lift["mean_b6_blanked"]
    table(lift, "04_depth_goal_lift")
    SUMMARY["depth_goal_lift"] = {int(r.t6c): dict(win_if_b6_blanked=pct(r.mean_b6_blanked),
                                                    win_if_b6_scored=pct(r.mean_b6_scored),
                                                    games_b6_scored=int(r.size_b6_scored),
                                                    games_b6_blanked=int(r.size_b6_blanked)) for r in lift.itertuples()}
    SUMMARY["b6_scores_at_least_one"] = pct((rs.B6_G > 0).mean())
    SUMMARY["win_pct_b6_scored_vs_not"] = [pct(rs[rs.B6_G > 0].win.mean()), pct(rs[rs.B6_G == 0].win.mean())]
    SUMMARY["t6_scores_at_most_one"] = pct((rs.T6_G <= 1).mean())

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = lift.t6c.values
    ax.plot(x, lift.mean_b6_blanked, color=MUTED, marker="o", label="Bottom 6 scored 0")
    ax.plot(x, lift.mean_b6_scored, color=C_B6, marker="o", label="Bottom 6 scored 1+")
    for xi, a, b in zip(x, lift.mean_b6_blanked, lift.mean_b6_scored):
        ax.annotate("", xy=(xi, b - 0.012), xytext=(xi, a + 0.012),
                    arrowprops=dict(arrowstyle="-|>", color=C_B6, lw=1.2))
        ax.text(xi + 0.08, (a + b) / 2, f"+{(b-a)*100:.0f} pts", color=INK, fontsize=9.5, va="center")
    ax.text(x[-1] + 0.12, lift.mean_b6_scored.iloc[-1], "Bottom 6 scored 1+", color=INK, va="center", fontsize=9.5)
    ax.text(x[-1] + 0.12, lift.mean_b6_blanked.iloc[-1], "Bottom 6 scored 0", color=INK2, va="center", fontsize=9.5)
    ax.set_xticks(x, [str(int(v)) if v < 4 else "4+" for v in x])
    ax.set_xlim(-0.3, x[-1] + 1.1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xlabel("Goals scored by the Top 6 that night")
    ax.set_ylabel("Win %")
    ax.legend(loc="upper left")
    ax.set_title("The depth-goal lift: one Bottom 6 goal changes the odds most when the stars are quiet", pad=26)
    S.subtitle(ax, "Win % at each Top 6 output, split by whether the Bottom 6 scored. Regular seasons 2021-26.", y=1.01)
    save(fig, "04_depth_goal_lift")

    # logistic model: is a Bottom 6 goal worth as much as a Top 6 goal? (offense only; GA left out because
    # GF = T6 + B6 + D goals would make win/loss almost perfectly separable)
    m = smf.logit("win ~ T6_G + B6_G + D_G + is_home", data=rs).fit(disp=0)
    coefs = pd.DataFrame({"coef": m.params, "se": m.bse, "odds_ratio": np.exp(m.params)}).reset_index()
    table(coefs, "05_logit_goals_by_group")
    SUMMARY["logit_goal_coefs"] = {k: dict(coef=round(v, 3), se=round(m.bse[k], 3), odds_ratio=round(np.exp(v), 3))
                                   for k, v in m.params.items()}

    # counterfactual 1 (exact): erase every Bottom 6 goal and re-score each game
    gf_new = rs.GF - rs.B6_G
    keep = rs.B6_G == 0
    win_new = np.where(keep, rs.win, np.where(gf_new > rs.GA, 1, np.where(gf_new == rs.GA, 0.5, 0)))
    pts_new = np.where(keep, rs.pts, np.where(gf_new > rs.GA, 2, np.where(gf_new == rs.GA, 1.5, 0)))
    flipped = (~keep) & (rs.win == 1) & (gf_new <= rs.GA)
    # counterfactual 2 (model): extra Top 6 goals per game needed to restore the win rate with no depth scoring
    base = m.predict(rs).mean()
    lo, hi = 0.0, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if m.predict(rs.assign(B6_G=0, T6_G=rs.T6_G + mid)).mean() < base:
            lo = mid
        else:
            hi = mid
    need = rs.T6_G.mean() + hi
    ts_t6 = rs.groupby(["season", "team"]).T6_G.mean()
    SUMMARY["no_depth_scoring"] = dict(
        actual_win_pct=pct(rs.win.mean()), win_pct_if_b6_goals_erased=pct(win_new.mean()),
        actual_pts_pct=pct(rs.pts.mean() / 2), pts_pct_if_b6_goals_erased=pct(pts_new.mean() / 2),
        wins_lost_per_82=round((rs.win.mean() - win_new.mean()) * 82, 1),
        standings_points_lost_per_82=round((rs.pts.mean() - pts_new.mean()) * 82, 1),
        share_of_wins_that_needed_a_b6_goal=pct(flipped.sum() / rs.win.sum()),
        extra_top6_goals_per_game_needed=round(hi, 3), b6_goals_per_game=round(rs.B6_G.mean(), 3),
        t6_goals_per_game=round(rs.T6_G.mean(), 3), t6_goals_needed_per_game=round(need, 3),
        t6_increase_needed_pct=pct(hi / rs.T6_G.mean()),
        share_of_team_seasons_whose_top6_reach_it=pct((ts_t6 >= need).mean()),
        best_top6_goals_per_game=round(ts_t6.max(), 3))
    return rs


# --------------------------------------------------------------------------------------------
# 4. Clutch: leverage of the goals each group scores
# --------------------------------------------------------------------------------------------
def clutch(goals, pg):
    grp = pg.set_index(["game_id", "player_id"])["group"]
    g = goals[goals.game_type == 2].copy()
    g["group"] = grp.reindex(pd.MultiIndex.from_arrays([g.game_id, g.scorer.fillna(-1).astype(int)])).values
    g = g[g.group.isin(["T6", "B6", "D"])]
    agg = g.groupby("group").agg(goals=("wpa", "size"), avg_wpa=("wpa", "mean"), gwg_rate=("gwg", "mean"),
                                 go_ahead_rate=("go_ahead", "mean"), tying_rate=("tying", "mean"),
                                 empty_net_rate=("empty_net", "mean"),
                                 third_or_ot=("period", lambda p: (p >= 3).mean()),
                                 pp_rate=("strength", lambda s: (s == "PP").mean())).reset_index()
    agg["lead_changing_rate"] = agg.go_ahead_rate + agg.tying_rate
    table(agg, "06_goal_leverage_by_group")
    SUMMARY["goal_leverage"] = {r.group: dict(avg_wpa=round(r.avg_wpa, 4), gwg_rate=pct(r.gwg_rate),
                                              lead_changing=pct(r.lead_changing_rate), empty_net=pct(r.empty_net_rate),
                                              pp_rate=pct(r.pp_rate)) for r in agg.itertuples()}

    order = ["T6", "B6", "D"]
    a = agg.set_index("group").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), gridspec_kw=dict(wspace=0.3))
    panels = [("avg_wpa", "Win probability added per goal", lambda v: f"{v*100:.1f}%"),
              ("lead_changing_rate", "Goals that tie or take the lead", lambda v: f"{v*100:.0f}%"),
              ("gwg_rate", "Goals that are game-winners", lambda v: f"{v*100:.0f}%")]
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.bar(range(3), a[col], color=[S.COLOR[k] for k in order], width=0.6, edgecolor=S.SURFACE, linewidth=2)
        for i, v in enumerate(a[col]):
            ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=10, color=INK, weight="bold")
        ax.set_xticks(range(3), ["Top 6", "Bottom 6", "Defense"])
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, a[col].max() * 1.2)
    fig.suptitle("Are Bottom 6 goals 'cheap'? Leverage of goals by who scores them", x=0.01, ha="left",
                 fontsize=13, weight="bold", y=1.04)
    fig.text(0.01, 0.955, "Win probability added = in-game win probability after the goal minus before it "
             "(model on clock + score). Regular seasons 2021-26.", color=INK2, fontsize=10)
    save(fig, "05_goal_leverage")
    return agg


# --------------------------------------------------------------------------------------------
# 5. Team-season level: what separates good teams?
# --------------------------------------------------------------------------------------------
def lmg(df, y, xs):
    """Lindeman-Merenda-Gold relative importance: average R^2 gain over all predictor orderings."""
    def r2(cols):
        if not cols:
            return 0.0
        return sm.OLS(df[y], sm.add_constant(df[list(cols)])).fit().rsquared
    cache = {}
    for k in range(len(xs) + 1):
        for c in itertools.combinations(xs, k):
            cache[frozenset(c)] = r2(c)
    imp = {}
    for x in xs:
        others = [o for o in xs if o != x]
        tot, w = 0.0, 0.0
        for k in range(len(others) + 1):
            for c in itertools.combinations(others, k):
                weight = 1 / (len(xs) * len(list(itertools.combinations(others, k))))
                tot += weight * (cache[frozenset(c) | {x}] - cache[frozenset(c)])
                w += weight
        imp[x] = tot
    return imp, cache[frozenset(xs)]


def team_season(ts, info):
    ts = ts.copy()
    xs = ["T6_esGD_pg", "B6_esGD_pg", "st_net_pg", "late_net_pg"]
    m = smf.ols("pts_pct ~ " + " + ".join(xs), data=ts).fit()
    imp, r2 = lmg(ts, "pts_pct", xs)
    z = ts[xs].std()
    stdb = (m.params[xs] * z / ts.pts_pct.std())
    res = pd.DataFrame(dict(term=xs, coef=m.params[xs].values, se=m.bse[xs].values,
                            std_beta=stdb.values, lmg_r2_share=[imp[x] for x in xs],
                            sd_across_teams=z.values))
    res["pct_of_explained"] = res.lmg_r2_share / r2
    table(res, "07_team_season_regression")
    SUMMARY["team_model"] = dict(r2=round(r2, 3), n=len(ts),
                                 terms={r.term: dict(coef=round(r.coef, 4), std_beta=round(r.std_beta, 3),
                                                     share_of_explained=pct(r.pct_of_explained),
                                                     sd=round(r.sd_across_teams, 3)) for r in res.itertuples()})
    # simple GD -> standings conversion used throughout the theory section
    g = smf.ols("pts_pct ~ GD_pg", data=ts).fit()
    SUMMARY["pts_pct_per_goal_diff_per_game"] = round(g.params["GD_pg"], 4)
    SUMMARY["standings_points_per_goal_diff"] = round(g.params["GD_pg"] * 2, 3)  # per season goal of GD
    # playoff cut line in points %
    lg = smf.logit("made_playoffs ~ pts_pct", data=ts).fit(disp=0)
    cut = -lg.params["Intercept"] / lg.params["pts_pct"]
    SUMMARY["playoff_line_pts_pct"] = round(cut, 4)

    # chart: relative importance + standardized effect
    labels = {"T6_esGD_pg": "Top 6 even-strength goal diff.", "B6_esGD_pg": "Bottom 6 even-strength goal diff.",
              "st_net_pg": "Special teams goal diff.", "late_net_pg": "Empty-net / extra-attacker goal diff."}
    colors = {"T6_esGD_pg": C_T6, "B6_esGD_pg": C_B6, "st_net_pg": NEUTRAL, "late_net_pg": NEUTRAL}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    r = res.set_index("term").loc[xs]
    for ax, col, title, fmt in ((axes[0], "pct_of_explained", "Share of the explained gap between teams", "{:.0f}%"),
                                (axes[1], "coef", "Standings points % per +1 goal/game of that group's GD", "{:.3f}")):
        vals = r[col] * (100 if col == "pct_of_explained" else 1)
        ax.barh(range(len(xs)), vals, color=[colors[x] for x in xs], height=0.6, edgecolor=S.SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=10, color=INK, weight="bold")
        ax.set_yticks(range(len(xs)), [labels[x] for x in xs])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(0, vals.max() * 1.3)
    axes[1].set_yticklabels([])
    fig.suptitle("Team standings: how much of the difference between teams each group explains",
                 x=0.01, ha="left", fontsize=13, weight="bold", y=1.05)
    fig.text(0.01, 0.955, f"OLS of points % on each group's estimated on-ice even-strength goal differential per game, "
             f"{len(ts)} team-seasons 2021-26, R² = {r2:.2f}. Importance = LMG decomposition of R².",
             color=INK2, fontsize=10)
    save(fig, "06_team_relative_importance")
    return ts, m


def quadrants(ts):
    t_med, b_med = ts.T6_esGD60.median(), ts.B6_esGD60.median()
    ts = ts.assign(top=np.where(ts.T6_esGD60 >= t_med, "Strong Top 6", "Weak Top 6"),
                   bot=np.where(ts.B6_esGD60 >= b_med, "Strong Bottom 6", "Weak Bottom 6"))
    q = ts.groupby(["top", "bot"]).agg(teams=("pts_pct", "size"), pts_pct=("pts_pct", "mean"),
                                       playoff_rate=("made_playoffs", "mean"),
                                       pts_per_82=("pts_pct", lambda s: s.mean() * 164)).reset_index()
    order = [("Strong Top 6", "Strong Bottom 6"), ("Strong Top 6", "Weak Bottom 6"),
             ("Weak Top 6", "Strong Bottom 6"), ("Weak Top 6", "Weak Bottom 6")]
    q = q.set_index(["top", "bot"]).loc[order].reset_index()
    table(q, "08_team_quadrants")
    SUMMARY["quadrants"] = {f"{r.top} + {r.bot}": dict(teams=int(r.teams), pts_pct=pct(r.pts_pct),
                                                       playoff_rate=pct(r.playoff_rate),
                                                       pts_per_82=round(r.pts_per_82, 1)) for r in q.itertuples()}
    names = ["Both groups\nstrong", "Strong Top 6\nonly", "Strong Bottom 6\nonly", "Both groups\nweak"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, title, fmt in ((axes[0], "pts_per_82", "Standings points per 82 games", "{:.0f}"),
                                (axes[1], "playoff_rate", "Made the playoffs", "{:.0%}")):
        ax.bar(range(4), q[col], color=[C_T6, C_T6, C_B6, NEUTRAL], width=0.62, edgecolor=S.SURFACE, linewidth=2)
        for i, v in enumerate(q[col]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=10.5, weight="bold", color=INK)
        ax.set_xticks(range(4), names, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, q[col].max() * 1.18)
    fig.suptitle("Depth matters at both ends of the lineup", x=0.01, ha="left", fontsize=13, weight="bold", y=1.05)
    fig.text(0.01, 0.955, "Team-seasons split at the league median of Top 6 and Bottom 6 even-strength on-ice goal differential "
             f"per 60 (n = {len(ts)}; {', '.join(str(int(t)) for t in q.teams)} teams per group).",
             color=INK2, fontsize=10)
    save(fig, "07_team_quadrants")
    return q


# --------------------------------------------------------------------------------------------
# 6. THEORY A - what would the Top 6 need if there were no (useful) Bottom 6?
# --------------------------------------------------------------------------------------------
def theory_requirements(ts):
    """Iso-standings lines: combinations of Top 6 and Bottom 6 on-ice GD/60 that yield the same points %."""
    s = ts.T6_es_share.mean()  # Top 6 share of even-strength line time
    m = smf.ols("pts_pct ~ T6_esGD60 + B6_esGD60 + st_net_pg + late_net_pg", data=ts).fit()
    a, bt, bb, bs = m.params["Intercept"], m.params["T6_esGD60"], m.params["B6_esGD60"], m.params["st_net_pg"]
    st_mean = ts.st_net_pg.mean() + m.params["late_net_pg"] / bs * ts.late_net_pg.mean()
    cut = SUMMARY["playoff_line_pts_pct"]
    b_p = {p: ts.B6_esGD60.quantile(p / 100) for p in (5, 10, 25, 50, 75, 90, 95)}
    t_p = {p: ts.T6_esGD60.quantile(p / 100) for p in (5, 10, 25, 50, 75, 90, 95)}

    def required_t6(target, b6):
        return (target - a - bs * st_mean - bb * b6) / bt

    req = []
    for label, b in (("Replacement-level Bottom 6 (5th pct)", b_p[5]), ("Weak Bottom 6 (25th pct)", b_p[25]),
                     ("Average Bottom 6 (median)", b_p[50]), ("Strong Bottom 6 (75th pct)", b_p[75]),
                     ("Elite Bottom 6 (95th pct)", b_p[95])):
        for tgt_label, tgt in (("Playoff line", cut), ("Contender (.620)", 0.62)):
            need = required_t6(tgt, b)
            req.append(dict(bottom6=label, bottom6_onGD60=b, target=tgt_label, target_pts_pct=tgt,
                            required_top6_onGD60=need,
                            share_of_real_top6_that_meet_it=(ts.T6_esGD60 >= need).mean()))
    req = pd.DataFrame(req)
    table(req, "09_required_top6")
    SUMMARY["theory_requirements"] = dict(
        top6_share_of_forward_toi=pct(s), model_coefs=dict(top6=round(bt, 4), bottom6=round(bb, 4)),
        bottom6_percentiles={k: round(v, 3) for k, v in b_p.items()},
        top6_percentiles={k: round(v, 3) for k, v in t_p.items()},
        rows=[dict(bottom6=r.bottom6, target=r.target, required_top6_onGD60=round(r.required_top6_onGD60, 3),
                   share_of_top6_groups_that_meet_it=pct(r.share_of_real_top6_that_meet_it)) for r in req.itertuples()])

    fig, ax = plt.subplots(figsize=(10.5, 7))
    made = ts.made_playoffs == 1
    ax.scatter(ts.B6_esGD60[~made], ts.T6_esGD60[~made], s=34, color=NEUTRAL, edgecolor=S.SURFACE, lw=1.2,
               label="Missed playoffs", zorder=3)
    ax.scatter(ts.B6_esGD60[made], ts.T6_esGD60[made], s=34, color=INK2, edgecolor=S.SURFACE, lw=1.2,
               label="Made playoffs", zorder=3)
    xs = np.linspace(ts.B6_esGD60.min() - 0.15, ts.B6_esGD60.max() + 0.15, 50)
    for tgt, lab, col, lw in ((0.45, ".450 points %", MUTED, 1), (cut, f"Playoff line (~{cut:.3f})", C_T6, 2.2),
                              (0.62, ".620 (contender)", MUTED, 1)):
        ax.plot(xs, required_t6(tgt, xs), color=col, lw=lw, zorder=2)
        ax.text(xs[-1] + 0.03, required_t6(tgt, xs[-1]), lab, color=INK if col == C_T6 else INK2, fontsize=9,
                va="center", weight="bold" if col == C_T6 else "normal")
    notes = []
    for letter, (p, lab) in zip("ABC", ((5, "Replacement-level Bottom 6 (5th percentile)"),
                                         (50, "Average Bottom 6 (median)"), (95, "Elite Bottom 6 (95th percentile)"))):
        bx = b_p[p]
        ax.axvline(bx, color=C_B6, lw=0.9, alpha=0.7, zorder=1)
        need = required_t6(cut, bx)
        ax.scatter([bx], [need], s=90, color=C_B6, edgecolor=S.SURFACE, lw=2, zorder=4)
        ax.annotate(letter, (bx, need), xytext=(7, 7), textcoords="offset points", fontsize=11, weight="bold",
                    color=INK, bbox=dict(boxstyle="round,pad=0.2", facecolor=S.SURFACE, edgecolor="none"))
        notes.append(f"{letter}  {lab}: Top 6 must be {need:+.2f} goals/60 to reach the playoff line - "
                     f"{(ts.T6_esGD60 >= need).mean()*100:.0f}% of real Top 6 groups get there")
    fig.text(0.01, -0.035, "\n".join(notes), fontsize=9.5, color=INK, va="top", ha="left", linespacing=1.6)
    ax.axhline(0, color=S.AXIS, lw=0.8)
    ax.axvline(0, color=S.AXIS, lw=0.8)
    ax.set_xlabel("Bottom 6 on-ice even-strength goal differential per 60 (estimated)")
    ax.set_ylabel("Top 6 on-ice even-strength goal differential per 60")
    ax.set_xlim(xs[0], xs[-1] + 0.75)
    ax.legend(loc="upper left")
    ax.set_title("Theory: how good must the Top 6 be to carry a team with no Bottom 6?", pad=26)
    S.subtitle(ax, f"Lines = equal standings points % from the team model. Dots = {len(ts)} real team-seasons "
                   "2021-26.", y=1.01)
    save(fig, "08_required_top6_isolines")
    return req, m


def theory_two_line(ts, tg):
    """What if only 6 forwards dressed and the Top 6 played every forward shift?"""
    s = ts.T6_es_team_sec.sum() / ts.es_sec.sum()  # Top 6 share of even-strength line time
    s_all = ts.T6_toi_sec.sum() / (ts.T6_toi_sec.sum() + ts.B6_toi_sec.sum())
    lg = dict(T6_GF60=ts.T6_esGF.sum() / (ts.T6_es_team_sec.sum() / 3600),
              T6_GA60=ts.T6_esGA.sum() / (ts.T6_es_team_sec.sum() / 3600),
              B6_GF60=ts.B6_esGF.sum() / (ts.B6_es_team_sec.sum() / 3600),
              B6_GA60=ts.B6_esGA.sum() / (ts.B6_es_team_sec.sum() / 3600))
    es_per_game = ts.es_sec.sum() / ts.GP.sum() / 3600  # fraction of a game played at even strength
    ppg = SUMMARY["pts_pct_per_goal_diff_per_game"] * 164  # standings points per +1 GD per game over 82 games

    # fatigue anchor 1: back-to-backs (team-season fixed effects)
    rs = tg[(tg.game_type == 2) & tg.rest_days.notna()].copy()
    rs["key"] = rs.team + rs.season.astype(str)
    fat = {}
    for col, lab in (("T6_ixG", "Top 6 xG"), ("T6_PTS", "Top 6 points"), ("T6_iCF", "Top 6 shot attempts"),
                     ("GF", "Team goals for"), ("GA", "Team goals against")):
        mm = smf.ols(f"{col} ~ b2b + C(key)", data=rs[rs.rest_days.between(1, 4)]).fit(cov_type="cluster",
                                                                                        cov_kwds={"groups": rs[rs.rest_days.between(1, 4)].key.astype("category").cat.codes})
        base = rs.loc[(rs.b2b == 0) & rs.rest_days.between(1, 4), col].mean()
        fat[lab] = dict(change=round(mm.params["b2b"] / base, 4), se=round(mm.bse["b2b"] / base, 4))
    SUMMARY["b2b_fatigue"] = fat

    # fatigue anchor 2: how per-minute output changes for Top 6 forwards as their ice time rises (within player)
    def break_even(t6_gf, t6_ga, b6_gf, b6_ga):
        # GD(two-line, d) = t6_gf*(1-d) - t6_ga*(1+d);  GD(actual) = s*(t6_gf-t6_ga) + (1-s)*(b6_gf-b6_ga)
        actual = (s * (t6_gf - t6_ga) + (1 - s) * (b6_gf - b6_ga)) * es_per_game
        return ((t6_gf - t6_ga) * es_per_game - actual) / ((t6_gf + t6_ga) * es_per_game), actual

    profiles = {}
    for name, tq, bq in (("Elite Top 6 + average Bottom 6", 0.9, 0.5), ("Average Top 6 + average Bottom 6", 0.5, 0.5),
                         ("Elite Top 6 + weak Bottom 6", 0.9, 0.1), ("Average Top 6 + elite Bottom 6", 0.5, 0.9)):
        t_gf, t_ga = ts.T6_esGF60.quantile(tq), ts.T6_esGA60.quantile(1 - tq)
        b_gf, b_ga = ts.B6_esGF60.quantile(bq), ts.B6_esGA60.quantile(1 - bq)
        d_star, actual = break_even(t_gf, t_ga, b_gf, b_ga)
        profiles[name] = dict(T6_GF60=t_gf, T6_GA60=t_ga, B6_GF60=b_gf, B6_GA60=b_ga, break_even_fatigue=d_star,
                              actual_GD_per_game=actual)
    d_star_lg, _ = break_even(lg["T6_GF60"], lg["T6_GA60"], lg["B6_GF60"], lg["B6_GA60"])
    SUMMARY["two_line_theory"] = dict(top6_share=pct(s), league_rates={k: round(v, 3) for k, v in lg.items()},
                                      league_break_even_fatigue=pct(d_star_lg),
                                      toi_per_top6_forward_now=round(SUMMARY["usage"]["T6"]["avg_toi_min"], 1),
                                      toi_per_top6_forward_two_lines=round(SUMMARY["usage"]["T6"]["avg_toi_min"] / s_all, 1),
                                      es_minutes_per_game=round(es_per_game * 60, 1),
                                      profiles={k: dict(break_even_fatigue=pct(v["break_even_fatigue"]),
                                                        **{kk: round(vv, 3) for kk, vv in v.items() if kk != "break_even_fatigue"})
                                                for k, v in profiles.items()})
    rows = []
    deltas = np.linspace(0, 0.30, 61)
    for name, v in profiles.items():
        for d in deltas:
            gd_two = (v["T6_GF60"] * (1 - d) - v["T6_GA60"] * (1 + d)) * es_per_game
            rows.append(dict(profile=name, fatigue=d, gd_change_per_game=gd_two - v["actual_GD_per_game"],
                             standings_points_change=(gd_two - v["actual_GD_per_game"]) * ppg))
    curves = pd.DataFrame(rows)
    table(curves, "10_two_line_theory_curves")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    styles = {"Elite Top 6 + average Bottom 6": (C_T6, "-"), "Average Top 6 + average Bottom 6": (INK2, "-"),
              "Elite Top 6 + weak Bottom 6": (C_T6, (0, (1, 0))), "Average Top 6 + elite Bottom 6": (C_B6, "-")}
    widths = {"Elite Top 6 + weak Bottom 6": 1.2}
    for name, grp in curves.groupby("profile", sort=False):
        col, ls = styles[name]
        ax.plot(grp.fatigue * 100, grp.standings_points_change, color=col, lw=widths.get(name, 2.2))
        de = profiles[name]["break_even_fatigue"]
        yend = grp.standings_points_change.iloc[-1]
        if de > 0:
            ax.scatter([de * 100], [0], s=60, color=col, edgecolor=S.SURFACE, lw=2, zorder=5)
            lab = f"{name}\n(break-even fatigue {de*100:.1f}%)"
        else:
            lab = f"{name}\n(worse even with zero fatigue)"
        ax.text(30.4, yend, lab, fontsize=8.5, color=INK, va="center")
    ax.axhline(0, color=INK, lw=1)
    lo = SUMMARY["b2b_fatigue"]["Top 6 xG"]["change"]
    ax.axvspan(0, max(abs(lo) * 100, 0.3), color=S.GRID, alpha=0.9, zorder=0)
    ax.text(max(abs(lo) * 100, 0.3) + 0.3, ax.get_ylim()[0] * 0.92 if ax.get_ylim()[0] < 0 else -5,
            f"Back-to-back game effect on Top 6 xG: {lo*100:+.1f}%\n(one night of fatigue; a two-line team doubles the "
            "minutes every night)", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(0, 30)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Fatigue penalty on the Top 6 (goals for -x%, goals against +x%) when they play every forward shift")
    ax.set_ylabel("Standings points per 82 games vs. the real 12-forward lineup")
    ax.set_title("Theory: a two-line team only works if the stars barely tire", pad=26)
    S.subtitle(ax, f"Top 6 would go from ~{SUMMARY['two_line_theory']['toi_per_top6_forward_now']:.0f} to "
                   f"~{SUMMARY['two_line_theory']['toi_per_top6_forward_two_lines']:.0f} min a night. Above 0 = two-line "
                   "team is better; dot = break-even.", y=1.01)
    save(fig, "09_two_line_theory")
    return curves


def goals_needed(rs):
    """How many goals do the Top 6 need to win when the Bottom 6 gets nothing?"""
    zero = rs[rs.B6_G == 0]
    g = zero.groupby(zero.T6_G.clip(upper=5)).agg(win_pct=("win", "mean"), games=("win", "size")).reset_index()
    dist = rs.T6_G.clip(upper=5).value_counts(normalize=True).sort_index().rename("freq_all_games").reset_index()
    g = g.merge(dist, on="T6_G")
    table(g, "11_goals_needed_without_depth")
    SUMMARY["goals_needed_no_depth"] = {int(r.T6_G): dict(win_pct=pct(r.win_pct), freq=pct(r.freq_all_games))
                                        for r in g.itertuples()}
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(g))
    ax.bar(x, g.win_pct, color=C_T6, width=0.6, edgecolor=S.SURFACE, linewidth=2)
    ax.plot(x, g.freq_all_games, color=INK2, marker="o", lw=1.5)
    for i, r in g.iterrows():
        ax.text(i, r.win_pct + 0.015, f"{r.win_pct*100:.0f}% win", ha="center", fontsize=10, weight="bold", color=INK)
    ax.text(x[-1] + 0.38, g.freq_all_games.iloc[-1], "How often\nthe Top 6\nscore that many", fontsize=8.5,
            color=INK2, va="center")
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.set_xlim(-0.5, x[-1] + 1.3)
    ax.set_xticks(x, [f"{int(v) if v < 5 else '5+'}\n{f*100:.0f}% of games" for v, f in zip(g.T6_G, g.freq_all_games)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Goals scored by the Top 6 forwards")
    ax.set_title("With no depth scoring, the Top 6 need 3 goals to be favorites - and rarely get them", pad=26)
    S.subtitle(ax, "Bars: win % in games where the Bottom 6 scored 0. Line: how often the Top 6 score that many "
                   "(all games).", y=1.01)
    save(fig, "10_goals_needed_without_depth")
    return g


# --------------------------------------------------------------------------------------------
# 7. THEORY B - what Bottom 6 traits help ANY team win?
# --------------------------------------------------------------------------------------------
B6_TRAITS = [
    ("B6_esGD60", "On-ice ES goal differential /60"), ("B6_esGF60", "On-ice ES goals for /60"),
    ("B6_esGA60", "On-ice ES goals against /60 (lower = better)"), ("B6_G60", "Goals /60"), ("B6_PTS60", "Points /60"),
    ("B6_ixG60", "Expected goals /60"), ("B6_iCF60", "Shot attempts /60"), ("B6_gax60", "Finishing (goals - xG) /60"),
    ("B6_pendiff60", "Penalty differential /60"), ("B6_takegive60", "Takeaways - giveaways /60"),
    ("B6_hits60", "Hits /60"), ("B6_blocks60", "Blocked shots /60"), ("B6_fo_pct", "Faceoff win %"),
    ("B6_dz_share", "Share of faceoffs in own zone"), ("B6_toi_share", "Share of forward ice time"),
]


def bottom6_recipe(ts, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    controls = ["T6_esGD60", "st_net_pg", "gsax_pg"]
    df = ts.dropna(subset=[c for c, _ in B6_TRAITS] + controls + ["pts_pct"]).reset_index(drop=True)
    Xc = sm.add_constant(df[controls])
    y_res = df.pts_pct - sm.OLS(df.pts_pct, Xc).fit().fittedvalues
    rows = []
    for col, lab in B6_TRAITS:
        x_res = df[col] - sm.OLS(df[col], Xc).fit().fittedvalues
        r = np.corrcoef(x_res, y_res)[0, 1]
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(df), len(df))
            bs.append(np.corrcoef(x_res.values[i], y_res.values[i])[0, 1])
        lo, hi = np.percentile(bs, [5, 95])
        # raw (no controls) correlation too
        raw = np.corrcoef(df[col], df.pts_pct)[0, 1]
        rows.append(dict(trait=lab, column=col, partial_r=r, ci90_lo=lo, ci90_hi=hi, raw_r=raw))
    out = pd.DataFrame(rows).sort_values("partial_r")
    table(out, "12_bottom6_recipe")
    SUMMARY["bottom6_recipe"] = {r.trait: dict(partial_r=round(r.partial_r, 3), ci90=[round(r.ci90_lo, 3), round(r.ci90_hi, 3)],
                                               raw_r=round(r.raw_r, 3)) for r in out.itertuples()}

    fig, ax = plt.subplots(figsize=(10.5, 7))
    y = np.arange(len(out))
    sig = (out.ci90_lo > 0) | (out.ci90_hi < 0)
    ax.hlines(y, out.ci90_lo, out.ci90_hi, color=MUTED, lw=1.5)
    ax.scatter(out.partial_r, y, s=70, color=[C_B6 if s_ else NEUTRAL for s_ in sig], edgecolor=S.SURFACE, lw=2, zorder=3)
    for i, r in enumerate(out.itertuples()):
        ax.text(max(r.ci90_hi, r.partial_r) + 0.02, i, f"{r.partial_r:+.2f}", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, out.trait)
    ax.set_xlabel("Partial correlation with standings points % (90% bootstrap interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Theory: which Bottom 6 traits help any team win?", pad=26)
    S.subtitle(ax, "Controlling for Top 6 quality, special teams and goaltending. Orange = interval excludes 0. "
                   f"{len(df)} team-seasons.", y=1.01)
    save(fig, "11_bottom6_recipe")
    return out


def value_curves(ts, m_iso):
    """Standings points gained/lost as each group moves through the league distribution."""
    bt, bb = m_iso.params["T6_esGD60"], m_iso.params["B6_esGD60"]
    ps = np.arange(5, 96, 5)
    rows = []
    for p in ps:
        tq, bq = ts.T6_esGD60.quantile(p / 100), ts.B6_esGD60.quantile(p / 100)
        rows.append(dict(percentile=p,
                         top6_pts=(tq - ts.T6_esGD60.median()) * bt * 164,
                         bottom6_pts=(bq - ts.B6_esGD60.median()) * bb * 164))
    v = pd.DataFrame(rows)
    table(v, "13_value_curves")
    SUMMARY["value_curves"] = dict(
        top6_10_to_90=round(v.loc[v.percentile == 90, "top6_pts"].item() - v.loc[v.percentile == 10, "top6_pts"].item(), 1),
        bottom6_10_to_90=round(v.loc[v.percentile == 90, "bottom6_pts"].item() - v.loc[v.percentile == 10, "bottom6_pts"].item(), 1),
        bottom6_25_to_75=round(v.loc[v.percentile == 75, "bottom6_pts"].item() - v.loc[v.percentile == 25, "bottom6_pts"].item(), 1),
        top6_25_to_75=round(v.loc[v.percentile == 75, "top6_pts"].item() - v.loc[v.percentile == 25, "top6_pts"].item(), 1),
        bottom6_median_to_breakeven=round((0 - ts.B6_esGD60.median()) * bb * 164, 1),
        bottom6_median_onGD60=round(ts.B6_esGD60.median(), 3),
        share_of_bottom6_at_or_above_zero=pct((ts.B6_esGD60 >= 0).mean()))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(v.percentile, v.top6_pts, color=C_T6, marker="o", markersize=5)
    ax.plot(v.percentile, v.bottom6_pts, color=C_B6, marker="o", markersize=5)
    ax.text(96, v.top6_pts.iloc[-1], f"Top 6  ({v.top6_pts.iloc[-1]:+.0f})", color=INK, va="center", fontsize=10)
    ax.text(96, v.bottom6_pts.iloc[-1], f"Bottom 6  ({v.bottom6_pts.iloc[-1]:+.0f})", color=INK, va="center", fontsize=10)
    ax.text(4, v.top6_pts.iloc[0], f"{v.top6_pts.iloc[0]:+.0f}", color=INK2, va="center", ha="right", fontsize=9)
    ax.text(4, v.bottom6_pts.iloc[0], f"{v.bottom6_pts.iloc[0]:+.0f}", color=INK2, va="center", ha="right", fontsize=9)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlim(0, 112)
    ax.set_xlabel("League percentile of that group's on-ice even-strength goal differential per 60")
    ax.set_ylabel("Standings points per 82 games vs. a median group")
    ax.legend(["Top 6 forwards", "Bottom 6 forwards"], loc="upper left")
    ax.set_title("What upgrading each group is worth in the standings", pad=26)
    S.subtitle(ax, "Holding the other group at the league median. From the team model, 2021-26.", y=1.01)
    save(fig, "12_value_curves")
    return v


# --------------------------------------------------------------------------------------------
# 8. Playoffs
# --------------------------------------------------------------------------------------------
def playoffs(tg, ts):
    po = tg[tg.game_type == 3].copy()
    if po.empty:
        SUMMARY["playoffs"] = "no playoff data"
        return None
    rs = tg[tg.game_type == 2]
    comp = []
    for lab, d in (("Regular season", rs), ("Playoffs", po)):
        comp.append(dict(phase=lab,
                         bottom6_goal_share=d.B6_G.sum() / d.GF.sum(), top6_goal_share=d.T6_G.sum() / d.GF.sum(),
                         defense_goal_share=d.D_G.sum() / d.GF.sum(),
                         bottom6_toi_share=d.B6_toi_sec.sum() / (d.B6_toi_sec.sum() + d.T6_toi_sec.sum()),
                         bottom6_wpa_share=d.B6_wpa_scored.sum() / (d.T6_wpa_scored.sum() + d.B6_wpa_scored.sum() + d.D_wpa_scored.sum()),
                         bottom6_gwg_share=d.B6_GWG.sum() / (d.T6_GWG.sum() + d.B6_GWG.sum() + d.D_GWG.sum())))
    comp = pd.DataFrame(comp)
    table(comp, "14_playoffs_vs_regular_season")

    # series: did the winner win the Top 6 battle or the Bottom 6 battle?
    po["series"] = po.game_id // 10
    ser = po.groupby(["series", "season", "team"]).agg(wins=("win", "sum"), T6=("T6_esGD", "sum"), B6=("B6_esGD", "sum"),
                                                      T6_G=("T6_G", "sum"), B6_G=("B6_G", "sum")).reset_index()
    rows = []
    for sid, g in ser.groupby("series"):
        if len(g) != 2:
            continue
        w, l = g.sort_values("wins", ascending=False).iloc[0], g.sort_values("wins", ascending=False).iloc[1]
        rows.append(dict(series=sid, season=w.season, winner=w.team, loser=l.team,
                         winner_better_T6_GD=int(w.T6 > l.T6), winner_better_B6_GD=int(w.B6 > l.B6),
                         winner_more_B6_goals=int(w.B6_G > l.B6_G), winner_more_T6_goals=int(w.T6_G > l.T6_G),
                         tie_B6_goals=int(w.B6_G == l.B6_G), tie_T6_goals=int(w.T6_G == l.T6_G)))
    sr = pd.DataFrame(rows)
    table(sr, "15_playoff_series")

    # champions: winner of the final series each season
    finals = po[po.game_id.astype(str).str[6:8] == "04"]
    champs = finals.groupby(["season", "team"]).win.sum().reset_index().sort_values("win").groupby("season").tail(1)
    ch = []
    for r in champs.itertuples():
        d = po[(po.season == r.season) & (po.team == r.team)]
        lg = po[po.season == r.season]
        oth = lg[lg.team != r.team]
        ch.append(dict(season=season_label(r.season), champion=r.team,
                       bottom6_goal_share=d.B6_G.sum() / d.GF.sum(),
                       league_playoff_bottom6_goal_share=lg.B6_G.sum() / lg.GF.sum(),
                       bottom6_onGD=d.B6_esGD.sum(), top6_onGD=d.T6_esGD.sum(), games=len(d),
                       bottom6_esGD_pg=d.B6_esGD.mean(), top6_esGD_pg=d.T6_esGD.mean(),
                       others_bottom6_esGD_pg=oth.B6_esGD.mean(), others_top6_esGD_pg=oth.T6_esGD.mean()))
    ch = pd.DataFrame(ch).sort_values("season").reset_index(drop=True)
    table(ch, "16_champions_depth")
    SUMMARY["playoffs"] = dict(
        comparison={r.phase: dict(bottom6_goal_share=pct(r.bottom6_goal_share), bottom6_toi_share=pct(r.bottom6_toi_share),
                                  bottom6_gwg_share=pct(r.bottom6_gwg_share), bottom6_wpa_share=pct(r.bottom6_wpa_share))
                    for r in comp.itertuples()},
        series=dict(n=len(sr), winner_better_T6_GD=pct(sr.winner_better_T6_GD.mean()),
                    winner_better_B6_GD=pct(sr.winner_better_B6_GD.mean()),
                    winner_more_B6_goals=pct(sr.winner_more_B6_goals.mean()),
                    winner_more_T6_goals=pct(sr.winner_more_T6_goals.mean())),
        champions=[dict(season=r.season, team=r.champion, games=int(r.games), bottom6_goal_share=pct(r.bottom6_goal_share),
                        league=pct(r.league_playoff_bottom6_goal_share), bottom6_esGD_total=round(r.bottom6_onGD, 1),
                        top6_esGD_total=round(r.top6_onGD, 1), bottom6_esGD_pg=round(r.bottom6_esGD_pg, 3),
                        others_bottom6_esGD_pg=round(r.others_bottom6_esGD_pg, 3)) for r in ch.itertuples()])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    vals = [sr.winner_better_T6_GD.mean(), sr.winner_better_B6_GD.mean()]
    ax.bar([0, 1], vals, color=[C_T6, C_B6], width=0.55, edgecolor=S.SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v*100:.0f}%", ha="center", va="bottom", weight="bold", fontsize=11)
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.text(1.32, 0.51, "coin flip", color=INK2, fontsize=8.5, va="bottom")
    ax.set_xticks([0, 1], ["Won the Top 6\nGD battle", "Won the Bottom 6\nGD battle"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 1.7)
    ax.set_title(f"Series winners ({len(sr)} series, 2022-26)", fontsize=10.5, pad=8)
    ax = axes[1]
    x = np.arange(len(ch))
    ax.bar(x - 0.18, ch.bottom6_esGD_pg, width=0.34, color=C_B6, edgecolor=S.SURFACE, linewidth=2, label="Champion")
    ax.bar(x + 0.18, ch.others_bottom6_esGD_pg, width=0.34, color=NEUTRAL, edgecolor=S.SURFACE, linewidth=2,
           label="Other playoff teams")
    for i, r in ch.iterrows():
        ax.text(i - 0.18, r.bottom6_esGD_pg + 0.01, f"{r.bottom6_esGD_pg:+.2f}", ha="center", va="bottom", fontsize=9.5,
                weight="bold")
        v = r.others_bottom6_esGD_pg
        ax.text(i + 0.18, v + (0.01 if v >= 0 else -0.01), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, [f"{r.champion}\n{r.season}" for r in ch.itertuples()])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    top = max(ch.bottom6_esGD_pg.max(), 0.1)
    ax.set_ylim(min(ch.others_bottom6_esGD_pg.min(), 0) - 0.08, top * 1.35)
    ax.set_title("Stanley Cup champions: Bottom 6 even-strength goal diff. per playoff game", fontsize=10.5, pad=8)
    fig.suptitle("Playoffs: depth shows up in the teams that win", x=0.01, ha="left", fontsize=13, weight="bold", y=1.05)
    fig.text(0.01, 0.955, "Goal differential (GD) = estimated on-ice even-strength goal differential "
             "over the series.", color=INK2, fontsize=10)
    save(fig, "13_playoffs")
    return comp, sr, ch


# --------------------------------------------------------------------------------------------
# 9. Robustness: define Top 6 by that night's TOI instead of season role
# --------------------------------------------------------------------------------------------
def robustness(tg, tg_alt):
    out = {}
    for lab, d in (("season_role", tg), ("game_toi", tg_alt)):
        rs = d[d.game_type == 2]
        m = smf.logit("win ~ T6_G + B6_G + D_G + is_home", data=rs).fit(disp=0)
        out[lab] = dict(bottom6_goal_share=pct(rs.B6_G.sum() / rs.GF.sum()),
                        bottom6_toi_share=pct(rs.B6_toi_sec.sum() / (rs.B6_toi_sec.sum() + rs.T6_toi_sec.sum())),
                        bottom6_pm_per_game=round(rs.B6_plus_minus.mean(), 3),
                        top6_pm_per_game=round(rs.T6_plus_minus.mean(), 3),
                        logit_T6_G=round(m.params["T6_G"], 3), logit_B6_G=round(m.params["B6_G"], 3),
                        win_pct_b6_scored=pct(rs[rs.B6_G > 0].win.mean()),
                        win_pct_b6_blanked=pct(rs[rs.B6_G == 0].win.mean()))
    SUMMARY["robustness"] = out
    table(pd.DataFrame(out).T.reset_index().rename(columns={"index": "definition"}), "17_robustness")


def main():
    S.setup()
    tg, tg_alt, ts, pg, goals, info = load()
    SUMMARY["data"] = dict(
        seasons=sorted(int(s) for s in tg.season.unique()),
        regular_season_games=int((tg.game_type == 2).sum() // 2), playoff_games=int((tg.game_type == 3).sum() // 2),
        team_seasons=int(len(ts)), skater_games=int(len(pg)), goals=int(len(goals)),
        xg_auc=round(float(np.mean([m["auc"] for m in info["xg"]])), 3), wp_auc=round(info["wp"]["auc"], 3),
        k_forwards_per_goal=round(info["k_forwards_per_goal"], 3))
    production_shares(tg, info)
    even_strength_on_ice(ts)
    per60_rates(tg, pg)
    rs = game_level(tg)
    clutch(goals, pg)
    ts, _ = team_season(ts, info)
    quadrants(ts)
    _, m_iso = theory_requirements(ts)
    theory_two_line(ts, tg)
    goals_needed(rs)
    bottom6_recipe(ts)
    value_curves(ts, m_iso)
    playoffs(tg, ts)
    robustness(tg, tg_alt)
    (RES / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
    print(json.dumps(SUMMARY, indent=2, default=float))


if __name__ == "__main__":
    main()
