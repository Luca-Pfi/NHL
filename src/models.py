"""Two helper models built from the play-by-play:

1. Expected goals (xG): probability an unblocked shot attempt becomes a goal, from shot distance,
   angle, type, rebound/rush context, manpower and empty net. Predictions are leave-one-season-out
   so no shot is scored by a model that saw it.
2. In-game win probability (WP): P(home team wins | game clock, score differential, playoffs).
   Used to value every goal by Win Probability Added (WPA) = WP after the goal - WP before it.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

UNBLOCKED = {"shot-on-goal", "missed-shot", "goal"}
SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around",
              "poke", "bat", "between-legs", "cradle"]


def _xg_features(s: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    X["distance"] = s["distance"]
    X["angle"] = s["angle"]
    X["shot_type"] = s["shot_type"].map({k: i for i, k in enumerate(SHOT_TYPES)}).fillna(len(SHOT_TYPES))
    X["rebound"] = s["rebound"]
    X["rush"] = s["rush"]
    X["strength"] = s["strength"].map({"ES": 0, "PP": 1, "SH": 2}).fillna(0)
    X["own_skaters"] = s["own_skaters"]
    X["opp_skaters"] = s["opp_skaters"]
    X["empty_net"] = 1 - s["opp_goalie_in"]
    X["ot"] = (s["period"] > 3).astype(int)
    X["score_diff"] = s["score_diff"].clip(-3, 3)
    X["prev_dt"] = s["prev_dt"].clip(upper=30).fillna(30)
    return X


def fit_xg(shots: pd.DataFrame):
    """Return (shots with an `xg` column, per-season validation metrics)."""
    shots = shots.copy()
    shots["xg"] = 0.0
    mask = shots["event"].isin(UNBLOCKED) & (shots["strength"] != "PS") & shots["distance"].notna()
    data = shots[mask]
    X, y = _xg_features(data), (data["event"] == "goal").astype(int)
    cat_idx = [list(X.columns).index("shot_type")]
    metrics = []
    seasons = sorted(data["season"].unique())
    if len(seasons) >= 3:
        fold = data["season"]
    else:  # too few seasons for leave-one-season-out: 5 folds grouped by game
        fold = (data["game_id"] % 5) + 1
    for season in sorted(fold.unique()):
        tr, te = fold != season, fold == season
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=200, l2_regularization=1.0,
                                               categorical_features=cat_idx, random_state=0)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        shots.loc[data.index[te.values], "xg"] = p
        metrics.append(dict(season=int(season), shots=int(te.sum()), goals=int(y[te].sum()),
                            xg=float(p.sum()), auc=roc_auc_score(y[te], p), logloss=log_loss(y[te], p)))
    # penalty shots / unlocated unblocked shots: league average conversion for that situation
    rest = shots["event"].isin(UNBLOCKED) & ~mask
    if rest.any():
        rate = (shots.loc[rest, "event"] == "goal").mean()
        shots.loc[rest, "xg"] = rate
    return shots, pd.DataFrame(metrics)


def _wp_features(t, home_diff, playoff):
    t = np.asarray(t, dtype=float)
    return pd.DataFrame({
        "t": np.minimum(t, 3600.0),  # any overtime state is 'tied, sudden death'
        "ot": (t >= 3600).astype(int),
        "home_diff": np.clip(home_diff, -5, 5),
        "playoff": np.asarray(playoff, dtype=int),
    })


class WinProb:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
            monotonic_cst=[0, 0, 1, 0], random_state=0)

    def fit(self, wp_states: pd.DataFrame):
        X = _wp_features(wp_states["t"], wp_states["home_diff"], wp_states["game_type"] == 3)
        self.model.fit(X, wp_states["home_win"])
        p = self.model.predict_proba(X)[:, 1]
        self.metrics = dict(states=len(X), auc=roc_auc_score(wp_states["home_win"], p),
                            logloss=log_loss(wp_states["home_win"], p))
        return self

    def home_wp(self, t, home_diff, playoff):
        return self.model.predict_proba(_wp_features(t, home_diff, playoff))[:, 1]


def goal_wpa(goals: pd.DataFrame, wp: WinProb) -> pd.Series:
    """Win probability added for the scoring team by each goal."""
    g = goals
    sign = np.where(g["is_home"] == 1, 1, -1)
    diff_before = (g["team_score_before"] - g["opp_score_before"]) * sign  # home perspective
    diff_after = diff_before + sign
    playoff = g["game_type"] == 3
    before = wp.home_wp(g["t"], diff_before, playoff)
    after = wp.home_wp(g["t"], diff_after, playoff)
    # an overtime goal ends the game
    ot = g["period"] > 3
    after = np.where(ot, np.where(sign == 1, 1.0, 0.0), after)
    team_before = np.where(sign == 1, before, 1 - before)
    team_after = np.where(sign == 1, after, 1 - after)
    return pd.Series(team_after - team_before, index=g.index)
