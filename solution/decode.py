"""Decode Home, Away, and dead-ball states with Viterbi."""
import numpy as np

from config import C

HOME_STATE = 0
AWAY_STATE = 1
DEAD_STATE = 2
STATE_LABELS = np.array(["H", "A", "D"])


def build_emission_scores(features, dead_mask, shape_logit, pass_chains=None, continuity_logit=None):
    """Build H/A/D log-scores from independent physical and tactical evidence."""
    n = len(dead_mask)
    team = C["w_control"] * (features.control[:, 0].astype(float) - features.control[:, 1].astype(float))
    if pass_chains is not None:
        team += C["w_chain"] * (pass_chains[:, 0] - pass_chains[:, 1])
    if continuity_logit is not None:
        team += C["w_control_bridge"] * continuity_logit
    team += C["w_shape"] * np.clip(shape_logit, -C["shape_logit_clip"], C["shape_logit_clip"])
    speed = np.nan_to_num(features.ball_speed, nan=0.0)
    live_ball = features.observed & ~features.outside & (speed >= C["live_ball_min_m_s"]) & ~dead_mask
    live = C["w_live_ball"] * live_ball + C["w_control"] * features.control.any(1)
    score = np.zeros((n, 3))
    score[:, HOME_STATE] = team / 2 + live
    score[:, AWAY_STATE] = -team / 2 + live
    score[:, DEAD_STATE] = C["dead_bias"] + C["w_dead_strong"] * dead_mask
    return score


def build_transition_scores():
    """Return the fixed three-state transition log-score matrix."""
    cost = np.zeros((3, 3))
    cost[HOME_STATE, AWAY_STATE] = cost[AWAY_STATE, HOME_STATE] = C["cost_switch"]
    cost[HOME_STATE, DEAD_STATE] = cost[AWAY_STATE, DEAD_STATE] = C["cost_enter_dead"]
    cost[DEAD_STATE, HOME_STATE] = cost[DEAD_STATE, AWAY_STATE] = C["cost_exit_dead"]
    return -cost


def decode_viterbi(score, transition_scores):
    """Return the highest-scoring state path for one contiguous video interval."""
    n = len(score)
    back = np.zeros((n, 3), np.int8)
    best = score[0].copy()
    trans = transition_scores
    for t in range(1, n):
        candidates = best[:, None] + trans          # from x to
        back[t] = candidates.argmax(0)
        best = candidates[back[t], np.arange(3)] + score[t]
        best -= best.max()
    path = np.empty(n, np.int8)
    path[-1] = best.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match_halves(score, halves, frame_count):
    """Decode each graded half independently and mark ungraded frames as ``X``."""
    labels = np.full(frame_count, "X", dtype="<U1")
    trans = build_transition_scores()
    for _, first, last in halves:
        labels[first:last + 1] = STATE_LABELS[decode_viterbi(score[first:last + 1], trans)]
    return labels
