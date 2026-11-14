"""Derive optional touch and pass-chain evidence from ball motion."""
import numpy as np

from config import C
from data import FPS
from features import Features, true_runs


def compute_ball_velocity(features: Features):
    """Estimate central-difference ball velocity inside observed tracking runs."""
    lag = C["ball_speed_lag_frames"]
    velocity = np.full((len(features.ball), 2), np.nan)
    velocity[lag:-lag] = (
        (features.ball[2 * lag:] - features.ball[:-2 * lag]) * FPS / (2 * lag)
    )
    return velocity


def detect_attributed_touches(features: Features):
    """Return ``(frame, team)`` pairs for unambiguous velocity-change touches."""
    velocity = compute_ball_velocity(features)
    k = C["touch_window_frames"]
    speed_change = np.full(len(velocity), np.nan)
    speed_change[k:-k] = np.linalg.norm(velocity[2 * k:] - velocity[:-2 * k], axis=1)
    candidate = np.nan_to_num(speed_change) >= C["touch_dv_m_s"]
    nearest = features.nearest.min(1)
    candidate &= nearest <= C["touch_distance_m"]
    with np.errstate(invalid="ignore"):
        margin = np.abs(features.nearest[:, 0] - features.nearest[:, 1])
    touches = []
    for start, stop in true_runs(candidate):
        # Use the largest velocity change in each candidate run.
        frame = start + int(np.nanargmax(speed_change[start:stop]))
        if margin[frame] < C["touch_team_margin_m"]:
            continue
        touches.append((frame, int(np.argmin(features.nearest[frame]))))
    return touches


def build_pass_chain_evidence(touches, frame_count):
    """Return per-frame team evidence between consecutive same-team touches."""
    evidence = np.zeros((frame_count, 2))
    limit = round(C["chain_max_s"] * FPS)
    for (a, team_a), (b, team_b) in zip(touches[:-1], touches[1:]):
        if team_a == team_b and 0 < b - a <= limit:
            evidence[a:b + 1, team_a] = 1.0
    return evidence
