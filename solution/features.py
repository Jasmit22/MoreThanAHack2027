"""Build per-frame physical features from tracking data."""
from dataclasses import dataclass
import warnings

import numpy as np

from config import C
from data import AWAY, FPS, HOME, PITCH, Match

TEAM_COUNT = 2


def true_runs(mask):
    """Return half-open intervals for contiguous true values in ``mask``."""
    mask = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def centered_nan_mean(values, width):
    """Return a centred moving mean that excludes missing values."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    kernel = np.ones(int(width))
    total = np.convolve(np.where(finite, values, 0.0), kernel, "same")
    count = np.convolve(finite.astype(float), kernel, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


@dataclass
class Features:
    observed: np.ndarray        # ball tracked this frame
    ball: np.ndarray            # (n, 2) metres, lightly smoothed, NaN when unobserved
    ball_speed: np.ndarray      # m/s, NaN when unobserved
    outside: np.ndarray         # ball observed clearly beyond a touch- or goal-line
    edge_distance: np.ndarray   # metres from the nearest line (negative outside), NaN unobserved
    nearest: np.ndarray         # (n, 2) nearest outfield-or-keeper distance per team, inf if none
    nearest_is_keeper: np.ndarray  # (n, 2) bool
    player_calm: np.ndarray     # 1-s mean of the median tracked-player speed
    control: np.ndarray         # (n, 2) bool, one team clearly controls a playable ball
    shape: np.ndarray           # (n, k) team-shape features, oriented by attacking direction
    shape_ok: np.ndarray        # enough players tracked for the shape features to mean anything
    half: np.ndarray            # 1 / 2 on graded frames, 0 elsewhere


def _build_ball_kinematics(ball):
    """Smooth each observed ball run and derive speed without crossing a gap."""
    observed = np.isfinite(ball).all(1)
    smooth = np.full_like(ball, np.nan)
    width = C["ball_smooth_frames"]
    for start, stop in true_runs(observed):
        segment = ball[start:stop]
        if len(segment) >= width:
            kernel = np.ones(width) / width
            pad = width // 2
            padded = np.pad(segment, ((pad, pad), (0, 0)), mode="edge")
            smooth[start:stop] = np.column_stack(
                [np.convolve(padded[:, k], kernel, "valid") for k in range(2)])
        else:
            smooth[start:stop] = segment
    lag = C["ball_speed_lag_frames"]
    speed = np.full(len(ball), np.nan)
    displacement = np.linalg.norm(smooth[2 * lag:] - smooth[:-2 * lag], axis=1)
    speed[lag:-lag] = displacement * FPS / (2 * lag)
    # A speed straddling a missing stretch is meaningless.
    contiguous = np.zeros(len(ball), bool)
    contiguous[lag:-lag] = observed[2 * lag:] & observed[:-2 * lag]
    speed[~contiguous] = np.nan
    return observed, smooth, speed


def find_nearest_team_players(players, ball, slot_team, goalkeeper):
    """Return each team's nearest tracked player and goalkeeper flag per frame."""
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.full((len(ball), TEAM_COUNT), np.inf)
    keeper = np.zeros((len(ball), TEAM_COUNT), bool)
    for team in (HOME, AWAY):
        cols = np.flatnonzero(slot_team == team)
        sub = distance[:, cols]
        idx = sub.argmin(1)
        nearest[:, team] = sub[np.arange(len(ball)), idx]
        keeper[:, team] = goalkeeper[cols][idx] & np.isfinite(nearest[:, team])
    return nearest, keeper


def _build_team_shape_features(match, half):
    """Team-shape descriptors, oriented so that +x is always towards the opponent's goal.

    A team in possession pushes up and spreads to make the pitch big; the defending side drops
    and narrows. These are generic properties of football, so they carry possession information
    on the 41% of frames where the ball itself is not tracked.
    """
    n = match.n
    x = match.players[:, :, 0] - PITCH[0] / 2
    y = match.players[:, :, 1]
    out = []
    ok = np.ones(n, bool)
    sign = np.zeros((n, 2))
    for h, (home_sign, away_sign) in match.attack_sign.items():
        sign[half == h] = (home_sign, away_sign)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team in (HOME, AWAY):
            cols = (match.slot_team == team) & ~match.goalkeeper
            tx = x[:, cols] * sign[:, team:team + 1]
            ty = y[:, cols]
            count = np.isfinite(tx).sum(1)
            ok &= count >= C["shape_min_players"]
            out.append(np.column_stack([
                np.nanmean(tx, 1),                       # how far up the pitch
                np.nanstd(tx, 1),                        # length of the block
                np.nanstd(ty, 1),                        # width of the block
                np.nanmax(tx, 1),                        # highest player
            ]))
    home, away = out
    shape = np.column_stack([home - away, home + away])
    # Velocities of the blocks (1-s difference) say who is going forward.
    velocity = np.zeros((n, 2))
    lag = FPS
    velocity[lag:] = (np.column_stack([home[:, 0], away[:, 0]])[lag:]
                      - np.column_stack([home[:, 0], away[:, 0]])[:-lag])
    shape = np.column_stack([shape, velocity[:, 0] - velocity[:, 1]])
    shape[~np.isfinite(shape)] = 0.0
    return shape, ok


def build_features(match: Match) -> Features:
    """Build every immutable feature array consumed by inference."""
    half = match.half_of()
    observed, ball, speed = _build_ball_kinematics(match.ball)
    margin = C["outside_margin_m"]
    with np.errstate(invalid="ignore"):
        edge = np.minimum.reduce([ball[:, 0], PITCH[0] - ball[:, 0], ball[:, 1], PITCH[1] - ball[:, 1]])
        outside = observed & (edge < -margin)
    nearest, keeper = find_nearest_team_players(
        match.players, ball, match.slot_team, match.goalkeeper
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_speed = np.nanmedian(match.player_speed, axis=1)
    calm = centered_nan_mean(median_speed, FPS)

    playable = observed & (np.nan_to_num(speed, nan=99.0) <= C["control_max_ball_speed"])
    control = np.zeros((match.n, 2), bool)
    for team in (HOME, AWAY):
        other = 1 - team
        with np.errstate(invalid="ignore"):  # inf - inf where neither team is tracked
            margin = nearest[:, other] - nearest[:, team]
        control[:, team] = (playable & (nearest[:, team] <= C["control_distance_m"])
                            & (margin >= C["control_margin_m"]))

    shape, shape_ok = _build_team_shape_features(match, half)
    return Features(observed, ball, speed, outside, edge, nearest, keeper, calm, control,
                    shape, shape_ok, half)
