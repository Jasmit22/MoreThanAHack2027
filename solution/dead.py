"""Find frames where the ball is out of play."""
import numpy as np

from config import C
from data import FPS, Match
from features import Features, centered_nan_mean, true_runs

RESTART_EVENT_CODES = {"SET PIECES", "GOALS"}


def calibrate_calm_speed(features: Features, match: Match):
    """Set a walking-speed threshold from visible live and restart frames."""
    graded = features.half > 0
    near_tag = np.zeros(match.n, bool)
    window = round(C["gap_tag_window_s"] * FPS)
    for tag in match.tags:
        if tag.code in RESTART_EVENT_CODES:
            near_tag[max(0, tag.start - window):tag.start + window] = True
    speed = np.nan_to_num(features.ball_speed, nan=0.0)
    live = graded & features.observed & ~features.outside & (speed > C["live_ball_min_m_s"]) & ~near_tag
    dead = graded & (features.outside | near_tag)
    live_calm = features.player_calm[live & np.isfinite(features.player_calm)]
    dead_calm = features.player_calm[dead & np.isfinite(features.player_calm)]
    grid = np.linspace(0.3, 3.0, 271)
    miss_live = np.array([(live_calm < t).mean() for t in grid])
    miss_dead = np.array([(dead_calm >= t).mean() for t in grid])
    threshold = float(grid[np.argmin(np.abs(miss_live - miss_dead))])
    return threshold, dict(n_pseudo_live=int(live.sum()), n_pseudo_dead=int(dead.sum()),
                           calm_threshold_m_s=threshold,
                           equal_error=float(miss_live[np.argmin(np.abs(miss_live - miss_dead))]))


def find_restart_release(features: Features, start, limit):
    """First frame at/after ``start`` where the ball leaves at restart speed from inside the pitch."""
    speed = np.nan_to_num(features.ball_speed, nan=0.0)
    fast = features.observed & ~features.outside & (speed >= C["release_speed_m_s"])
    need = C["release_frames"]
    stop = min(len(speed), start + limit)
    count = 0
    for frame in range(start, stop):
        count = count + 1 if fast[frame] else 0
        if count >= need:
            return frame - need + 1
    return None


def find_stoppage_start(features: Features, start, limit):
    """Walk backward over observed frames where the ball is not in play."""
    speed = np.nan_to_num(features.ball_speed, nan=0.0)
    in_play = features.observed & ~features.outside & (speed >= C["onset_active_ball_m_s"])
    frame = start
    lo = max(0, start - limit)
    while frame > lo and features.observed[frame - 1] and not in_play[frame - 1]:
        frame -= 1
    return frame


def add_dead_interval(mask, reasons, start, stop, reason):
    """Clip and record one dead-ball interval in the local evidence accumulator."""
    start, stop = int(max(0, start)), int(min(len(mask), stop))
    if stop > start:
        mask[start:stop] = True
        reasons.append((start, stop, reason))


def mark_out_of_play(features: Features, mask, reasons):
    """Mark visible out-of-play runs through their observed restart release."""
    search = round(C["release_search_s"] * FPS)
    for start, stop in true_runs(features.outside):
        if stop - start < C["release_frames"]:
            continue
        release = find_restart_release(features, stop, search)
        # Leave an unobserved release to the gap classifier.
        later_gap = np.flatnonzero(~features.observed[stop:stop + search])
        end = release if release is not None else stop + search
        if len(later_gap) and (release is None or stop + later_gap[0] < release):
            end = stop + later_gap[0]
        add_dead_interval(mask, reasons, start, end, "out_of_play")


def mark_placed_balls(features: Features, mask, reasons):
    """Mark unattended, stationary visible balls as restart setup intervals."""
    speed = features.ball_speed
    still = features.observed & np.isfinite(speed) & (speed <= C["placed_speed_m_s"])
    nearest = features.nearest.min(1)
    keeper_holds = (features.nearest_is_keeper & (features.nearest <= C["control_distance_m"])).any(1)
    unattended = (nearest >= C["placed_player_m"]) | (features.player_calm <= C["placed_calm_m_s"])
    candidate = still & unattended & ~keeper_holds
    need = round(C["placed_min_s"] * FPS)
    search = round(C["release_search_s"] * FPS)
    for start, stop in true_runs(candidate):
        if stop - start < need:
            continue
        release = find_restart_release(features, stop, search)
        add_dead_interval(mask, reasons, start, release if release is not None else stop, "placed_ball")


def mark_tagged_restarts(features: Features, match: Match, mask, reasons):
    """Mark goal and set-piece intervals using tags plus physical restart release."""
    search = round(C["release_search_s"] * FPS)
    onset_limit = round(C["onset_search_s"] * FPS)
    kickoffs = [t for t in match.tags if t.labels.get("Type", "").casefold().startswith("kick off")]
    for tag in match.tags:
        kind = tag.labels.get("Type", "").casefold()
        if tag.code == "GOALS":
            # A goal remains dead until the next kick-off release.
            after = [k for k in kickoffs if k.start >= tag.start]
            anchor = after[0].start if after else tag.end
            release = find_restart_release(features, anchor, search)
            add_dead_interval(mask, reasons, tag.start, release if release is not None else tag.end, "goal")
        elif tag.code == "SET PIECES" and not kind.startswith("kick off (start)"):
            onset = find_stoppage_start(features, tag.start, onset_limit)
            release = find_restart_release(features, tag.start, search)
            if release is None:
                continue  # the gap classifier decides when the ball never reappears
            add_dead_interval(mask, reasons, onset, release, "restart_tag")


def classify_tracking_gaps(features: Features, match: Match, calm_threshold, mask, reasons):
    """Score ball-track gaps and mark the dead portion of qualifying gaps."""
    tag_starts = np.array([t.start for t in match.tags if t.code in RESTART_EVENT_CODES])
    phase_ends = np.array([t.end for t in match.tags if t.code not in RESTART_EVENT_CODES])
    window = round(C["gap_tag_window_s"] * FPS)
    search = round(C["release_search_s"] * FPS)
    calm3 = centered_nan_mean(features.player_calm, 3 * FPS)
    speed = np.nan_to_num(features.ball_speed, nan=np.nan)
    table = []
    for half, first, last in match.halves:
        missing = ~features.observed.copy()
        missing[:first] = False
        missing[last + 1:] = False
        for start, stop in true_runs(missing):
            duration = (stop - start) / FPS
            score, cues = 0.0, []
            if len(tag_starts) and np.any((tag_starts >= start - window) & (tag_starts <= stop + window)):
                score += 2.0; cues.append("tag")
            elif (C["use_gap_phase_end"] and len(phase_ends)
                  and np.any((phase_ends >= start) & (phase_ends < stop))):
                score += 1.0; cues.append("phase_end")
            before = np.flatnonzero(features.observed[max(first, start - 2 * FPS):start])
            exited = False
            if len(before):
                edge = features.edge_distance[max(first, start - 2 * FPS) + before[-1]]
                if edge <= C["gap_edge_m"]:
                    score += 1.5; cues.append("edge")
                # A ball last seen near a line may have gone out before the gap.
                exited = edge <= C["gap_exit_m"]
            after = speed[stop:min(stop + FPS, last + 1)]
            returned_moving = False
            if np.isfinite(after).sum() >= FPS // 2:
                if np.nanmedian(after) <= C["gap_placed_m_s"]:
                    score += 1.0; cues.append("placed_return")
                returned_moving = np.nanmedian(after) >= C["gap_live_return_m_s"]
            calm = np.nanmedian(features.player_calm[start:stop])
            if calm <= calm_threshold:
                score += 1.0; cues.append("calm")
            elif calm >= C["gap_active_m_s"]:
                score -= 1.5; cues.append("active")
            if duration >= C["gap_break_s"]:
                score += 3.0; cues.append("break")
            elif duration >= C["gap_long_s"]:
                score += 1.0; cues.append("long")
            dead = score >= C["gap_dead_threshold"]
            # Require a restart-specific cue when this optional guard is enabled.
            if C["gap_require_specific_cue"] and not set(cues) & {"tag", "edge", "placed_return", "break"}:
                dead = False
            table.append(dict(half=half, start=int(start), stop=int(stop), seconds=round(duration, 2),
                              score=score, cues="+".join(cues), dead=bool(dead), dead_from=None))
            if not dead:
                continue
            # Without a visible exit, start at the calm tail of the gap.
            if C["gap_onset_needs_exit"]:
                at_line = exited
            else:
                at_line = "edge" in cues
            lo = start if at_line else find_calm_tail(calm3[start:stop], calm_threshold, start)
            # A moving return can show that the restart happened inside the gap.
            if C["gap_end_on_moving_return"] and returned_moving:
                hi = min(stop, last + 1)
            else:
                release = find_restart_release(features, stop, search)
                hi = min(release if release is not None else stop, last + 1)
            add_dead_interval(mask, reasons, lo, hi, "gap")
            table[-1]["dead_from"] = int(lo)
    return table


def find_calm_tail(calm, threshold, offset):
    """Return the first point whose remaining gap is mostly calm."""
    is_calm = np.nan_to_num(calm, nan=threshold) <= threshold + C["tail_calm_slack_m_s"]
    # Calm-frame fraction from each frame to the end of the gap.
    remaining = np.cumsum(is_calm[::-1])[::-1] / np.arange(len(is_calm), 0, -1)
    ok = np.flatnonzero(remaining >= C["tail_calm_fraction"])
    return offset + (int(ok[0]) if len(ok) else len(calm))


def mark_visible_calm(features: Features, calm_threshold, mask, reasons):
    """Mark long visible slow-ball and walking intervals."""
    calm3 = centered_nan_mean(features.player_calm, 3 * FPS)
    ball = centered_nan_mean(np.where(features.observed, features.ball_speed, np.nan), FPS)
    quiet = (features.observed & (np.nan_to_num(calm3, nan=9) <= calm_threshold - C["visible_calm_slack_m_s"])
             & (np.nan_to_num(ball, nan=9) <= C["visible_calm_ball_m_s"]) & (features.half > 0))
    need = round(C["visible_calm_min_s"] * FPS)
    search = round(C["release_search_s"] * FPS)
    for start, stop in true_runs(quiet):
        if stop - start < need:
            continue
        release = find_restart_release(features, stop, search)
        add_dead_interval(mask, reasons, start, release if release is not None else stop, "visible_calm")


def build_dead_evidence(features: Features, match: Match):
    """Return dead-ball mask, provenance, gap diagnostics, and calibration metadata."""
    calm_threshold, calibration = calibrate_calm_speed(features, match)
    mask = np.zeros(match.n, bool)
    reasons = []
    mark_out_of_play(features, mask, reasons)
    mark_placed_balls(features, mask, reasons)
    mark_tagged_restarts(features, match, mask, reasons)
    if C["use_visible_calm"]:
        mark_visible_calm(features, calm_threshold, mask, reasons)
    gaps = classify_tracking_gaps(features, match, calm_threshold, mask, reasons)
    return mask, reasons, gaps, calibration
