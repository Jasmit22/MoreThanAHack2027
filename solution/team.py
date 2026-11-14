"""Build team-possession evidence from control, tags, and team shape."""
import numpy as np

from config import C
from data import AWAY, FPS, HOME, Match
from features import Features, true_runs

ATTACKING_PHASE_CODES = {
    "ATTACKING TRANSITION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "BUILD UP",
    "CREATING CHANCES", "LONG BALL", "PLAYERS IN THE BOX", "PROGRESSION",
}
DEFENDING_PHASE_CODES = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION",
}
HOME_PSEUDO_LABEL = 1
AWAY_PSEUDO_LABEL = 0
UNKNOWN_PSEUDO_LABEL = -1


def build_phase_evidence(match: Match):
    """Return team evidence supplied by possession-oriented tactical phase tags."""
    evidence = np.zeros((match.n, 2))
    for tag in match.tags:
        if tag.team is None:
            continue
        if tag.code in ATTACKING_PHASE_CODES:
            evidence[tag.start:tag.end, tag.team] = 1.0
        elif tag.code in DEFENDING_PHASE_CODES:
            evidence[tag.start:tag.end, 1 - tag.team] = 1.0
    return evidence


def build_shape_model_features(features: Features, match: Match):
    """Combine physical shape features with signed tactical-phase evidence."""
    phase_evidence = build_phase_evidence(match)
    return np.column_stack([
        features.shape,
        phase_evidence[:, HOME] - phase_evidence[:, AWAY],
    ])


def build_control_pseudo_labels(features: Features, dead_mask):
    """Label sustained, clear control as Home/Away and leave all other frames unknown."""
    need = round(C["pseudo_min_control_s"] * FPS)
    labels = np.full(len(dead_mask), UNKNOWN_PSEUDO_LABEL, np.int8)
    for team in (HOME, AWAY):
        for start, stop in true_runs(features.control[:, team]):
            if stop - start >= need:
                labels[start:stop] = HOME_PSEUDO_LABEL if team == HOME else AWAY_PSEUDO_LABEL
    labels[dead_mask | (features.half == 0) | ~features.shape_ok] = UNKNOWN_PSEUDO_LABEL
    return labels


def build_gap_continuity_logit(features: Features, dead_mask):
    """Carry possession through a short gap when both edges agree."""
    pseudo = build_control_pseudo_labels(features, dead_mask)
    out = np.zeros(len(pseudo))
    edge = round(C["control_bridge_edge_s"] * FPS)
    maximum = round(C["control_bridge_max_s"] * FPS)
    report = dict(candidate_gaps=0, bridged_gaps=0, bridged_frames=0)
    gaps = (~features.observed) & (features.half > 0)
    for start, stop in true_runs(gaps):
        duration = stop - start
        if duration > maximum or dead_mask[start:stop].any():
            continue
        report["candidate_gaps"] += 1
        before = pseudo[max(0, start - edge):start]
        after = pseudo[stop:min(len(pseudo), stop + edge)]
        before = before[before >= 0]
        after = after[after >= 0]
        if not len(before) or not len(after) or before[-1] != after[0]:
            continue
        # Convert the pseudo-label to signed Home-minus-Away evidence.
        out[start:stop] = 1.0 if before[-1] == HOME_PSEUDO_LABEL else -1.0
        report["bridged_gaps"] += 1
        report["bridged_frames"] += int(duration)
    return out, report


def fit_logistic_model(features, labels, l2, iterations):
    """Fit L2-regularised logistic regression with Newton updates."""
    mean, std = features.mean(0), features.std(0)
    std[std == 0] = 1.0
    z = np.column_stack([np.ones(len(features)), (features - mean) / std])
    w = np.zeros(z.shape[1])
    penalty = np.eye(z.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        gradient = z.T @ (p - labels) + penalty @ w
        hessian = (z * (p * (1 - p))[:, None]).T @ z + penalty
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w, mean, std


def predict_logits(model, features):
    """Return Home-versus-Away logits from a fitted logistic model."""
    w, mean, std = model
    return np.column_stack([np.ones(len(features)), (features - mean) / std]) @ w


def fit_shape_possession_model(features: Features, match: Match, dead_mask):
    """Fit one self-supervised shape model per half and return full-match logits."""
    model_features = build_shape_model_features(features, match)
    pseudo_labels = build_control_pseudo_labels(features, dead_mask)
    train = pseudo_labels >= 0
    # Fit each half separately to allow changes in team shape after half-time.
    logit = np.zeros(match.n)
    report = {}
    for half, first, last in match.halves:
        sel = train.copy()
        sel[:first] = False
        sel[last + 1:] = False
        model = fit_logistic_model(
            model_features[sel], pseudo_labels[sel].astype(float),
            C["logistic_l2"], C["logistic_iterations"],
        )
        logit[first:last + 1] = predict_logits(model, model_features[first:last + 1])
        probability = 1 / (1 + np.exp(-predict_logits(model, model_features[sel])))
        report[half] = dict(n_pseudo=int(sel.sum()), home_share=float(pseudo_labels[sel].mean()),
                            train_agreement=float(((probability > 0.5) == (pseudo_labels[sel] == 1)).mean()),
                            weights=[round(float(v), 3) for v in model[0]])
    logit[~features.shape_ok] = 0.0
    return logit, report
