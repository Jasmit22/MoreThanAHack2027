"""Label possession from tracking, tactical tags, and Viterbi decoding.

    python solution/pipeline.py
    python solution/pipeline.py --to-submission

"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decode  # noqa: E402
import data  # noqa: E402
import dead  # noqa: E402
import features  # noqa: E402
import team  # noqa: E402
import touches  # noqa: E402
from config import C  # noqa: E402

ROOT = HERE.parent
PIPELINE_NAME = "FC YYC"
HOME_LABEL = "H"
AWAY_LABEL = "A"
DEAD_LABEL = "D"
GRADED_LABELS = (HOME_LABEL, AWAY_LABEL, DEAD_LABEL)


def infer_labels(match):
    """Run feature extraction, scoring, and decoding for one match."""
    feature_set = features.build_features(match)
    dead_mask, dead_reasons, gaps, calibration = dead.build_dead_evidence(feature_set, match)
    shape_logit, shape_report = team.fit_shape_possession_model(feature_set, match, dead_mask)
    continuity_logit, bridge_report = team.build_gap_continuity_logit(feature_set, dead_mask)
    pass_chains = None
    if C["use_pass_chains"]:
        touches_found = touches.detect_attributed_touches(feature_set)
        pass_chains = touches.build_pass_chain_evidence(touches_found, match.n)
    score = decode.build_emission_scores(
        feature_set, dead_mask, shape_logit, pass_chains, continuity_logit
    )
    labels = decode.decode_match_halves(score, match.halves, match.n)
    diagnostics = dict(calibration=calibration, shape_model=shape_report,
                       control_bridge=bridge_report,
                       dead_sources=summarize_dead_sources(dead_reasons),
                       gaps=dict(total=len(gaps), dead=sum(g["dead"] for g in gaps)),
                       ball_observed_graded=float(feature_set.observed[feature_set.half > 0].mean()))
    return labels, diagnostics, gaps


def summarize_dead_sources(reasons):
    """Aggregate dead-ball intervals by their evidence source."""
    out = {}
    for start, stop, kind in reasons:
        entry = out.setdefault(kind, {"count": 0, "frames": 0})
        entry["count"] += 1
        entry["frames"] += int(stop - start)
    return out


def summarize_labels(labels, halves):
    """Calculate counts and durations from decoded labels."""
    graded = np.zeros(len(labels), bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    spells = []
    for _, first, last in halves:
        seg = labels[first:last + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.r_[0, change, len(seg)]
        spells += [(first + a, first + b, seg[a]) for a, b in zip(bounds[:-1], bounds[1:])]
    possessions = [(b - a) / data.FPS for a, b, label in spells if label in (HOME_LABEL, AWAY_LABEL)]
    stoppages = [(b - a) / data.FPS for a, b, label in spells if label == DEAD_LABEL]
    g = labels[graded]
    return dict(
        graded_frames=int(graded.sum()),
        share={label: round(float((g == label).mean()), 4) for label in GRADED_LABELS},
        possessions=len(possessions),
        possession_seconds_mean=round(float(np.mean(possessions)), 2),
        possession_seconds_median=round(float(np.median(possessions)), 2),
        possessions_under_2s=int(np.sum(np.array(possessions) < 2)),
        stoppages=len(stoppages),
        stoppage_seconds_median=round(float(np.median(stoppages)), 2),
        home_share_of_live=round(float((g == HOME_LABEL).sum() / max(1, np.isin(g, [HOME_LABEL, AWAY_LABEL]).sum())), 4),
    ), spells


def sha256_file(path):
    """Return the SHA-256 digest for a local output or source file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_outputs(output_dir, labels, spells, gaps, summary, diagnostics):
    """Write submission files and return the submission path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = output_dir / f"{PIPELINE_NAME}.csv"
    with submission.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))
    with (output_dir / "segments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["start_frame", "end_frame_exclusive", "label", "duration_seconds"])
        writer.writerows((a, b, label, round((b - a) / data.FPS, 2)) for a, b, label in spells)
    with (output_dir / "gaps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gaps[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(gaps)
    report = dict(version=PIPELINE_NAME, config=C, summary=summary, diagnostics=diagnostics,
                  python=sys.version.split()[0], numpy=np.__version__,
                  code_sha256={path.name: sha256_file(path) for path in sorted(HERE.glob("*.py"))},
                  submission_sha256=sha256_file(submission),
                  note="No ground truth or hand label is read by inference.")
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--to-submission", action="store_true",
                        help="copy the CSV to submission/FC YYC.csv")
    args = parser.parse_args()

    match = data.load_match(args.data_dir)
    labels, diagnostics, gaps = infer_labels(match)
    summary, spells = summarize_labels(labels, match.halves)
    submission = write_outputs(args.output_dir, labels, spells, gaps, summary, diagnostics)
    if args.to_submission:
        final_dir = ROOT / "submission"
        final_dir.mkdir(exist_ok=True)
        shutil.copyfile(submission, final_dir / f"{PIPELINE_NAME}.csv")
    print(json.dumps(dict(summary=summary, diagnostics=diagnostics), indent=2))
    print(submission)


if __name__ == "__main__":
    main()
