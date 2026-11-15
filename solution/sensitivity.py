"""Report label changes after scaling one prior at a time.

Each listed constant is scaled by 0.5 and 1.5. The output shows the fraction of graded frames
that change plus possession and stoppage counts. The script reads no labels.

    python solution/sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data  # noqa: E402
from config import C  # noqa: E402
from pipeline import infer_labels, summarize_labels  # noqa: E402

KEYS = ["control_distance_m", "control_margin_m", "control_max_ball_speed", "w_shape",
        "cost_switch", "cost_enter_dead", "cost_exit_dead", "w_dead_strong", "dead_bias",
        "placed_min_s", "release_speed_m_s", "gap_long_s", "gap_edge_m", "tail_calm_fraction"]


def main():
    """Run and print the configured label-free one-prior-at-a-time sensitivity sweep."""
    match = data.load_match()
    base, _, _ = infer_labels(match)
    graded = base != "X"
    ref, _ = summarize_labels(base, match.halves)
    print(f"reference: possessions {ref['possessions']}  stoppages {ref['stoppages']}  "
          f"D share {ref['share']['D']:.3f}")
    print(f"{'constant':24s} {'scale':>5s} {'changed':>8s} {'poss':>5s} {'stops':>6s} {'D share':>8s}")
    rows = []
    for key in KEYS:
        original = C[key]
        for scale in (0.5, 1.5):
            if key.endswith("_fraction"):
                # Scale the complement to keep a fraction in range.
                C[key] = 1 - (1 - original) * scale
            elif isinstance(original, int):
                C[key] = type(original)(original * scale)
            else:
                C[key] = original * scale
            try:
                labels, _, _ = infer_labels(match)
            finally:
                C[key] = original
            s, _ = summarize_labels(labels, match.halves)
            changed = float((labels[graded] != base[graded]).mean())
            rows.append((key, scale, changed))
            print(f"{key:24s} {scale:5.1f} {changed:8.2%} {s['possessions']:5d} {s['stoppages']:6d} "
                  f"{s['share']['D']:8.3f}")
    worst = sorted(rows, key=lambda r: -r[2])[:5]
    print("\nmost sensitive:", ", ".join(f"{k} x{s} ({c:.1%})" for k, s, c in worst))


if __name__ == "__main__":
    main()
