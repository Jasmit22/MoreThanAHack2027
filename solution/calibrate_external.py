"""Measure control geometry on Metrica's public sample games.

The script reads only the supplied Metrica directory, not challenge-match files.

Usage:
    python solution/calibrate_external.py /path/to/metrica-sample-data/data
"""
import argparse
import csv
from pathlib import Path

import numpy as np

PITCH = np.array([105.0, 68.0])
EVENT_FILENAME = "Sample_Game_{game}_RawEventsData.csv"
HOME_TRACKING_FILENAME = "Sample_Game_{game}_RawTrackingData_Home_Team.csv"
AWAY_TRACKING_FILENAME = "Sample_Game_{game}_RawTrackingData_Away_Team.csv"
RECOVERY_EVENT = "RECOVERY"
TEAM_FIELD = "Team"
TYPE_FIELD = "Type"
PERIOD_FIELD = "Period"
FRAME_FIELD = "Start Frame"
PERCENTILES = (50, 75, 90, 95)


def load_tracking_arrays(path):
    """Load normalized Metrica tracking rows into metre-space arrays."""
    table = np.genfromtxt(path, delimiter=",", skip_header=3, dtype=float)
    players = table[:, 3:-2].reshape(len(table), -1, 2) * PITCH
    ball = table[:, -2:] * PITCH
    key_to_row = {(int(period), int(frame)): row
                  for row, (period, frame) in enumerate(table[:, :2])}
    return players, ball, key_to_row


def iter_recovery_events(path):
    """Yield team and tracking-clock coordinates for external recovery events."""
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row[TYPE_FIELD] == RECOVERY_EVENT:
                yield row[TEAM_FIELD], int(row[PERIOD_FIELD]), int(row[FRAME_FIELD])


def measure_recovery_geometry(data_dir, game):
    """Measure team-to-ball distance and advantage at each external recovery."""
    root = Path(data_dir) / f"Sample_Game_{game}"
    home, ball, index = load_tracking_arrays(root / HOME_TRACKING_FILENAME.format(game=game))
    away, _, _ = load_tracking_arrays(root / AWAY_TRACKING_FILENAME.format(game=game))
    own, other = [], []
    for team, period, frame in iter_recovery_events(root / EVENT_FILENAME.format(game=game)):
        row = index.get((period, frame))
        if row is None or not np.isfinite(ball[row]).all():
            continue
        first, second = (home, away) if team == "Home" else (away, home)
        own.append(np.nanmin(np.linalg.norm(first[row] - ball[row], axis=1)))
        other.append(np.nanmin(np.linalg.norm(second[row] - ball[row], axis=1)))
    own, other = np.asarray(own), np.asarray(other)
    good = np.isfinite(own) & np.isfinite(other)
    return own[good], (other - own)[good]


def percentile_summary(values):
    """Return the stable percentile summary printed by the calibration CLI."""
    return {f"p{q}": round(float(np.percentile(values, q)), 3) for q in PERCENTILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    all_distance, all_margin = [], []
    for game in (1, 2):
        distance, margin = measure_recovery_geometry(args.data_dir, game)
        all_distance.append(distance)
        all_margin.append(margin)
        print(f"game {game}: {len(distance)} recoveries")
        print("  nearest recovering player (m):", percentile_summary(distance))
        print("  recovering-player advantage (m):", percentile_summary(margin))
    print("combined:")
    print("  nearest recovering player (m):", percentile_summary(np.concatenate(all_distance)))
    print("  recovering-player advantage (m):", percentile_summary(np.concatenate(all_margin)))


if __name__ == "__main__":
    main()
