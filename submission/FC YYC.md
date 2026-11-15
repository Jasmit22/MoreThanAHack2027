# FC YYC: Possession labelling system

## Scope

The submission file is produced by `solution/pipeline.py`. It labels each video frame as Home
(`H`), Away (`A`), dead ball (`D`), or ungraded (`X`).

## Inputs

The pipeline uses the supplied ATD tracking and metadata plus automatic Smart Tagging. Tracking
provides player and ball positions, player speed, goalkeeper slots, team assignments, pitch size,
and attacking direction. Smart Tagging provides automatic tactical phase, set-piece, and goal tags.

It does not use the public ground-truth CSV, private ground truth, manual annotations, or
human-produced data for this match. No external labelled dataset is used by the submitted model.
`solution/calibrate_external.py` is an optional utility for public Metrica samples and is not run
by the submission pipeline.

## Method

The feature stage smooths ball positions within each uninterrupted ball track. It calculates ball
speed, distance from each team to the ball, whether a goalkeeper is nearest, player calmness, and
team shape. Team shape is oriented toward the opponent's goal in both halves.

Direct possession requires a team to be close to a playable ball and clearly closer than the
opponent. Runs of direct control form algorithmic pseudo-labels. A regularised logistic model,
fitted separately in each half, estimates possession from team shape and automatic tactical-phase
tags where the ball is missing. A short missing-ball interval is bridged only when the same team
has clear control on both sides.

Dead-ball evidence includes a visible ball outside the pitch, a stationary unattended ball,
automatic restart or goal tags, and scored tracking gaps. A tracking gap uses nearby restart tags,
the last ball position, player calmness, duration, and the ball's return. The calmness threshold is
estimated from visible live and restart frames.

The final frame sequence is selected with a three-state Viterbi decoder. Switching teams or
entering and leaving dead ball has a fixed cost. This prevents isolated proximity changes from
creating a separate possession.

## Possession decisions

- A nearby player does not control a fast ball in flight.
- A single touch or deflection does not automatically change possession.
- A team change needs sustained evidence.
- A visible exit or restart setup is dead ball until release.
- A missing-ball interval is labelled dead only when its evidence reaches the configured threshold.

## Limitations

The model depends on the provider's tracking and tag quality. Long ball gaps can hide a turnover or
restart. The team-shape model can inherit errors from its direct-control pseudo-labels. Fixed
physical thresholds may not transfer unchanged to a different tracking provider or frame rate.

## Reproduction

From the repository root, with the competition data available in `data/`:

```bash
python solution/pipeline.py --to-submission
python evaluation/validate_submission.py "submission/FC YYC.csv"
python -m unittest solution/test_pipeline.py
```

The upload files are `submission/FC YYC.csv` and `submission/FC YYC.md`.
