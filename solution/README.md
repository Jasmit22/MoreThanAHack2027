# Solution

This program labels each graded tracking frame as Home (`H`), Away (`A`), or dead ball (`D`).

## Run

```bash
venv/bin/python solution/pipeline.py --to-submission
venv/bin/python -m unittest solution/test_pipeline.py
venv/bin/python evaluation/validate_submission.py "submission/FC YYC.csv"
venv/bin/python solution/sensitivity.py
```

The pipeline writes `submission/FC YYC.csv`. It also writes possession segments, tracking-gap
diagnostics, and a JSON report to `solution/artifacts/`; that directory is ignored by Git.

## Method

`features.py` converts the tracking feed into ball speed, player-to-ball distance, player calmness,
and team-shape measurements. It smooths only within uninterrupted ball tracks, so a speed estimate
never spans a missing-ball interval. Team shape is flipped at half-time so positive field direction
always points toward the opponent's goal.

The model calls clear direct control only when one team is close to the ball, clearly closer than
the other, and the ball is slow enough to be playable. Sustained control supplies pseudo-labels
for a regularised Home-versus-Away shape model, fitted separately for each half. Possession is
carried through a short missing-ball gap only when the same team has sustained clear control on
both sides of the gap.

Dead-ball evidence comes from visible exits, stationary unattended balls, restart and goal tags,
and missing-ball gaps. A gap is scored from nearby restart tags, the last visible ball position,
player calmness, its duration, and the ball's return. The calm threshold is derived from visible
live and dead anchors in the tracking data. The final H/A/D path is selected with Viterbi decoding;
switching state has a cost, so isolated touches and tracking noise do not become separate spells.

The values in `config.py` are football or measurement priors. They do not refer to a frame, minute,
or event in the challenge match. The public five-minute truth is not read by this program and is not
used to select settings. This matches the restrictions in `RULES.md`.

## Files

| File | Purpose |
|---|---|
| `data.py` | Load tracking and tactical tags |
| `features.py` | Derive physical measurements |
| `dead.py` | Build dead-ball evidence |
| `team.py` | Build control, shape, and gap evidence |
| `touches.py` | Optional touch and pass-chain evidence |
| `decode.py` | Score states and decode the frame sequence |
| `pipeline.py` | Run inference and write outputs |
| `sensitivity.py` | Check how selected priors affect the output |
| `calibrate_external.py` | Measure recovery geometry on public Metrica sample games |

`calibrate_external.py` uses the public Metrica sample data at
https://github.com/metrica-sports/sample-data. It is separate from challenge-match inference.
