"""Synthetic unit tests: no match data and no labels needed.

    python -m unittest solution/test_pipeline.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode  # noqa: E402
import dead  # noqa: E402
import team  # noqa: E402
from config import C  # noqa: E402
from features import Features, centered_nan_mean, find_nearest_team_players, true_runs  # noqa: E402


def fake_features(n, observed=None, speed=None, outside=None):
    observed = np.ones(n, bool) if observed is None else observed
    speed = np.full(n, 1.0) if speed is None else speed
    outside = np.zeros(n, bool) if outside is None else outside
    return Features(
        observed=observed, ball=np.zeros((n, 2)), ball_speed=np.where(observed, speed, np.nan),
        outside=outside, edge_distance=np.full(n, 30.0), nearest=np.full((n, 2), 5.0),
        nearest_is_keeper=np.zeros((n, 2), bool), player_calm=np.full(n, 1.5),
        control=np.zeros((n, 2), bool), shape=np.zeros((n, 3)), shape_ok=np.ones(n, bool),
        half=np.ones(n, np.int8))


class Helpers(unittest.TestCase):
    def test_true_runs(self):
        self.assertEqual(true_runs([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        self.assertEqual(true_runs([]), [])

    def test_centered_nan_mean_ignores_nan(self):
        out = centered_nan_mean([1.0, np.nan, 3.0], 3)
        self.assertAlmostEqual(out[1], 2.0)

    def test_find_nearest_team_players(self):
        players = np.array([[[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]]])
        ball = np.array([[0.5, 0.0]])
        nearest, keeper = find_nearest_team_players(
            players, ball, np.array([0, 0, 1]), np.array([False, True, False])
        )
        np.testing.assert_allclose(nearest[0], [0.5, 0.5])
        self.assertFalse(keeper[0].any())


class Decoder(unittest.TestCase):
    def test_brief_contrary_evidence_does_not_switch(self):
        score = np.zeros((100, 3))
        score[:, decode.HOME_STATE] = 1.0
        score[50:55, decode.AWAY_STATE] = 2.0  # A touch is not enough to change possession.
        path = decode.decode_viterbi(score, decode.build_transition_scores())
        self.assertTrue((path == decode.HOME_STATE).all())

    def test_sustained_evidence_switches_at_its_onset(self):
        score = np.zeros((200, 3))
        score[:100, decode.HOME_STATE] = 1.0
        score[100:, decode.AWAY_STATE] = 1.0
        path = decode.decode_viterbi(score, decode.build_transition_scores())
        self.assertEqual(int(np.flatnonzero(path == decode.AWAY_STATE)[0]), 100)

    def test_outside_halves_is_x(self):
        score = np.zeros((10, 3))
        labels = decode.decode_match_halves(score, [(1, 2, 5)], 10)
        self.assertEqual("".join(labels[:2]), "XX")
        self.assertEqual("".join(labels[6:]), "XXXX")


class DeadBall(unittest.TestCase):
    def test_release_needs_consecutive_fast_frames(self):
        speed = np.full(50, 1.0)
        speed[20] = 9.0                    # single-frame spike is jitter
        speed[30:35] = 9.0
        f = fake_features(50, speed=speed)
        self.assertEqual(dead.find_restart_release(f, 0, 50), 30)

    def test_onset_does_not_cross_untracked_frames(self):
        observed = np.ones(100, bool)
        observed[40:60] = False
        f = fake_features(100, observed=observed, speed=np.full(100, 0.2))
        self.assertEqual(dead.find_stoppage_start(f, 80, 100), 60)

    def test_onset_stops_at_ball_in_play(self):
        speed = np.full(100, 0.2)
        speed[30] = 10.0
        f = fake_features(100, speed=speed)
        self.assertEqual(dead.find_stoppage_start(f, 80, 100), 31)

    def test_stoppage_tail_is_calm_end_of_gap(self):
        calm = np.r_[np.full(50, 3.0), np.full(50, 0.5)]
        self.assertEqual(dead.find_calm_tail(calm, 1.0, 1000), 1050 - int(50 * (1 - C["tail_calm_fraction"]) / C["tail_calm_fraction"]))

    def test_placed_ball_excludes_keeper_holding(self):
        n = 200
        f = fake_features(n, speed=np.full(n, 0.1))
        f.nearest[:] = 0.5
        f.nearest_is_keeper[:, 0] = True
        mask = np.zeros(n, bool)
        dead.mark_placed_balls(f, mask, [])
        self.assertFalse(mask.any())


class Touches(unittest.TestCase):
    def test_same_team_touches_make_a_chain(self):
        import touches
        chains = touches.build_pass_chain_evidence([(10, 0), (30, 0), (50, 1), (60, 0)], 100)
        self.assertTrue((chains[10:31, 0] == 1).all())
        self.assertFalse(chains[31:, :].any())      # H->A and A->H pairs give nobody evidence

    def test_chain_needs_touches_close_in_time(self):
        import touches
        chains = touches.build_pass_chain_evidence([(0, 1), (500, 1)], 600)
        self.assertFalse(chains.any())


class ShapeModel(unittest.TestCase):
    def test_logistic_separates(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 2))
        y = (x[:, 0] > 0).astype(float)
        model = team.fit_logistic_model(x, y, 1.0, 50)
        accuracy = ((team.predict_logits(model, x) > 0) == (y == 1)).mean()
        self.assertGreater(accuracy, 0.95)

    def test_pseudo_labels_need_sustained_control(self):
        f = fake_features(100)
        f.control[10:12, 0] = True         # 2 frames: too short
        f.control[40:60, 1] = True         # 0.8 s: long enough
        labels = team.build_control_pseudo_labels(f, np.zeros(100, bool))
        self.assertTrue((labels[10:12] == -1).all())
        self.assertTrue((labels[40:60] == 0).all())

    def test_control_bridge_needs_same_team_on_both_gap_edges(self):
        f = fake_features(300)
        f.observed[100:200] = False
        f.control[40:100, 0] = True
        f.control[200:260, 0] = True
        bridge, report = team.build_gap_continuity_logit(f, np.zeros(300, bool))
        self.assertTrue((bridge[100:200] == 1).all())
        self.assertEqual(report["bridged_gaps"], 1)

        f.control[200:260] = False
        f.control[200:260, 1] = True
        bridge, _ = team.build_gap_continuity_logit(f, np.zeros(300, bool))
        self.assertFalse(bridge.any())



if __name__ == "__main__":
    unittest.main()
