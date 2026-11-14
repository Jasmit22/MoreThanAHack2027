"""Model constants and short explanations for their values.

They use football and measurement priors, not labels or event-specific settings from this match.
"""

C = dict(
    # ---- ball kinematics -------------------------------------------------------------------
    # 5-frame (0.2 s) smoothing removes tracker jitter without blurring a kick.
    ball_smooth_frames=5,
    # Speed from a +/-2 frame central difference of the smoothed track.
    ball_speed_lag_frames=2,
    # Ball radius plus tracker noise makes this distance beyond a line out of play.
    outside_margin_m=0.5,

    # ---- control ----------------------------------------------------------------------------
    # A ball travelling faster than ~12 m/s is a pass or shot in flight, not under control.
    control_max_ball_speed=12.0,
    # A controlled ball should be within a stride of the player's feet.
    control_distance_m=1.5,
    # ...and that player is clearly closer than any opponent (otherwise it is a duel).
    control_margin_m=1.0,

    # ---- team shape -----------------------------------------------------------------------
    # Below 7 tracked outfield players a team-shape statistic is mostly noise.
    shape_min_players=7,

    # ---- dead ball: whole-gap classification ----------------------------------------------
    # Ball-track gaps in this ATD are never shorter than ~4 s (the provider fills short
    # occlusions), so each gap is classified as a unit from the evidence around it.
    gap_tag_window_s=3.0,          # a restart/goal tag this close to a gap explains it
    gap_edge_m=3.0,                # ball last seen this close to a line probably went out
    gap_calm_m_s=1.0,              # median player speed of a stoppage: walking
    gap_active_m_s=2.0,            # median player speed of live play under pressure
    gap_long_s=15.0,               # live occlusions rarely last this long
    gap_break_s=45.0,              # nothing in open play loses the ball for 45 s
    gap_placed_m_s=1.0,            # ball reappears at walking pace -> being placed for a restart
    gap_dead_threshold=2.0,        # evidence score needed to call a gap a stoppage
    # The stoppage inside a dead gap is its calm tail: the earliest point after which at least
    # 80% of frames have walking-pace players (calibrated threshold + 0.2 m/s slack for jogging).
    tail_calm_fraction=0.8,
    tail_calm_slack_m_s=0.2,
    # Optional guard for low-specificity long-gap evidence.
    gap_require_specific_cue=False,
    # Optional rule: start a gap stoppage at the gap only after a visible exit.
    gap_exit_m=1.0,
    gap_onset_needs_exit=False,
    # End a gap stoppage when the ball returns already moving.
    gap_live_return_m_s=3.0,
    gap_end_on_moving_return=False,

    # ---- dead ball: visible ball ----------------------------------------------------------
    # A visible ball that has not moved faster than 0.5 m/s for 2 s, while nobody is playing
    # it, has been placed for a restart. A keeper holding the ball is excluded: that is live.
    placed_speed_m_s=0.5,
    placed_min_s=2.0,
    placed_player_m=1.0,           # nobody within this of the ball...
    placed_calm_m_s=1.5,           # ...or the players around are walking
    # A restart is the ball leaving a stoppage at kicking/throwing speed.
    release_speed_m_s=5.0,
    release_frames=3,
    # Searches for the release after a stoppage and for its onset before a tag.
    release_search_s=40.0,
    onset_search_s=20.0,
    onset_active_ball_m_s=6.0,     # a ball moving this fast before the stop was still in play

    # ---- touches and pass chains (touches.py) ---------------------------------------------
    # A kick, reception or deflection changes the ball's velocity by several m/s within
    # 0.25 s; the player responsible is within reach (2 m) and clearly nearer than an opponent.
    # Disabled until validated on accessible labels from another match.
    use_pass_chains=False,
    touch_window_frames=3,
    touch_dv_m_s=4.0,
    touch_distance_m=2.0,
    touch_team_margin_m=0.5,
    # Two same-team touches up to 4 s apart = a pass or a carry; longer is not one action.
    chain_max_s=4.0,
    w_chain=0.6,                   # per frame between the two touches

    # ---- extra stoppage cues ----------------------------------------------------------------
    # Phase boundaries are weaker than restart tags.
    use_gap_phase_end=False,
    # Slow visible ball while players walk may indicate restart setup.
    use_visible_calm=False,
    visible_calm_min_s=8.0,
    visible_calm_ball_m_s=3.0,
    visible_calm_slack_m_s=0.2,    # below the calibrated calm threshold by this much

    # ---- emissions (log-score units per frame) --------------------------------------------
    w_control=1.0,                 # one frame of clear control
    w_shape=0.05,                  # per unit of shape-model logit (clipped to +/-3)
    shape_logit_clip=3.0,
    w_dead_strong=2.0,             # out of play, goal, placed ball, classified dead gap
    w_live_ball=1.0,               # ball visibly moving at play speed, inside the pitch
    live_ball_min_m_s=3.0,
    dead_bias=-0.3,                # prior: the ball is live more often than dead

    # ---- transitions (log-score costs) ----------------------------------------------------
    # Changing team needs ~0.5 s (12 frames) of net clear control: a touch is not possession.
    cost_switch=12.0,
    # Entering or leaving a stoppage needs ~0.5 s of strong evidence.
    cost_enter_dead=25.0,
    cost_exit_dead=25.0,

    # ---- self-trained shape model ---------------------------------------------------------
    # Pseudo-labels: frames inside a run of clear control lasting at least this long.
    pseudo_min_control_s=0.4,
    logistic_l2=1.0,
    logistic_iterations=50,

    # ---- unobserved ball continuity ---------------------------------------------------------
    # Carry control across a short gap only when the same team controls both edges.
    control_bridge_max_s=8.0,
    control_bridge_edge_s=1.0,
    # Two matching control observations make the bridge stronger than one.
    w_control_bridge=2.0,
)
