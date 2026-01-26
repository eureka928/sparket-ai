"""E2E test assertions for scoring and weights."""

from .scoring import (
    ScoringAssertionResult,
    assert_clv_scores_valid,
    assert_brier_scores_valid,
    assert_pss_scores_valid,
    assert_rolling_scores_valid,
    assert_no_infinite_values,
    run_all_scoring_assertions,
)

from .weights import (
    WeightAssertionResult,
    assert_weights_sum_to_one,
    assert_weights_non_negative,
    assert_weights_match_skill_ranking,
    assert_no_nan_weights,
    run_all_weight_assertions,
)

__all__ = [
    # Scoring
    "ScoringAssertionResult",
    "assert_clv_scores_valid",
    "assert_brier_scores_valid",
    "assert_pss_scores_valid",
    "assert_rolling_scores_valid",
    "assert_no_infinite_values",
    "run_all_scoring_assertions",
    # Weights
    "WeightAssertionResult",
    "assert_weights_sum_to_one",
    "assert_weights_non_negative",
    "assert_weights_match_skill_ranking",
    "assert_no_nan_weights",
    "run_all_weight_assertions",
]
