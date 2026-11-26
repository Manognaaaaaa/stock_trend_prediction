"""Backtester sanity checks (docs/STEPS.md, Step 3)."""
import pytest

SKIP = pytest.mark.skip(reason="Step 3 not built yet")


@SKIP
def test_random_scores_match_universe_minus_costs():
    """Random scores: net excess return over the equal-weight universe is about
    zero minus costs."""


@SKIP
def test_perfect_foresight_gives_very_large_returns():
    """Perfect foresight: using the real next-week target as the score gives
    very large returns."""


@SKIP
def test_momentum_only_gives_believable_result():
    """Momentum only: using ret_12m as the score gives a small, believable result."""
