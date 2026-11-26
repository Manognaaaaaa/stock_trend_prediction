"""Leakage tests for features, target, folds and market data (docs/STEPS.md, Step 2.5)."""
import pytest

SKIP = pytest.mark.skip(reason="Step 2 not built yet")


@SKIP
def test_shift_features_use_only_past_data():
    """Shift test: recompute features for 20 random rows using only data up to
    that date; the values must match the stored features."""


@SKIP
def test_target_uses_only_future_prices():
    """Target test: the target at week t uses prices strictly after week t's close."""


@SKIP
def test_folds_do_not_overlap_and_respect_embargo():
    """Fold test: no date appears in both train and test, and there is a gap of
    at least 5 trading days between blocks."""


@SKIP
def test_oil_and_vix_dated_before_rebalance_day():
    """Oil alignment test: every Brent and VIX value is dated strictly before
    the Saudi rebalance day."""


@SKIP
def test_random_labels_give_zero_rank_ic():
    """Random-label test: LightGBM trained on shuffled targets gives mean Rank IC
    within about plus or minus 0.01."""
