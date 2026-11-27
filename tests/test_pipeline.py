"""Pipeline checks on the synthetic data (docs/SPEC_STEPS_0_2.md, Sections 3.3 and 4.6)."""
import io

import numpy as np
import pandas as pd
import pytest

from src.clean import check_week_ends, to_local_dates
from src.data_download import load_universe, yahoo_symbol
from src.utils import write_raw
from tests.conftest import CA_JUMPS, DELISTED, IPO_JUMP, LISTED_2013, REIT_SYMBOL


# ---------------------------------------------------------------- calendar and weeks


def test_week_end_never_friday_and_in_calendar(synthetic, cfg):
    """No week_end is a Friday; every week_end is a TASI trading day; no duplicates."""
    checks = check_week_ends(synthetic["panel"], synthetic["calendar"], cfg)
    assert checks["friday"] == 0
    assert checks["not_in_calendar"] == 0
    assert checks["thursday_before_change"] == 0
    assert checks["dup_symbol_week"] == 0


def test_wednesday_week_end_after_change_only_when_thursday_is_holiday(synthetic, cfg):
    """After the weekend change, a Wednesday week_end means Thursday was not a trading day."""
    change = pd.Timestamp(cfg["calendar"]["weekend_change_date"])
    we = pd.Series(synthetic["panel"]["week_end"].unique())
    weds = we[(we >= change) & (we.dt.day_of_week == 2)]
    assert len(weds) > 0  # the fixture plants such holidays
    assert not (weds + pd.Timedelta(days=1)).isin(synthetic["calendar"]).any()


def test_to_local_dates_uses_exchange_time_zone():
    """A UTC timestamp late in the evening belongs to the next Riyadh day."""
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01 22:00", "2024-01-01 22:00"]).tz_localize("UTC"),
                       "Close": [1.0, 2.0]})
    out, n_dup = to_local_dates(df, "Asia/Riyadh")
    assert n_dup == 1
    assert out["date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert out["Close"].tolist() == [2.0]  # keep last
    assert out["date"].dtype == "datetime64[ns]"


# ---------------------------------------------------------------- cleaning rules


def test_no_duplicate_symbol_date(synthetic):
    """daily has one row per (symbol, date) and all dates are TASI trading days."""
    daily = synthetic["daily"]
    assert not daily.duplicated(["symbol", "date"]).any()
    assert daily["date"].isin(synthetic["calendar"]).all()


def test_cleaning_counts(synthetic):
    """R0 drops the planted NaN and off-calendar rows; R1 drops zero-volume days."""
    counts = synthetic["counts"]
    assert counts["r0_nan_price"] == 1
    assert counts["r0_not_in_calendar"] == 1
    assert counts["duplicate_dates"] >= 1
    assert counts["r1_zero_volume"] > 0
    assert (synthetic["daily"]["volume"] > 0).all()


def test_fake_corporate_actions_are_flagged(synthetic):
    """The two +40% jumps are flagged and their ret is NaN; the IPO jump is not flagged."""
    daily = synthetic["daily"].set_index(["symbol", "date"])
    for symbol, day in CA_JUMPS:
        row = daily.loc[(symbol, pd.Timestamp(day))]
        assert bool(row["ca_flag"])
        assert np.isnan(row["ret"])
    ipo = synthetic["daily"].loc[synthetic["daily"]["symbol"] == LISTED_2013[0]].sort_values("date")
    assert ipo["raw_ret"].iloc[1] == pytest.approx(IPO_JUMP)
    assert not ipo["ca_flag"].iloc[:3].any()
    assert synthetic["counts"]["r2_ca_flags"] == len(CA_JUMPS)


def test_returns_and_limit_up(synthetic, cfg):
    """ret is the log of adjusted close ratio; limit_up uses the raw close."""
    d = synthetic["daily"]
    one = d.loc[(d["symbol"] == "1003")].sort_values("date")
    expected = np.log(one["adj_close"] / one["adj_close"].shift(1))
    ok = ~one["ca_flag"]
    assert np.allclose(one.loc[ok, "ret"], expected[ok], equal_nan=True)
    assert (d.loc[d["limit_up"], "raw_ret"] >= cfg["universe"]["limit_up_tolerance"]).all()


def test_market_value_uses_only_past_share_counts(synthetic):
    """'full' rows use the latest share count dated on or before the day."""
    d = synthetic["daily"]
    full = synthetic["shares_full"].copy()
    full["date"] = full["date"].dt.tz_convert("Asia/Riyadh").dt.tz_localize(None)
    sym = full["symbol"].iloc[0]
    rows = d.loc[(d["symbol"] == sym) & (d["shares_source"] == "full")]
    sample = rows.sample(20, random_state=0)
    for _, r in sample.iterrows():
        known = full.loc[(full["symbol"] == sym) & (full["date"] <= r["date"])]
        assert r["shares_out"] == known.sort_values("date")["shares"].iloc[-1]
    before_first = d.loc[(d["symbol"] == sym) & (d["date"] < full["date"].min())]
    assert (before_first["shares_source"] != "full").all()
    assert set(d["shares_source"].unique()) == {"full", "snapshot", "none"}


def test_reit_excluded_and_sectors_attached(synthetic):
    """The REIT never reaches daily; every row has an industry and broad group."""
    d = synthetic["daily"]
    assert REIT_SYMBOL not in set(d["symbol"])
    assert d["industry_group"].notna().all()
    assert d["broad_group"].notna().all()


# ---------------------------------------------------------------- weekly panel


def test_delisted_stop_and_new_listing_needs_history(synthetic, cfg):
    """Delisted stocks stop after their last date; the 2013 listing needs 60 rows to be eligible."""
    panel = synthetic["panel"]
    for symbol, last in DELISTED.items():
        assert panel.loc[panel["symbol"] == symbol, "last_date"].max() <= pd.Timestamp(last)
    new = panel.loc[panel["symbol"] == LISTED_2013[0]]
    assert new["week_end"].min() >= pd.Timestamp(LISTED_2013[1])
    assert (new.loc[new["eligible"], "history_rows"] >= cfg["universe"]["min_history_days"]).all()
    assert (~new.loc[new["history_rows"] < cfg["universe"]["min_history_days"], "eligible"]).all()


def test_panel_columns_are_consistent(synthetic, cfg):
    """days_traded, flags and week_ret follow their definitions."""
    p = synthetic["panel"]
    assert ((p["days_traded"] >= 1) & (p["days_traded"] <= 6)).all()
    assert (p["traded_on_week_end"] == (p["last_date"] == p["week_end"])).all()
    assert (~p["limit_up_on_week_end"] | p["traded_on_week_end"]).all()
    fails = p["days_traded"] < cfg["universe"]["min_days_traded_per_week"]
    assert not p.loc[fails, "eligible"].any()
    one = p.loc[p["symbol"] == "1003"].sort_values("week_end").reset_index(drop=True)
    i = one["week_ret"].first_valid_index() + 5
    assert one.loc[i, "week_ret"] == pytest.approx(np.log(one.loc[i, "adj_close"] / one.loc[i - 1, "adj_close"]))


def test_market_daily(synthetic):
    """market_daily holds the three series with a daily log return."""
    m = synthetic["market_daily"]
    assert set(m["series"].unique()) == {"tasi", "brent", "vix"}
    brent = m.loc[m["series"] == "brent"].sort_values("date")
    assert brent["date"].dt.day_of_week.max() <= 4
    assert np.allclose(brent["ret"].iloc[1:], np.log(brent["close"] / brent["close"].shift(1)).iloc[1:])


# ---------------------------------------------------------------- download helpers and io


def test_load_universe(tmp_path, cfg):
    """Symbols keep leading zeros, REITs are dropped and an empty file raises."""
    path = tmp_path / "universe.csv"
    path.write_text("symbol,name,industry_group,status,listed_date,delisted_date\n"
                    "0042,A,Banks,listed,,\n4330,B,REITs,listed,,\n1111,C,Energy,delisted,,2015-01-01\n")
    uni = load_universe(path, cfg)
    assert list(uni["symbol"]) == ["0042", "1111"]
    assert uni["delisted_date"].dtype == "datetime64[ns]"
    assert yahoo_symbol("2222", cfg) == "2222.SR"
    empty = tmp_path / "empty.csv"
    empty.write_text("symbol,name,industry_group,status,listed_date,delisted_date\n")
    with pytest.raises(ValueError, match="Section 5.1"):
        load_universe(empty, cfg)


def test_write_raw_never_overwrites(tmp_path):
    """write_raw writes once and then refuses to overwrite."""
    path = tmp_path / "x.parquet"
    assert write_raw(pd.DataFrame({"a": [1]}), path)
    assert not write_raw(pd.DataFrame({"a": [2]}), path)
    assert pd.read_parquet(path)["a"].tolist() == [1]


# ---------------------------------------------------------------- TASI reference CSV


INVESTING_SAMPLE = '''"Date","Price","Open","High","Low","Vol.","Change %"
"01/03/2024","11,967.45","11,980.10","12,001.20","11,940.00","245.67M","-0.10%"
"01/02/2024","11,979.62","11,967.39","12,010.55","11,950.31","1.20B","0.10%"
"01/01/2024","11,967.39","11,960.00","11,990.00","11,930.00","-","0.05%"
'''


def test_parse_investing_csv(tmp_path, cfg):
    """Investing.com exports become a sorted Yahoo-shaped table; parts are joined without duplicates."""
    from src.clean import parse_investing_csv, read_tasi_reference, to_local_dates
    df = pd.read_csv(io.StringIO(INVESTING_SAMPLE), dtype="string")
    out = parse_investing_csv(df, cfg["tasi_csv"]["date_format"])
    assert out["Date"].tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert out["Close"].tolist() == [11967.39, 11979.62, 11967.45]
    assert out["Volume"].iloc[1] == pytest.approx(1.2e9)
    assert np.isnan(out["Volume"].iloc[0])
    local, _ = to_local_dates(out, cfg["calendar"]["saudi_tz"])
    assert local["date"].dtype == "datetime64[ns]"
    (tmp_path / "tasi_part1.csv").write_text(INVESTING_SAMPLE, encoding="utf-8-sig")
    (tmp_path / "tasi_part2.csv").write_text(INVESTING_SAMPLE, encoding="utf-8")
    joined = read_tasi_reference(tmp_path, cfg)
    assert len(joined) == 3
    short = {**cfg, "data": {**cfg["data"], "end_date": "2024-01-02"}}
    assert read_tasi_reference(tmp_path, short)["Date"].max() == pd.Timestamp("2024-01-02")


def test_missing_tasi_csv_raises_clear_error(tmp_path, cfg):
    """Without Yahoo TASI and without a reference CSV, the error says where to get it."""
    from src.clean import read_tasi_reference
    with pytest.raises(FileNotFoundError, match="investing.com"):
        read_tasi_reference(tmp_path, cfg)


def test_tasi_filler_rows_dropped():
    """Rows with no volume and an unchanged close are dropped; real no-volume rows stay."""
    from src.clean import drop_tasi_filler
    tasi = pd.DataFrame({"Close": [100.0, 101.0, 101.0, 101.0, 102.0],
                         "Volume": [np.nan, np.nan, np.nan, 5.0, np.nan]})
    out, n = drop_tasi_filler(tasi)
    assert n == 1
    assert out["Close"].tolist() == [100.0, 101.0, 101.0, 102.0]


def test_date_suspect_flag(synthetic, cfg):
    """Stock rows before yahoo_dates_reliable_from are flagged, later ones are not."""
    cut = pd.Timestamp(cfg["calendar"]["yahoo_dates_reliable_from"])
    d, p = synthetic["daily"], synthetic["panel"]
    assert (d["date_suspect"] == (d["date"] < cut)).all()
    assert (p["date_suspect"] == (p["last_date"] < cut)).all()


# ---------------------------------------------------------------- Step 2 tables


def test_ranked_features_in_range_without_nan(synthetic):
    """Ranked features lie in [-1, 1] and have no NaN."""
    from src.features import FEATURES
    x = synthetic["model_table"][FEATURES]
    assert x.notna().all().all()
    assert ((x >= -1) & (x <= 1)).all().all()


def test_sector_groups_have_min_size(synthetic, cfg):
    """Every (week_end, sector_group) except Other has at least min_sector_size stocks."""
    mt = synthetic["model_table"]
    sizes = mt.groupby(["week_end", "sector_group"]).size().reset_index(name="n")
    small = sizes.loc[sizes["sector_group"] != "Other", "n"]
    assert (small >= cfg["universe"]["min_sector_size"]).all()
    assert "Other" in set(sizes["sector_group"])  # Energy (2) and Insurance (3) cannot reach 5
