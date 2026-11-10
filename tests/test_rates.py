"""The risk-free lookup's fallback order, offline (requests is faked)."""
import pandas as pd
import pytest

import kelly as ks

REAL_RATE = ks.get_risk_free_rate      # conftest swaps in a fake once tests start

FRED_CSV = "observation_date,DGS2\n2026-11-06,3.61\n2026-11-09,\n"
TREASURY_CSV = ('"Date","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr"\n'
                "11/06/2026,3.70,3.60,3.55,3.70,3.90,4.10\n"
                "11/09/2026,3.72,3.62,3.57,3.71,3.91,4.12\n")


class Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def fake_get(fred_ok: bool, treasury_ok: bool):
    calls = []

    def get(url, **kw):
        calls.append(url)
        if "fred" in url and fred_ok:
            return Resp(FRED_CSV)
        if "treasury.gov" in url and treasury_ok:
            return Resp(TREASURY_CSV)
        raise TimeoutError("read timed out")
    return get, calls


@pytest.mark.parametrize("fred_ok, treasury_ok, want", [
    (True, True, (0.0362, "Treasury.gov 2 Yr")),         # Treasury.gov first
    (True, False, (0.0361, "FRED DGS2")),                # skips FRED's blank latest day
    (False, False, (ks.FALLBACK_RF, "fallback constant")),
])
def test_fallback_order(monkeypatch, fred_ok, treasury_ok, want):
    get, calls = fake_get(fred_ok, treasury_ok)
    monkeypatch.setattr(ks.requests, "get", get)
    monkeypatch.setattr("yfinance.Ticker", lambda s: (_ for _ in ()).throw(TimeoutError()))
    rf, source = REAL_RATE(2)
    assert (round(rf, 4), source) == want
    if not treasury_ok:
        assert sum("treasury.gov" in c for c in calls) >= ks.RATE_TRIES   # retried


def test_treasury_csv_newest_row_wins_even_if_listed_first(monkeypatch):
    flipped = TREASURY_CSV.splitlines()
    csv = "\n".join([flipped[0], flipped[2], flipped[1]]) + "\n"
    monkeypatch.setattr(ks.requests, "get", lambda url, **kw: Resp(csv))
    assert ks._treasury_yield(10) == pytest.approx(0.0412)


def test_a_ticker_that_comes_back_blank_is_fetched_again(monkeypatch):
    """yfinance can return a column of NaN (and only log the error) when its
    cache locks; the download must retry that name rather than drop it."""
    import yfinance as yf
    from conftest import fake_download
    calls = []

    def flaky(tickers, **kw):
        calls.append(tickers)
        out = fake_download(tickers, **kw)
        if isinstance(tickers, list) and "AAA" in tickers:
            out.loc[:, ("Close", "AAA")] = float("nan")       # blank on the bulk call
        return out
    monkeypatch.setattr(yf, "download", flaky)
    prices = ks.download_monthly_closes(["AAA", "BBB", "^GSPC"], 10)
    assert prices["AAA"].notna().sum() >= ks.MIN_MONTHS
    assert "AAA" in calls                                      # refetched on its own
