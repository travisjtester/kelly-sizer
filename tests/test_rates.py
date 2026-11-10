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
    (True, True, (0.0361, "FRED DGS2")),                 # skips FRED's blank latest day
    (False, True, (0.0362, "Treasury.gov 2 Yr")),        # newest row, whatever the order
    (False, False, (ks.FALLBACK_RF, "fallback constant")),
])
def test_fallback_order(monkeypatch, fred_ok, treasury_ok, want):
    get, calls = fake_get(fred_ok, treasury_ok)
    monkeypatch.setattr(ks.requests, "get", get)
    monkeypatch.setattr("yfinance.Ticker", lambda s: (_ for _ in ()).throw(TimeoutError()))
    rf, source = REAL_RATE(2)
    assert (round(rf, 4), source) == want
    if not fred_ok:
        assert sum("fred" in c for c in calls) == ks.RATE_TRIES       # retried once


def test_treasury_csv_newest_row_wins_even_if_listed_first(monkeypatch):
    flipped = TREASURY_CSV.splitlines()
    csv = "\n".join([flipped[0], flipped[2], flipped[1]]) + "\n"
    monkeypatch.setattr(ks.requests, "get", lambda url, **kw: Resp(csv))
    assert ks._treasury_yield(10) == pytest.approx(0.0412)
