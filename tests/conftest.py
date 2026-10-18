"""
Offline test fixtures: a deterministic fake of Yahoo Finance so the suite
runs without network and gives identical numbers every run.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kelly as ks  # noqa: E402

N_MONTHS = 120
SHORT_HISTORY = {"NEWCO": 20}          # a recent IPO
_IDX = pd.date_range(end=pd.Timestamp.today().normalize() - pd.DateOffset(months=1),
                     periods=N_MONTHS, freq="MS")
_MARKET = np.random.default_rng(1).normal(0.008, 0.045, N_MONTHS)


def fake_returns(ticker: str) -> np.ndarray:
    if ticker in ("^GSPC", "VOO"):
        return _MARKET.copy()
    seed = zlib.crc32(ticker.encode())
    rng = np.random.default_rng(seed)
    beta = 0.5 + (seed % 15) / 10
    r = beta * _MARKET + rng.normal(0, 0.05, N_MONTHS)
    if ticker in SHORT_HISTORY:
        r[: N_MONTHS - SHORT_HISTORY[ticker]] = np.nan
    return r


def fake_prices(ticker: str) -> np.ndarray:
    r = fake_returns(ticker)
    p = 100 * np.cumprod(1 + np.nan_to_num(r))
    p[np.isnan(r)] = np.nan
    return p


def fake_download(tickers, **kwargs):
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    close = pd.DataFrame({t: fake_prices(t) for t in tickers}, index=_IDX)
    return pd.concat({"Close": close}, axis=1)


class FakeTicker:
    def __init__(self, t):
        self.info = {"shortName": f"{t} Inc", "sector": "Technology"}


CONSTITUENTS = pd.DataFrame({
    "ticker": ["AAPL", "MSFT", "NVDA", "KO", "JNJ", "XOM"],
    "name": ["Apple", "Microsoft", "Nvidia", "Coca-Cola", "J&J", "Exxon"],
    "sector": ["IT", "IT", "IT", "Staples", "Health Care", "Energy"],
})


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ks, "get_risk_free_rate", lambda: (0.045, "test"))
    monkeypatch.setattr(ks, "get_sp500_constituents", lambda: CONSTITUENTS.copy())
    monkeypatch.setattr(ks.time, "sleep", lambda s: None)
    monkeypatch.setattr(ks, "log", lambda m: None)


@pytest.fixture
def prices():
    return fake_download(["^GSPC", "AAA", "BBB", "CCC", "DDD", "NEWCO"])["Close"]
