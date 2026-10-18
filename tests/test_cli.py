import pandas as pd

import kelly as ks

EXAMPLE = (ks.Path(__file__).resolve().parents[1] / "views.example.csv").read_text()


def run(tmp_path, monkeypatch, *args):
    monkeypatch.chdir(tmp_path)
    return ks.main(list(args) + ["--out-dir", str(tmp_path / "out")])


def test_sizes_basket_and_saves(tmp_path, monkeypatch, capsys):
    assert run(tmp_path, monkeypatch, "NVDA", "KO", "MSFT", "VOO") == 0
    out = capsys.readouterr().out
    assert "Chance of loss (3y)" in out and "every name uses CAPM" in out
    w = pd.read_csv(next((tmp_path / "out").glob("kelly_portfolio_2*.csv")))
    assert abs(w.filter(like="weight_half").sum().iloc[0] - 1) < 1e-9


def test_uses_views_csv_and_logs_to_journal(tmp_path, monkeypatch, capsys):
    (tmp_path / "views.csv").write_text(EXAMPLE)
    assert run(tmp_path, monkeypatch, "NVDA", "ASML", "KO", "VOO", "--log") == 0
    out = capsys.readouterr().out
    assert "Using your views from views.csv" in out and "Sized on your research for 3" in out
    j = ks.load_journal(tmp_path / "journal.csv")
    assert set(j.ticker) == {"NVDA", "ASML", "KO"} and j.entry_price.notna().all()


def test_no_tickers_sizes_the_views(tmp_path, monkeypatch, capsys):
    (tmp_path / "views.csv").write_text(EXAMPLE)
    assert run(tmp_path, monkeypatch, "--no-save") == 0
    assert "Sized on your research for 3 of 3" in capsys.readouterr().out


def test_report_only(tmp_path, monkeypatch, capsys):
    (tmp_path / "views.csv").write_text(EXAMPLE)
    run(tmp_path, monkeypatch, "NVDA", "KO", "--log", "--no-save")
    capsys.readouterr()
    assert run(tmp_path, monkeypatch, "--report") == 0
    assert "Conviction journal: 2 calls" in capsys.readouterr().out


def test_screen_is_opt_in_and_labeled(tmp_path, monkeypatch, capsys):
    assert run(tmp_path, monkeypatch, "--screen", "--extra", "TSM") == 0
    out = capsys.readouterr().out
    assert "no-edge baseline" in out
    res = pd.read_csv(next((tmp_path / "out").glob("kelly_screen_*.csv")))
    assert "TSM" in set(res.ticker) and "cagr_gap" not in res.columns


def test_no_args_and_no_views_shows_help(tmp_path, monkeypatch, capsys):
    assert run(tmp_path, monkeypatch) == 2
    assert "python kelly.py NVDA ASML KO" in capsys.readouterr().out


def test_warns_above_book_ceiling(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, "KO", "MSFT", "--max-weight", "0.6", "--no-save")
    assert "above the book's 40% ceiling" in capsys.readouterr().out
