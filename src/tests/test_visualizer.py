import json
from pathlib import Path

import pandas as pd

from agents.visualizer import main as viz_main


def test_visualizer_outputs_pngs(tmp_path, monkeypatch):
    ticker = "TEST"

    # minimal prices to get a 50-DMA line
    pdir = tmp_path / "data" / ticker
    pdir.mkdir(parents=True)
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    df = pd.DataFrame({"Close": range(100, 160)}, index=dates)
    df.to_parquet(pdir / "prices.parquet")

    # minimal sentiment JSON
    sdir = tmp_path / "output" / "sentiment" / ticker
    sdir.mkdir(parents=True)
    (sdir / "sentiment_20250101T000000Z.json").write_text(
        json.dumps([
            {"published": "2025-01-01", "score": 0.15},
            {"published": "2025-01-02", "score": 0.35},
        ])
    )

    # run in tmp root so paths match repo layout
    monkeypatch.chdir(tmp_path)
    viz_main()

    out_dir = tmp_path / "output" / "visuals" / ticker
    assert (out_dir / "price_ma50.png").exists()
    assert (out_dir / "sentiment.png").exists()
