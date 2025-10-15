from analyst.compose import build_facts, render_analyst_take

def test_analyst_fallback_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYST_ENABLED", "0")
    facts = {
        "ticker": "TEST",
        "price": {"last": 100.0, "pct30": 4.5},
        "tech": {"above_ma50": True, "above_ma200": False, "overbought": False, "oversold": False},
        "sent": {"n": 20, "avg": 0.1, "pos_pct": 45.0, "neg_pct": 15.0},
        "fund": {"rev_yoy": 0.12, "ni_yoy": 0.08}
    }
    txt = render_analyst_take(facts)
    assert isinstance(txt, str) and len(txt) > 10
