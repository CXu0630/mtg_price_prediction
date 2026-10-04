import numpy as np
import pandas as pd
import pytest

from mtg_price_prediction import market
from mtg_price_prediction.cards import CustomCard, colors_of, custom_rows, mana_value


def _dataset() -> pd.DataFrame:
    """A tiny dataset-shaped frame with the columns custom_rows reads."""
    return pd.DataFrame({
        "scryfall_id": ["s1", "s2"], "oracle_id": ["o1", "o1"], "name": ["Sol Ring", "Sol Ring"],
        "edhrec_rank": [1.0, 1.0], "edhrec_salt": [1.5, 1.5], "oracle_tags": [["ramp"], ["ramp"]],
        "legalities.commander": ["legal", "legal"], "legalities.standard": ["not_legal", "not_legal"],
        "play_rate_percent": [np.nan, np.nan], "collector_rate_percent": [np.nan, np.nan],
    })


def test_mana_value_and_colors():
    assert mana_value("{2}{U}{U}") == 4
    assert mana_value("{X}{R}") == 1
    assert mana_value("{2/W}{G/U}") == 3
    assert colors_of("{2}{W}{U/B}") == ["W", "U", "B"]


def test_custom_rows_defaults_and_reprint(tmp_path, monkeypatch):
    defaults = pd.DataFrame({"rarity": ["rare"], "finish": ["nonfoil"], "play_rate_percent": [0.1],
                             "collector_rate_percent": [0.3]})
    path = tmp_path / "defaults.parquet"
    defaults.to_parquet(path)
    monkeypatch.setattr("mtg_price_prediction.cards.PULL_RATE_DEFAULTS", path)
    cards = [CustomCard(name="New Card", rarity="rare", type_line="Legendary Creature — Elf", mana_cost="{1}{G}",
                        demand="staple"),
             CustomCard(name="Reprint", rarity="rare", type_line="Artifact", reprint_of="Sol Ring", set_type="commander")]
    rows = custom_rows(cards, _dataset(), pd.Timestamp("2026-10-03"))
    new, reprint = rows.iloc[0], rows.iloc[1]
    assert new["printing_key"] == "custom:0" and new["oracle_id"] == "custom:0"
    assert new["edhrec_rank"] == 650.0 and new["can_lead_commander"]
    assert new["play_rate_percent"] == 0.1 and new["collector_rate_percent"] == 0.3  # booster set: play + collector
    assert new["legalities.standard"] == "legal"  # expansion -> standard-era formats
    assert reprint["oracle_id"] == "o1" and reprint["reprint"] and reprint["edhrec_rank"] == 1.0
    assert reprint["oracle_tags"] == ["ramp"] and reprint["num_paper_printings"] == 3
    assert np.isnan(reprint["play_rate_percent"])  # commander set: no boosters


def test_custom_rows_rejects_unknown_reprint():
    with pytest.raises(ValueError, match="reprint_of"):
        custom_rows([CustomCard(name="x", rarity="rare", type_line="Artifact", reprint_of="No Such Card")],
                    _dataset(), pd.Timestamp("2026-10-03"))


def test_move_model_interval_is_ordered_even_with_negative_margin():
    rng = np.random.default_rng(0)
    n = 3000
    rows = pd.DataFrame({c: rng.normal(size=n) for c in market.B_NUMS})
    for c in market.CATS:
        rows[c] = rng.choice(["a", "b"], n)
    rows["release_status"] = rng.choice(market.CAT_LEVELS["release_status"], n)
    rows["finish_group"] = rng.choice(market.CAT_LEVELS["finish_group"], n)
    rows["snapshot_date"] = pd.Timestamp("2026-08-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D")
    rows["ret"] = 0.1 * rows["log_price"] + rng.normal(scale=0.05, size=n)
    model = market.model_b(rows).fit(rows)
    model.q = -1.0  # a negative CQR margin used to push lo above the median
    p = model.predict(rows.head(200))
    assert (p["move_lo"] <= p["move_med"]).all() and (p["move_med"] <= p["move_hi"]).all()
