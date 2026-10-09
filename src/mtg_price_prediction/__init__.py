"""MTG card price prediction. Fair value (`fair_value`, Model A), 14-day forecasts (`market`,
Models B and C), the deep GP (`deep_gp`, `gp_features`), custom cards (`cards`) and the nightly
scoring (`scoring`). Modules import their own dependencies, so the nightly pipeline runs without torch."""
