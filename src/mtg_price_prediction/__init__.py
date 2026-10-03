"""MTG card price-range prediction. Inference wrappers for the two published
models live in `catboost_model` (Model 1) and `deep_gp` (Model 2); they're
imported lazily so each only needs its own dependencies."""
