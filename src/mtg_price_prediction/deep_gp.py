"""Model 2 -- wide & deep feature extractor + sparse variational GP (deep
kernel learning), SPECS.md S4.2.

Holds the model classes and the tabular preprocessing so that training
(scripts/train_model2.py) and inference (`DeepGPPredictor`, also what the
published Hugging Face model loads) run exactly the same code.
"""
from __future__ import annotations

import json
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mtg_price_prediction import calibration

CATEGORICAL_COLS = [
    "lang", "set_type", "rarity", "finish", "layout", "border_color", "frame",
    "watermark", "security_stamp", "artist",
]
# power/toughness/loyalty/defense are NaN whenever a card type doesn't have
# that stat at all (not just the already-flagged "*"/"X"-style variable
# case) -- the neural net can't consume raw NaN the way CatBoost could, so
# these need an explicit fresh _isna flag + impute here.
NEEDS_ISNA_FLAG = ["power", "toughness", "loyalty", "defense"]
# All continuous (non-boolean) numeric columns -- standardized on train
# stats. Everything else in the wide feature set is already 0/1 boolean.
NUMERIC_TO_SCALE = (
    ["conv_mana_cost", "power", "toughness", "loyalty", "defense",
     "edhrec_rank", "edhrec_salt", "penny_rank", "num_paper_printings"]
)

Z_80 = 1.2816  # standard-normal z for the 10th/90th percentiles


# --- Preprocessing --------------------------------------------------------

def _build_vocab(series: pd.Series) -> dict:
    values = series.fillna("__missing__").astype(str)
    vocab = {v: i + 1 for i, v in enumerate(sorted(values.unique()))}
    vocab["__unk__"] = 0
    return vocab


def _encode_categorical(series: pd.Series, vocab: dict) -> np.ndarray:
    values = series.fillna("__missing__").astype(str)
    return values.map(lambda v: vocab.get(v, vocab["__unk__"])).to_numpy(dtype=np.int64)


def _add_isna_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NEEDS_ISNA_FLAG:
        df[f"{col}__isna"] = df[col].isna()
    return df


def fit_preprocessor(df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    """Categorical vocabs, wide-column order, impute medians and
    standardization stats -- all fit on the train split only. JSON-
    serializable, so it ships alongside the published weights."""
    df = _add_isna_flags(df)
    train = df.iloc[train_idx]

    rate_cols = [c for c in df.columns if c.endswith("_rate_percent")]
    numeric_to_scale = NUMERIC_TO_SCALE + rate_cols
    wide_cols = [c for c in df.columns if c not in CATEGORICAL_COLS and not c.startswith("__")]

    medians = {}
    filled = train[wide_cols].copy()
    for col in numeric_to_scale:
        median = filled[col].median()
        medians[col] = 0.0 if pd.isna(median) else float(median)
        filled[col] = filled[col].fillna(medians[col])
    filled = filled.astype(np.float32)
    scale_cols = [c for c in numeric_to_scale if c in filled.columns]
    means = filled[scale_cols].mean().fillna(0.0)
    stds = filled[scale_cols].std().replace(0, 1.0).fillna(1.0)

    return {
        "categorical_cols": list(CATEGORICAL_COLS),
        "cat_vocabs": {col: _build_vocab(train[col]) for col in CATEGORICAL_COLS},
        "wide_cols": wide_cols,
        "impute_medians": medians,
        "scale_means": {c: float(v) for c, v in means.items()},
        "scale_stds": {c: float(v) for c, v in stds.items()},
    }


def transform(df: pd.DataFrame, prep: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> (categorical index matrix, standardized wide float matrix)."""
    df = _add_isna_flags(df)
    cat_idx = np.stack([_encode_categorical(df[col], prep["cat_vocabs"][col])
                        for col in prep["categorical_cols"]], axis=1)

    missing = [c for c in prep["wide_cols"] if c not in df.columns]
    if missing:
        raise KeyError(f"{len(missing)} wide feature columns missing from input, e.g. {missing[:5]}")
    wide_df = df[prep["wide_cols"]].copy()
    for col, median in prep["impute_medians"].items():
        wide_df[col] = wide_df[col].fillna(median)
    wide_df = wide_df.astype(np.float32)
    scale_cols = list(prep["scale_means"])
    means = pd.Series(prep["scale_means"], dtype=np.float32)
    stds = pd.Series(prep["scale_stds"], dtype=np.float32)
    wide_df[scale_cols] = (wide_df[scale_cols] - means) / stds
    return cat_idx, wide_df.to_numpy(dtype=np.float32)


# --- Model ----------------------------------------------------------------

def _emb_dim(vocab_size: int) -> int:
    return int(min(32, max(2, round(vocab_size ** 0.5))))


def _linear(in_dim: int, out_dim: int, spectral_norm: bool) -> nn.Module:
    layer = nn.Linear(in_dim, out_dim)
    return torch.nn.utils.spectral_norm(layer) if spectral_norm else layer


class _ResidualSpectralMLP(nn.Module):
    """`Linear(in,hidden) -> [x + spectral_norm(Linear(hidden,hidden))(relu(x))] * n_blocks
    -> Linear(hidden,out)`. The residual connection is what makes this
    bi-Lipschitz: spectral norm alone caps the block's Lipschitz constant
    from above (bounding expansion), but the `x + ...` skip guarantees the
    output can't collapse arbitrarily close together either, since the
    identity path always carries the input's own separation through
    unchanged. Plain spectral-normed feedforward layers (`_linear` above)
    have no such floor."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_blocks: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            torch.nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_proj(x))
        for block in self.blocks:
            x = x + block(self.act(x))
        return self.out_proj(x)


class WideAndDeepFeatureExtractor(nn.Module):
    def __init__(self, n_wide: int, cat_vocab_sizes: list[int], text_dim: int, image_dim: int, joint_dim: int,
                 use_text: bool = True, use_image: bool = True, use_tabular: bool = True,
                 spectral_norm: bool = False, residual_spectral_norm: bool = False,
                 residual_hidden_dim: int = 128, residual_n_blocks: int = 2):
        super().__init__()
        self.use_text = use_text
        self.use_image = use_image
        self.use_tabular = use_tabular
        combined_dim = 0
        sn = spectral_norm

        if use_tabular:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, _emb_dim(vocab_size)) for vocab_size in cat_vocab_sizes
            ])
            combined_dim += sum(_emb_dim(v) for v in cat_vocab_sizes)
            if residual_spectral_norm:
                self.wide_branch = _ResidualSpectralMLP(n_wide, residual_hidden_dim, 128, residual_n_blocks)
            else:
                self.wide_branch = nn.Sequential(_linear(n_wide, 256, sn), nn.ReLU(), _linear(256, 128, sn), nn.ReLU())
            combined_dim += 128
        if use_text:
            if residual_spectral_norm:
                self.text_branch = _ResidualSpectralMLP(text_dim, residual_hidden_dim, 64, residual_n_blocks)
            else:
                self.text_branch = nn.Sequential(_linear(text_dim, 128, sn), nn.ReLU(), _linear(128, 64, sn))
            combined_dim += 64
        if use_image:
            if residual_spectral_norm:
                self.image_branch = _ResidualSpectralMLP(image_dim, residual_hidden_dim, 64, residual_n_blocks)
            else:
                self.image_branch = nn.Sequential(_linear(image_dim, 128, sn), nn.ReLU(), _linear(128, 64, sn))
            combined_dim += 64

        assert combined_dim > 0, "feature extractor needs at least one of use_text/use_image/use_tabular"
        if residual_spectral_norm:
            self.head = _ResidualSpectralMLP(combined_dim, residual_hidden_dim, joint_dim, residual_n_blocks)
        else:
            self.head = nn.Sequential(_linear(combined_dim, 128, sn), nn.ReLU(), _linear(128, joint_dim, sn))

    def forward(self, cat_idx: torch.Tensor, wide: torch.Tensor, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_tabular:
            cat_out = torch.cat([emb(cat_idx[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=-1)
            parts.append(cat_out)
            parts.append(self.wide_branch(wide))
        if self.use_text:
            parts.append(self.text_branch(text_emb))
        if self.use_image:
            parts.append(self.image_branch(image_emb))
        return self.head(torch.cat(parts, dim=-1))


class GPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1)))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor: WideAndDeepFeatureExtractor, inducing_points: torch.Tensor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPLayer(inducing_points)

    def forward(self, cat_idx, wide, text_emb, image_emb):
        z = self.feature_extractor(cat_idx, wide, text_emb, image_emb)
        return self.gp_layer(z)


def predict_moments(model: DKLModel, likelihood, cat, wide, text_emb, image_emb,
                    batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Posterior predictive (mean, std) in log-price. GaussianLikelihood's
    marginal() is closed-form: pred.mean/variance are already (batch,).
    Non-Gaussian likelihoods (e.g. StudentTLikelihood) instead approximate
    the marginal by *sampling* -- pred.mean/variance come back shaped
    (num_samples, batch). Collapsed here via the law of total variance:
    Var[y] = E[Var(y|f)] + Var[E(y|f)]."""
    means, variances = [], []
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        for start in range(0, len(cat), batch_size):
            sl = slice(start, start + batch_size)
            pred = likelihood(model(cat[sl], wide[sl], text_emb[sl], image_emb[sl]))
            pred_mean, pred_var = pred.mean, pred.variance
            if pred_mean.dim() == 2:  # sample dimension present (non-Gaussian likelihood)
                pred_var = pred_var.mean(dim=0) + pred_mean.var(dim=0, unbiased=False)
                pred_mean = pred_mean.mean(dim=0)
            means.append(pred_mean)
            variances.append(pred_var)
    mean = torch.cat(means).numpy()
    std = torch.cat(variances).clamp_min(1e-8).sqrt().numpy()
    return mean, std


# --- Published-model inference -------------------------------------------

class DeepGPPredictor:
    """Loads an exported Model 2 (scripts/export_models.py layout: config.json,
    preprocessor.json, model.safetensors) and predicts calibrated 80% price
    ranges. Gaussian likelihood only (the only one published)."""

    def __init__(self, model: DKLModel, likelihood, prep: dict, config: dict):
        self.model, self.likelihood, self.prep, self.config = model, likelihood, prep, config
        self.corrections = {int(k): v for k, v in config["calibration_corrections"].items()}

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str) -> "DeepGPPredictor":
        from safetensors.torch import load_file

        path = Path(repo_id_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id_or_path))
        config = json.loads((path / "config.json").read_text())
        prep = json.loads((path / "preprocessor.json").read_text())
        state = load_file(path / "model.safetensors")

        arch = config["architecture"]
        feature_extractor = WideAndDeepFeatureExtractor(
            n_wide=len(prep["wide_cols"]),
            cat_vocab_sizes=[len(prep["cat_vocabs"][c]) for c in prep["categorical_cols"]],
            text_dim=arch["text_dim"], image_dim=arch["image_dim"], joint_dim=arch["joint_dim"],
            use_text=arch["use_text"], use_image=arch["use_image"], use_tabular=arch["use_tabular"],
        )
        inducing = state["model.gp_layer.variational_strategy.inducing_points"]
        model = DKLModel(feature_extractor, torch.zeros_like(inducing))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
        likelihood.load_state_dict({k[len("likelihood."):]: v for k, v in state.items() if k.startswith("likelihood.")})
        return cls(model, likelihood, prep, config)

    def predict(self, features: pd.DataFrame, text_emb: np.ndarray, image_emb: np.ndarray) -> pd.DataFrame:
        """`features`: rows shaped like training/features_model2.parquet
        (scripts/prepare_training_data_model2.py). `text_emb`: (n, 768)
        all-mpnet-base-v2 oracle-text embeddings. `image_emb`: (n, 512) CLIP
        ViT-B-32-quickgelu art_crop embeddings. Returns USD price_low /
        price_mid / price_high (calibrated 80% interval + posterior mean)."""
        cat_idx, wide = transform(features, self.prep)
        mean, std = predict_moments(
            self.model, self.likelihood, torch.from_numpy(cat_idx), torch.from_numpy(wide),
            torch.as_tensor(text_emb, dtype=torch.float32), torch.as_tensor(image_emb, dtype=torch.float32),
        )
        lo, hi = calibration.apply_corrections(mean, mean - Z_80 * std, mean + Z_80 * std, self.corrections)
        return pd.DataFrame({"price_low": np.exp(lo), "price_mid": np.exp(mean), "price_high": np.exp(hi)},
                            index=features.index)
