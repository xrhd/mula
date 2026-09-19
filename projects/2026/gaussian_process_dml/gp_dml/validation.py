"""Validação causal do CATE via econml.validate.DRTester (BLP, calibração, Qini, TOC)."""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.validate import DRTester


def binarize_treatment(T: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Converte tratamento contínuo em binário (control=0, tratado=1).

    Se há amostras com |T| < eps (T concentrado perto do zero), corta em T > eps;
    caso contrário (ex.: sim, T ~ N(5+x, 1)), usa a mediana como corte.
    """
    T = np.asarray(T, dtype=float)
    if np.any(np.abs(T) < eps):
        return (T > eps).astype(int)
    return (T > np.median(T)).astype(int)


def make_split(data: dict, val_frac: float = 0.25, seed: int = 42):
    """Split treino/validação estratificado por tratamento discreto."""
    rng = np.random.default_rng(seed)
    D = binarize_treatment(data["T"]) if np.unique(data["T"]).size > 2 else np.asarray(data["T"]).astype(int)
    idx = np.arange(len(data["Y"]))
    train_idx = np.concatenate(
        [rng.choice(idx[D == t], size=int(len(idx[D == t]) * (1 - val_frac)), replace=False)
         for t in np.unique(D)]
    )
    val_idx = np.setdiff1d(idx, train_idx)
    return train_idx, val_idx, D


def _nuisance_models(seed: int = 42):
    common = dict(random_state=seed, verbose=-1, force_col_wise=True)
    return LGBMRegressor(n_estimators=50, max_depth=3, **common), LGBMClassifier(
        n_estimators=50, max_depth=3, **common
    )


def run_drtester(dml, data: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                 D: np.ndarray, seed: int = 42, n_bootstrap: int = 100):
    """Executa a bateria completa de validação (BLP + Cal + Qini + TOC)."""
    model_regression, model_propensity = _nuisance_models(seed)
    tester = DRTester(model_regression=model_regression, model_propensity=model_propensity,
                      cate=dml, cv=5)
    X, Y = data["X"], data["Y"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        tester.fit_nuisance(X[val_idx], D[val_idx], np.asarray(Y)[val_idx],
                            X[train_idx], D[train_idx], np.asarray(Y)[train_idx])
        tester.get_cate_preds(X[val_idx], X[train_idx])
        res = tester.evaluate_all(Xval=X[val_idx], Xtrain=X[train_idx], n_bootstrap=n_bootstrap)

    tmt = tester.treatments[1]
    return res, res.summary(), tmt


def save_plots(res, tmt, out_dir) -> list:
    """Salva cal.png, qini.png e toc.png no diretório informado."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = (("cal", res.plot_cal(tmt=tmt)),
             ("qini", res.plot_qini(tmt=tmt)),
             ("toc", res.plot_toc(tmt=tmt)))
    paths = []
    for name, fig in plots:
        fig.figure.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        paths.append(out_dir / f"{name}.png")
    return paths


def validate_dml(data: dict, dml, seed: int = 42, n_bootstrap: int = 100, splits: dict = None):
    """Pipeline completo: split -> DRTester -> métricas + plots.

    `splits` opcional (dict com train_idx/val_idx/D) para reutilizar o mesmo
    split em vários estimadores — essencial para comparações justas.

    Retorna (res, summary_df, tmt, splits).
    """
    if splits is None:
        train_idx, val_idx, D = make_split(data, seed=seed)
    else:
        train_idx, val_idx, D = splits["train_idx"], splits["val_idx"], splits["D"]
    res, summary_df, tmt = run_drtester(dml, data, train_idx, val_idx, D, seed, n_bootstrap)
    return res, summary_df, tmt, {"train_idx": train_idx, "val_idx": val_idx, "D": D}


def compare_summaries(gp_df: pd.DataFrame, lm_df: pd.DataFrame) -> pd.DataFrame:
    """Junta as métricas do DML+GP e do LinearDML lado a lado (colunas gp_* / lm_*)."""
    metric_cols = ["blp_est", "blp_se", "blp_pval",
                   "qini_est", "qini_pval",
                   "autoc_est", "autoc_pval",
                   "cal_r_squared"]
    out = pd.concat(
        [gp_df.set_index("treatment").add_prefix("gp_"),
         lm_df.set_index("treatment").add_prefix("lm_")],
        axis=1,
    )
    return out[[f"gp_{c}" for c in metric_cols] + [f"lm_{c}" for c in metric_cols if f"lm_{c}" in out.columns]]

    rows = []
    for name, df in summaries.items():
        df = df.copy()
        df.insert(0, "test", name)
        rows.append(df.reset_index() if df.index.name else df)
    summary_df = pd.concat(rows, ignore_index=True)
    return res, summary_df, tmt, {"train_idx": train_idx, "val_idx": val_idx, "D": D}
