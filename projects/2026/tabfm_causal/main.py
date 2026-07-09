"""Compare the R-learner with LinearRegression, XGBoost and TabFM as the
base (effect) learner, following the causalml meta-learners synthetic-data
example.

Reference:
https://causalml.readthedocs.io/en/latest/examples/meta_learners_with_synthetic_data.html

Design note: the R-learner cross-fits the *outcome* model with sklearn's
`cross_val_predict`, which clones/deepcopies the estimator. TabFM's JAX model
does not survive that machinery, so we keep the outcome model fixed
(LinearRegression) and only swap the *effect* (tau) learner
{LinearRegression, XGBoost, TabFM} -- which is exactly the base learner we
want to compare. The true propensity `e` is passed in.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from causalml.dataset import synthetic_data
from causalml.inference.meta import BaseRRegressor

from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")


class TabFMAdapter:
  """sklearn-compatible wrapper that feeds TabFM a DataFrame and batches
  predictions to bound memory. The underlying TabFM model is shared on
  `deepcopy` (it is not stateful between fit/predict)."""

  def __init__(self, regressor):
    object.__setattr__(self, "_reg", regressor)

  def fit(self, X, y, sample_weight=None):
    self._reg.fit(self._to_df(X), y)
    return self

  def predict(self, X):
    df = self._to_df(X)
    out = []
    for i in range(0, len(df), 128):
      out.append(np.asarray(self._reg.predict(df.iloc[i : i + 128])).reshape(-1))
    return np.concatenate(out)

  def _to_df(self, X):
    if isinstance(X, pd.DataFrame):
      return X
    X = np.asarray(X)
    cols = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)

  def get_params(self, deep=True):
    return {"regressor": self._reg}

  def set_params(self, **params):
    if "regressor" in params:
      object.__setattr__(self, "_reg", params["regressor"])
    return self

  def __deepcopy__(self, memo):
    # Share the (stateless-after-fit) model; avoid copying the JAX checkpoint.
    return TabFMAdapter(self._reg)

  def __getattr__(self, name):
    reg = object.__getattribute__(self, "_reg")
    return getattr(reg, name)


def make_tabfm_learner():
  model = tabfm_v1_0_0.load(model_type="regression")
  reg = TabFMRegressor(
      model=model,
      n_estimators=2,
      batch_size=16,
      max_num_rows=200,
      random_state=42,
      verbose=False,
  )
  return TabFMAdapter(reg)


def main():
  y, X, treatment, tau, b, e = synthetic_data(mode=1, n=1000, p=8, sigma=1.0)
  true_ate = float(tau.mean())

  outcome_learner = LinearRegression()
  effect_learners = {
      "R-Learner (LR)": LinearRegression(),
      "R-Learner (XGB)": XGBRegressor(),
      "R-Learner (TabFM)": make_tabfm_learner(),
  }

  results = {}
  print(f"True ATE: {true_ate:.4f}\n")
  print(f"{'Learner':<24} {'ATE':>8} {'MSE':>10}")
  print("-" * 44)

  for name, effect_learner in effect_learners.items():
    print(f"Training {name} ...", flush=True)
    learner = BaseRRegressor(
        outcome_learner=outcome_learner,
        effect_learner=effect_learner,
        n_fold=5,
        random_state=42,
    )
    cate = learner.fit_predict(X=X, treatment=treatment, y=y, p=e)
    cate = np.asarray(cate).reshape(-1)
    ate = float(np.mean(cate))
    mse = float(np.mean((tau - cate) ** 2))
    results[name] = cate
    print(f"{name:<24} {ate:>8.4f} {mse:>10.4f}", flush=True)

  plt.figure(figsize=(10, 6))
  for name, cate in results.items():
    plt.hist(cate, alpha=0.4, bins=30, label=name)
  plt.vlines(
      true_ate,
      0,
      plt.gca().get_ylim()[1],
      label="True ATE",
      linestyles="dashed",
      colors="black",
      linewidth=2,
  )
  plt.title("R-Learner CATE predictions by base (effect) learner")
  plt.xlabel("Individual Treatment Effect (CATE)")
  plt.ylabel("# of Samples")
  plt.legend()
  plt.tight_layout()
  out_path = "r_learner_comparison.png"
  plt.savefig(out_path, dpi=120)
  print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
  main()