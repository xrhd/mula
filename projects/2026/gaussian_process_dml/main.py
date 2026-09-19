"""DML + Gaussian Process: efeito causal com bandas de incerteza.

Exemplos:
    uv run main.py                          # IHDP surface B (dataset real)
    uv run main.py --dataset ihdp-a         # IHDP surface A (efeito constante)
    uv run main.py --dataset sim            # DGP simulado original
    uv run main.py --plot                   # salva PNG e abre janela
"""

import argparse
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gp_dml.data import load_dataset
from gp_dml.modeling import fit_dml, fit_linear_dml
from gp_dml.uncertainty import GPScoreExtractor
from gp_dml.validation import compare_summaries, make_split, validate_dml

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["sim", "ihdp-a", "ihdp-b"], default="ihdp-b")
    p.add_argument("--n-samples", type=int, default=800, help="só para --dataset sim")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kernel", choices=["auto", "default", "sim", "smooth"], default="auto",
                   help="kernel do GP final ('auto' usa preset por dataset)")
    p.add_argument("--plot", action="store_true", help="abrir janela do matplotlib no fim")
    return p.parse_args(argv)


def param_grid_structure(data: dict) -> dict:
    """Define X_test e efeito de referência conforme o dataset."""
    if data["name"] == "sim":
        X_test = np.linspace(data["X"].min(), data["X"].max(), 100).reshape(-1, 1)
        true_te_grid = data["true_te_fn"](X_test)
        return {"X_test": X_test, "true_te_grid": true_te_grid, "X_eval": data["X"]}
    X_eval = data["X"]
    true_te = data["true_te"]
    order = np.argsort(true_te)
    return {"X_test": X_eval[order], "true_te_grid": true_te[order], "X_eval": X_eval}


def plot_results(data: dict, grid: dict, cate, lower, upper, bench_cate=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if data["name"] == "sim":
        x = grid["X_test"][:, 0]
        ax.plot(x, grid["true_te_grid"], "r--", label="Efeito Real Oculto")
        ax.plot(x, cate, "b-", label="DML + GP")
        ax.fill_between(x, lower, upper, color="blue", alpha=0.2, label="Confiança 95% (GP)")
        ax.set_xlabel("Feature X")
    else:
        x = np.arange(len(cate))
        ax.plot(x, grid["true_te_grid"], "r--", lw=1.2, label="Efeito Real (true_TE)")
        ax.plot(x, cate, "b-", lw=1.2, label="DML + GP")
        ax.fill_between(x, lower, upper, color="blue", alpha=0.2, label="Confiança 95% (GP)")
        ax.set_xlabel("Amostras de IHDP ordenadas por efeito real")
    if bench_cate is not None:
        ax.plot(x, bench_cate, "g", lw=1.2, ls="--", label="LinearDML (benchmark)")
    ax.set_title("Efeito Causal (DML + GP) com Bandas de Incerteza")
    ax.set_ylabel("Efeito de T em Y")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = RESULTS_DIR / args.dataset
    print(f"[1/6] Carregando dataset '{args.dataset}'...")
    data = load_dataset(args.dataset, args.n_samples, args.seed)
    print(f"      n={data['X'].shape[0]}, d_x={data['X'].shape[1]}, "
          f"unique(T)={np.unique(data['T']).size}")

    print("[2/6] Split treino/validação (validação honesta do DRTester)...")
    train_idx, val_idx, D_split = make_split(data, seed=args.seed)
    splits = {"train_idx": train_idx, "val_idx": val_idx, "D": D_split}
    print(f"      n_train={len(train_idx)}, n_val={len(val_idx)}")
    data_train = {**data, "Y": np.asarray(data["Y"])[train_idx],
                  "T": np.asarray(data["T"])[train_idx], "X": data["X"][train_idx]}

    print("[3/6] Treinando DML+GP e LinearDML (mesmos nuisances LGBM)...")
    dml = fit_dml(data_train, args.seed, args.kernel)
    print(f"      kernel ajustado: {dml.model_final_.kernel_}")
    lm = fit_linear_dml(data_train, args.seed)

    print("[4/6] Extraindo score de confiança (GPScoreExtractor)...")
    scorer = GPScoreExtractor(dml)
    grid = param_grid_structure(data)
    cate, std = scorer.effect_and_std(grid["X_test"])
    lower, upper = scorer.confidence_band(grid["X_test"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bench_cate = np.ravel(lm.effect(grid["X_test"], T0=0, T1=1))

    if "true_te" in data:
        metrics = scorer.coverage(data["X"][val_idx], data["true_te"][val_idx])
        print(f"      RMSE(CATE vs real, val): {metrics['rmse']:.3f}")
        print(f"      cobertura banda {int(metrics['level']*100)}% (val): {metrics['coverage']:.1%}")

    print("[5/6] Validando causalmente (DRTester: DML+GP vs LinearDML)...")
    res, summary_df, tmt, _ = validate_dml(data, dml, args.seed, splits=splits)
    _, summary_lm_df, _, _ = validate_dml(data, lm, args.seed, splits=splits)
    comp = compare_summaries(summary_df, summary_lm_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    comp.to_csv(out_dir / "comparison.csv")
    print(_indent(comp.round(3).to_string(), "      "))
    print(f"      summaries: {out_dir}/summary.csv, comparison.csv")

    print("[6/6] Gerando figuras...")
    fig = plot_results(data, grid, cate, lower, upper, bench_cate)
    effect_path = out_dir / "effect_bands.png"
    fig.savefig(effect_path, dpi=150, bbox_inches="tight")
    print("      figura salva em " + str(effect_path))
    for name, fig_val in (("cal", res.plot_cal(tmt=tmt)),
                          ("qini", res.plot_qini(tmt=tmt)),
                          ("toc", res.plot_toc(tmt=tmt))):
        path = out_dir / f"{name}.png"
        fig_val.figure.savefig(path, dpi=150, bbox_inches="tight")
        print(f"      figura salva em {path}")

    if args.plot:
        plt.show()
    print("Concluído!")
    return 0


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
