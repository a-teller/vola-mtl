#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX table comparing forecasting model vs baseline model.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.reports.one_vs_one_model_performance_tables import parse_result_filename
from src.reports.one_vs_one_model_performance_tables import load_predictions_csv


def mse_series(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute Mean Squared Error of two series.
    """
    e = y - f
    return e * e


def qlike_series(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute QLike loss of two series.
    """
    r = y / f
    return r - np.log(r) - 1.0


def load_all_predictions(
    results_dir: Path,
    file_glob_patterns: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load pediction results across targets and models.
    """
    paths = []
    for pattern in file_glob_patterns:
        paths.extend(Path(results_dir).glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No files matched in {results_dir}")

    predictions_by_target = {}

    for p in paths:
        target, spec = parse_result_filename(p)
        df = load_predictions_csv(p)
        y = df["actuals"].to_numpy(dtype=float)
        f = df["predictions"].to_numpy(dtype=float)
        model_name = spec.prettify_name()

        predictions_by_target.setdefault(target, {"actuals": None, "preds": {}})
        if predictions_by_target[target]["actuals"] is None:
            predictions_by_target[target]["actuals"] = y
        else:
            yref = predictions_by_target[target]["actuals"]
            if len(yref) != len(y) or not np.allclose(yref, y, equal_nan=True):
                raise ValueError(
                    f"Actuals mismatch within target {target} for model {model_name}"
                )
        predictions_by_target[target]["preds"][model_name] = f

    return predictions_by_target


def base_class(model_name: str) -> str:
    """
    Map model identifier to its base model class.
    """
    if model_name == "RW":
        return "RW"
    if " HAR-" in model_name:
        return "HAR"
    if " FNN-" in model_name:
        return "FNN"
    if " XGB-" in model_name:
        return "XGB"
    return "OTHER"


def window_mean_loss(
    y: np.ndarray, f: np.ndarray, s: int, horizon: int, metric: str
) -> float:
    """
    Compute the mean forecast loss over a fixed evaluation window.
    """
    y_w = y[s : s + horizon]
    f_w = f[s : s + horizon]
    if metric == "MSE":
        return float(np.mean(mse_series(y_w, f_w)))
    elif metric == "QLIKE":
        return float(np.mean(qlike_series(y_w, f_w)))
    else:
        raise ValueError("metric must be MSE or QLIKE")


def compute_relative_table_values(
    data: Dict[str, Dict[str, np.ndarray]],
    eval_list: List[int],
    horizon: int,
    benchmark: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-target average relative loss tables at multiple start points.
    """
    targets = sorted(data.keys())
    models = sorted({m for t in targets for m in data[t]["preds"].keys()})

    missing = [t for t in targets if benchmark not in data[t]["preds"]]
    if missing:
        raise ValueError(
            f"Benchmark '{benchmark}' missing for targets: "
            f"{missing[:5]}{'...' if len(missing)>5 else ''}"
        )

    rel_mse = pd.DataFrame(
        index=models, columns=[f"s = {s}" for s in eval_list], dtype=float
    )
    rel_ql = pd.DataFrame(
        index=models, columns=[f"s = {s}" for s in eval_list], dtype=float
    )
    for s in eval_list:
        col = f"s = {s}"
        for m in models:
            mse_ratios = []
            ql_ratios = []
            for t in targets:
                y = data[t]["actuals"]
                preds = data[t]["preds"]
                if m not in preds:
                    continue

                bm = preds[benchmark]
                fm = preds[m]

                bm_mse = window_mean_loss(y, bm, s, horizon, "MSE")
                fm_mse = window_mean_loss(y, fm, s, horizon, "MSE")
                bm_ql = window_mean_loss(y, bm, s, horizon, "QLIKE")
                fm_ql = window_mean_loss(y, fm, s, horizon, "QLIKE")

                mse_ratios.append(fm_mse / bm_mse)
                ql_ratios.append(fm_ql / bm_ql)

            rel_mse.loc[m, col] = float(np.mean(mse_ratios)) if mse_ratios else np.nan
            rel_ql.loc[m, col] = float(np.mean(ql_ratios)) if ql_ratios else np.nan

    return rel_mse, rel_ql


def fmt_val(x: float) -> str:
    """
    Format value.
    """
    if x > 99:
        return r"$>$ 99"
    return f"{x:.3f}"


def blue(s: str) -> str:
    """
    Apply blue coloring.
    """
    return r"\textcolor{blue}{" + s + "}"


def bold_blue(s: str) -> str:
    """
    Format bold and apply blue coloring.
    """
    return r"\textbf{\textcolor{blue}{" + s + "}}"


def render_relative_table_latex(
    rel_mse: pd.DataFrame,
    rel_ql: pd.DataFrame,
    model_order: List[str],
    eval_list: List[int],
    caption: str,
    label: str,
    benchmark: str,
) -> str:
    """
    Render a LaTeX table of relative forecast losses across evaluation windows.
    """
    cols = [f"s = {s}" for s in eval_list]

    classes = ["HAR", "FNN", "XGB"]

    competitors = [m for m in model_order if m in rel_mse.index and m != benchmark]

    best_class_mse = {(c, col): None for c in classes for col in cols}
    best_class_ql = {(c, col): None for c in classes for col in cols}
    best_overall_mse = {col: None for col in cols}
    best_overall_ql = {col: None for col in cols}

    for col in cols:
        vals = [
            (m, rel_mse.loc[m, col])
            for m in competitors
            if np.isfinite(rel_mse.loc[m, col])
        ]
        if vals:
            best_overall_mse[col] = min(vals, key=lambda z: z[1])[0]
        vals = [
            (m, rel_ql.loc[m, col])
            for m in competitors
            if np.isfinite(rel_ql.loc[m, col])
        ]
        if vals:
            best_overall_ql[col] = min(vals, key=lambda z: z[1])[0]

        for c in classes:
            mset = [m for m in competitors if base_class(m) == c]
            vals = [
                (m, rel_mse.loc[m, col])
                for m in mset
                if np.isfinite(rel_mse.loc[m, col])
            ]
            if vals:
                best_class_mse[(c, col)] = min(vals, key=lambda z: z[1])[0]
            vals = [
                (m, rel_ql.loc[m, col]) for m in mset if np.isfinite(rel_ql.loc[m, col])
            ]
            if vals:
                best_class_ql[(c, col)] = min(vals, key=lambda z: z[1])[0]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\scalebox{0.66}{")
    colspec = "p{4.0cm}" + "p{1.75cm}" * (2 * len(eval_list))
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\hline")
    lines.append(r"\hline")
    header1 = (
        r"\multicolumn{1}{c}{Model} & "
        + " & ".join([rf"\multicolumn{{2}}{{c}}{{s = {s}}}" for s in eval_list])
        + r"\\"
    )
    header2 = " & " + " & ".join(["MSE & QLIKE" for _ in eval_list]) + r"\\"
    lines.append(header1)
    lines.append(header2)
    lines.append(r"\hline")

    for m in model_order:
        if m not in rel_mse.index:
            continue
        row = [m]
        for s in eval_list:
            col = f"s = {s}"

            x = rel_mse.loc[m, col]
            sx = fmt_val(x)

            y = rel_ql.loc[m, col]
            sy = fmt_val(y)

            mc = base_class(m)
            if m == best_overall_mse[col] and sx not in ["—", r"$>$ 99"]:
                sx = bold_blue(sx)
            elif (
                mc in classes
                and m == best_class_mse[(mc, col)]
                and sx not in ["—", r"$>$ 99"]
            ):
                sx = blue(sx)

            if m == best_overall_ql[col] and sy not in ["—", r"$>$ 99"]:
                sy = bold_blue(sy)
            elif (
                mc in classes
                and m == best_class_ql[(mc, col)]
                and sy not in ["—", r"$>$ 99"]
            ):
                sy = blue(sy)

            row.extend([sx, sy])

        lines.append(" & ".join(row) + r"\\")
        if m == "RW":
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def create_rel_to_benchmark_model_perfomance_table(
    results_dp: str, reports_dp: str, complete_model_order: List[str], benchmark: str
):
    """
    Generate a LaTeX table of model performance relative to a benchmark model.
    """

    results_dp = Path(results_dp)
    reports_dp = Path(reports_dp)

    eval_list = [0, 100, 200, 300, 400]
    eval_horizon = 100

    complete_model_order.remove(benchmark)

    file_glob_patterns = [
        "*random-walk*50_*_5_predictions_complete*",
        "*har*50_std_5_predictions_complete*",
        "*xgboost*50_std_5_predictions_complete*",
        "*feedforward-neural-network*50_std_5_predictions_complete*",
        "*har*50_ext_5_predictions_complete*",
        "*xgboost*50_ext_5_predictions_complete*",
        "*feedforward-neural-network*50_ext_5_predictions_complete*",
    ]
    data = load_all_predictions(results_dp, file_glob_patterns)

    rel_mse, rel_ql = compute_relative_table_values(
        data=data,
        eval_list=eval_list,
        horizon=eval_horizon,
        benchmark=benchmark,
    )

    caption = (
        "MSEs and QLIKE losses of each forecasting model relative to the MSE (QLIKE) of "
        "the TO HAR-STD model, averaged over all new issues and spin-offs considered. "
        "The MSE and QLIKE metrics are obtained for 100 rolling 1-day-ahead forecasts "
        "based on $s=50$, $150$, $250$, $350$, and $450$ trading days after the "
        "distribution day of the new issue/spin-off. Blue values indicate, for each "
        "sample period ($s$) and error metric, the best-performing model within each "
        "models class (HAR, FNN, XGB). The overall best-performing model is 'highlighted "
        'in bold blue. "$>$ 99" represents values that exceed 99.'
    )

    latex_table = render_relative_table_latex(
        rel_mse=rel_mse,
        rel_ql=rel_ql,
        model_order=complete_model_order,
        eval_list=eval_list,
        caption=caption,
        label="tab:forecasting_results_rel_toharstd",
        benchmark=benchmark,
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    with (reports_dp / "rel_to_benchmark_model_performance.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(latex_table)
