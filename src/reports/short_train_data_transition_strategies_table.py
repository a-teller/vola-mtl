#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the "strategy" LaTeX tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.reports.rel_to_benchmark_model_performane_table import mse_series, qlike_series
from src.reports.short_train_data_rel_to_benchmark_model_performance_table import (
    FileInfo,
    parse_file,
)


def model_label(info: FileInfo) -> str:
    """
    Convert FileInfo into labels.
    """
    tl_map = {
        "target-only": "TO",
        "naive-pooling": "NP",
        "mtl-25": "MTL-25",
        "mtl-50": "MTL-50",
        "mtl-75": "MTL-75",
    }
    tl_tag = tl_map.get(info.tl, info.tl.upper())
    if info.base == "RW":
        return "RW"
    return f"{tl_tag} {info.base}-{info.pred_set.upper()}"


def family_from_model_label(model: str) -> str:
    """
    Extract base family.
    """
    if model == "RW":
        return "RW"
    return model.split()[1].split("-")[0]


def fmt_value(x: float) -> str:
    """Format a numeric value for LaTeX."""
    if not np.isfinite(x):
        return "—"
    if x > 99:
        return r"$> 99$"
    return f"{x:.3f}"


def bold(s: str) -> str:
    """
    Wrap LaTeX cell in bold.
    """
    return r"\textbf{" + s + "}"


def discover_prediction_files(
    results_dir: str, file_glob_patterns: List[str]
) -> List[Path]:
    """
    Discover prediction CSV files.
    """
    results_dir = Path(results_dir)

    paths = []
    for pattern in file_glob_patterns:
        paths.extend(results_dir.glob(pattern))

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched in {results_dir}")
    return paths


def index_files_by_model(
    paths: List[Path],
) -> Dict[Tuple[str, int, int], Dict[str, Path]]:
    """
    Build index by_model.
    """
    by_model = {}

    for p in paths:
        info = parse_file(p)
        key = (info.target, info.eval_period, info.spec)
        m = model_label(info)

        by_model.setdefault(key, {})
        by_model[key].setdefault(m, p)

    return by_model


def align_model_and_benchmark(
    model_df: pd.DataFrame,
    bm_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align y, f_model, f_bench for loss computation.
    """
    y = model_df["actuals"].to_numpy(float)
    f = model_df["predictions"].to_numpy(float)

    yb = bm_df["actuals"].to_numpy(float)
    fb = bm_df["predictions"].to_numpy(float)

    if len(y) == len(yb):
        return y, f, fb

    if "date" in model_df.columns and "date" in bm_df.columns:
        tmp = model_df.merge(bm_df, on="date", suffixes=("", "_bm"))
        if tmp.empty:
            return np.array([]), np.array([]), np.array([])
        y2 = tmp["actuals"].to_numpy(float)
        f2 = tmp["predictions"].to_numpy(float)
        fb2 = tmp["predictions_bm"].to_numpy(float)
        return y2, f2, fb2

    return np.array([]), np.array([]), np.array([])


def compute_strategy_relative_by_family_weighted(
    by_model: Dict[Tuple[str, int, int], Dict[str, Path]],
    eval_to_spec: Dict[int, int],
    approach_prefix: str,
    pred_set: str,
    benchmark_eval_to_spec: Dict[int, int],
    benchmark_model: str = "TO HAR-STD",
) -> Dict[str, Dict[str, float]]:
    """
    Compute cross-sectional average relative MSE and QLIKE by family
    for a transition strategy using pooled (length-weighted) losses.
    """
    families = ["HAR", "FNN", "XGB"]

    ratios = {fam: {"MSE": [], "QLIKE": []} for fam in families}

    targets = sorted({t for (t, _, _) in by_model.keys()})

    for t in targets:
        sum_model = {fam: {"MSE": 0.0, "QLIKE": 0.0} for fam in families}
        sum_bench = {fam: {"MSE": 0.0, "QLIKE": 0.0} for fam in families}
        n_used = {fam: {"MSE": 0, "QLIKE": 0} for fam in families}

        for e, spec in eval_to_spec.items():
            bm_spec = benchmark_eval_to_spec.get(e)
            bm_key = (t, e, bm_spec)
            bm_path = by_model.get(bm_key, {}).get(benchmark_model)
            bm_df = pd.read_csv(bm_path)
            key = (t, e, spec)
            model_map = by_model.get(key, {})

            for mname, mpath in model_map.items():
                if not mname.startswith(f"{approach_prefix} "):
                    continue
                if not mname.endswith(f"-{pred_set}"):
                    continue

                fam = family_from_model_label(mname)
                if fam not in families:
                    continue

                df = pd.read_csv(mpath)
                if not {"predictions", "actuals"}.issubset(df.columns):
                    continue

                y, f_model, f_bench = align_model_and_benchmark(df, bm_df)

                lm = mse_series(y, f_model)
                lb = mse_series(y, f_bench)
                mask = np.isfinite(lm) & np.isfinite(lb)
                if mask.any():
                    sum_model[fam]["MSE"] += float(np.nansum(lm[mask]))
                    sum_bench[fam]["MSE"] += float(np.nansum(lb[mask]))
                    n_used[fam]["MSE"] += int(mask.sum())

                qm = qlike_series(y, f_model)
                qb = qlike_series(y, f_bench)
                maskq = np.isfinite(qm) & np.isfinite(qb)
                if maskq.any():
                    sum_model[fam]["QLIKE"] += float(np.nansum(qm[maskq]))
                    sum_bench[fam]["QLIKE"] += float(np.nansum(qb[maskq]))
                    n_used[fam]["QLIKE"] += int(maskq.sum())

        for fam in families:
            if n_used[fam]["MSE"] > 0 and sum_bench[fam]["MSE"] > 0:
                ratios[fam]["MSE"].append(sum_model[fam]["MSE"] / sum_bench[fam]["MSE"])
            if n_used[fam]["QLIKE"] > 0 and sum_bench[fam]["QLIKE"] > 0:
                ratios[fam]["QLIKE"].append(
                    sum_model[fam]["QLIKE"] / sum_bench[fam]["QLIKE"]
                )

    return {
        fam: {
            "MSE": float(np.mean(ratios[fam]["MSE"])) if ratios[fam]["MSE"] else np.nan,
            "QLIKE": (
                float(np.mean(ratios[fam]["QLIKE"])) if ratios[fam]["QLIKE"] else np.nan
            ),
        }
        for fam in families
    }


def render_strategy_table(
    rows: List[Tuple[str, Dict[str, Dict[str, float]]]],
    caption: Optional[str] = None,
    label: Optional[str] = None,
    highlight_best: bool = True,
    scalebox: Optional[float] = None,
    add_table_env: bool = False,
    dash_after: Optional[set[str]] = None,
    hline_after: Optional[set[str]] = None,
) -> str:
    """
    Render the LaTeX table.
    """
    dash_after = dash_after or set()
    hline_after = hline_after or set()

    families = ["HAR", "FNN", "XGB"]
    metrics = ["MSE", "QLIKE"]

    cell_txt = {}
    cell_num = {}

    def txt_to_num(s: str) -> float:
        if s in ("—", r"$> 99$"):
            return np.inf
        try:
            return float(s)
        except ValueError:
            return np.inf

    for i, (_, vals) in enumerate(rows):
        for metric in metrics:
            for fam in families:
                x = vals.get(fam, {}).get(metric, np.nan)
                s = fmt_value(x)
                cell_txt[(i, metric, fam)] = s
                cell_num[(i, metric, fam)] = txt_to_num(s)

    best_idx = {}
    if highlight_best:
        for metric in metrics:
            for fam in families:
                candidates = [(i, cell_num[(i, metric, fam)]) for i in range(len(rows))]
                i_best, v_best = min(candidates, key=lambda z: z[1])
                best_idx[(metric, fam)] = None if not np.isfinite(v_best) else i_best

    lines = []

    if add_table_env:
        lines.append(r"\begin{table}[t]")
        if caption:
            lines.append(r"\caption{" + caption + "}")
        lines.append(r"\centering")
        if scalebox is not None:
            lines.append(rf"\scalebox{{{scalebox}}}{{")

    lines.append(
        r"\begin{tabular}[t]{p{5cm}|p{1.5cm}p{1.5cm}p{1.5cm}|p{1.5cm}p{1.5cm}p{1.5cm}}"
    )
    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(
        r"& \multicolumn{3}{c}{\makecell{MSE relative to \\ Str. 1-1-1 TO HAR-STD}} |"
        r"& \multicolumn{3}{c}{\makecell{QLIKE relative to \\ Str. 1-1-1 TO HAR-STD}}\\"
    )
    lines.append(r"\hline")
    lines.append(r"& HAR & FNN & XGB & HAR & FNN & XGB \\")
    lines.append(r"\hline")

    for i, (row_label, _) in enumerate(rows):
        mse_cells = {}
        ql_cells = {}

        for fam in families:
            s_mse = cell_txt[(i, "MSE", fam)]
            s_ql = cell_txt[(i, "QLIKE", fam)]

            if (
                highlight_best
                and best_idx.get(("MSE", fam)) == i
                and s_mse not in ("—", r"$> 99$")
            ):
                s_mse = bold(s_mse)
            if (
                highlight_best
                and best_idx.get(("QLIKE", fam)) == i
                and s_ql not in ("—", r"$> 99$")
            ):
                s_ql = bold(s_ql)

            mse_cells[fam] = s_mse
            ql_cells[fam] = s_ql

        lines.append(
            f"{row_label} & "
            f"{mse_cells['HAR']} & {mse_cells['FNN']} & {mse_cells['XGB']} & "
            f"{ql_cells['HAR']} & {ql_cells['FNN']} & {ql_cells['XGB']} \\\\"
        )

        if row_label in dash_after:
            lines.append(r"\hdashline")
        if row_label in hline_after:
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    if add_table_env:
        if scalebox is not None:
            lines.append(r"}")
        if label:
            lines.append(r"\label{" + label + "}")
        lines.append(r"\end{table}")
    else:
        if label:
            lines.append(r"\label{" + label + "}")

    return "\n".join(lines)


def create_strategy_table(
    results_dir: str,
    reports_dp: str,
    benchmark_model: str = "TO HAR-STD",
) -> None:
    """
    Build the strategy table.
    """
    reports_dp = Path(reports_dp)
    file_glob_patterns = [
        "*har*1_*_1_predictions_1*",
        "*har*1_*_1_predictions_2*",
        "*har*1_*_1_predictions_3*",
        "*har*5_*_1_predictions_2*",
        "*har*5_*_1_predictions_3*",
        "*har*22_*_1_predictions_3*",
        "*xgboost*1_*_1_predictions_1*",
        "*xgboost*1_*_1_predictions_2*",
        "*xgboost*1_*_1_predictions_3*",
        "*xgboost*5_*_1_predictions_2*",
        "*xgboost*5_*_1_predictions_3*",
        "*xgboost*22_*_1_predictions_3*",
        "*feedforward*1_*_1_predictions_1*",
        "*feedforward*1_*_1_predictions_2*",
        "*feedforward*1_*_1_predictions_3*",
        "*feedforward*5_*_1_predictions_2*",
        "*feedforward*5_*_1_predictions_3*",
        "*feedforward*22_*_1_predictions_3*",
    ]

    paths = discover_prediction_files(results_dir, file_glob_patterns)
    by_model = index_files_by_model(paths)

    benchmark_eval_to_spec = {1: 1, 2: 1, 3: 1}

    strategy_defs: List[Tuple[str, Dict[int, int], str, str]] = [
        ("Str. 1-1-1 TO-STD", {1: 1, 2: 1, 3: 1}, "TO", "STD"),
        ("Str. 1-1-1 TO-EXT", {1: 1, 2: 1, 3: 1}, "TO", "EXT"),
        ("Str. 1-5-5 TO-STD", {1: 1, 2: 5, 3: 5}, "TO", "STD"),
        ("Str. 1-5-5 TO-EXT", {1: 1, 2: 5, 3: 5}, "TO", "EXT"),
        ("Str. 1-5-22 TO-STD", {1: 1, 2: 5, 3: 22}, "TO", "STD"),
        ("Str. 1-5-22 TO-EXT", {1: 1, 2: 5, 3: 22}, "TO", "EXT"),
        ("Str. 1-5-22 NP-STD", {1: 1, 2: 5, 3: 22}, "NP", "STD"),
        ("Str. 1-5-22 NP-EXT", {1: 1, 2: 5, 3: 22}, "NP", "EXT"),
        ("Str. 1-5-22 MTL-25-STD", {1: 1, 2: 5, 3: 22}, "MTL-25", "STD"),
        ("Str. 1-5-22 MTL-25-EXT", {1: 1, 2: 5, 3: 22}, "MTL-25", "EXT"),
        ("Str. 1-5-22 MTL-50-STD", {1: 1, 2: 5, 3: 22}, "MTL-50", "STD"),
        ("Str. 1-5-22 MTL-50-EXT", {1: 1, 2: 5, 3: 22}, "MTL-50", "EXT"),
        ("Str. 1-5-22 MTL-75-STD", {1: 1, 2: 5, 3: 22}, "MTL-75", "STD"),
        ("Str. 1-5-22 MTL-75-EXT", {1: 1, 2: 5, 3: 22}, "MTL-75", "EXT"),
    ]

    rows = []
    for label, eval_to_spec, approach, pred_set in strategy_defs:
        vals = compute_strategy_relative_by_family_weighted(
            by_model=by_model,
            eval_to_spec=eval_to_spec,
            approach_prefix=approach,
            pred_set=pred_set,
            benchmark_eval_to_spec=benchmark_eval_to_spec,
            benchmark_model=benchmark_model,
        )
        rows.append((label, vals))

    caption = (
        "Performance of transition strategies for TO models compared to the best-performing "
        "MTL and NP model strategy (Str. 1-5-22). Reported are the cross-sectional average "
        "MSEs (QLIKE losses) for each forecasting model (HAR, FNN, XGB) under different "
        "transition strategies, predictor sets (STD, EXT), and forecasting approaches "
        "(TO, NP, MTL) relative to the Str. 1-1-1 TO HAR-STD. The MSEs (QLIKE losses) are "
        "based on one-day-ahead realized variance forecasts pooled over the evaluation "
        "periods included in each strategy. “—” denotes not applicable. The best-performing "
        'model within each model class is marked in bold. "$>$ 99" represents values that '
        "exceed 99."
    )

    dash_after = {
        "Str. 1-1-1 TO-EXT",
        "Str. 1-5-5 TO-EXT",
        "Str. 1-5-22 TO-EXT",
        "Str. 1-5-22 MTL-25-EXT",
        "Str. 1-5-22 MTL-50-EXT",
    }
    hline_after = {
        "Str. 1-5-22 TO-EXT",
        "Str. 1-5-22 NP-EXT",
    }

    tex = render_strategy_table(
        rows=rows,
        caption=caption,
        label="tab:strat_forecast_results",
        highlight_best=True,
        scalebox=0.8,
        add_table_env=True,
        dash_after=dash_after,
        hline_after=hline_after,
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    (reports_dp / "strategy_table.txt").write_text(tex, encoding="utf-8")
