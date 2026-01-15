#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make avg. transition strategy performance table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.reports.rel_to_benchmark_model_performane_table import mse_series, qlike_series
from src.reports.short_train_data_rel_to_benchmark_model_performance_table import (
    FileInfo,
    parse_file,
)


def model_label(info: FileInfo) -> str:
    """
    Generate label for a model configuration.
    """
    tl_map = {
        "target-only": "TO",
        "naive-pooling": "NP",
        "mtl-25": "MTL-25",
        "mtl-50": "MTL-50",
        "mtl-75": "MTL-75",
    }
    tl_tag = tl_map.get(info.tl, info.tl.upper())
    return f"{tl_tag} {info.base}-{info.pred_set.upper()}"


def fmt_value(x: float) -> str:
    """
    Format value.
    """
    if not np.isfinite(x):
        return "—"
    if x > 99:
        return r"$> 99$"
    return f"{x:.3f}"


def discover_prediction_files(
    results_dir: str, file_glob_patterns: List[str]
) -> List[Path]:
    """
    Discover prediction result files matching one or more glob patterns.
    """
    results_dir = Path(results_dir)
    paths = []
    for pat in file_glob_patterns:
        paths.extend(results_dir.glob(pat))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched in {results_dir}")
    return paths


def index_files_by_model(
    paths: List[Path],
) -> Dict[Tuple[str, int, int], Dict[str, Path]]:
    """
    Index files.
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
    other_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align y, f_model, f_other (other strategy) for loss computation.
    """
    y = model_df["actuals"].to_numpy(float)
    f = model_df["predictions"].to_numpy(float)

    y2 = other_df["actuals"].to_numpy(float)
    f2 = other_df["predictions"].to_numpy(float)

    if len(y) == len(y2):
        return y, f, f2

    if "date" in model_df.columns and "date" in other_df.columns:
        tmp = model_df.merge(other_df, on="date", suffixes=("", "_o"))
        if tmp.empty:
            return np.array([]), np.array([]), np.array([])
        y_a = tmp["actuals"].to_numpy(float)
        f_a = tmp["predictions"].to_numpy(float)
        f2_a = tmp["predictions_o"].to_numpy(float)
        return y_a, f_a, f2_a

    return np.array([]), np.array([]), np.array([])


def pooled_losses_for_strategy(
    by_model: Dict[Tuple[str, int, int], Dict[str, Path]],
    eval_to_spec: Dict[int, int],
    include_prefixes: Tuple[str, ...] = ("NP", "MTL-25", "MTL-50", "MTL-75"),
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Pool per-observation loss sums across eval periods for each
    (target, model) under a strategy.
    """
    targets = sorted({t for (t, _, _) in by_model.keys()})
    out = {}

    for t in targets:
        models_for_t = set()
        for (tt, e, _), mm in by_model.items():
            if tt == t:
                models_for_t.update(mm.keys())

        candidates = [
            m
            for m in models_for_t
            if any(m.startswith(pref + " ") for pref in include_prefixes)
        ]

        for mname in sorted(candidates):
            mse_sum = 0.0
            ql_sum = 0.0
            mse_n = 0
            ql_n = 0

            for e, spec in eval_to_spec.items():
                key = (t, e, spec)
                p = by_model.get(key, {}).get(mname)
                if p is None:
                    continue

                df = pd.read_csv(p)
                if not {"predictions", "actuals"}.issubset(df.columns):
                    continue

                y = df["actuals"].to_numpy(float)
                f = df["predictions"].to_numpy(float)

                lm = mse_series(y, f)
                m = np.isfinite(lm)
                if m.any():
                    mse_sum += float(np.nansum(lm[m]))
                    mse_n += int(m.sum())

                lq = qlike_series(y, f)
                mq = np.isfinite(lq)
                if mq.any():
                    ql_sum += float(np.nansum(lq[mq]))
                    ql_n += int(mq.sum())

            if mse_n > 0 and ql_n > 0:
                out[(t, mname)] = (mse_sum / mse_n, ql_sum / ql_n)

    return out


def strategy_relative_matrix(
    by_model: Dict[Tuple[str, int, int], Dict[str, Path]],
    strategies: Dict[str, Dict[int, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the matrix.
    """
    pooled = {}
    for strat_name, eval_to_spec in strategies.items():
        pooled[strat_name] = pooled_losses_for_strategy(by_model, eval_to_spec)

    names = list(strategies.keys())
    rel_mse = pd.DataFrame(index=names, columns=names, dtype=float)
    rel_ql = pd.DataFrame(index=names, columns=names, dtype=float)

    for r in names:
        for c in names:
            if r == c:
                rel_mse.loc[r, c] = 1.0
                rel_ql.loc[r, c] = 1.0
                continue

            ratios_mse = []
            ratios_ql = []

            common_keys = set(pooled[r].keys()) & set(pooled[c].keys())
            for k in common_keys:
                rm, rq = pooled[r][k]
                cm, cq = pooled[c][k]
                ratios_mse.append(cm / rm)
                ratios_ql.append(cq / rq)

            rel_mse.loc[r, c] = float(np.mean(ratios_mse)) if ratios_mse else np.nan
            rel_ql.loc[r, c] = float(np.mean(ratios_ql)) if ratios_ql else np.nan

    return rel_mse, rel_ql


def render_transition_matrix_latex(
    rel_mse: pd.DataFrame,
    rel_ql: pd.DataFrame,
    caption: str,
    label: str,
    scalebox: float = 0.8,
) -> str:
    """
    Render latex table.
    """
    names = list(rel_mse.index)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\centering")
    lines.append(rf"\scalebox{{{scalebox}}}{{")
    lines.append(r"\begin{tabular}[t]{p{2cm}|p{2cm}p{2cm}p{2cm}|p{2cm}p{2cm}p{2cm}}")
    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(
        r"& \multicolumn{3}{c}{relative MSE} |& \multicolumn{3}{c}{relative QLIKE}\\"
    )
    lines.append(r"\hline")

    header = "Strategy & " + " & ".join(names) + " & " + " & ".join(names) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for r in names:
        row_mse = [fmt_value(rel_mse.loc[r, c]) for c in names]
        row_ql = [fmt_value(rel_ql.loc[r, c]) for c in names]
        lines.append(r + " & " + " & ".join(row_mse + row_ql) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def create_avg_transition_strategy_table(results_dir: str, reports_dp: str) -> None:
    """
    Create table.
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

    strategies = {
        "Str. 1-1-1": {1: 1, 2: 1, 3: 1},
        "Str. 1-5-5": {1: 1, 2: 5, 3: 5},
        "Str. 1-5-22": {1: 1, 2: 5, 3: 22},
    }

    rel_mse, rel_ql = strategy_relative_matrix(by_model, strategies)

    caption = (
        "Performance of transition strategies. Reported are the averages of the MSEs "
        "(QLIKE losses) of the column strategy relative to the MSEs (QLIKE losses) "
        "of the row strategy. MSEs (QLIKE losses) are based on one-day-ahead realized "
        "variance forecasts following the first trading day of a new issue/spin-off. "
        "The averages are computed across all new issues and spin-offs, as well as "
        "across all NP and MTL models."
    )

    tex = render_transition_matrix_latex(
        rel_mse=rel_mse,
        rel_ql=rel_ql,
        caption=caption,
        label="tab:predictor_transitioning",
        scalebox=0.8,
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    (reports_dp / "avg_strategy_table.txt").write_text(tex, encoding="utf-8")
