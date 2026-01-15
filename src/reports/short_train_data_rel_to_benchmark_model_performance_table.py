#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing and rendering relative forecast performance table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.reports.rel_to_benchmark_model_performane_table import mse_series, qlike_series


@dataclass(frozen=True)
class FileInfo:
    """
    Container for metadata parsed from a prediction result
    filename.
    """

    target: str
    base: str
    tl: str
    spec: int
    pred_set: str
    eval_period: int


def parse_file(p: Path) -> FileInfo:
    """
    Parse filenames.
    """
    parts = p.stem.split("_")
    if len(parts) < 8 or parts[-2] != "predictions":
        raise ValueError(f"Unexpected filename format: {p.name}")

    target = parts[0]
    base_raw = parts[1].lower()
    tl = parts[2]
    spec = int(parts[3])
    pred_set = parts[4].lower()
    eval_period = int(parts[-1])

    base = {
        "har": "HAR",
        "xgboost": "XGB",
        "feedforward-neural-network": "FNN",
        "random-walk": "RW",
    }.get(base_raw, base_raw.upper())

    return FileInfo(
        target=target,
        base=base,
        tl=tl,
        spec=spec,
        pred_set=pred_set,
        eval_period=eval_period,
    )


def spec_label(spec: int) -> str:
    """
    Map spec integers to Q labels used in the LaTeX table.
    """
    return {1: "Q1", 5: "Q5", 22: "Q22"}.get(spec, f"Q{spec}")


def prettify_model(base: str, tl: str, pred_set: str) -> str:
    """
    Convert parsed (base, tl, pred_set) into the row label
    used in the table.
    """
    tl_map = {
        "target-only": "TO",
        "naive-pooling": "NP",
        "mtl-25": "MTL-25",
        "mtl-50": "MTL-50",
        "mtl-75": "MTL-75",
    }
    tl_tag = tl_map.get(tl, tl.upper())

    if base == "RW":
        return "RW"
    return f"{tl_tag} {base}-{pred_set.upper()}"


def compute_relative_tables(
    results_dir: str,
    file_glob_patterns: List[str],
    eval_to_s: Dict[int, int],
    specs: List[int],
    eval_periods: List[int],
    benchmark_model: str = "TO HAR-STD",
    benchmark_spec: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional average relative MSE and QLIKE.
    """
    results_dir = Path(results_dir)
    paths = []
    for pattern in file_glob_patterns:
        paths.extend(Path(results_dir).glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No files matched in {results_dir}")

    losses = {}

    for p in paths:
        info = parse_file(p)
        model_name = prettify_model(info.base, info.tl, info.pred_set)

        df = pd.read_csv(p)
        y = df["actuals"].to_numpy(float)
        f = df["predictions"].to_numpy(float)

        key = (info.target, info.eval_period, info.spec)
        losses.setdefault(key, {})
        losses[key][model_name] = (
            float(np.mean(mse_series(y, f))),
            float(np.mean(qlike_series(y, f))),
        )

    targets = sorted({t for (t, _, _) in losses.keys()})
    models = sorted({m for d in losses.values() for m in d.keys()})

    if eval_to_s is None:
        eval_to_s = {e: e for e in eval_periods}

    cols = pd.MultiIndex.from_tuples(
        [(eval_to_s[e], spec_label(q)) for e in eval_periods for q in specs],
        names=["s", "spec"],
    )
    rel_mse = pd.DataFrame(index=models, columns=cols, dtype=float)
    rel_ql = pd.DataFrame(index=models, columns=cols, dtype=float)

    acc_mse = {}
    acc_ql = {}

    for t in targets:
        for e in eval_periods:
            bm = losses.get((t, e, benchmark_spec), {}).get(benchmark_model)
            bm_mse, bm_ql = bm

            s_val = eval_to_s[e]

            for q in specs:
                qlab = spec_label(q)
                model_map = losses.get((t, e, q), {})
                for model_name, (mse_v, ql_v) in model_map.items():
                    acc_mse.setdefault((model_name, s_val, qlab), []).append(
                        mse_v / bm_mse
                    )
                    acc_ql.setdefault((model_name, s_val, qlab), []).append(
                        ql_v / bm_ql
                    )

    for (m, s_val, qlab), vals in acc_mse.items():
        rel_mse.loc[m, (s_val, qlab)] = float(np.mean(vals)) if vals else np.nan
    for (m, s_val, qlab), vals in acc_ql.items():
        rel_ql.loc[m, (s_val, qlab)] = float(np.mean(vals)) if vals else np.nan

    return rel_mse, rel_ql


def fmt_value(x: float) -> str:
    """
    Format value.
    """
    if not np.isfinite(x):
        return "—"
    if x > 99:
        return r"$> 99$"
    return f"{x:.3f}"


def blue(s: str) -> str:
    """
    Format blue.
    """
    return r"\textcolor{blue}{" + s + "}"


def bold_blue(s: str) -> str:
    """
    Format bold blue.
    """
    return r"\textbf{\textcolor{blue}{" + s + "}}"


def render_all_in_one_table(
    rel_mse: pd.DataFrame,
    rel_ql: pd.DataFrame,
    model_order: List[str],
    caption: str,
    label: str,
    s_values: List[int],
    q_specs: List[str],
) -> str:
    """
    Render the LaTeX layout.
    """
    best_within = {}
    best_overall = {}

    def _best_model_for(table: pd.DataFrame, s: int, spec: str) -> Optional[str]:
        best_m, best_v = None, np.inf
        for m in model_order:
            if m not in table.index or (s, spec) not in table.columns:
                continue
            v = table.loc[m, (s, spec)]
            if np.isfinite(v) and v < best_v:
                best_v, best_m = v, m
        return best_m

    def _best_model_overall(table: pd.DataFrame, s: int) -> Optional[str]:
        best_m, best_v = None, np.inf
        for m in model_order:
            if m not in table.index:
                continue
            for spec in q_specs:
                if (s, spec) not in table.columns:
                    continue
                v = table.loc[m, (s, spec)]
                if np.isfinite(v) and v < best_v:
                    best_v, best_m = v, m
        return best_m

    for s in s_values:
        best_overall[(s, "MSE")] = _best_model_overall(rel_mse, s)
        best_overall[(s, "QLIKE")] = _best_model_overall(rel_ql, s)
        for spec in q_specs:
            best_within[(s, "MSE", spec)] = _best_model_for(rel_mse, s, spec)
            best_within[(s, "QLIKE", spec)] = _best_model_for(rel_ql, s, spec)

    lines = []
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\scalebox{0.63}{")
    lines.append(r"\begin{tabular}{l *{18}{c}}")
    lines.append(r"\hline")
    lines.append(r"\hline")

    lines.append(
        r"& "
        + " & ".join(
            [rf"\multicolumn{{6}}{{c}}{{\textbf{{$s={s}$}}}}" for s in s_values]
        )
        + r"\\"
    )
    lines.append(r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-19}")
    lines.append(
        r"& "
        + " & ".join(
            [r"\multicolumn{3}{c}{MSE} & \multicolumn{3}{c}{QLIKE}" for _ in s_values]
        )
        + r"\\"
    )
    lines.append(
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"
        r"\cmidrule(lr){8-10}\cmidrule(lr){11-13}"
        r"\cmidrule(lr){14-16}\cmidrule(lr){17-19}"
    )
    lines.append(
        r"\diagbox{\textbf{Model}}{\textbf{Spec.}} & "
        + " & ".join(
            [
                r"\textbf{Q1} & \textbf{Q5} & \textbf{Q22} & \textbf{Q1} & "
                r"\textbf{Q5} & \textbf{Q22}"
            ]
            * len(s_values)
        )
        + r"\\"
    )
    lines.append(r"\hline")

    for m in model_order:
        if m not in rel_mse.index and m not in rel_ql.index:
            continue
        row = [m]
        for s in s_values:
            for spec in q_specs:
                x = (
                    rel_mse.loc[m, (s, spec)]
                    if (m in rel_mse.index and (s, spec) in rel_mse.columns)
                    else np.nan
                )
                sx = fmt_value(x)
                if sx not in ["—", r"$> 99$"]:
                    if m == best_overall[(s, "MSE")]:
                        sx = bold_blue(sx)
                    elif m == best_within[(s, "MSE", spec)]:
                        sx = blue(sx)
                row.append(sx)

            for spec in q_specs:
                y = (
                    rel_ql.loc[m, (s, spec)]
                    if (m in rel_ql.index and (s, spec) in rel_ql.columns)
                    else np.nan
                )
                sy = fmt_value(y)
                if sy not in ["—", r"$> 99$"]:
                    if m == best_overall[(s, "QLIKE")]:
                        sy = bold_blue(sy)
                    elif m == best_within[(s, "QLIKE", spec)]:
                        sy = blue(sy)
                row.append(sy)

        lines.append(" & ".join(row) + r" \\")
        if m == "RW":
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_table_from_dir(
    results_dir: str,
    file_glob_patterns: List[str],
    complete_model_order: List[str],
    caption: str,
    label: str = "tab:all-in-one",
    benchmark_model: str = "TO HAR-STD",
) -> None:
    """
    Compute relative MSE/QLIKE tables and write the rendered LaTeX table.
    """
    eval_to_s = {1: 1, 2: 5, 3: 22}

    rel_mse, rel_ql = compute_relative_tables(
        results_dir=results_dir,
        file_glob_patterns=file_glob_patterns,
        eval_to_s=eval_to_s,
        specs=[1, 5, 22],
        eval_periods=[1, 2, 3],
        benchmark_model=benchmark_model,
        benchmark_spec=1,
    )

    tex = render_all_in_one_table(
        rel_mse=rel_mse,
        rel_ql=rel_ql,
        model_order=complete_model_order,
        caption=caption,
        label=label,
        s_values=[1, 5, 22],
        q_specs=["Q1", "Q5", "Q22"],
    )
    return tex


def short_rel_to_benchmark_model_perfomance_table(
    results_dp: str, reports_dp: str, complete_model_order: List[str]
) -> None:
    """
    Generate and save a LaTeX table summarizing relative forecast performance
    across models and specifications.
    """
    results_dp = Path(results_dp)
    reports_dp = Path(reports_dp)

    file_glob_patterns = [
        "*random-walk*1_*_1_predictions_1*",
        "*random-walk*1_*_1_predictions_2*",
        "*random-walk*1_*_1_predictions_3*",
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

    caption = (
        r"1-day-ahead cross-sectional average relative MSEs and QLIKE losses by model, "
        r"sample period $s\in\{1,5,22\}$, and specification (Q1, Q5, Q22). "
        r"Reported are the MSEs (QLIKE losses) of forecasting models relative to the MSE "
        r"(QLIKE) of the TO HAR-STD model with Q1 specification, averaged across all "
        r"new issues and spin-offs. “—” denotes not applicable. "
        r"The best-performing model within each specification is marked in blue, and the "
        r'best overall is highlighted in bold blue. "$> 99$" represents values that exceed 99.'
    )

    latex_table = build_table_from_dir(
        results_dir=results_dp,
        file_glob_patterns=file_glob_patterns,
        complete_model_order=complete_model_order,
        caption=caption,
        label="tab:all-in-one",
        benchmark_model="TO HAR-STD",
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    with (
        reports_dp / "rel_to_benchmark_after_first_trading_day_model_performance.txt"
    ).open("w", encoding="utf-8") as f:
        f.write(latex_table)
