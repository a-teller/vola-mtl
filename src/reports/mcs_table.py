#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Model Confidence Set table comparing forecasting models.
"""

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from src.reports.mcs import ModelConfidenceSet

from src.reports.rel_to_benchmark_model_performane_table import (
    mse_series,
    qlike_series,
    load_all_predictions,
)

np.random.seed(1337)


def choose_block_length(
    loss_df: pd.DataFrame, max_lags: int = 15, alpha: float = 0.05
) -> int:
    """
    Fit AR(max_lags) on each model's loss series, take max significant lag.
    Returns at least 1.
    """
    c = 0
    for col in loss_df.columns:
        x = loss_df[col].astype(float).to_numpy()
        ar_model = AutoReg(x, lags=max_lags, old_names=False).fit()
        pvals = ar_model.pvalues
        for lag in range(1, len(pvals)):
            if pvals[lag] < alpha:
                c = max(c, lag)
    return max(c, 1)


def run_mcs_for_window(
    y: np.ndarray,
    preds: Dict[str, np.ndarray],
    start: int,
    length: int,
    metric: str,
    mcs_alpha: float,
    num_b: int,
    algorithm: str = "SQ",
) -> List[str]:
    """
    Run Model Confidence Set (MCS) generation on a rolling evaluation window.
    """
    end = min(start + length, len(y))
    y_w = y[start:end]

    data = {}
    for name, f in preds.items():
        f_w = f[start:end]
        if metric == "MSE":
            data[name] = mse_series(y_w, f_w)
        elif metric == "QLIKE":
            data[name] = qlike_series(y_w, f_w)
        else:
            raise ValueError("metric must be MSE or QLIKE")

    loss_df = pd.DataFrame(data)

    w = choose_block_length(loss_df, max_lags=15, alpha=0.05)
    mcs = ModelConfidenceSet(loss_df, mcs_alpha, num_b, w, algorithm=algorithm).run()
    return list(mcs.included)


def fmt_cell(mse_count: int, qlike_count: int, highlight_nonzero: bool = True) -> str:
    """
    Format cell.
    """
    a = str(mse_count)
    b = str(qlike_count)
    if highlight_nonzero:
        if mse_count != 0:
            a = r"\textcolor{blue}{" + a + "}"
        if qlike_count != 0:
            b = r"\textcolor{blue}{" + b + "}"
    return f"{a}/{b}"


def bold_blue(cell_part: str) -> str:
    """
    Format value in bold blue.
    """
    num = re.sub(r"\\textcolor\{blue\}\{(\d+)\}", r"\1", cell_part)
    return r"\textbf{\textcolor{blue}{" + num + "}}"


def render_mcs_frequency_table(
    model_order: List[str],
    window_labels: List[str],
    counts_mse: Dict[str, Dict[str, int]],
    counts_qlike: Dict[str, Dict[str, int]],
    caption: str,
    label: str,
) -> str:
    """
    Render a LaTeX table summarizing Model Confidence Set (MCS)
    inclusion frequencies.
    """
    col_max_mse = {
        w: max(counts_mse.get(m, {}).get(w, 0) for m in model_order)
        for w in window_labels
    }
    col_max_q = {
        w: max(counts_qlike.get(m, {}).get(w, 0) for m in model_order)
        for w in window_labels
    }

    lines = []
    lines.append(r"\begin{table}[hbt!]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\scalebox{0.8}{")
    colspec = "p{4.0cm}" + "".join(["p{1.0cm}" for _ in window_labels])
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\hline")
    lines.append(r"\hline")
    header = (
        r"\multicolumn{1}{c}{Model} & "
        + " & ".join([rf"\multicolumn{{1}}{{c}}{{{w}}}" for w in window_labels])
        + r"\\"
    )
    lines.append(header)
    lines.append(r"\hline")

    for m in model_order:
        row = [m]
        for w in window_labels:
            a = counts_mse.get(m, {}).get(w, 0)
            b = counts_qlike.get(m, {}).get(w, 0)

            cell = fmt_cell(a, b, highlight_nonzero=True)
            a_txt, b_txt = cell.split("/")

            if a == col_max_mse[w] and a != 0:
                a_txt = bold_blue(a_txt)
            if b == col_max_q[w] and b != 0:
                b_txt = bold_blue(b_txt)

            row.append(f"{a_txt}/{b_txt}")

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


def create_mcs_table(results_dp: str, reports_dp: str, model_order: List[str]) -> None:
    """
    Generate and write a LaTeX table summarizing Model Confidence Set (MCS)
    frequencies.
    """
    results_dp = Path(results_dp)
    reports_dp = Path(reports_dp)

    eval_list = [0, 100, 200, 300, 400]
    window_length = 100

    mcs_alpha = 0.95
    num_b = 5000
    algorithm = "SQ"

    window_labels = [f"s = {s}" for s in eval_list] + [
        "Agg.",
        r"$\textnormal{s = s}^{*}$",
    ]

    counts_mse = {m: {w: 0 for w in window_labels} for m in model_order}
    counts_qlike = {m: {w: 0 for w in window_labels} for m in model_order}

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

    targets = sorted(data.keys())
    if not targets:
        raise RuntimeError("No targets loaded.")

    for target in targets:
        y = data[target]["actuals"]
        preds = data[target]["preds"]

        for s in eval_list:
            label = f"s = {s}"

            inc_mse = run_mcs_for_window(
                y,
                preds,
                start=s,
                length=window_length,
                metric="MSE",
                mcs_alpha=mcs_alpha,
                num_b=num_b,
                algorithm=algorithm,
            )
            inc_q = run_mcs_for_window(
                y,
                preds,
                start=s,
                length=window_length,
                metric="QLIKE",
                mcs_alpha=mcs_alpha,
                num_b=num_b,
                algorithm=algorithm,
            )

            for m in inc_mse:
                if m in counts_mse:
                    counts_mse[m][label] += 1
            for m in inc_q:
                if m in counts_qlike:
                    counts_qlike[m][label] += 1

        inc_mse_full = run_mcs_for_window(
            y,
            preds,
            start=0,
            length=len(y),
            metric="MSE",
            mcs_alpha=mcs_alpha,
            num_b=num_b,
            algorithm=algorithm,
        )
        inc_q_full = run_mcs_for_window(
            y,
            preds,
            start=0,
            length=len(y),
            metric="QLIKE",
            mcs_alpha=mcs_alpha,
            num_b=num_b,
            algorithm=algorithm,
        )

        for m in inc_mse_full:
            if m in counts_mse:
                counts_mse[m][r"$\textnormal{s = s}^{*}$"] += 1
        for m in inc_q_full:
            if m in counts_qlike:
                counts_qlike[m][r"$\textnormal{s = s}^{*}$"] += 1

    for m in model_order:
        counts_mse[m]["Agg."] = sum(counts_mse[m][f"s = {s}"] for s in eval_list)
        counts_qlike[m]["Agg."] = sum(counts_qlike[m][f"s = {s}"] for s in eval_list)

    caption = (
        "Frequency of inclusion of forecasting models in MCSs based on MSE and QLIKE "
        "of all considered new issues and spin-offs for individual sample periods. "
        "Each cell reports the number of MCS inclusions for a given model under MSE "
        "and QLIKE (denoted as MSE/QLIKE). The frequency of inclusion of each model over "
        "all individual sample periods is aggregated in a separate column (Agg.). The "
        "column $s=s^{*}$ reports the inclusion of individual forecasting models in the MCSs "
        "of new issues/spin-offs for the entire test period."
    )

    latex_table = render_mcs_frequency_table(
        model_order=model_order,
        window_labels=window_labels,
        counts_mse=counts_mse,
        counts_qlike=counts_qlike,
        caption=caption,
        label="tab:mcs",
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    with (reports_dp / "model_confidence_sets.txt").open("w", encoding="utf-8") as f:
        f.write(latex_table)
