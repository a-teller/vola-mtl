#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables comparing forecasting model performance (1 vs. 1).
"""

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from dieboldmariano import dm_test


@dataclass(frozen=True)
class ModelSpec:
    """
    Forecasting model specification.
    """

    approach: str
    base: str
    predictors: str

    def prettify_name(self) -> str:
        """
        Convert model name.
        """
        if self.base == "RW":
            return "RW"
        return f"{self.approach} {self.base}-{self.predictors}"


def parse_result_filename(p: Path) -> Tuple[str, ModelSpec]:
    """
    Parse a result csv filename and extract target, base,
    approach, and predictor sets.
    """
    name = p.stem
    name = re.sub(r"_predictions_complete$", "", name)

    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected file name: {p.name}")

    target = parts[0]

    base_token = parts[1].lower()
    if base_token.startswith("har"):
        base = "HAR"
    elif base_token.startswith("feedforward"):
        base = "FNN"
    elif base_token.startswith("xgb"):
        base = "XGB"
    elif base_token.startswith("random-walk"):
        base = "RW"
    else:
        raise ValueError(f"Could not parse base model from: {p.name}")

    rest = "_".join(parts[2:]).lower()

    if "mtl-25" in rest:
        approach = "MTL-25"
    elif "mtl-50" in rest:
        approach = "MTL-50"
    elif "mtl-75" in rest:
        approach = "MTL-75"
    elif re.search(r"(?:^|_)naive-pooling(?:_|$)", rest):
        approach = "NP"
    elif re.search(r"(?:^|_)target-only(?:_|$)", rest):
        approach = "TO"
    else:
        approach = "TO"

    predictors = "EXT" if re.search(r"(?:^|_)ext(?:_|$)", rest) else "STD"

    return target, ModelSpec(approach=approach, base=base, predictors=predictors)


def mse_point(y_true: float, y_pred: float) -> float:
    """
    Scalar Mean Squared Error.
    """
    return (y_true - y_pred) ** 2


def qlike_point(y_true: float, y_pred: float) -> float:
    """
    Scalar QLIKE loss.
    """
    r = y_true / y_pred
    return r - math.log(r) - 1.0


def mean_loss(y: np.ndarray, f: np.ndarray, loss_fn) -> float:
    """
    Compute mean pointwise loss using a scalar loss function.
    """
    return float(
        np.mean(np.fromiter((loss_fn(yt, ft) for yt, ft in zip(y, f)), dtype=float))
    )


def load_predictions_csv(path: Path) -> pd.DataFrame:
    """
    Read result dataframe.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "predictions", "actuals"]]


def load_all_predictions(
    results_dir: Path,
    file_glob_patterns: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Load forecast and actual values for all target assets and models.
    """
    actuals = {}
    preds = {}

    paths = []
    for pattern in file_glob_patterns:
        paths.extend(Path(results_dir).glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No files matched in {results_dir}")

    for p in paths:
        target, spec = parse_result_filename(p)
        df = load_predictions_csv(p)
        y = df["actuals"].to_numpy(dtype=float)
        f = df["predictions"].to_numpy(dtype=float)
        model_name = spec.prettify_name()

        preds.setdefault(target, {})[model_name] = f

        if target not in actuals:
            actuals[target] = y
        else:
            if len(actuals[target]) != len(y) or not np.allclose(
                actuals[target], y, equal_nan=True
            ):
                raise ValueError(
                    f"Actuals mismatch for target {target} in file {p.name}"
                )
    return actuals, preds


def cross_section_relative_matrix(
    actuals: Dict[str, np.ndarray],
    preds: Dict[str, Dict[str, np.ndarray]],
    row_models: List[str],
    col_models: List[str],
    loss_fn,
    alpha: float = 0.05,
    h: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional relative performance and significance matrices
    between forecasting models.
    """
    targets = sorted(actuals.keys())

    values = pd.DataFrame(index=row_models, columns=col_models, dtype=float)
    stars = pd.DataFrame(index=row_models, columns=col_models, dtype=bool)

    for r in row_models:
        for c in col_models:
            if r == c:
                values.loc[r, c] = np.nan
                stars.loc[r, c] = False
                continue

            rels = []
            sig_better = 0
            tested = 0

            for t in targets:
                if t not in preds or r not in preds[t] or c not in preds[t]:
                    continue

                y = actuals[t]
                pr = preds[t][r]
                pc = preds[t][c]

                mr = mean_loss(y, pr, loss_fn)
                mc = mean_loss(y, pc, loss_fn)

                rels.append(mc / mr)

                res = dm_test(
                    y.tolist(),
                    pc.tolist(),
                    pr.tolist(),
                    one_sided=True,
                    h=h,
                    loss=loss_fn,
                )

                p = res[1]
                tested += 1
                if p < alpha:
                    sig_better += 1

            values.loc[r, c] = float(np.mean(rels)) if rels else np.nan
            stars.loc[r, c] = (tested > 0) and ((sig_better / tested) > 0.5)

    return values, stars


def fmt_cell(x: float, star: bool) -> str:
    """
    Format table cell.
    """
    if not np.isfinite(x):
        return "—"
    s = f"{x:.3f}"
    if star:
        return r"\textcolor{blue}{" + s + "*}"
    return s


def latex_header_for_columns(col_models: List[str]) -> str:
    """
    Create latex column header.
    """
    out = []
    for name in col_models:
        if name == "RW":
            out.append("RW")
        else:
            a, rest = name.split(" ", 1)
            out.append(
                r"\begin{tabular}[c]{@{}c@{}}" + a + r" \\ " + rest + r"\end{tabular}"
            )
    return " & ".join(out)


def render_latex_table_block(
    values: pd.DataFrame,
    stars: pd.DataFrame,
    caption: str,
    label: Optional[str],
    col_format: str,
    split_hlines_after_rows: Optional[List[str]] = None,
    continued: bool = False,
) -> str:
    """
    Render a LaTeX table environment for a matrix of values
    with significance markers.
    """
    split_hlines_after_rows = split_hlines_after_rows or []

    col_models = list(values.columns)
    row_models = list(values.index)

    lines: List[str] = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    if continued:
        lines.append(r"\ContinuedFloat")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\scalebox{0.515}{")
    lines.append(r"\begin{tabular}{" + col_format + "}")
    lines.append(r"\toprule")

    lines.append(" & " + latex_header_for_columns(col_models) + r" \\")
    lines.append(r"\midrule")

    for r in row_models:
        row_cells = [
            fmt_cell(values.loc[r, c], bool(stars.loc[r, c])) for c in col_models
        ]
        lines.append(r + " & " + " & ".join(row_cells) + r" \\")
        if r in split_hlines_after_rows:
            lines.append(r"\hline")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_tables(
    results_dir: str,
    row_models: List[str],
    cols_std_block: List[str],
    cols_ext_block: List[str],
    row_hline_after: List[str],
    colformat_std: str,
    colformat_ext: str,
) -> Dict[str, str]:
    """
    Build LaTeX tables summarizing cross-sectional forecasting performance.
    """
    actuals, preds = load_all_predictions(
        Path(results_dir),
        file_glob_patterns=[
            "*random-walk*50_*_5_predictions_complete*",
            "*har*50_std_5_predictions_complete*",
            "*xgboost*50_std_5_predictions_complete*",
            "*feedforward-neural-network*50_std_5_predictions_complete*",
            "*har*50_ext_5_predictions_complete*",
            "*xgboost*50_ext_5_predictions_complete*",
            "*feedforward-neural-network*50_ext_5_predictions_complete*",
        ],
    )

    all_cols = sorted(set(cols_std_block + cols_ext_block))

    mse_vals, mse_stars = cross_section_relative_matrix(
        actuals=actuals,
        preds=preds,
        row_models=row_models,
        col_models=all_cols,
        loss_fn=mse_point,
        alpha=0.05,
        h=1,
    )

    q_vals, q_stars = cross_section_relative_matrix(
        actuals=actuals,
        preds=preds,
        row_models=row_models,
        col_models=all_cols,
        loss_fn=qlike_point,
        alpha=0.05,
        h=1,
    )

    mse_std_vals, mse_std_stars = (
        mse_vals.loc[row_models, cols_std_block],
        mse_stars.loc[row_models, cols_std_block],
    )
    mse_ext_vals, mse_ext_stars = (
        mse_vals.loc[row_models, cols_ext_block],
        mse_stars.loc[row_models, cols_ext_block],
    )

    q_std_vals, q_std_stars = (
        q_vals.loc[row_models, cols_std_block],
        q_stars.loc[row_models, cols_std_block],
    )
    q_ext_vals, q_ext_stars = (
        q_vals.loc[row_models, cols_ext_block],
        q_stars.loc[row_models, cols_ext_block],
    )

    caption_mse = (
        "1-day-ahead cross-sectional average relative MSEs. Each value represents "
        "the cross-sectional average of the model MSE in the selected column relative "
        "to the benchmark in the selected row. ($*$) denotes whether the Diebold–Mariano "
        "test is rejected for more than 50\\% of target assets (5\\% level), indicating "
        "the model in the column has significantly lower loss; these values are "
        "colored blue. “—” denotes not applicable."
    )

    caption_q = (
        "1-day-ahead cross-sectional average relative QLIKE losses. Each value represents "
        "the cross-sectional average of the model QLIKE loss in the selected column "
        "relative to the benchmark in the selected row. ($*$) denotes whether the "
        "Diebold–Mariano test is rejected for more than 50\\% of target assets (5\\% level), "
        "indicating the model in the column has significantly lower loss; these values are "
        "colored blue. “—” denotes not applicable."
    )

    tex: Dict[str, str] = {}

    tex["mse_page1"] = render_latex_table_block(
        values=mse_std_vals,
        stars=mse_std_stars,
        caption=caption_mse,
        label=None,
        col_format=colformat_std,
        split_hlines_after_rows=row_hline_after,
        continued=False,
    )

    tex["mse_page2"] = render_latex_table_block(
        values=mse_ext_vals,
        stars=mse_ext_stars,
        caption=caption_mse + " (continued).",
        label="tab:forecasting_results_mse",
        col_format=colformat_ext,
        split_hlines_after_rows=row_hline_after,
        continued=True,
    )

    tex["qlike_page1"] = render_latex_table_block(
        values=q_std_vals,
        stars=q_std_stars,
        caption=caption_q,
        label=None,
        col_format=colformat_std,
        split_hlines_after_rows=row_hline_after,
        continued=False,
    )

    tex["qlike_page2"] = render_latex_table_block(
        values=q_ext_vals,
        stars=q_ext_stars,
        caption=caption_q + " (continued).",
        label="tab:forecasting_results_qlike",
        col_format=colformat_ext,
        split_hlines_after_rows=row_hline_after,
        continued=True,
    )

    return tex


def create_one_vs_one_model_performance_tables(
    predictions_dp: str,
    reports_dp: str,
    complete_model_oder: str,
    std_model_order: str,
    ext_model_order: str,
) -> None:
    """
    Generate and write one-vs-one model performance tables in LaTeX format.
    """
    predictions_dp = Path(predictions_dp)
    reports_dp = Path(reports_dp)

    row_hline_after = ["RW", "MTL-75 HAR-EXT", "MTL-75 FNN-EXT"]

    colformat_std = (
        r"lC{2cm}|C{2cm}C{2cm}C{2cm}C{2cm}C{2cm}|"
        r"C{2cm}C{2cm}C{2cm}C{2cm}C{2cm}|C{2cm}C{2cm}C{2cm}C{2cm}C{2cm}"
    )
    colformat_ext = (
        r"lC{2cm}C{2cm}C{2cm}C{2cm}C{2cm}|C{2cm}C{2cm}C{2cm}C{2cm}C{2cm}|"
        r"C{2cm}C{2cm}C{2cm}C{2cm}C{2cm}"
    )

    tables = build_tables(
        predictions_dp,
        complete_model_oder,
        std_model_order,
        ext_model_order,
        row_hline_after,
        colformat_std,
        colformat_ext,
    )

    reports_dp.mkdir(parents=True, exist_ok=True)

    with (reports_dp / "one_vs_one_mse1.txt").open("w", encoding="utf-8") as f:
        f.write(tables["mse_page1"])
    with (reports_dp / "one_vs_one_mse2.txt").open("w", encoding="utf-8") as f:
        f.write(tables["mse_page2"])
    with (reports_dp / "one_vs_one_qlike1.txt").open("w", encoding="utf-8") as f:
        f.write(tables["qlike_page1"])
    with (reports_dp / "one_vs_one_qlike2.txt").open("w", encoding="utf-8") as f:
        f.write(tables["qlike_page2"])
