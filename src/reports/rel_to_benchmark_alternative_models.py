#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX table comparing HAR-based extensions vs two benchmarks:
(1) TO HAR-STD (5-day re-estimation) and (2) MTL-75 XGB-EXT.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.reports.one_vs_one_model_performance_tables import ModelSpec
from src.reports.one_vs_one_model_performance_tables import load_predictions_csv
from src.reports.rel_to_benchmark_model_performane_table import mse_series, qlike_series


def extract_reestimation_tag(rest: str) -> Optional[str]:
    """
    Extract re-estimation frequency k.
    """
    r = rest.lower()
    m = re.search(
        r"(?:^|_)(std-q|ext-q|std-semi|ext-semi|std-mean|ext-mean|std|ext)_(\d+)(?:_|$)",
        r,
    )
    return m.group(2) if m else None


def _parse_predictors(rest: str) -> str:
    """
    Map filename tokens to predictor tags.
    """
    r = rest.lower()

    if re.search(r"(?:^|_)std-q(?:_|$)", r):
        return "STD-Q"
    if re.search(r"(?:^|_)ext-q(?:_|$)", r):
        return "EXT-Q"
    if re.search(r"(?:^|_)std-semi(?:_|$)", r):
        return "STD-SEMI"
    if re.search(r"(?:^|_)ext-semi(?:_|$)", r):
        return "EXT-SEMI"
    if re.search(r"(?:^|_)std-mean(?:_|$)", r):
        return "STD-MEAN"
    if re.search(r"(?:^|_)ext-mean(?:_|$)", r):
        return "EXT-MEAN"
    if re.search(r"(?:^|_)std(?:_|$)", r):
        return "STD"
    if re.search(r"(?:^|_)ext(?:_|$)", r):
        return "EXT"

    raise ValueError(f"Could not parse predictors from: {rest}")


def parse_result_filename(p: Path) -> Tuple[str, ModelSpec]:
    """
    Parse a result csv filename and extract:
    """
    name = p.stem
    name = re.sub(r"_predictions_complete$", "", name)

    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected file name: {p.name}")

    target = parts[0]
    base_token = parts[1].lower()
    rest = "_".join(parts[2:]).lower()

    if "mtl-25" in rest:
        approach = "MTL-25"
    elif "mtl-50" in rest:
        approach = "MTL-50"
    elif "mtl-75" in rest:
        approach = "MTL-75"
    elif "navie-pooling" in rest:
        approach = "NP"
    elif "target-only" in rest:
        approach = "TO"
    else:
        approach = "TO"

    if base_token.startswith("har"):
        base = "HAR"
    elif base_token.startswith("xgboost") or base_token.startswith("xgb"):
        base = "XGB"
    elif base_token.startswith("feedforward"):
        base = "FNN"
    elif base_token.startswith("random-walk"):
        base = "RW"
    elif base_token.startswith("lasso"):
        base = "HAR"
    else:
        raise ValueError(f"Could not parse base model from: {p.name}")

    predictors = _parse_predictors(rest)
    reest = extract_reestimation_tag(rest)

    is_daily_har = base_token.startswith("har") and approach == "TO" and reest == "1"

    if predictors in ("STD-Q", "EXT-Q"):
        base = "HARQ"
        predictors = "STD" if predictors == "STD-Q" else "EXT"
    elif predictors in ("STD-SEMI", "EXT-SEMI"):
        base = "HAR-RS"
        predictors = "STD" if predictors == "STD-SEMI" else "EXT"
    elif predictors in ("STD-MEAN", "EXT-MEAN"):
        base = "HAR-Mean"
        predictors = "STD" if predictors == "STD-MEAN" else "EXT"

    if base_token.startswith("lasso"):
        base = "HAR-LASSO"

    if is_daily_har:
        base = "HAR-1"

    return target, ModelSpec(approach=approach, base=base, predictors=predictors)


def canonical_model_name(spec: ModelSpec) -> str:
    """
    Create the exact model names used in the table.
    """
    approach = spec.approach
    base = spec.base
    pred = spec.predictors

    if base == "HAR-Mean":
        return f"HAR-Mean-{pred}"

    return f"{approach} {base}-{pred}"


def load_all_predictions(
    results_dir: Path,
    file_glob_patterns: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load prediction results across targets and models.
    """
    paths: List[Path] = []
    for pattern in file_glob_patterns:
        paths.extend(Path(results_dir).glob(pattern))
    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(f"No files matched in {results_dir}")

    predictions_by_target: Dict[str, Dict[str, object]] = {}

    for p in paths:
        target, spec = parse_result_filename(p)
        df = load_predictions_csv(p)

        y = df["actuals"].to_numpy(dtype=float)
        f = df["predictions"].to_numpy(dtype=float)

        model_name = canonical_model_name(spec)

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


def mean_loss_full_period(
    y: np.ndarray, f: np.ndarray, s_star: int, metric: str
) -> float:
    """
    Mean loss over the entire evaluation period (from s_star to end).
    """
    y_w = y[s_star:]
    f_w = f[s_star:]
    if metric == "MSE":
        return float(np.mean(mse_series(y_w, f_w)))
    if metric == "QLIKE":
        return float(np.mean(qlike_series(y_w, f_w)))
    raise ValueError("metric must be MSE or QLIKE")


def compute_relative_table_values_two_benchmarks(
    data: Dict[str, Dict[str, np.ndarray]],
    s_star: int,
    benchmarks: List[str],
) -> pd.DataFrame:
    """
    Cross-target average relative losses over the full evaluation period,
    for two benchmarks.
    """
    if len(benchmarks) != 2:
        raise ValueError("benchmarks must contain exactly 2 benchmark model names")

    targets = sorted(data.keys())
    models = sorted({m for t in targets for m in data[t]["preds"].keys()})

    for b in benchmarks:
        missing = [t for t in targets if b not in data[t]["preds"]]
        if missing:
            raise ValueError(
                f"Benchmark '{b}' missing for targets: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    cols = pd.MultiIndex.from_product(
        [benchmarks, ["MSE", "QLIKE"]], names=["benchmark", "metric"]
    )
    rel = pd.DataFrame(index=models, columns=cols, dtype=float)

    for m in models:
        for b in benchmarks:
            mse_ratios = []
            ql_ratios = []

            for t in targets:
                y = data[t]["actuals"]
                preds = data[t]["preds"]

                if m not in preds:
                    continue

                fb = preds[b]
                fm = preds[m]

                bm_mse = mean_loss_full_period(y, fb, s_star, "MSE")
                fm_mse = mean_loss_full_period(y, fm, s_star, "MSE")
                bm_ql = mean_loss_full_period(y, fb, s_star, "QLIKE")
                fm_ql = mean_loss_full_period(y, fm, s_star, "QLIKE")

                mse_ratios.append(fm_mse / bm_mse)
                ql_ratios.append(fm_ql / bm_ql)

            rel.loc[m, (b, "MSE")] = (
                float(np.mean(mse_ratios)) if mse_ratios else np.nan
            )
            rel.loc[m, (b, "QLIKE")] = (
                float(np.mean(ql_ratios)) if ql_ratios else np.nan
            )

    return rel


def fmt_val(x: float) -> str:
    """
    Format value.
    """
    if not np.isfinite(x):
        return "—"
    if x > 99:
        return r"$>$ 99"
    return f"{x:.3f}"


def render_two_benchmark_table_latex(
    rel: pd.DataFrame,
    model_order: List[str],
    benchmark_display_names: Tuple[str, str],
    benchmarks_internal: Tuple[str, str],
    caption: str,
    label: str,
    row_breaks_after: List[str] = None,
) -> str:
    """
    Render latex table.
    """
    b1_int, b2_int = benchmarks_internal
    b1_disp, b2_disp = benchmark_display_names

    cols = [(b1_int, "MSE"), (b1_int, "QLIKE"), (b2_int, "MSE"), (b2_int, "QLIKE")]

    displayed = [m for m in model_order if m in rel.index]
    best = {}

    for c in cols:
        vals = [(m, rel.loc[m, c]) for m in displayed if np.isfinite(rel.loc[m, c])]
        best[c] = min(vals, key=lambda z: z[1])[0] if vals else None

    row_breaks_after_set = set(row_breaks_after or [])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\centering")
    lines.append(r"\scalebox{0.8}{")
    lines.append(
        r"\begin{tabular}{l"
        r"  >{\centering\arraybackslash}p{1.4cm}"
        r"  >{\centering\arraybackslash}p{1.4cm}"
        r"  >{\centering\arraybackslash}p{1.4cm}"
        r"  >{\centering\arraybackslash}p{1.4cm}"
        r"}"
    )
    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(
        r"& \multicolumn{2}{c}{\makecell{relative to \\ " + b1_disp + r"}}"
        r" & \multicolumn{2}{c}{\makecell{relative to \\ " + b2_disp + r"}}\\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r"Model & MSE & QLIKE & MSE & QLIKE \\")
    lines.append(r"\midrule")

    for m in model_order:
        if m not in rel.index:
            continue

        out_cells: List[str] = []
        for c in cols:
            s = fmt_val(rel.loc[m, c])
            if best.get(c) == m and s not in ["—", r"$>$ 99"]:
                s = r"\textbf{" + s + "}"
            out_cells.append(s)

        lines.append(m + " & " + " & ".join(out_cells) + r" \\")
        if m in row_breaks_after_set:
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def create_har_extensions_table(results_dp: str, reports_dp: str):
    """
    Create HAR extensions table.
    """
    results_dp = Path(results_dp)
    reports_dp = Path(reports_dp)

    file_glob_patterns = [
        "*har_target-only_50_std_5_predictions_complete*",
        "*xgboost_mtl-75_50_ext_5_predictions_complete*",
        "*har_target-only_50_std_1_predictions_complete*",
        "*har_target-only_50_ext_1_predictions_complete*",
        "*har_target-only_50_std-q_5_predictions_complete*",
        "*har_target-only_50_ext-q_5_predictions_complete*",
        "*har_target-only_50_std-semi_5_predictions_complete*",
        "*har_target-only_50_ext-semi_5_predictions_complete*",
        "*har_target-only_50_std-mean_5_predictions_complete*",
        "*har_target-only_50_ext-mean_5_predictions_complete*",
        "*lasso_target-only_50_std_5_predictions_complete*",
        "*lasso_target-only_50_ext_5_predictions_complete*",
    ]

    benchmark_to_har_std_5 = "TO HAR-STD"
    benchmark_mtl_75_xgb_ext = "MTL-75 XGB-EXT"

    model_order = [
        "TO HAR-1-STD",
        "TO HAR-1-EXT",
        "TO HAR-LASSO-STD",
        "TO HAR-LASSO-EXT",
        "TO HAR-RS-STD",
        "TO HAR-RS-EXT",
        "TO HARQ-STD",
        "TO HARQ-EXT",
        "HAR-Mean-STD",
        "HAR-Mean-EXT",
    ]

    data = load_all_predictions(results_dp, file_glob_patterns)

    rel = compute_relative_table_values_two_benchmarks(
        data=data,
        s_star=0,
        benchmarks=[benchmark_to_har_std_5, benchmark_mtl_75_xgb_ext],
    )

    caption = (
        "Performance of HAR-based extensions and the daily re-estimated "
        "TO HAR-STD model compared with the TO HAR-STD model (re-estimated "
        "every 5 days) and the best-performing MTL model, i.e., MTL-75 XGB-EXT. "
        "Reported are the cross-sectional average MSEs (QLIKE losses) from "
        "rolling 1-day-ahead forecasts for each HAR-based extension and the "
        "daily re-estimated TO HAR-STD model relative to both the TO HAR-STD "
        "(5-day re-estimation) and the MTL-75 XGB-EXT benchmark. The MSEs "
        "(QLIKE losses) are based on the entire evaluation period ($s=s^{*}$), "
        "with the MSE (QLIKE loss) criteria of the best-performing model for "
        'each benchmark marked in bold. "$>$ 99" represents values that exceed '
        "99. Model names reflect both the forecasting approach and predictor set: "
        "TO denotes target only models trained solely on the target data; the "
        "second part of the name indicates the base model class, with HAR "
        'referring to the heterogeneous autoregressive model, suffix "1" '
        "indicates that the HAR model has been re-estimated daily, LASSO refers "
        "to the least absolute shrinkage and selection operator, HAR-RS denotes "
        "the semivariance HAR model, and HARQ the quarticity-based extension of "
        "the HAR model. HAR-Mean extends the HAR model by the equally weighted "
        "average of the previous day’s realized variances across all source "
        "assets and the target."
    )

    latex_table = render_two_benchmark_table_latex(
        rel=rel,
        model_order=model_order,
        benchmark_display_names=("TO HAR-STD", "MTL-75 XGB-EXT"),
        benchmarks_internal=(benchmark_to_har_std_5, benchmark_mtl_75_xgb_ext),
        caption=caption,
        label="tab:har_extensions",
        row_breaks_after=[
            "TO HAR-1-EXT",
            "TO HAR-LASSO-EXT",
            "TO HAR-RS-EXT",
            "TO HARQ-EXT",
            "HAR-Mean-EXT",
        ],
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    with (reports_dp / "har_extensions.txt").open("w", encoding="utf-8") as f:
        f.write(latex_table)
