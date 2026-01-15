#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX table for relative selection percentages.
"""

import math
import pickle

from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def invert_target_to_source(
    data_by_target: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Invert a nested dictionary from target->source->value to source->target->value.
    """
    data_by_source = defaultdict(dict)
    for target, src_map in data_by_target.items():
        for source, value in src_map.items():
            data_by_source[source][target] = value
    return dict(data_by_source)


def _is_nan(x: object) -> bool:
    """
    Check if x is NaN.
    """
    return isinstance(x, float) and math.isnan(x)


def _fmt_value(x: Optional[float], blue_threshold: float, red_threshold: float) -> str:
    """
    Return formatted numeric cell with blue/red thresholds; em dash if missing.
    """
    if x is None or _is_nan(x):
        return "—"
    s = f"{x:.3f}"
    if x >= blue_threshold:
        return r"\color{blue} " + s
    if x <= red_threshold:
        return r"\color{red} " + s
    return s


def _display_ticker(t: str) -> str:
    """
    BRK.B shown as BRK-B.
    """
    if t == "BRK.B":
        return "BRK-B"
    return t


def make_latex_table(
    data: Dict[str, Dict[str, float]],
    target_assets: List[str],
    sector_to_sources: Dict[str, List[str]],
    sector_highlight_target: Dict[str, str],
    blue_threshold: float,
    red_threshold: float,
    target_col_width: str,
) -> str:
    """
    Create LaTeX table for relative selection percentages.
    """
    col_format = "ll" + ("x{" + target_col_width + "}") * len(target_assets)

    lines = []
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\hline \hline")
    lines.append(
        r"\textbf{GICS Sector} & \textbf{Source Asset} & "
        + rf"\multicolumn{{{len(target_assets)}}}{{c}}{{\textbf{{Target Asset}}}} \\"
    )
    lines.append(
        r"&  & " + " & ".join([rf"\textbf{{{t}}}" for t in target_assets]) + r" \\"
    )
    lines.append(r"\hline")

    for sector, sources in sector_to_sources.items():
        n = len(sources)
        highlight_target = sector_highlight_target.get(sector)

        for i, src in enumerate(sources):
            sector_cell = rf"\multirow{{{n}}}{{*}}{{{sector}}}" if i == 0 else ""
            row = [sector_cell, _display_ticker(src)]

            for target_asset in target_assets:
                val = data.get(src, {}).get(target_asset, None)
                cell = _fmt_value(val, blue_threshold, red_threshold)

                if highlight_target is not None and target_asset == highlight_target:
                    cell = r"\cellcolor{lightgray} " + cell

                row.append(cell)

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\hline")

    lines[-1] = r"\hline \hline"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def create_rel_selection_perc_table(
    meta_data_dp: str,
    reports_dp: str,
    tl_perc: int,
    backtest_start_offset: int,
    blue_threshold: float,
    red_threshold: float,
) -> None:
    """
    Create relative selection percentage table.
    """

    target_col_width = "1.75cm"

    meta_data_dp = Path(meta_data_dp)
    reports_dp = Path(reports_dp)

    data_by_target = {}

    files = [
        f
        for f in meta_data_dp.iterdir()
        if f.is_file()
        and "selection" in f.name
        and "std" in f.name
        and str(tl_perc) in f.name.split("_")[-4]
        and str(backtest_start_offset) in f.name.split("_")[-3]
    ]
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)

        sums = defaultdict(float)
        counts = defaultdict(int)
        for d in data:
            for k, v in d.items():
                sums[k] += v
                counts[k] += 1
        averages = {k: sums[k] / counts[k] for k in sums}

        data_by_target[file.name.split("_")[0]] = averages

    data_by_source = invert_target_to_source(data_by_target)

    target_assets_order = [
        "TWTR",
        "NCLH",
        "LW",
        "PSX",
        "SYF",
        "MRNA",
        "CARR",
        "DXC",
        "CTVA",
        "INVH",
    ]

    sector_to_sources = {
        "Communication Services": ["DISH", "EA", "GOOGL", "LUMN", "META", "TTWO"],
        "Consumer Discretionary": ["AMZN", "EBAY", "LEN", "MHK", "NWL", "TSLA"],
        "Consumer Staples": ["CPB", "HSY", "KMB", "PG", "TAP", "WMT"],
        "Energy": ["APA", "CVX", "DVN", "EQT", "HES", "XOM"],
        "Financials": ["AIZ", "BRK.B", "JPM", "LNC", "MTB", "NDAQ"],
        "Health Care": ["BIIB", "DVA", "DXCM", "JNJ", "UNH", "XRAY"],
        "Industrials": ["ALK", "FAST", "GNRC", "GWW", "HON", "UPS"],
        "Information Technology": ["AAPL", "FFIV", "FTNT", "MSFT", "QRVO", "TEL"],
        "Materials": ["APD", "IFF", "LIN", "SEE", "VMC", "WRK"],
        "Real Estate": ["AMT", "AVB", "FRT", "PLD", "VNO", "WY"],
        "Utilities": ["AWK", "DUK", "ES", "NEE", "NRG", "PNW"],
    }

    sector_highlight_target = {
        "Communication Services": "TWTR",
        "Consumer Discretionary": "NCLH",
        "Consumer Staples": "LW",
        "Energy": "PSX",
        "Financials": "SYF",
        "Health Care": "MRNA",
        "Industrials": "CARR",
        "Information Technology": "DXC",
        "Materials": "CTVA",
        "Real Estate": "INVH",
    }

    latex_table = make_latex_table(
        data_by_source,
        target_assets_order,
        sector_to_sources,
        sector_highlight_target,
        blue_threshold,
        red_threshold,
        target_col_width,
    )

    reports_dp.mkdir(parents=True, exist_ok=True)
    output_fp = reports_dp / f"relative_source_asset_selection_table-{tl_perc}.txt"
    with output_fp.open("w", encoding="utf-8") as f:
        f.write(latex_table)
