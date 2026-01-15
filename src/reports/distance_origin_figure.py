#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures showing the distribution of temporal distances from origin.
"""

import pickle

from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable

import matplotlib.pyplot as plt


def fill_missing_buckets(
    d: Dict[int, int], step: int = 100, max_key: int = 3400
) -> Dict[int, int]:
    """
    Fill missing buckets in a dictionary with zero values.
    """
    return {k: d.get(k, 0) for k in range(0, max_key + step, step)}


def mean_per_bucket(dicts: Iterable[Dict[int, float]]) -> Dict[int, float]:
    """
    Calculate the mean value per bucket across multiple dictionaries.
    """
    sums = defaultdict(float)
    counts = defaultdict(int)

    for d in dicts:
        for bucket, value in d.items():
            sums[bucket] += value
            counts[bucket] += 1

    return {bucket: sums[bucket] / counts[bucket] for bucket in sums}


def get_histogram(
    data: Iterable[int], max_value: int, bucket_size: int
) -> Dict[int, int]:
    """
    Get the histogram of the data.
    """
    hist = {}

    for x in data:
        if 0 <= x < max_value:
            bucket = (x // bucket_size) * bucket_size
            hist[bucket] = hist.get(bucket, 0) + 1

    return hist


def reindex_bucket_dict(d: Dict[int, int], bucket_size: int = 100) -> Dict[int, int]:
    """
    Reindex the keys of the bucket dictionary by dividing by the bucket size.
    """
    return {k // bucket_size: v for k, v in d.items()}


def create_distance_origin_figure(
    meta_data_dp: str, figure_dp: str, tl_perc: int, backtest_start_offset: int
) -> None:
    """
    Create figures showing the distribution of temporal distances from origin.
    """
    figure_dp = Path(figure_dp)
    asset_distance_distributions = {}

    files = [
        f
        for f in Path(meta_data_dp).iterdir()
        if f.is_file()
        and "origin_dist" in f.name
        and str(tl_perc) in f.name.split("_")[-5]
        and str(backtest_start_offset) in f.name.split("_")[-4]
    ]
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            avg = mean_per_bucket(data)
            asset_distance_distributions[file.name.split("_")[0]] = avg

    _, axes = plt.subplots(nrows=10, ncols=1, figsize=(15, 30))
    order = ["TWTR", "NCLH", "LW", "PSX", "SYF", "MRNA", "CARR", "DXC", "CTVA", "INVH"]
    counter = 0
    for key in order:
        max_value = max(asset_distance_distributions[key].keys())
        asset_distance_distributions[key] = fill_missing_buckets(
            asset_distance_distributions[key], step=100, max_key=3400
        )

        axes[counter].bar(
            range(len(asset_distance_distributions[key].keys())),
            asset_distance_distributions[key].values(),
            color="#607c8e",
        )
        axes[counter].set_title(key)
        axes[counter].set_ylabel("Percentage (%)")
        axes[counter].set_ylim(0, 100)
        axes[counter].set_yticks([0, 25, 50, 75, 100])
        axes[counter].set_xticks(range(len(asset_distance_distributions[key])))

        if counter != 9:
            axes[counter].set_xticklabels([])
        axes[counter].grid(True)
        axes[counter].set_xlim(-1, len(asset_distance_distributions[key]))

        if max_value < 3400:
            axes[counter].axvspan(
                max_value / 100 + 1,
                len(asset_distance_distributions[key]),
                color="grey",
                alpha=0.3,
            )

        if counter == 9:
            axes[counter].set_xticklabels(
                [
                    str(label) + "-" + str(label + 100)
                    for label in asset_distance_distributions[key].keys()
                ]
            )
            axes[counter].set_xlabel("Temporal Distance (Trading Days)")
            axes[counter].tick_params(axis="x", labelrotation=90)
        counter += 1

    if not figure_dp.exists():
        figure_dp.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(figure_dp / f"dist_origin-{tl_perc}.png", dpi=300)
