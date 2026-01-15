#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities.
"""

from typing import Dict, List, Tuple, TypedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path
from src.models.utils import get_histogram, get_relative_percentages_of_dicts


class DataLoader:
    """
    Loading data for model training and evaluation.
    """

    def __init__(
        self,
        data_dp: str,
        target_assets: List[str],
        source_assets: List[str],
        retrain_intervall: int,
    ):
        self.data_dp = Path(data_dp)
        self.target_assets = target_assets
        self.source_assets = source_assets
        self.retrain_intv = retrain_intervall

    def load_target_data(
        self, asset: str, eval_point: int, retrain_point: int, partial_data_dp: str
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        np.ndarray,
    ]:
        """
        Load target data for a specific asset.
        """
        file_path = self.data_dp / f"{partial_data_dp}{asset}.csv"
        df = pd.read_csv(file_path)
        df = df[: eval_point + retrain_point]
        cut_off_date = df.iloc[-self.retrain_intv - 1]["date"]

        xt = df.values[: -self.retrain_intv, 1:-1]
        yt = df.values[: -self.retrain_intv, -1]
        xtst = df.values[-self.retrain_intv :, 1:-1]
        ytst = df.values[-self.retrain_intv :, -1]
        ytst_dates = df["date"].values[-self.retrain_intv :]

        return xt, yt, xtst, ytst, cut_off_date, ytst_dates

    def load_source_data(
        self, asset: str, cut_off_date: str, partial_data_dp: str
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Load source data for a specific asset.
        """
        file_path = self.data_dp / f"{partial_data_dp}{asset}.csv"
        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(cut_off_date)]
        return df.values[:, 1:-1], df.values[:, -1]


class SourceSubsequence(TypedDict):
    """
    Metadata and data for a source-asset subsequence used in
    similarity matching.
    """

    asset: str
    dist: float
    x_subsequence: np.ndarray
    y_subsequence: np.ndarray
    source_asset_x_len: int
    dist_to_forecast_origin: int


class MTLSelector:
    """
    Multi-Source Transfer Learning (MTL) for Instance Selection.
    """

    def __init__(self, data_dp, source_assets, distance_percentile: int):
        self.source_assets = source_assets
        self.distance_percentile = distance_percentile
        self.data_dp = Path(data_dp)

    def select_source_data(
        self,
        recent_target_obs: np.ndarray,
        cut_off_date: str,
        partial_data_dp: str,
        num_feat: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Dict[str, float],
        Dict[int, float],
    ]:
        """
        Select source data based on DTW distance.
        """
        selected_perc_source = {}
        selected_dist_to_forecast_origin = []
        all_dist_to_forecast_origin = []

        subsequences = self._generate_source_data_subsequences(
            recent_target_obs, cut_off_date, partial_data_dp, num_feat
        )

        dists = [subsequence["dist"] for subsequence in subsequences]
        threshold = np.percentile(dists, self.distance_percentile)
        source_xt, source_yt = [], []

        for subsequence in subsequences:
            if subsequence["asset"] not in selected_perc_source:
                selected_perc_source[subsequence["asset"]] = 0.0
            if subsequence["dist"] <= threshold:
                source_xt.append(subsequence["x_subsequence"])
                source_yt.append(subsequence["y_subsequence"])

                selected_perc_source[subsequence["asset"]] += (
                    len(subsequence["y_subsequence"])
                    / subsequence["source_asset_x_len"]
                )
                selected_dist_to_forecast_origin.append(
                    subsequence["dist_to_forecast_origin"]
                )

            all_dist_to_forecast_origin.append(subsequence["dist_to_forecast_origin"])

        all_dist_origin_hist = get_histogram(
            all_dist_to_forecast_origin, max(all_dist_to_forecast_origin), 100
        )
        selected_dist_origin_hist = get_histogram(
            selected_dist_to_forecast_origin, max(all_dist_to_forecast_origin), 100
        )
        perc_dist_origin_hist = get_relative_percentages_of_dicts(
            all_dist_origin_hist, selected_dist_origin_hist
        )

        return (
            np.concatenate(source_xt),
            np.concatenate(source_yt),
            selected_perc_source,
            perc_dist_origin_hist,
        )

    def _generate_source_data_subsequences(
        self,
        recent_target_obs: np.ndarray,
        cut_off_date: str,
        partial_data_dp: str,
        num_feat: int,
    ) -> List[SourceSubsequence]:
        """
        Generate source data subsequences.
        """
        subsequences = []

        for asset in self.source_assets:
            path = self.data_dp / f"{partial_data_dp}{asset}.csv"
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[df["date"] <= pd.Timestamp(cut_off_date)].values

            cut_off = len(df) % len(recent_target_obs)
            source_asset_x = df[cut_off:, 1:-1]
            source_asset_y = df[cut_off:, -1]

            if len(source_asset_x) // len(recent_target_obs) == 0:
                continue

            source_asset_x_subsequences = np.split(
                source_asset_x, len(source_asset_x) // len(recent_target_obs)
            )
            source_asset_y_subsequences = np.split(
                source_asset_y, len(source_asset_x) // len(recent_target_obs)
            )

            for i, x_subsequence in enumerate(source_asset_x_subsequences):
                _, dist = dtw_path(x_subsequence[:, :num_feat], recent_target_obs)
                subsequences.append(
                    {
                        "asset": asset,
                        "dist": dist,
                        "x_subsequence": x_subsequence,
                        "y_subsequence": source_asset_y_subsequences[i],
                        "source_asset_x_len": len(source_asset_x),
                        "dist_to_forecast_origin": len(source_asset_x_subsequences)
                        * len(recent_target_obs)
                        - i * len(recent_target_obs),
                    }
                )
        return subsequences
