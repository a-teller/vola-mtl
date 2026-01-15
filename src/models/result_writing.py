#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result writers for model predictions and meta data.
"""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.model_factory import ModelFactory


class MTLMetaDataWriter:
    """
    Writer for MTL meta data.
    """

    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        target_asset: str,
        amount_meta: List[Dict[str, float]],
        origin_meta: List[Dict[int, float]],
        model_factory: ModelFactory,
        distance_percentile: int,
        is_first_eval_point: bool,
    ) -> None:
        """
        Saves selection ratios of sources assets and subsequence to
        forecast origin distance distributions.
        """
        file_name_source_selection = (
            f"{target_asset}_"
            f"{model_factory.tl_method}-{distance_percentile}_"
            f"{model_factory.backtest_start_day}_"
            f"{model_factory.predictor_set}_selection.xyz"
        )
        file_name_origin_dist = (
            f"{target_asset}_"
            f"{model_factory.tl_method}-{distance_percentile}_"
            f"{model_factory.backtest_start_day}_"
            f"{model_factory.predictor_set}_origin_dist.xyz"
        )

        if is_first_eval_point:
            with open(self.save_path / file_name_source_selection, "wb") as f:
                pickle.dump(amount_meta, f)
            with open(self.save_path / file_name_origin_dist, "wb") as f:
                pickle.dump(origin_meta, f)
        else:
            with open(self.save_path / file_name_source_selection, "rb") as f:
                stored_amount_meta = pickle.load(f)
            stored_amount_meta = stored_amount_meta + amount_meta
            with open(self.save_path / file_name_source_selection, "wb") as f:
                pickle.dump(stored_amount_meta, f)

            with open(self.save_path / file_name_origin_dist, "rb") as f:
                stored_origin_meta = pickle.load(f)
            stored_origin_meta = stored_origin_meta + origin_meta
            with open(self.save_path / file_name_origin_dist, "wb") as f:
                pickle.dump(stored_origin_meta, f)


class PredictionsWriter:
    """
    Writer for model predictions and actual data.
    """

    def __init__(self, save_path: str, retrain_intervall: int):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.retrain_intervall = retrain_intervall

    def write(
        self,
        eval_point_index: int,
        target_asset: str,
        preds: np.ndarray,
        actuals: np.ndarray,
        dates: List[str],
        model_factory: ModelFactory,
        distance_percentile: int,
        is_first_eval_point: bool,
    ) -> None:
        """
        Writes model predictions and actual data to CSV files.
        """
        result_df = pd.DataFrame(
            {"date": dates, "predictions": preds, "actuals": actuals}
        )
        if distance_percentile:
            file_name_complete = (
                f"{target_asset}_"
                f"{model_factory.model_type}_"
                f"{model_factory.tl_method}-{distance_percentile}_"
                f"{model_factory.backtest_start_day}_"
                f"{model_factory.predictor_set}_"
                f"{self.retrain_intervall}_"
                "predictions_complete.csv"
            )

            file_name_partial = (
                f"{target_asset}_"
                f"{model_factory.model_type}_"
                f"{model_factory.tl_method}-{distance_percentile}_"
                f"{model_factory.backtest_start_day}_"
                f"{model_factory.predictor_set}_"
                f"{self.retrain_intervall}_"
                f"predictions_{eval_point_index}.csv"
            )

        else:
            file_name_complete = (
                f"{target_asset}_"
                f"{model_factory.model_type}_"
                f"{model_factory.tl_method}_"
                f"{model_factory.backtest_start_day}_"
                f"{model_factory.predictor_set}_"
                f"{self.retrain_intervall}_"
                f"predictions_complete.csv"
            )
            file_name_partial = (
                f"{target_asset}_"
                f"{model_factory.model_type}_"
                f"{model_factory.tl_method}_"
                f"{model_factory.backtest_start_day}_"
                f"{model_factory.predictor_set}_"
                f"{self.retrain_intervall}_"
                f"predictions_{eval_point_index}.csv"
            )

        if is_first_eval_point:
            result_df.to_csv(self.save_path / file_name_complete, index=False)
            result_df.to_csv(self.save_path / file_name_partial, index=False)
        else:
            result_df.to_csv(self.save_path / file_name_partial, index=False)
            prev_result_df = pd.read_csv(self.save_path / file_name_complete)
            result_df = pd.concat([prev_result_df, result_df], ignore_index=True)
            result_df.to_csv(self.save_path / file_name_complete, index=False)
