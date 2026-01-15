#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model predictor for training, predicting, and evaluating models.
"""

import os
import random

from typing import Dict, Optional
from itertools import chain

import numpy as np

from numpy.random import seed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.models.data_handling import DataLoader
from src.models.model_factory import ModelFactory
from src.models.result_writing import PredictionsWriter, MTLMetaDataWriter
from src.models.data_handling import MTLSelector
from src.models.utils import qlike_loss

os.environ["PYTHONHASHSEED"] = str(256)
random.seed(256)
seed(256)


def estimate_optimal_nn_iterations(
    model_factory: ModelFactory,
    data_loader: DataLoader,
    mtl_selector: MTLSelector,
    tl_method: str,
    retrain_intervall: int,
    partial_data_dp: str,
    target_asset: str,
    eval_point: int,
) -> int:
    """
    Estimate optimal number of iterations for neural network model.
    """
    xt, yt, xtst, _, target_date, _ = data_loader.load_target_data(
        target_asset, eval_point, retrain_intervall, partial_data_dp
    )
    if tl_method == "mtl":
        har_comp_feat_map = {1: 1, 5: 2, 22: 3, 50: 3}
        num_feat = har_comp_feat_map[model_factory.backtest_start_day]
        recent_obs = np.append(xt[-21:, :num_feat], xtst[0:1, :num_feat], axis=0)
        source_xt, source_yt, _, _ = mtl_selector.select_source_data(
            recent_obs, target_date, partial_data_dp, num_feat
        )
        xt = np.concatenate((xt, source_xt))
        yt = np.concatenate((yt, source_yt))

    if tl_method == "naive-pooling":
        for source_asset in data_loader.source_assets:
            source_xt, source_yt = data_loader.load_source_data(
                source_asset, target_date, partial_data_dp
            )
            xt = np.concatenate((xt, source_xt))
            yt = np.concatenate((yt, source_yt))

    scaler = StandardScaler()
    xt = scaler.fit_transform(xt)
    xtst = scaler.transform(xtst)

    model = model_factory._create_nn(
        nn_early=True,
        iters=None,
    )
    model.fit(xt, yt)
    best_iter = model.n_iter_

    return best_iter


class Evaluator:
    """
    Evaluator for model predictions.
    """

    def __init__(self):
        pass

    def evaluate(self, preds: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model predictions using MSE and MAE.
        """

        preds = preds.astype(np.float64)
        actuals = actuals.astype(np.float64)

        mse = mean_squared_error(actuals, preds)
        qlike = np.mean(qlike_loss(actuals, preds))

        return {
            "mse_model": mse,
            "qlike_model": qlike,
        }


class PredictionRunner:
    """
    Runner for training, predicting, and evaluating models.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        data_loader: DataLoader,
        evaluator: Evaluator,
        predictions_writer: PredictionsWriter,
        meta_data_writer: MTLMetaDataWriter,
        mtl_selector: Optional[MTLSelector] = None,
    ):
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.mtl_selector = mtl_selector
        self.predictions_writer = predictions_writer
        self.meta_data_writer = meta_data_writer

    def run(
        self,
        target_asset: str,
        eval_point: int,
        test_set_len: int,
        partial_data_dp: str,
        eval_point_index: int,
        is_first_eval_point: bool,
        scale: int,
        iters_override: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run the prediction process.
        """
        retrain_intv = self.data_loader.retrain_intv
        preds_all, actuals_all, dates, meta_selected_all, meta_origin_all = (
            [],
            [],
            [],
            [],
            [],
        )

        for offset in range(retrain_intv, test_set_len + retrain_intv, retrain_intv):
            xt, yt, xtst, ytst, target_date, ytst_dates = (
                self.data_loader.load_target_data(
                    target_asset, eval_point, offset, partial_data_dp
                )
            )

            if self.model_factory.tl_method == "mtl":
                har_comp_feat_map = {1: 1, 5: 2, 22: 3, 50: 3}
                num_feat = har_comp_feat_map[self.model_factory.backtest_start_day]
                recent_obs = np.append(
                    xt[-21:, :num_feat], xtst[0:1, :num_feat], axis=0
                )
                source_xt, source_yt, meta_selected, meta_origin = (
                    self.mtl_selector.select_source_data(
                        recent_obs, target_date, partial_data_dp, num_feat
                    )
                )
                meta_selected_all.append(meta_selected)
                meta_origin_all.append(meta_origin)
                xt = np.concatenate((xt, source_xt))
                yt = np.concatenate((yt, source_yt))

            if self.model_factory.tl_method == "naive-pooling":
                for source_asset in self.data_loader.source_assets:
                    source_xt, source_yt = self.data_loader.load_source_data(
                        source_asset, target_date, partial_data_dp
                    )
                    xt = np.concatenate((xt, source_xt))
                    yt = np.concatenate((yt, source_yt))

            if self.model_factory.model_type == "feedforward-neural-network":
                scaler = StandardScaler()
                xt = scaler.fit_transform(xt)
                xtst = scaler.transform(xtst)

            model = self.model_factory.create_model(iters=iters_override)
            if self.model_factory.model_type != "random-walk":
                model.fit(xt, yt)
                preds = model.predict(xtst)
                if self.model_factory.model_type == "feedforward-neural-network":
                    xt = scaler.inverse_transform(xt)
                preds[preds < 0] = np.min(xt[:, 0])
            else:
                preds = np.append(yt[-1:], ytst[:-1])

            preds = preds / scale
            ytst = ytst / scale

            preds_all.append(preds)
            actuals_all.append(ytst)
            dates.append(ytst_dates.tolist())

        complete_preds = np.concatenate(preds_all)
        complete_actuals = np.concatenate(actuals_all)
        dates = list(chain.from_iterable(dates))

        distance_percentile = None
        if self.mtl_selector:
            distance_percentile = self.mtl_selector.distance_percentile

        self.predictions_writer.write(
            eval_point_index,
            target_asset,
            complete_preds,
            complete_actuals,
            dates,
            self.model_factory,
            distance_percentile,
            is_first_eval_point,
        )
        if self.mtl_selector:
            self.meta_data_writer.write(
                target_asset,
                meta_selected_all,
                meta_origin_all,
                self.model_factory,
                distance_percentile,
                is_first_eval_point,
            )

        return self.evaluator.evaluate(complete_preds, complete_actuals)
