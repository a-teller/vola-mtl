#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory for creating forecasting models.
"""

from typing import Optional, Union

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


class ModelFactory:
    """
    Factory for creating forecasting models.
    """

    def __init__(
        self,
        model_type: str,
        tl_method: str,
        backtest_start_day: int,
        predictor_set: str,
    ):
        self.model_type = model_type
        self.tl_method = tl_method
        self.backtest_start_day = backtest_start_day
        self.predictor_set = predictor_set

    def create_model(
        self, iters: Optional[int] = None, nn_early: bool = False
    ) -> Optional[
        Union[
            LinearRegression,
            Lasso,
            RandomForestRegressor,
            xgb.XGBRegressor,
            MLPRegressor,
        ]
    ]:
        """
        Create and return the specified model.
        """
        if self.model_type == "har":
            return LinearRegression()
        if self.model_type == "lasso":
            return Lasso()
        if self.model_type == "random-forest":
            return RandomForestRegressor(
                n_estimators=500,
                n_jobs=-1,
                max_features=1 if self.predictor_set == "std" else 4,
                min_samples_split=5,
                random_state=256,
            )
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                verbosity=0,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.75,
                gamma=0.1,
                n_estimators=40,
                seed_per_iteration=True,
                seed=256,
            )
        if self.model_type == "feedforward-neural-network":
            return self._create_nn(iters, nn_early)
        if self.model_type == "random-walk":
            return None
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_nn(self, iters: Optional[int], nn_early: bool) -> MLPRegressor:
        """
        Create and return a feedforward neural network model.
        """
        batch_size = self._determine_batch_size()

        return MLPRegressor(
            random_state=512,
            max_iter=iters or 500,
            early_stopping=nn_early,
            shuffle=not nn_early,
            verbose=False,
            n_iter_no_change=100,
            batch_size=batch_size,
            learning_rate_init=0.001,
            tol=0.0,
            hidden_layer_sizes=[8, 4, 2],
        )

    def _determine_batch_size(self) -> int:
        """
        Determine batch size based on transfer learning method and backtest start day.
        """
        if self.tl_method == "target-only" and self.backtest_start_day in {1, 5, 22}:
            return 1
        if self.tl_method == "target-only":
            return 4
        if self.tl_method == "mtl":
            return 512
        return 1024
