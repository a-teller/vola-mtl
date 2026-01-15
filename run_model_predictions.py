#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for generating model predictions.
"""

import argparse
import logging
import sys

from src.models.predictions import (
    Evaluator,
    PredictionRunner,
    estimate_optimal_nn_iterations,
)
from src.models.data_handling import DataLoader
from src.models.data_handling import MTLSelector
from src.models.model_factory import ModelFactory
from src.models.result_writing import PredictionsWriter, MTLMetaDataWriter
from src.data.utils import load_scaling_factor_from_config
from src.data.utils import load_assets_from_config
from src.models.utils import (
    get_test_set_sizes_from_config,
    get_eval_points_from_config,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run predictions using specified forecasting model and "
            "transfer learning method."
        )
    )
    parser.add_argument(
        "--model",
        choices=[
            "random-walk",
            "har",
            "xgboost",
            "feedforward-neural-network",
            "random-forest",
            "lasso",
        ],
        required=True,
        help=(
            "Forecast model to use: 'random-walk', 'har', 'xgboost', "
            "'feedforward-neural-network', randmom-forest or lasso."
        ),
    )
    parser.add_argument(
        "--tl_method",
        choices=["target-only", "naive-pooling", "mtl"],
        required=True,
        help="Transfer learning method: 'target-only', 'naive-pooling', or 'mtl'.",
    )
    parser.add_argument(
        "--backtest_start_offset",
        type=int,
        choices=[1, 5, 22, 50],
        required=True,
        help=(
            "Backtest start offset (in days) from the initial trading day. "
            "Choices: 1, 5, 22, or 50."
        ),
    )
    parser.add_argument(
        "--predictor_set",
        choices=[
            "std",
            "ext",
            "std-q",
            "std-semi",
            "ext-q",
            "ext-semi",
            "std-mean",
            "ext-mean",
        ],
        required=True,
        help="Predictor set type: 'std', 'ext', 'std-q, 'std-semi', 'ext-q', 'ext-semi', 'std-mean', 'ext-mean'",
    )
    parser.add_argument(
        "--tl_perc",
        type=int,
        choices=[25, 50, 75],
        help="Percentile threshold for MTL DTW filtering.",
    )
    parser.add_argument(
        "--retrain_intervall",
        type=int,
        choices=[1, 5],
        help="After how many days the model should be retrained.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    """
    if args.model == "random-walk":
        if args.tl_method != "target-only":
            raise ValueError(
                "random-walk can only be used with --tl_method target-only."
            )
        if args.predictor_set != "std":
            raise ValueError("random-walk can only be used with --predictor_set std.")
    if args.predictor_set in ("std-mean", "ext-mean"):
        if args.tl_method != "target-only":
            raise ValueError(
                "predictor_set 'mean' can only be used with 'target-only'."
            )
    if args.tl_method == "mtl" and args.tl_perc is None:
        raise ValueError("--tl_perc must be specified when --tl_method is 'mtl'.")


def main() -> None:
    """
    Main function to run predictions.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/model_predictions.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    args = parse_arguments()

    try:
        validate_args(args)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(2)

    target_assets, source_assets = load_assets_from_config()
    scaling_factor = load_scaling_factor_from_config()

    model_factory = ModelFactory(
        model_type=args.model,
        tl_method=args.tl_method,
        backtest_start_day=args.backtest_start_offset,
        predictor_set=args.predictor_set,
    )
    data_loader = DataLoader(
        data_dp="data/processed/",
        target_assets=target_assets,
        source_assets=source_assets,
        retrain_intervall=args.retrain_intervall,
    )
    evaluator = Evaluator()
    predictions_writer = PredictionsWriter(
        "results/predictions/", args.retrain_intervall
    )
    meta_data_writer = MTLMetaDataWriter("results/meta/")

    mtl_selector = None
    if args.tl_method == "mtl":
        mtl_selector = MTLSelector(
            data_dp="data/processed/",
            source_assets=source_assets,
            distance_percentile=args.tl_perc,
        )

    runner = PredictionRunner(
        model_factory=model_factory,
        data_loader=data_loader,
        evaluator=evaluator,
        predictions_writer=predictions_writer,
        meta_data_writer=meta_data_writer,
        mtl_selector=mtl_selector,
    )

    partial_data_dp = f"{min(22, args.backtest_start_offset)}/{args.predictor_set}/"

    test_set_sizes = get_test_set_sizes_from_config()
    eval_points = get_eval_points_from_config()

    logger.info("Running predictions with settings: %s", args)
    for target_asset in target_assets:
        is_first_eval_point = True
        for eval_point_idx, eval_point, test_set_size in zip(
            eval_points[args.backtest_start_offset].keys(),
            eval_points[args.backtest_start_offset].values(),
            test_set_sizes[args.backtest_start_offset],
        ):

            optimal_iters = None
            if (
                args.model == "feedforward-neural-network"
                and args.backtest_start_offset in (1, 5, 22)
                and is_first_eval_point
            ):
                optimal_iters = 500
            elif args.model == "feedforward-neural-network":
                optimal_iters = estimate_optimal_nn_iterations(
                    model_factory=model_factory,
                    data_loader=data_loader,
                    mtl_selector=mtl_selector,
                    tl_method=args.tl_method,
                    retrain_intervall=args.retrain_intervall,
                    partial_data_dp=partial_data_dp,
                    target_asset=target_asset,
                    eval_point=eval_point,
                )
                logging.info("Estimated optimal NN iterations: %s", optimal_iters)

            result = runner.run(
                target_asset=target_asset,
                eval_point=eval_point,
                test_set_len=test_set_size,
                partial_data_dp=partial_data_dp,
                eval_point_index=eval_point_idx,
                is_first_eval_point=is_first_eval_point,
                scale=scaling_factor,
                iters_override=optimal_iters,
            )

            logger.info("Evaluation results for %s (%s):", target_asset, eval_point_idx)
            for key, value in result.items():
                logger.info("%s: %.3e", key, value)

            is_first_eval_point = False


if __name__ == "__main__":
    main()
