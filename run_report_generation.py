#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for report and figure generation.
"""
import configparser
import logging
import json
import sys

from src.reports.distance_origin_figure import create_distance_origin_figure
from src.reports.source_asset_selection_table import create_rel_selection_perc_table
from src.reports.one_vs_one_model_performance_tables import (
    create_one_vs_one_model_performance_tables,
)
from src.reports.rel_to_benchmark_model_performane_table import (
    create_rel_to_benchmark_model_perfomance_table,
)
from src.reports.mcs_table import create_mcs_table

from src.reports.rel_to_benchmark_alternative_models import create_har_extensions_table

from src.reports.short_train_data_rel_to_benchmark_model_performance_table import (
    short_rel_to_benchmark_model_perfomance_table,
)
from src.reports.short_train_data_transition_strategies_table import (
    create_strategy_table,
)

from src.reports.avg_short_train_data_transition_strategies_table import (
    create_avg_transition_strategy_table,
)


def load_model_orders_from_config(config_file: str = "config.ini") -> tuple[list, list]:
    """
    Load target and source assets names from the configuration file.
    """
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_file)
        complete_model_order = json.loads(config.get("reports", "complete_model_order"))
        std_model_order = json.loads(config.get("reports", "std_model_order"))
        ext_model_order = json.loads(config.get("reports", "ext_model_order"))
    except (configparser.Error, json.JSONDecodeError) as e:
        logging.error("Failed to load configuration or parse JSON: %s", e)
        sys.exit(1)
    return complete_model_order, std_model_order, ext_model_order


def main():
    """
    Report and figure generation.
    """
    prediction_results_dp = "results/predictions/"
    meta_results_dp = "results/meta/"
    reports_dp = "reports/"
    figure_dp = "figures/"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/report_generation.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    complete_model_order, std_model_order, ext_model_order = (
        load_model_orders_from_config()
    )

    logger.info("Creating figures.")
    create_distance_origin_figure(meta_results_dp, figure_dp, 25, 50)
    create_distance_origin_figure(meta_results_dp, figure_dp, 50, 50)
    create_distance_origin_figure(meta_results_dp, figure_dp, 75, 50)

    logger.info("Creating MTL source selection tables.")
    create_rel_selection_perc_table(meta_results_dp, reports_dp, 25, 50, 0.45, 0.05)
    create_rel_selection_perc_table(meta_results_dp, reports_dp, 50, 50, 0.70, 0.30)
    create_rel_selection_perc_table(meta_results_dp, reports_dp, 75, 50, 0.95, 0.55)

    logger.info("Create one-vs-one model performance tables.")
    create_one_vs_one_model_performance_tables(
        prediction_results_dp,
        reports_dp,
        complete_model_order,
        std_model_order,
        ext_model_order,
    )

    logger.info("Create relative to benchmark model performance tables.")
    create_rel_to_benchmark_model_perfomance_table(
        prediction_results_dp, reports_dp, complete_model_order, "TO HAR-STD"
    )

    logger.info("Create MCS table.")
    create_mcs_table(prediction_results_dp, reports_dp, complete_model_order)

    logger.info("Create HAR extensions table.")
    create_har_extensions_table(prediction_results_dp, reports_dp)

    logger.info(
        "Create short training data relative to benchmark model performance table."
    )
    short_rel_to_benchmark_model_perfomance_table(
        prediction_results_dp, reports_dp, complete_model_order
    )

    logger.info("Create short training data transition strategy tables.")
    create_strategy_table(
        prediction_results_dp,
        reports_dp,
        "TO HAR-STD",
    )

    logger.info("Create average transition strategy table.")
    create_avg_transition_strategy_table(prediction_results_dp, reports_dp)


if __name__ == "__main__":
    main()
