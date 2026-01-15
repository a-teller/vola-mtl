#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for processing raw financial data.
"""

import logging

from src.data.raw_data_processing import RawDataProcessor
from src.data.raw_data_processing import MeanDatasetGenerator
from src.data.utils import load_scaling_factor_from_config
from src.data.utils import load_assets_from_config


def main() -> None:
    """
    Run data processing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/data_processing.log"),
            logging.StreamHandler(),
        ],
    )

    scaling_factor = load_scaling_factor_from_config()

    rdp = RawDataProcessor(
        raw_stock_data_dp="data/raw/stock_data/",
        raw_macro_data_dp="data/raw/macro_data/",
        raw_earning_announce_data_dp="data/raw/earning_announce_data/",
        processed_data_dp="data/processed/",
    )

    rdp.process_raw_data(
        "2010-01-01",
        [1, 5, 22],
        ["std", "ext", "std-semi", "ext-semi", "std-q", "ext-q"],
        scaling_factor,
    )

    target_assets, source_assets = load_assets_from_config()
    mdg = MeanDatasetGenerator(data_dp="data/processed/22/")
    mdg.create_mean_dataset(target_assets, source_assets, "std")
    mdg.create_mean_dataset(target_assets, source_assets, "ext")


if __name__ == "__main__":
    main()
