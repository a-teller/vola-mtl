#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for model predictions and meta data analysis.
"""

import configparser
import json
import logging
import math
import sys

from typing import Dict, List

import numpy as np


def get_test_set_sizes_from_config(
    config_file: str = "config.ini",
) -> Dict[int, List[int]]:
    """
    Load test set sizes from the configuration file.
    """
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_file)
        test_set_sizes = json.loads(config.get("test_sets", "sizes"))
        test_set_sizes = {int(k): v for k, v in test_set_sizes.items()}
    except (configparser.Error, json.JSONDecodeError) as e:
        logging.error("Failed to load configuration or parse JSON: %s", e)
        sys.exit(1)
    return test_set_sizes


def get_eval_points_from_config(
    config_file: str = "config.ini",
) -> Dict[int, Dict[int, int]]:
    """
    Load evaluation points from the configuration file.
    """
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_file)
        eval_points = json.loads(config.get("eval_periods", "eval_points"))
        eval_points = {
            int(outer_key): {
                int(inner_key): value for inner_key, value in inner_dict.items()
            }
            for outer_key, inner_dict in eval_points.items()
        }
    except (configparser.Error, json.JSONDecodeError) as e:
        logging.error("Failed to load configuration or parse JSON: %s", e)
        sys.exit(1)
    return eval_points


def get_histogram(
    data: List[int], max_value_inclusive: int, bucket_size: int
) -> Dict[int, int]:
    """
    Create histogram with fixed bucket size up to a maximum value.
    """
    n_buckets = math.floor(max_value_inclusive / bucket_size) + 1
    hist = {i * bucket_size: 0 for i in range(n_buckets)}

    for val in data:
        if 0 <= val <= max_value_inclusive:
            bucket = (val // bucket_size) * bucket_size
            hist[bucket] += 1

    return hist


def get_relative_percentages_of_dicts(
    dict_a: Dict[int, int], dict_b: Dict[int, int]
) -> Dict[int, float]:
    """
    Calculate the relative percentage of values in two dictionaries.
    """
    rel_percentages = {}
    for k in dict_a:
        if dict_a[k] == 0:
            rel_percentages[k] = 0.0
        else:
            rel_percentages[k] = 100 * dict_b.get(k, 0) / dict_a[k]
    return rel_percentages


def qlike_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute QLike loss of two series.
    """
    r = y / f
    return r - np.log(r) - 1.0
