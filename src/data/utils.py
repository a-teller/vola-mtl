#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for data processing.
"""

import configparser
import json
import logging
import sys


def load_scaling_factor_from_config(config_file: str = "config.ini") -> int:
    """
    Load scaling factor from the configuration file.
    """
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_file)
        scaling_factor = config.getint("data", "scaling_factor")
    except (configparser.Error, ValueError) as e:
        logging.error("Failed to load configuration or parse value: %s", e)
        sys.exit(1)
    return scaling_factor


def load_assets_from_config(config_file: str = "config.ini") -> tuple[list, list]:
    """
    Load target and source assets names from the configuration file.
    """
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_file)
        target_assets = json.loads(config.get("assets", "target_assets"))
        source_assets = json.loads(config.get("assets", "source_assets"))
    except (configparser.Error, json.JSONDecodeError) as e:
        logging.error("Failed to load configuration or parse JSON: %s", e)
        sys.exit(1)
    return target_assets, source_assets
