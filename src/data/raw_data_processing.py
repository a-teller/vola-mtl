#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw financial data processing module.
"""

from itertools import chain
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RawDataProcessor:
    """
    Processes raw financial data.
    """

    def __init__(
        self,
        raw_stock_data_dp: str,
        raw_macro_data_dp: str,
        raw_earning_announce_data_dp: str,
        processed_data_dp: str,
    ) -> None:
        self.raw_stock_data_dp = Path(raw_stock_data_dp)
        self.raw_macro_data_dp = Path(raw_macro_data_dp)
        self.raw_earning_announce_data_dp = Path(raw_earning_announce_data_dp)
        self.processed_data_dp = Path(processed_data_dp)
        self._external_data: Dict[str, pd.DataFrame] = {}

    def process_raw_data(
        self,
        start_date: str,
        vola_comps: List[int],
        predictor_sets: List[str],
        scale: int,
    ) -> None:
        """Process raw stock data files."""
        start_date = pd.Timestamp(start_date)

        logger.info(
            "Processing marco data",
        )
        self._process_external_macro_data(start_date)

        for file_path in self.raw_stock_data_dp.iterdir():
            if file_path.is_file():
                ticker_name = self._extract_ticker_name(file_path)
                logger.info("Processing ticker: %s", ticker_name)

                for vola_comp in vola_comps:
                    for predictor_set in predictor_sets:
                        processed_df = self._process_stock_data(
                            file_path,
                            ticker_name,
                            vola_comp,
                            predictor_set,
                            start_date,
                            scale,
                        )
                        self._save_processed_data(
                            processed_df, ticker_name, vola_comp, predictor_set
                        )

    @staticmethod
    def _extract_ticker_name(file_path: Path) -> str:
        """Extract ticker name from file path."""
        return file_path.stem.split("_")[0]

    def _process_external_macro_data(self, start_date: str) -> None:
        """Process external macro-economic data sources."""
        self._external_data = {
            "ads": _process_ads_data(self.raw_macro_data_dp / "ads.xlsx", start_date),
            "vix": _process_vix_data(self.raw_macro_data_dp / "vix.csv", start_date),
            "3mtb": _process_threemtb_data(
                self.raw_macro_data_dp / "3mtb.csv", start_date
            ),
            "epu": _process_epu_data(self.raw_macro_data_dp / "epu.xlsx", start_date),
            "hsi": _process_hsi_data(self.raw_macro_data_dp / "hsi.csv", start_date),
        }

    def _add_earning_announce_data(
        self, df: pd.DataFrame, ticker_name: str
    ) -> pd.DataFrame:
        """Add earnings announcement dates to the dataframe."""
        earnings_fp = self.raw_earning_announce_data_dp / f"{ticker_name}.csv"
        df_earning_dates = pd.read_csv(earnings_fp, names=["date"])
        df_earning_dates[["date", "mins"]] = df_earning_dates["date"].str.split(
            " ", expand=True
        )
        df_earning_dates["date"] = pd.to_datetime(
            df_earning_dates["date"], format="%Y-%m-%d"
        )
        df["ea"] = df["date"].isin(df_earning_dates["date"]).astype(int)
        df["ea"] = df["ea"].shift(-1)

        return df

    def _process_stock_data(
        self,
        fp: Path,
        ticker_name: str,
        backtest_start_day: int,
        predictor_set: str,
        start_date: pd.Timestamp,
        scale: int,
    ) -> pd.DataFrame:
        """Process individual stock data file."""
        df = _read_data(fp, ["date", "open", "high", "low", "close", "volume"])

        pipeline = PreprocessingPipeline(
            _filter_weekends,
            _group_five_min,
            _group_daily,
            _temporal_rv_comps_generation,
            _make_rv_scaling(scale),
            _dollar_volume_generation,
            _one_week_mom_generation,
            _target_feature_generation,
        )

        df = pipeline.process(df)

        for external_df in self._external_data.values():
            df = df.merge(external_df, how="left", on="date")

        df["hsi"] = df["hsi"].ffill()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= start_date]

        df = self._add_earning_announce_data(df, ticker_name)
        df = _select_features(df, backtest_start_day, predictor_set)
        df = df.dropna()

        return df

    def _save_processed_data(
        self, df: pd.DataFrame, ticker_name: str, vola_comp: int, predictor_set: str
    ) -> None:
        """Save processed dataframe to CSV file."""
        output_dir = self.processed_data_dp / str(vola_comp) / predictor_set
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{ticker_name}.csv", index=False)


class PreprocessingPipeline:
    """A preprocessing pipeline for sequentially applying functions to data."""

    def __init__(self, *functions: Callable[..., Any]) -> None:
        self._functions = functions

    def process(self, data: Any) -> Any:
        """Process the data using the functions in the pipeline."""
        for function in self._functions:
            data = function(data)

        return data


def _read_data(fp: Path, cols: List[str]) -> pd.DataFrame:
    """Read data from CSV."""
    df = pd.read_csv(fp, names=cols, delimiter=",", header=None)
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    return df


def _filter_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out weekends from a DataFrame and set date as index."""
    df = df[~(df["date"]).dt.day_name().isin(["Saturday", "Sunday"])]
    df.set_index("date", inplace=True)
    return df


def _group_five_min(df: pd.DataFrame) -> pd.DataFrame:
    """Group DataFrame into 5-minute intervals and aggregate OHLCV data."""
    df = (
        df.between_time("09:30", "16:00")
        .groupby(pd.Grouper(freq="5Min"))
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "volume": "sum",
                "close": "last",
            }
        )
    )
    return df


def _rv_generation(series: pd.Series) -> List[float]:
    """Generate realized volatility from a series of closing prices."""
    ls = series.tolist()
    returns = [np.log(ls[i]) - np.log(ls[i - 1]) for i in range(1, len(ls))]
    rv = sum(r**2 for r in returns)
    pos_semi_var = sum([r**2 for r in returns if r > 0])
    neg_semi_var = sum([r**2 for r in returns if r < 0])
    rq = sum([r**4 for r in returns]) * (78 / 3)

    close = ls[-1] if ls else np.nan
    return [close, rv, pos_semi_var, neg_semi_var, rq]


def _group_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate financial market data on a daily basis."""
    df = (
        df.between_time("09:30", "16:00")
        .groupby(pd.Grouper(freq="D"))
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "volume": "sum",
                "close": _rv_generation,
            }
        )
    )

    df[["close", "rv", "pos_semi_var", "neg_semi_var", "rq"]] = pd.DataFrame(
        df.close.tolist(), index=df.index
    )
    df.dropna(inplace=True)
    return df


def _temporal_rv_comps_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rolling mean components for realized volatility."""
    df["rv_w_mean"] = df["rv"].rolling(5).mean()
    df["rv_m_mean"] = df["rv"].rolling(22).mean()
    return df


def _target_feature_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Generate target feature 'y' based on shifted values of 'rv'."""
    df["y"] = df["rv"].shift(-1)
    return df


def _dollar_volume_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Generate dollar volume feature."""
    df["dollar_volume"] = df["volume"] * df["close"]
    df["dollar_volume"] = np.log(df["dollar_volume"])
    df["dollar_volume"] = df["dollar_volume"].diff()
    return df


def _one_week_mom_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Generate one-week momentum feature."""
    df["week_mom"] = np.log(df["close"]) - np.log(df["close"].shift(5))
    return df


def _make_rv_scaling(scale: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create an RV scaling function with a configurable scale factor.
    """

    def _rv_scaling(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "rv",
            "rv_w_mean",
            "rv_m_mean",
            "pos_semi_var",
            "neg_semi_var",
            "rq",
        ]
        for col in cols:
            if col in df.columns:
                df[col] *= scale
        return df

    return _rv_scaling


def _select_features(
    df: pd.DataFrame, backtest_start_day: int, predictor_set: str
) -> Optional[pd.DataFrame]:
    """Select specific features based on backtest start day and predictor set."""
    feature_map = {
        (1, "std"): ["date", "rv", "y"],
        (1, "std-q"): ["date", "rv", "rq", "y"],
        (1, "std-semi"): ["date", "pos_semi_var", "neg_semi_var", "y"],
        (1, "ext"): ["date", "rv", "ads", "epu", "hsi", "vix", "3mtb", "ea", "y"],
        (1, "ext-q"): [
            "date",
            "rv",
            "rq",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (1, "ext-semi"): [
            "date",
            "pos_semi_var",
            "neg_semi_var",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (5, "std"): ["date", "rv", "rv_w_mean", "y"],
        (5, "std-q"): ["date", "rv", "rq", "rv_w_mean", "y"],
        (5, "std-semi"): ["date", "pos_semi_var", "neg_semi_var", "rv_w_mean", "y"],
        (5, "ext"): [
            "date",
            "rv",
            "rv_w_mean",
            "dollar_volume",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (5, "ext-q"): [
            "date",
            "rv",
            "rv_w_mean",
            "rq",
            "dollar_volume",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (5, "ext-semi"): [
            "date",
            "pos_semi_var",
            "neg_semi_var",
            "rv_w_mean",
            "dollar_volume",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (22, "std"): ["date", "rv", "rv_w_mean", "rv_m_mean", "y"],
        (22, "std-q"): ["date", "rv", "rv_w_mean", "rv_m_mean", "rq", "y"],
        (22, "std-semi"): [
            "date",
            "pos_semi_var",
            "neg_semi_var",
            "rv_w_mean",
            "rv_m_mean",
            "y",
        ],
        (22, "ext"): [
            "date",
            "rv",
            "rv_w_mean",
            "rv_m_mean",
            "dollar_volume",
            "week_mom",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (22, "ext-q"): [
            "date",
            "rv",
            "rv_w_mean",
            "rv_m_mean",
            "rq",
            "dollar_volume",
            "week_mom",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
        (22, "ext-semi"): [
            "date",
            "pos_semi_var",
            "neg_semi_var",
            "rv_w_mean",
            "rv_m_mean",
            "dollar_volume",
            "week_mom",
            "ads",
            "epu",
            "hsi",
            "vix",
            "3mtb",
            "ea",
            "y",
        ],
    }

    cols = feature_map.get((backtest_start_day, predictor_set))
    return df[cols] if cols is not None else None


def _process_ads_data(fp: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """Process ADS (Aruoba, Diebold, and Scotti business conditions) index data."""
    df = pd.read_excel(fp)
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "ads"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y:%m:%d")
    df = df[df["date"] >= start_date]
    return df


def _process_epu_data(fp: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """Process Economic Policy Uncertainty (EPU) index data."""
    df = pd.read_excel(fp)
    df.rename(
        columns={df.columns[0]: "year", df.columns[1]: "month", df.columns[2]: "epu"},
        inplace=True,
    )
    df = df[["year", "month", "epu"]]
    df["epu"] = df["epu"].shift(1)
    df.dropna(inplace=True)
    df["Year"] = df["year"].astype(int)
    df["Month"] = df["month"].astype(int)

    def expand_to_daily(row: pd.Series) -> pd.DatetimeIndex:
        start = pd.Timestamp(year=row["Year"], month=row["Month"], day=1)
        end = start + pd.offsets.MonthEnd(1)
        return pd.date_range(start=start, end=end, freq="D")

    date_ranges = df.apply(expand_to_daily, axis=1)
    df = df.loc[df.index.repeat(date_ranges.str.len())].reset_index(drop=True)
    df["date"] = list(chain.from_iterable(date_ranges))

    df = df[["date", "epu"]]
    df = df[df["date"] >= start_date]
    return df


def _process_threemtb_data(fp: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """Process 3-Month Treasury Bill (3MTB) data."""
    df = pd.read_csv(fp)
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "3mtb"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["3mtb"] = df["3mtb"].replace(".", np.nan)
    df["3mtb"] = df["3mtb"].ffill()
    df["3mtb"] = df["3mtb"].astype(float)
    df["3mtb"] = df["3mtb"].diff()
    df = df[df["date"] >= start_date]
    return df


def _process_vix_data(fp: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """Process VIX (Volatility Index) data."""
    df = pd.read_csv(fp)
    df.rename(columns={df.columns[0]: "date", df.columns[4]: "vix"}, inplace=True)
    df = df[["date", "vix"]]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df[df["date"] >= start_date]
    return df


def _process_hsi_data(fp: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """Process Hang Seng Index (HSI) data."""
    df = pd.read_csv(fp)
    df.rename(columns={df.columns[0]: "date", df.columns[4]: "hsi"}, inplace=True)
    df = df[["date", "hsi"]]
    df[["date", "mins"]] = df["date"].str.split(" ", expand=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["hsi"] = np.log(df["hsi"]) - np.log(df["hsi"].shift(1))
    df = df[df["date"] >= start_date]
    df = df[["date", "hsi"]]
    return df


class MeanDatasetGenerator:
    """
    Generates mean datasets from multiple asset data files.
    """

    def __init__(self, data_dp: str) -> None:
        self.data_dp = Path(data_dp)

    def create_mean_dataset(
        self, target_assets: List[str], source_assets: List[str], predictor_set: str
    ) -> None:
        """
        Generate mean datasets for target assets based on source assets.
        """
        logger.info("Creating mean dataset")
        for target_asset in target_assets:
            dfs = []
            file_path = self.data_dp / f"{predictor_set}/{target_asset}.csv"
            target_df = pd.read_csv(file_path, parse_dates=["date"])
            target_df["date"] = pd.to_datetime(target_df["date"])
            target_partial_df = target_df[["date", "rv"]]
            dfs.append(target_partial_df)

            for source_asset in source_assets:
                file_path = self.data_dp / f"{predictor_set}/{source_asset}.csv"
                df = pd.read_csv(file_path, parse_dates=["date"])
                df["date"] = pd.to_datetime(df["date"])
                partial_df = df[["date", "rv"]]
                dfs.append(partial_df)

            concated_df = pd.concat(dfs, ignore_index=True)
            mean_df = concated_df.groupby("date", as_index=False).mean()
            mean_df["mean_rv"] = mean_df["rv"]

            extended_df = target_df.merge(
                mean_df[["date", "mean_rv"]], on="date", how="left"
            )
            extended_df = extended_df[
                [c for c in extended_df.columns if c != "y"] + ["y"]
            ]

            output_dir = Path(f"data/processed/22/{predictor_set}-mean")
            output_dir.mkdir(parents=True, exist_ok=True)
            extended_df.to_csv(output_dir / f"{target_asset}.csv", index=False)
