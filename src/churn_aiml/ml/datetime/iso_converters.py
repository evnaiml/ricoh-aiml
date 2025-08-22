"""
ISO Week Date Converters (iso_converters.py)

A comprehensive module containing both ISO Week Date Converter and Week Midpoint Converter classes
for converting between ISO week dates and calendar dates with proper ISO 8601 compliance.

Includes full pandas integration for efficient batch processing and date calculations.
Requires pandas, numpy, math, and follows churn_aiml logging patterns.
"""
# %%
# -----------------------------------------------------------------------------
# Author: Evgeni Nikolaev
# emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-08
# CREATED ON: 2025-08-05
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh-USA. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh-USA and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh-USA
# -----------------------------------------------------------------------------
# %%
from datetime import datetime, timedelta, date
from typing import Union, Literal, Tuple, Optional
import re
import math

import pandas as pd
import numpy as np
from omegaconf import DictConfig
from churn_aiml.loggers.loguru.config import setup_logger, get_logger


class ISOWeekDateConverter:
    """
    A class for converting between ISO 8601 week dates and calendar dates.

    This class provides methods for:
    - Converting YYYYWW format to calendar dates
    - Converting calendar dates to ISO week dates
    - Handling ISO 8601 week date standards properly
    - Validating dates and week numbers

    ISO 8601 Week Date Rules:
    - Week 1 is the first week that contains at least 4 days of the new year
    - Weeks start on Monday (day 1) and end on Sunday (day 7)
    - Years can have 52 or 53 weeks
    - January 4th is always in week 1

    Example:
        >>> converter = ISOWeekDateConverter(config)
        >>> converter.convert_yyyywk_to_date(202401)
        '2024-01-01'
        >>> converter.convert_yyyywk_to_date(202401, weekday=7)
        '2024-01-07'
    """

    def __init__(self, config: DictConfig, log_operations: bool = False) -> None:
        """
        Initialize the ISO Week Date Converter.

        Args:
            config (DictConfig): Hydra configuration object
            log_operations: Whether to log conversion operations (default: False)
        """
        self.config = config
        self.log_operations = log_operations
        self.logger = get_logger()
        self.logger.info("Initialized ISOWeekDateConverter with log_operations={}", log_operations)

    def convert_yyyywk_to_date(
        self,
        yyyywk: Union[int, str],
        weekday: int = 1
    ) -> datetime:
        """
        Convert a year-week format (YYYYWW) to a calendar date.

        Uses proper ISO 8601 week date calculations for accurate results.

        Args:
            yyyywk: Year and week number in format YYYYWW (e.g., 202452 for 2024, week 52).
                   Can be passed as int or str.
            weekday: Day of week (1=Monday, 2=Tuesday, ..., 7=Sunday). Defaults to 1 (Monday).

        Returns:
            datetime: Date as datetime object

        Raises:
            ValueError: If yyyywk format is invalid, week is out of range, or weekday is invalid.

        Example:
            >>> converter = ISOWeekDateConverter(config)
            >>> converter.convert_yyyywk_to_date(202401)
            datetime.datetime(2024, 1, 1, 0, 0)
            >>> converter.convert_yyyywk_to_date(202452, weekday=5)
            datetime.datetime(2024, 12, 27, 0, 0)
        """
        try:
            # Input validation and parsing
            year, week = self._parse_yyyywk(yyyywk)
            self._validate_week_and_weekday(year, week, weekday)

            # Convert to calendar date using ISO 8601 rules
            date_obj = self._iso_week_to_date(year, week, weekday)

            if self.log_operations:
                self.logger.debug(
                    "Converted YYYYWK {} weekday {} to: {}",
                    yyyywk, weekday, date_obj.strftime("%Y-%m-%d")
                )

            return date_obj

        except Exception as e:
            self.logger.error(
                "Error converting YYYYWK {} weekday {} to date: {}",
                yyyywk, weekday, str(e)
            )
            raise

    def _parse_yyyywk(self, yyyywk: Union[int, str]) -> Tuple[int, int]:
        """Parse YYYYWW format into year and week components."""
        yyyywk_str = str(yyyywk)

        if not re.match(r'^\d{6}$', yyyywk_str):
            raise ValueError(f"Invalid YYYYWW format: {yyyywk}. Expected 6 digits (e.g., 202401)")

        year = int(yyyywk_str[:4])
        week = int(yyyywk_str[4:6])

        if year < 1 or year > 9999:
            raise ValueError(f"Year must be between 1 and 9999, got: {year}")

        self.logger.debug("Parsed YYYYWK {} into year: {}, week: {}", yyyywk, year, week)
        return year, week

    def _validate_week_and_weekday(self, year: int, week: int, weekday: int) -> None:
        """Validate week number and weekday for the given year."""
        if not (1 <= weekday <= 7):
            raise ValueError(f"Weekday must be between 1 and 7, got: {weekday}")

        max_weeks = self.get_weeks_in_year(year)
        if not (1 <= week <= max_weeks):
            raise ValueError(
                f"Week must be between 1 and {max_weeks} for year {year}, got: {week}"
            )

    def _iso_week_to_date(self, year: int, week: int, weekday: int) -> datetime:
        """Convert ISO year, week, and weekday to calendar date.
        
        Returns datetime object for internal consistency.
        String conversion happens at the public interface level.
        """
        jan_4 = datetime(year, 1, 4)
        days_to_monday = jan_4.weekday()
        week_1_monday = jan_4 - timedelta(days=days_to_monday)
        target_week_monday = week_1_monday + timedelta(weeks=week - 1)
        target_date = target_week_monday + timedelta(days=weekday - 1)
        
        if self.log_operations:
            self.logger.debug("ISO week {}-W{:02d}-{} -> {}", 
                            year, week, weekday, target_date.date())
        
        return target_date

    def get_week_range(self, yyyywk: Union[int, str]) -> Tuple[str, str]:
        """Get the Monday and Sunday dates for a given ISO week."""
        monday = self.convert_yyyywk_to_date(yyyywk, weekday=1)
        sunday = self.convert_yyyywk_to_date(yyyywk, weekday=7)

        if self.log_operations:
            self.logger.debug("Week {} range: {} to {}", yyyywk, monday, sunday)

        return monday, sunday

    def date_to_iso_week(self, date: Union[str, datetime]) -> Tuple[int, int, int]:
        """Convert a calendar date to ISO week date."""
        try:
            if isinstance(date, str):
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            else:
                date_obj = date

            iso_year, iso_week, iso_weekday = date_obj.isocalendar()

            if self.log_operations:
                self.logger.debug(
                    "Converted date {} to ISO: year={}, week={}, weekday={}",
                    date, iso_year, iso_week, iso_weekday
                )

            return iso_year, iso_week, iso_weekday

        except ValueError as e:
            self.logger.error("Invalid date format. Expected 'YYYY-MM-DD', got: {}", date)
            raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD', got: {date}") from e

    def convert_date_to_yyyywk(self, date: Union[str, datetime, pd.Timestamp]) -> int:
        """
        Convert a date to YYYYWW format using proper ISO 8601 calculation.

        Args:
            date: Date to convert. Can be:
                - String in "YYYY-MM-DD" format
                - datetime object
                - pandas Timestamp

        Returns:
            int: Week in YYYYWW format

        Example:
            >>> converter = ISOWeekDateConverter(config)
            >>> converter.convert_date_to_yyyywk("2024-01-01")
            202401
            >>> converter.convert_date_to_yyyywk(pd.to_datetime("2024-07-15"))
            202429
        """
        try:
            # Convert to datetime if needed
            if isinstance(date, str):
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            elif isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            else:
                date_obj = date

            # Use ISO calendar to get proper week number
            iso_year, iso_week, _ = date_obj.isocalendar()
            result = iso_year * 100 + iso_week

            if self.log_operations:
                self.logger.debug("Converted date {} to YYYYWK: {}", date, result)

            return result

        except Exception as e:
            self.logger.error("Error converting date {} to YYYYWK: {}", date, str(e))
            raise

    def convert_date_to_yyyywk_pandas(self, date: Union[str, datetime, pd.Timestamp, pd.Series]) -> Union[int, pd.Series]:
        """
        Convert date(s) to YYYYWW format with pandas optimization for batch processing.

        Efficiently handles both single dates and pandas Series.

        Args:
            date: Date(s) to convert. Can be single date or pandas Series.

        Returns:
            Union[int, pd.Series]: Week(s) in YYYYWW format

        Example:
            >>> converter = ISOWeekDateConverter(config)
            >>> dates = pd.Series(["2024-01-01", "2024-07-15", "2024-12-31"])
            >>> converter.convert_date_to_yyyywk_pandas(dates)
            0    202401
            1    202429
            2    202501
            dtype: int64
        """
        try:
            if isinstance(date, pd.Series):
                # Batch processing for pandas Series
                self.logger.info("Processing batch conversion for {} dates", len(date))
                dt_series = pd.to_datetime(date)
                iso_years = dt_series.dt.isocalendar().year
                iso_weeks = dt_series.dt.isocalendar().week
                result = iso_years * 100 + iso_weeks

                if self.log_operations:
                    self.logger.debug("Batch converted {} dates to YYYYWK format", len(date))

                return result
            else:
                # Single date processing
                return self.convert_date_to_yyyywk(date)

        except Exception as e:
            self.logger.error("Error in pandas batch conversion: {}", str(e))
            raise

    def date_to_yyyywk(self, date: Union[str, datetime]) -> int:
        """Convert a calendar date to YYYYWW format."""
        iso_year, iso_week, _ = self.date_to_iso_week(date)
        return iso_year * 100 + iso_week

    def get_weeks_in_year(self, year: int) -> int:
        """Get the number of ISO weeks in a given year (52 or 53)."""
        dec_28 = datetime(year, 12, 28)
        _, last_week, _ = dec_28.isocalendar()

        if self.log_operations:
            self.logger.debug("Year {} has {} ISO weeks", year, last_week)

        return last_week

    def is_valid_yyyywk(self, yyyywk: Union[int, str]) -> bool:
        """Check if a YYYYWW format is valid."""
        try:
            year, week = self._parse_yyyywk(yyyywk)
            max_weeks = self.get_weeks_in_year(year)
            is_valid = 1 <= week <= max_weeks

            if self.log_operations:
                self.logger.debug("Validation for YYYYWK {}: {}", yyyywk, is_valid)

            return is_valid
        except ValueError:
            if self.log_operations:
                self.logger.debug("Validation for YYYYWK {}: False (invalid format)", yyyywk)
            return False

    def get_current_yyyywk(self) -> int:
        """Get the current week in YYYYWW format."""
        current = self.date_to_yyyywk(datetime.now())
        self.logger.info("Current YYYYWK: {}", current)
        return current

    def get_week_info(self, yyyywk: Union[int, str]) -> dict:
        """Get comprehensive information about a given week."""
        try:
            year, week = self._parse_yyyywk(yyyywk)
            monday, sunday = self.get_week_range(yyyywk)

            info = {
                'year': year,
                'week': week,
                'yyyywk': int(f"{year}{week:02d}"),
                'monday': monday,
                'tuesday': self.convert_yyyywk_to_date(yyyywk, 2),
                'wednesday': self.convert_yyyywk_to_date(yyyywk, 3),
                'thursday': self.convert_yyyywk_to_date(yyyywk, 4),
                'friday': self.convert_yyyywk_to_date(yyyywk, 5),
                'saturday': self.convert_yyyywk_to_date(yyyywk, 6),
                'sunday': sunday,
                'total_weeks_in_year': self.get_weeks_in_year(year)
            }

            if self.log_operations:
                self.logger.debug("Generated week info for YYYYWK {}: {} to {}", yyyywk, monday, sunday)

            return info

        except Exception as e:
            self.logger.error("Error generating week info for YYYYWK {}: {}", yyyywk, str(e))
            raise

    @staticmethod
    def is_leap_year(year: int) -> bool:
        """Check if a year is a leap year using complete rules."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


class WeekMidpointConverter:
    """
    A class for converting YYYYWW format to actual midpoint dates of weeks.

    This class provides multiple interpretations of "mid date":
    - Wednesday (middle weekday): Most common interpretation
    - Arithmetic midpoint: Exact middle between Monday and Sunday
    - Thursday (ISO week middle): Day 4 of 7 in ISO week

    Uses proper ISO 8601 week calculations for accurate, standards-compliant results.

    Example:
        >>> converter = WeekMidpointConverter(config)
        >>> converter.convert_yyyywk_to_actual_mid_date(202401)
        '2024-01-03'  # Wednesday of week 1, 2024
    """

    def __init__(self, config: DictConfig, log_operations: bool = False) -> None:
        """
        Initialize the Week Midpoint Converter.

        Args:
            config (DictConfig): Hydra configuration object
            log_operations: Whether to log conversion operations (default: False)
        """
        self.config = config
        self.log_operations = log_operations
        self.logger = get_logger()
        self.logger.info("Initialized WeekMidpointConverter with log_operations={}", log_operations)

    def convert_yyyywk_to_actual_mid_date(
        self,
        yyyywk: Union[int, str],
        midpoint_type: Literal["wednesday", "arithmetic", "thursday"] = "wednesday"
    ) -> datetime:
        """
        Convert a year-week format (YYYYWW) to the actual midpoint date of that week.

        Args:
            yyyywk: Year and week number in format YYYYWW (e.g., 202452 for 2024, week 52).
            midpoint_type: Type of midpoint calculation:
                - "wednesday": Wednesday of the week (most common interpretation)
                - "arithmetic": Exact arithmetic midpoint between Monday and Sunday
                - "thursday": Thursday of the week (ISO day 4 of 7)

        Returns:
            datetime: Midpoint date as datetime object

        Raises:
            ValueError: If yyyywk format is invalid or week is out of range.
        """
        try:
            year, week = self._parse_yyyywk(yyyywk)
            self._validate_week(year, week)

            if midpoint_type == "wednesday":
                result = self._iso_week_to_date(year, week, 3)
            elif midpoint_type == "thursday":
                result = self._iso_week_to_date(year, week, 4)
            elif midpoint_type == "arithmetic":
                result = self._calculate_arithmetic_midpoint(year, week)
            else:
                raise ValueError(f"Invalid midpoint_type: {midpoint_type}")

            if self.log_operations:
                self.logger.debug(
                    "Converted YYYYWK {} to {} midpoint: {}",
                    yyyywk, midpoint_type, result.strftime("%Y-%m-%d")
                )

            return result

        except Exception as e:
            self.logger.error(
                "Error converting YYYYWK {} to {} midpoint: {}",
                yyyywk, midpoint_type, str(e)
            )
            raise

    def get_days_diff(
        self,
        yyyywk: Union[int, str],
        last_yyyywk: Union[int, str],
        midpoint_type: Literal["wednesday", "arithmetic", "thursday"] = "wednesday"
    ) -> float:
        """
        Calculate the difference in days between two week midpoints using pandas.

        Args:
            yyyywk: Current week in YYYYWW format
            last_yyyywk: Previous week in YYYYWW format
            midpoint_type: Type of midpoint to use for calculation

        Returns:
            float: Difference in days (positive if last_yyyywk is after yyyywk)

        Example:
            >>> converter = WeekMidpointConverter(config)
            >>> converter.get_days_diff(202401, 202402)
            7.0  # One week difference
            >>> converter.get_days_diff(202402, 202401)
            -7.0  # Negative because we're going backwards
        """
        try:
            curr_date = self.convert_yyyywk_to_actual_mid_date(yyyywk, midpoint_type)
            last_date = self.convert_yyyywk_to_actual_mid_date(last_yyyywk, midpoint_type)

            # Use pandas for date calculation
            curr = pd.Timestamp(curr_date)
            last = pd.Timestamp(last_date)
            days_diff = (last - curr) / np.timedelta64(1, 'D')
            result = float(days_diff)

            if self.log_operations:
                self.logger.debug(
                    "Days difference between YYYYWK {} and {}: {:.1f} days",
                    yyyywk, last_yyyywk, result
                )

            return result

        except Exception as e:
            self.logger.error(
                "Error calculating days difference between YYYYWK {} and {}: {}",
                yyyywk, last_yyyywk, str(e)
            )
            raise

    def _parse_yyyywk(self, yyyywk: Union[int, str]) -> Tuple[int, int]:
        """Parse YYYYWW format into year and week components."""
        yyyywk_str = str(yyyywk)

        if not re.match(r'^\d{6}$', yyyywk_str):
            raise ValueError(f"Invalid YYYYWW format: {yyyywk}")

        year = int(yyyywk_str[:4])
        week = int(yyyywk_str[4:6])

        if year < 1 or year > 9999:
            raise ValueError(f"Year must be between 1 and 9999, got: {year}")

        return year, week

    def _validate_week(self, year: int, week: int) -> None:
        """Validate week number for the given year."""
        max_weeks = self.get_weeks_in_year(year)
        if not (1 <= week <= max_weeks):
            raise ValueError(f"Week must be between 1 and {max_weeks} for year {year}, got: {week}")

    def _iso_week_to_date(self, year: int, week: int, weekday: int) -> datetime:
        """Convert ISO year, week, and weekday to calendar date.
        
        Returns datetime object for consistency.
        """
        jan_4 = datetime(year, 1, 4)
        days_to_monday = jan_4.weekday()
        week_1_monday = jan_4 - timedelta(days=days_to_monday)
        target_week_monday = week_1_monday + timedelta(weeks=week - 1)
        target_date = target_week_monday + timedelta(days=weekday - 1)
        return target_date

    def _calculate_arithmetic_midpoint(self, year: int, week: int) -> datetime:
        """Calculate the exact arithmetic midpoint between Monday and Sunday.
        
        Returns datetime object for consistency.
        """
        monday_date = self._iso_week_to_date(year, week, 1)
        sunday_date = self._iso_week_to_date(year, week, 7)
        days_difference = (sunday_date - monday_date).days
        midpoint_days = days_difference / 2.0
        midpoint_date = monday_date + timedelta(days=math.floor(midpoint_days + 0.5))
        return midpoint_date

    def get_weeks_in_year(self, year: int) -> int:
        """Get the number of ISO weeks in a given year (52 or 53)."""
        dec_28 = datetime(year, 12, 28)
        _, last_week, _ = dec_28.isocalendar()
        return last_week

    def get_week_range(self, yyyywk: Union[int, str]) -> Tuple[str, str]:
        """Get the Monday and Sunday dates for a given ISO week."""
        year, week = self._parse_yyyywk(yyyywk)
        monday = self._iso_week_to_date(year, week, 1)
        sunday = self._iso_week_to_date(year, week, 7)
        return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")

    def convert_date_to_yyyywk(self, date: Union[str, datetime, pd.Timestamp]) -> int:
        """
        Convert a date to YYYYWW format using proper ISO 8601 calculation.

        Args:
            date: Date to convert (string, datetime, or pandas Timestamp)

        Returns:
            int: Week in YYYYWW format

        Example:
            >>> converter = WeekMidpointConverter(config)
            >>> converter.convert_date_to_yyyywk("2024-01-01")
            202401
        """
        try:
            # Convert to datetime if needed
            if isinstance(date, str):
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            elif isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            else:
                date_obj = date

            # Use ISO calendar to get proper week number
            iso_year, iso_week, _ = date_obj.isocalendar()
            result = iso_year * 100 + iso_week

            if self.log_operations:
                self.logger.debug("Converted date {} to YYYYWK: {}", date, result)

            return result

        except Exception as e:
            self.logger.error("Error converting date {} to YYYYWK: {}", date, str(e))
            raise

    def convert_date_to_yyyywk_pandas(self, date: Union[str, datetime, pd.Timestamp, pd.Series]) -> Union[int, pd.Series]:
        """
        Convert date(s) to YYYYWW format with pandas optimization for batch processing.

        Efficiently handles both single dates and pandas Series.

        Args:
            date: Date(s) to convert. Can be single date or pandas Series.

        Returns:
            Union[int, pd.Series]: Week(s) in YYYYWW format
        """
        if isinstance(date, pd.Series):
            dt_series = pd.to_datetime(date)
            iso_years = dt_series.dt.isocalendar().year
            iso_weeks = dt_series.dt.isocalendar().week
            return iso_years * 100 + iso_weeks
        else:
            return self.convert_date_to_yyyywk(date)

    def get_week_midpoint_info(self, yyyywk: Union[int, str]) -> dict:
        """Get comprehensive midpoint information for a given week."""
        try:
            year, week = self._parse_yyyywk(yyyywk)

            monday = self._iso_week_to_date(year, week, 1)
            tuesday = self._iso_week_to_date(year, week, 2)
            wednesday = self._iso_week_to_date(year, week, 3)
            thursday = self._iso_week_to_date(year, week, 4)
            friday = self._iso_week_to_date(year, week, 5)
            saturday = self._iso_week_to_date(year, week, 6)
            sunday = self._iso_week_to_date(year, week, 7)

            arithmetic_midpoint = self._calculate_arithmetic_midpoint(year, week)

            info = {
                'year': year,
                'week': week,
                'yyyywk': int(f"{year}{week:02d}"),
                'monday': monday,
                'tuesday': tuesday,
                'wednesday': wednesday,
                'thursday': thursday,
                'friday': friday,
                'saturday': saturday,
                'sunday': sunday,
                'wednesday_midpoint': wednesday,
                'thursday_midpoint': thursday,
                'arithmetic_midpoint': arithmetic_midpoint,
                'total_weeks_in_year': self.get_weeks_in_year(year)
            }

            if self.log_operations:
                self.logger.debug("Generated midpoint info for YYYYWK {}", yyyywk)

            return info

        except Exception as e:
            self.logger.error("Error generating midpoint info for YYYYWK {}: {}", yyyywk, str(e))
            raise

    def is_valid_yyyywk(self, yyyywk: Union[int, str]) -> bool:
        """Check if a YYYYWW format is valid."""
        try:
            year, week = self._parse_yyyywk(yyyywk)
            max_weeks = self.get_weeks_in_year(year)
            return 1 <= week <= max_weeks
        except ValueError:
            return False

    def get_current_yyyywk(self) -> int:
        """Get the current week in YYYYWW format."""
        now = datetime.now()
        iso_year, iso_week, _ = now.isocalendar()
        return iso_year * 100 + iso_week

    @staticmethod
    def is_leap_year(year: int) -> bool:
        """Check if a year is a leap year using complete rules."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# %%
# Example usage and comprehensive demonstration
if __name__ == "__main__":
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from churn_aiml.utils.find_paths import ProjectRootFinder
    # %%
    # Set paths
    churn_aiml_dir = ProjectRootFinder().find_path()
    conf_dir = churn_aiml_dir / "conf"
    print(f"config path: {conf_dir}")
    # %%
    # Load Hydra configuration manually
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config")
    # %%
    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("Starting ISO Week Date Converters demonstration")

    print("=" * 70)
    print("COMPREHENSIVE WEEK DATE CONVERTERS DEMONSTRATION")
    print("=" * 70)
    # %%
    # Create converter instances with logging enabled for demo
    iso_converter = ISOWeekDateConverter(config=cfg, log_operations=True)
    midpoint_converter = WeekMidpointConverter(config=cfg, log_operations=True)

    logger.info("Created converter instances")

    print("\n" + "=" * 50)
    print("1. ISO WEEK DATE CONVERTER EXAMPLES")
    print("=" * 50)
    # %%
    # Basic date conversion examples
    print("\n📅 Basic YYYYWW to Date Conversions:")
    test_weeks = [202401, 202410, 202452]

    for yyyywk in test_weeks:
        monday = iso_converter.convert_yyyywk_to_date(yyyywk, weekday=1)
        wednesday = iso_converter.convert_yyyywk_to_date(yyyywk, weekday=3)
        friday = iso_converter.convert_yyyywk_to_date(yyyywk, weekday=5)
        print(f"   {yyyywk}: Mon={monday}, Wed={wednesday}, Fri={friday}")
    # %%
    # Week range examples
    print(f"\n📊 Week Range Information:")
    for yyyywk in [202401, 202452]:
        monday, sunday = iso_converter.get_week_range(yyyywk)
        weeks_in_year = iso_converter.get_weeks_in_year(int(str(yyyywk)[:4]))
        print(f"   Week {yyyywk}: {monday} to {sunday} (Year has {weeks_in_year} weeks)")

    # %%
    # Reverse conversion examples
    print(f"\n🔄 Date to Week Conversions:")
    test_dates = ["2024-01-01", "2024-07-15", "2024-12-31"]
    for date in test_dates:
        yyyywk = iso_converter.convert_date_to_yyyywk(date)
        iso_year, iso_week, iso_weekday = iso_converter.date_to_iso_week(date)
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_name = weekday_names[iso_weekday - 1]
        print(f"   {date} → {yyyywk} (ISO {iso_year}-W{iso_week:02d}-{iso_weekday} {weekday_name})")
    # %%
    # Pandas batch conversion example
    print(f"\n📊 Pandas Batch Date Conversions:")
    dates_series = pd.Series(["2024-01-01", "2024-02-15", "2024-06-30", "2024-12-25"])
    yyyywk_series = iso_converter.convert_date_to_yyyywk_pandas(dates_series)
    for date, yyyywk in zip(dates_series, yyyywk_series):
        print(f"   {date} → {yyyywk}")

    print("\n" + "=" * 50)
    print("2. WEEK MIDPOINT CONVERTER EXAMPLES")
    print("=" * 50)
    # %%
    # Midpoint conversion examples
    print(f"\n🎯 YYYYWW to Midpoint Date Conversions:")
    for yyyywk in test_weeks:
        wednesday = midpoint_converter.convert_yyyywk_to_actual_mid_date(yyyywk, "wednesday")
        arithmetic = midpoint_converter.convert_yyyywk_to_actual_mid_date(yyyywk, "arithmetic")
        thursday = midpoint_converter.convert_yyyywk_to_actual_mid_date(yyyywk, "thursday")
        print(f"   {yyyywk}:")
        print(f"     Wednesday midpoint:  {wednesday}")
        print(f"     Arithmetic midpoint: {arithmetic}")
        print(f"     Thursday midpoint:   {thursday}")
        print()

    # %%
    # Days difference examples (using your enhanced function)
    print(f"📏 Days Difference Between Week Midpoints:")
    week_pairs = [(202401, 202402), (202401, 202405), (202410, 202408)]

    for yyyywk1, yyyywk2 in week_pairs:
        days_diff = midpoint_converter.get_days_diff(yyyywk1, yyyywk2)

        print(f"   Between {yyyywk1} and {yyyywk2}: {days_diff:.1f} days")

        # Show the actual dates for context
        date1 = midpoint_converter.convert_yyyywk_to_actual_mid_date(yyyywk1)
        date2 = midpoint_converter.convert_yyyywk_to_actual_mid_date(yyyywk2)
        print(f"     {date1} → {date2}")
        print()

    # %%
    # Date to week conversion examples for midpoint converter
    print(f"📅 Date to Week Conversions (Midpoint Converter):")
    sample_dates = ["2024-03-06", "2024-08-14", "2024-11-20"]  # Various midpoint dates
    for date in sample_dates:
        yyyywk = midpoint_converter.convert_date_to_yyyywk(date)
        midpoint = midpoint_converter.convert_yyyywk_to_actual_mid_date(yyyywk, "wednesday")
        print(f"   {date} → Week {yyyywk} (Wednesday midpoint: {midpoint})")
    print()

    print("\n" + "=" * 50)
    print("3. COMPREHENSIVE WEEK INFORMATION")
    print("=" * 50)
    # %%
    # Detailed week information
    sample_week = 202410
    print(f"\n📋 Complete Information for Week {sample_week}:")
    # %%
    # ISO converter info
    iso_info = iso_converter.get_week_info(sample_week)
    print(f"   📅 ISO Week Info:")
    print(f"     Year: {iso_info['year']}, Week: {iso_info['week']}")
    print(f"     Monday:    {iso_info['monday']}")
    print(f"     Wednesday: {iso_info['wednesday']}")
    print(f"     Friday:    {iso_info['friday']}")
    print(f"     Sunday:    {iso_info['sunday']}")
    # %%
    # Midpoint converter info
    midpoint_info = midpoint_converter.get_week_midpoint_info(sample_week)
    print(f"   🎯 Midpoint Info:")
    print(f"     Wednesday midpoint:  {midpoint_info['wednesday_midpoint']}")
    print(f"     Arithmetic midpoint: {midpoint_info['arithmetic_midpoint']}")
    print(f"     Thursday midpoint:   {midpoint_info['thursday_midpoint']}")

    print("\n" + "=" * 50)
    print("4. VALIDATION AND UTILITY EXAMPLES")
    print("=" * 50)
    # %%
    # Validation examples
    print(f"\n✅ Validation Examples:")
    validation_tests = [202401, 202453, 202454, 202400, "invalid"]
    for test in validation_tests:
        iso_valid = iso_converter.is_valid_yyyywk(test)
        midpoint_valid = midpoint_converter.is_valid_yyyywk(test)
        status = "✅ Valid" if iso_valid else "❌ Invalid"
        print(f"   {test}: {status}")
    # %%
    # Current week
    current_week = iso_converter.get_current_yyyywk()
    print(f"\n📅 Current Week: {current_week}")
    # %%
    # Leap year examples
    print(f"\n🗓️  Leap Year Examples:")
    leap_test_years = [2024, 2023, 2000, 1900]
    for year in leap_test_years:
        is_leap = ISOWeekDateConverter.is_leap_year(year)
        weeks_in_year = iso_converter.get_weeks_in_year(year)
        status = "✅ Leap" if is_leap else "❌ Not Leap"
        print(f"   {year}: {status}, {weeks_in_year} weeks")

    print("\n" + "=" * 50)
    print("5. PRACTICAL USE CASES")
    print("=" * 50)

    print(f"\n💼 Business Applications:")
    # %%
    # Weekly reporting scenario
    current_week = 202410
    previous_week = 202409

    current_range = iso_converter.get_week_range(current_week)
    previous_range = iso_converter.get_week_range(previous_week)

    print(f"   📊 Weekly Reporting:")
    print(f"     Current Week {current_week}: {current_range[0]} to {current_range[1]}")
    print(f"     Previous Week {previous_week}: {previous_range[0]} to {previous_range[1]}")

    # Calculate days between midpoints for time-series analysis
    days_between = midpoint_converter.get_days_diff(previous_week, current_week)
    print(f"     Days between midpoints: {days_between:.0f} days")
    # %%
    # Weekly KPI tracking scenario
    print(f"\n   📈 Weekly KPI Tracking:")
    kpi_weeks = [202408, 202409, 202410, 202411]
    for i, week in enumerate(kpi_weeks):
        week_info = iso_converter.get_week_info(week)
        midpoint = midpoint_converter.convert_yyyywk_to_actual_mid_date(week)
        print(f"     Week {week}: {week_info['monday']} to {week_info['sunday']} (Midpoint: {midpoint})")

        if i > 0:
            days_diff = midpoint_converter.get_days_diff(kpi_weeks[i-1], week)
            print(f"       → {days_diff:.0f} days from previous week")
    # %%
    # Batch date processing example
    print(f"\n   🔄 Batch Date Processing with Pandas:")
    sample_transactions = pd.DataFrame({
        'transaction_date': ['2024-01-15', '2024-01-22', '2024-02-05', '2024-02-12', '2024-02-26'],
        'amount': [100, 250, 175, 300, 125]
    })
    # %%
    # Convert all transaction dates to weeks at once
    sample_transactions['yyyywk'] = iso_converter.convert_date_to_yyyywk_pandas(sample_transactions['transaction_date'])

    print(f"     📊 Transaction Analysis by Week:")
    for _, row in sample_transactions.iterrows():
        week_info = iso_converter.get_week_info(row['yyyywk'])
        print(f"       {row['transaction_date']} → Week {row['yyyywk']} (${row['amount']})")
        print(f"         Week span: {week_info['monday']} to {week_info['sunday']}")
    # %%$
    # Weekly aggregation example
    weekly_totals = sample_transactions.groupby('yyyywk')['amount'].sum()
    print(f"\n     📈 Weekly Totals:")
    for yyyywk, total in weekly_totals.items():
        print(f"       Week {yyyywk}: ${total}")

    print()

    print(f"\n   📋 Data Requirements:")
    print(f"     ✅ Pandas/NumPy - full functionality enabled")
    print(f"     ✅ Date-to-week conversion methods available")
    print(f"     ✅ Batch processing with pandas Series supported")
    print(f"     ✅ All date calculations use pandas for consistency and performance")
    print(f"     ✅ Loguru logging integrated for operations tracking")

    logger.info("Completed ISO Week Date Converters demonstration")

    print(f"\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("Both converters are ready for production use!")
    print("✅ Week-to-date conversions (ISO 8601 compliant)")
    print("✅ Date-to-week conversions (ISO 8601 standard)")
    print("✅ Midpoint calculations with multiple methods")
    print("✅ Days difference calculations")
    print("✅ Pandas integration for batch processing")
    print("✅ Comprehensive validation and utility functions")
    print("✅ Loguru logging for operations tracking and debugging")
    print("=" * 70)
# %%
