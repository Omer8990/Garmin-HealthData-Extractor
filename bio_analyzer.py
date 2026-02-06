"""
Bio Analyzer - The Logic Core (Expert Filters)
Implements the Blueprint, Huberman, Attia, and Bio-Budget metrics.
Derives biomarkers from raw Garmin data using pandas and numpy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HRVMetrics:
    """HRV-related metrics following Bryan Johnson's Blueprint protocol.
    Extended with 60d and 90d windows for comprehensive trend analysis."""
    today_ms: float
    baseline_7d_avg: float
    baseline_14d_avg: float
    baseline_30d_avg: float
    baseline_60d_avg: float  # Extended: 2-month perspective
    baseline_90d_avg: float  # Extended: 3-month perspective
    deviation_from_baseline: float  # percentage vs 30d
    deviation_from_90d: float  # percentage vs 90d (long-term deviation)
    z_score: float  # vs 30d
    z_score_90d: float  # vs 90d (how unusual in longer context)
    recovery_status: str  # "Optimal", "Strained", "Recovering"
    balance_classification: str
    trend_slope_7d: str  # Short-term trend
    trend_slope_30d: str  # Medium-term trend
    trend_slope_90d: str  # Long-term trend


@dataclass
class RHRMetrics:
    """Resting Heart Rate metrics with extended historical context."""
    today_bpm: float
    lowest_30d: float
    lowest_90d: float  # Extended: 3-month low
    avg_7d: float
    avg_30d: float
    avg_60d: float  # Extended: 2-month average
    avg_90d: float  # Extended: 3-month average
    delta_7d: float  # bpm change vs 7 days ago
    delta_30d: float  # bpm change vs 30 days ago
    trend_slope_7d: str  # "ascending", "descending", "stable"
    trend_slope_30d: str  # Medium-term trend
    sympathetic_drive: bool  # True if elevated >3bpm from 30d low


@dataclass
class SleepMetrics:
    """Sleep architecture metrics following Huberman's protocol."""
    score: int
    quality: str
    total_minutes: int
    deep_sleep_minutes: int
    deep_sleep_percentage: float
    deep_sleep_30d_avg: float
    rem_sleep_minutes: int
    rem_sleep_percentage: float
    rem_sleep_30d_avg: float
    light_sleep_minutes: int
    awake_minutes: int
    interruptions_count: int
    glymphatic_efficiency: bool  # Deep > 15%
    cognitive_repair: bool  # REM > 20%
    bed_time: str
    wake_time: str
    wake_time_variance_7d: float  # minutes
    circadian_disruption: bool


@dataclass
class StressMetrics:
    """Stress and Body Battery metrics."""
    daily_stress_avg: float
    high_stress_duration_min: int
    rest_stress_duration_min: int
    stress_balance_ratio: float
    am_charge_level: int
    pm_drain_level: int
    actual_recharge: int
    body_battery_floor: int
    resource_trend: str  # "positive", "negative", "stable"


@dataclass
class ActivityMetrics:
    """Activity and metabolic health metrics following Attia's protocol."""
    active_calories: int
    total_calories: int
    zone_2_minutes: int  # Moderate intensity
    zone_5_minutes: int  # Vigorous intensity
    weekly_zone_2_total: int
    weekly_zone_2_goal_progress: float  # percentage
    total_steps: int
    steps_30d_avg: int
    steps_delta_percent: float
    floors_climbed: int
    sedentary_flag: bool


class BioAnalyzer:
    """
    Expert-level biomarker analyzer.
    Processes raw Garmin data to derive health insights.
    """
    
    # Huberman protocol thresholds
    DEEP_SLEEP_TARGET_PERCENT = 15.0
    REM_SLEEP_TARGET_PERCENT = 20.0
    WAKE_TIME_VARIANCE_THRESHOLD_MINS = 30
    
    # Blueprint protocol thresholds
    RHR_ELEVATION_THRESHOLD_BPM = 3
    
    # Attia protocol goals
    WEEKLY_ZONE_2_GOAL_MINUTES = 180  # 3 hours/week
    SEDENTARY_STEPS_THRESHOLD = 5000
    
    def __init__(self, raw_data: Dict[str, Any]):
        """
        Initialize the analyzer with raw Garmin data.
        
        Args:
            raw_data: Dictionary containing daily_data from GarminCollector
        """
        self.raw_data = raw_data
        self.daily_data = raw_data.get("daily_data", {})
        self.activities = raw_data.get("activities", [])
        self._build_dataframes()
    
    def _build_dataframes(self) -> None:
        """Convert raw data into pandas DataFrames for analysis."""
        # HRV DataFrame
        hrv_records = []
        for date_str, day_data in self.daily_data.items():
            hrv = day_data.get("hrv")
            if hrv and isinstance(hrv, dict):
                # Handle various HRV data structures from Garmin
                hrv_value = self._extract_hrv_value(hrv)
                if hrv_value is not None:
                    hrv_records.append({
                        "date": pd.to_datetime(date_str),
                        "hrv_ms": hrv_value
                    })
        self.hrv_df = pd.DataFrame(hrv_records).sort_values("date") if hrv_records else pd.DataFrame()
        
        # RHR DataFrame
        rhr_records = []
        for date_str, day_data in self.daily_data.items():
            rhr = day_data.get("rhr")
            if rhr and isinstance(rhr, dict):
                rhr_value = self._extract_rhr_value(rhr)
                if rhr_value is not None:
                    rhr_records.append({
                        "date": pd.to_datetime(date_str),
                        "rhr_bpm": rhr_value
                    })
        self.rhr_df = pd.DataFrame(rhr_records).sort_values("date") if rhr_records else pd.DataFrame()
        
        # Sleep DataFrame
        sleep_records = []
        for date_str, day_data in self.daily_data.items():
            sleep = day_data.get("sleep")
            if sleep and isinstance(sleep, dict):
                sleep_record = self._extract_sleep_data(sleep, date_str)
                if sleep_record:
                    sleep_records.append(sleep_record)
        self.sleep_df = pd.DataFrame(sleep_records).sort_values("date") if sleep_records else pd.DataFrame()
        
        # Stats DataFrame (steps, calories, floors)
        stats_records = []
        for date_str, day_data in self.daily_data.items():
            stats = day_data.get("stats")
            if stats and isinstance(stats, dict):
                stats_record = self._extract_stats_data(stats, date_str)
                if stats_record:
                    stats_records.append(stats_record)
        self.stats_df = pd.DataFrame(stats_records).sort_values("date") if stats_records else pd.DataFrame()
        
        # Stress/Body Battery DataFrame
        stress_records = []
        for date_str, day_data in self.daily_data.items():
            stress = day_data.get("stress")
            body_battery = day_data.get("body_battery")
            stress_record = self._extract_stress_data(stress, body_battery, date_str)
            if stress_record:
                stress_records.append(stress_record)
        self.stress_df = pd.DataFrame(stress_records).sort_values("date") if stress_records else pd.DataFrame()
    
    def _extract_hrv_value(self, hrv_data: Dict[str, Any]) -> Optional[float]:
        """Extract HRV rMSSD value from various Garmin data structures."""
        # Try different possible keys
        if "hrvSummary" in hrv_data:
            summary = hrv_data["hrvSummary"]
            if isinstance(summary, dict):
                return summary.get("lastNightAvg") or summary.get("weeklyAvg")
        if "lastNightAvg" in hrv_data:
            return hrv_data["lastNightAvg"]
        if "hrvValue" in hrv_data:
            return hrv_data["hrvValue"]
        if "value" in hrv_data:
            return hrv_data["value"]
        # Check for nested structure
        if "dailyHrv" in hrv_data and hrv_data["dailyHrv"]:
            return hrv_data["dailyHrv"].get("hrvValue")
        return None
    
    def _extract_rhr_value(self, rhr_data: Dict[str, Any]) -> Optional[float]:
        """Extract RHR value from Garmin data."""
        # Check for nested structure from get_rhr_day endpoint
        # Format: allMetrics.metricsMap.WELLNESS_RESTING_HEART_RATE[0].value
        if "allMetrics" in rhr_data:
            metrics_map = rhr_data.get("allMetrics", {}).get("metricsMap", {})
            rhr_list = metrics_map.get("WELLNESS_RESTING_HEART_RATE", [])
            if rhr_list and isinstance(rhr_list, list) and len(rhr_list) > 0:
                return rhr_list[0].get("value")
        # Fallback to other common structures
        if "restingHeartRate" in rhr_data:
            return rhr_data["restingHeartRate"]
        if "value" in rhr_data:
            return rhr_data["value"]
        if "allDayRHR" in rhr_data:
            return rhr_data["allDayRHR"]
        return None
    
    def _extract_sleep_data(self, sleep_data: Dict[str, Any], date_str: str) -> Optional[Dict[str, Any]]:
        """Extract sleep metrics from Garmin sleep data."""
        try:
            # Handle different Garmin sleep data structures
            daily_sleep = sleep_data.get("dailySleepDTO") or sleep_data
            
            total_seconds = daily_sleep.get("sleepTimeSeconds", 0)
            deep_seconds = daily_sleep.get("deepSleepSeconds", 0)
            rem_seconds = daily_sleep.get("remSleepSeconds", 0)
            light_seconds = daily_sleep.get("lightSleepSeconds", 0)
            awake_seconds = daily_sleep.get("awakeSleepSeconds", 0)
            
            # Get sleep times
            sleep_start = daily_sleep.get("sleepStartTimestampGMT") or daily_sleep.get("sleepStartTimestampLocal")
            sleep_end = daily_sleep.get("sleepEndTimestampGMT") or daily_sleep.get("sleepEndTimestampLocal")
            
            # Extract bed/wake times
            bed_time = ""
            wake_time = ""
            if sleep_start:
                if isinstance(sleep_start, (int, float)):
                    bed_time = datetime.fromtimestamp(sleep_start / 1000).strftime("%H:%M")
                else:
                    bed_time = str(sleep_start)
            if sleep_end:
                if isinstance(sleep_end, (int, float)):
                    wake_time = datetime.fromtimestamp(sleep_end / 1000).strftime("%H:%M")
                else:
                    wake_time = str(sleep_end)
            
            # Count sleep interruptions
            sleep_levels = daily_sleep.get("sleepLevels", [])
            interruptions = sum(1 for level in sleep_levels if level.get("activityLevel") == 0) if sleep_levels else 0
            
            return {
                "date": pd.to_datetime(date_str),
                "score": daily_sleep.get("sleepScores", {}).get("overall", {}).get("value", 0) if isinstance(daily_sleep.get("sleepScores"), dict) else daily_sleep.get("sleepScore", 0),
                "total_seconds": total_seconds,
                "deep_seconds": deep_seconds,
                "rem_seconds": rem_seconds,
                "light_seconds": light_seconds,
                "awake_seconds": awake_seconds,
                "bed_time": bed_time,
                "wake_time": wake_time,
                "interruptions": interruptions
            }
        except Exception as e:
            print(f"Error extracting sleep data for {date_str}: {e}")
            return None
    
    def _extract_stats_data(self, stats_data: Dict[str, Any], date_str: str) -> Optional[Dict[str, Any]]:
        """Extract daily stats from Garmin data."""
        try:
            return {
                "date": pd.to_datetime(date_str),
                "steps": stats_data.get("totalSteps", 0) or 0,
                "active_calories": stats_data.get("activeKilocalories", 0) or 0,
                "total_calories": stats_data.get("totalKilocalories", 0) or 0,
                "floors_climbed": stats_data.get("floorsAscended", 0) or 0,
                "moderate_intensity_minutes": stats_data.get("moderateIntensityMinutes", 0) or 0,
                "vigorous_intensity_minutes": stats_data.get("vigorousIntensityMinutes", 0) or 0,
                "intensity_minutes_goal": stats_data.get("intensityMinutesGoal", 150) or 150
            }
        except Exception as e:
            print(f"Error extracting stats data for {date_str}: {e}")
            return None
    
    def _extract_stress_data(self, stress_data: Optional[Dict[str, Any]], 
                             body_battery: Optional[Any], 
                             date_str: str) -> Optional[Dict[str, Any]]:
        """Extract stress and body battery metrics."""
        try:
            result = {
                "date": pd.to_datetime(date_str),
                "avg_stress": 0,
                "high_stress_minutes": 0,
                "rest_minutes": 0,
                "body_battery_high": 0,
                "body_battery_low": 100
            }
            
            if stress_data and isinstance(stress_data, dict):
                result["avg_stress"] = stress_data.get("avgStressLevel", 0) or 0
                
                # Try direct duration fields first
                high_stress = stress_data.get("highStressDuration", 0)
                rest_stress = stress_data.get("restStressDuration", 0)
                
                # If not available, calculate from stressValuesArray
                # Stress levels: Rest (1-25), Low (26-50), Medium (51-75), High (76-100)
                # Each reading is ~3 minutes apart (180 seconds)
                if (not high_stress and not rest_stress) and "stressValuesArray" in stress_data:
                    stress_values = stress_data.get("stressValuesArray", [])
                    high_count = 0
                    rest_count = 0
                    reading_interval_minutes = 3  # Garmin typically records every 3 minutes
                    
                    for reading in stress_values:
                        if isinstance(reading, list) and len(reading) >= 2:
                            level = reading[1]
                            if level is not None and level > 0:  # Valid reading
                                if level >= 76:  # High stress
                                    high_count += 1
                                elif level <= 25:  # Rest/recovery
                                    rest_count += 1
                    
                    result["high_stress_minutes"] = high_count * reading_interval_minutes
                    result["rest_minutes"] = rest_count * reading_interval_minutes
                else:
                    result["high_stress_minutes"] = high_stress or 0
                    result["rest_minutes"] = rest_stress or 0
            
            # Extract body battery from stress data's bodyBatteryValuesArray if available
            # Format: [timestamp, status, level, delta] e.g., [1770328800000, 'MEASURED', 72, 3.0]
            if stress_data and "bodyBatteryValuesArray" in stress_data:
                bb_values_array = stress_data.get("bodyBatteryValuesArray", [])
                bb_levels = []
                for reading in bb_values_array:
                    if isinstance(reading, list) and len(reading) >= 3:
                        level = reading[2]  # Body battery level is at index 2
                        if level is not None and isinstance(level, (int, float)) and level > 0:
                            bb_levels.append(level)
                if bb_levels:
                    result["body_battery_high"] = max(bb_levels)
                    result["body_battery_low"] = min(bb_levels)
            
            # Fallback to separate body_battery parameter
            if result["body_battery_high"] == 0 and body_battery:
                if isinstance(body_battery, list) and body_battery:
                    # Find highest (morning) and lowest (evening) values
                    bb_values = [bb.get("bodyBatteryLevel", bb.get("charged", 0)) 
                                for bb in body_battery if isinstance(bb, dict)]
                    if bb_values:
                        result["body_battery_high"] = max(bb_values)
                        result["body_battery_low"] = min(bb_values)
                elif isinstance(body_battery, dict):
                    result["body_battery_high"] = body_battery.get("charged", body_battery.get("bodyBatteryHigh", 0))
                    result["body_battery_low"] = body_battery.get("drained", body_battery.get("bodyBatteryLow", 100))
            
            return result
        except Exception as e:
            print(f"Error extracting stress data for {date_str}: {e}")
            return None
    
    def _calculate_z_score(self, value: float, series: pd.Series) -> float:
        """Calculate Z-score for a value against a series."""
        if series.empty or series.std() == 0:
            return 0.0
        return (value - series.mean()) / series.std()
    
    def _calculate_trend_slope(self, series: pd.Series, window: int = 7) -> Tuple[float, str]:
        """
        Calculate the trend slope using linear regression.
        
        Returns:
            Tuple of (slope_value, trend_classification)
        """
        if len(series) < 2:
            return 0.0, "stable"
        
        recent = series.tail(window).values
        if len(recent) < 2:
            return 0.0, "stable"
        
        x = np.arange(len(recent))
        try:
            slope, _ = np.polyfit(x, recent, 1)
            
            # Classify trend based on slope magnitude
            if abs(slope) < 0.5:
                classification = "stable"
            elif slope > 0:
                classification = "ascending"
            else:
                classification = "descending"
            
            return float(slope), classification
        except Exception:
            return 0.0, "stable"
    
    def _get_rolling_average(self, df: pd.DataFrame, column: str, window: int) -> float:
        """Get rolling average for a column."""
        if df.empty or column not in df.columns:
            return 0.0
        return df[column].tail(window).mean()
    
    def _parse_wake_time_to_minutes(self, wake_time: str) -> Optional[int]:
        """Convert wake time string to minutes from midnight."""
        try:
            if not wake_time:
                return None
            parts = wake_time.split(":")
            if len(parts) >= 2:
                return int(parts[0]) * 60 + int(parts[1])
            return None
        except Exception:
            return None
    
    def analyze_hrv(self, target_date: str) -> HRVMetrics:
        """
        Analyze HRV metrics following Blueprint protocol with extended historical context.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to analyze
            
        Returns:
            HRVMetrics dataclass with all calculated values including 60d/90d perspectives
        """
        target_dt = pd.to_datetime(target_date)
        
        # Get today's HRV
        today_hrv = 0.0
        if not self.hrv_df.empty:
            today_row = self.hrv_df[self.hrv_df["date"] == target_dt]
            if not today_row.empty:
                today_hrv = today_row["hrv_ms"].values[0]
        
        # Calculate baselines across multiple timeframes
        baseline_7d = self._get_rolling_average(self.hrv_df, "hrv_ms", 7)
        baseline_14d = self._get_rolling_average(self.hrv_df, "hrv_ms", 14)
        baseline_30d = self._get_rolling_average(self.hrv_df, "hrv_ms", 30)
        baseline_60d = self._get_rolling_average(self.hrv_df, "hrv_ms", 60)
        baseline_90d = self._get_rolling_average(self.hrv_df, "hrv_ms", 90)
        
        # Calculate deviations (short-term vs 30d, long-term vs 90d)
        deviation_30d = ((today_hrv - baseline_30d) / baseline_30d * 100) if baseline_30d > 0 else 0
        deviation_90d = ((today_hrv - baseline_90d) / baseline_90d * 100) if baseline_90d > 0 else 0
        
        # Calculate Z-scores for different timeframes
        z_score_30d = self._calculate_z_score(today_hrv, self.hrv_df["hrv_ms"].tail(30)) if len(self.hrv_df) > 0 else 0
        z_score_90d = self._calculate_z_score(today_hrv, self.hrv_df["hrv_ms"].tail(90)) if len(self.hrv_df) > 0 else 0
        
        # Calculate trend slopes for multiple timeframes
        _, trend_7d = self._calculate_trend_slope(self.hrv_df["hrv_ms"], 7) if not self.hrv_df.empty else (0, "stable")
        _, trend_30d = self._calculate_trend_slope(self.hrv_df["hrv_ms"], 30) if not self.hrv_df.empty else (0, "stable")
        _, trend_90d = self._calculate_trend_slope(self.hrv_df["hrv_ms"], 90) if not self.hrv_df.empty else (0, "stable")
        
        # Determine recovery status (based on 30d baseline)
        if deviation_30d < -10:
            recovery_status = "Strained"
        elif deviation_30d > 10:
            recovery_status = "Peaking"
        else:
            recovery_status = "Optimal"
        
        # Balance classification
        if z_score_30d < -1.5:
            balance = "Unbalanced (Sympathetic Dominance)"
        elif z_score_30d > 1.5:
            balance = "Unbalanced (Parasympathetic Dominance)"
        else:
            balance = "Balanced"
        
        return HRVMetrics(
            today_ms=round(today_hrv, 1),
            baseline_7d_avg=round(baseline_7d, 1),
            baseline_14d_avg=round(baseline_14d, 1),
            baseline_30d_avg=round(baseline_30d, 1),
            baseline_60d_avg=round(baseline_60d, 1),
            baseline_90d_avg=round(baseline_90d, 1),
            deviation_from_baseline=round(deviation_30d, 1),
            deviation_from_90d=round(deviation_90d, 1),
            z_score=round(z_score_30d, 2),
            z_score_90d=round(z_score_90d, 2),
            recovery_status=recovery_status,
            balance_classification=balance,
            trend_slope_7d=trend_7d,
            trend_slope_30d=trend_30d,
            trend_slope_90d=trend_90d
        )
    
    def analyze_rhr(self, target_date: str) -> RHRMetrics:
        """
        Analyze Resting Heart Rate metrics with extended historical context.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to analyze
            
        Returns:
            RHRMetrics dataclass with all calculated values including extended windows
        """
        target_dt = pd.to_datetime(target_date)
        
        # Get today's RHR
        today_rhr = 0.0
        if not self.rhr_df.empty:
            today_row = self.rhr_df[self.rhr_df["date"] == target_dt]
            if not today_row.empty:
                today_rhr = today_row["rhr_bpm"].values[0]
        
        # Calculate baselines across multiple timeframes
        lowest_30d = self.rhr_df["rhr_bpm"].tail(30).min() if not self.rhr_df.empty else 0
        lowest_90d = self.rhr_df["rhr_bpm"].tail(90).min() if not self.rhr_df.empty else 0
        avg_7d = self._get_rolling_average(self.rhr_df, "rhr_bpm", 7)
        avg_30d = self._get_rolling_average(self.rhr_df, "rhr_bpm", 30)
        avg_60d = self._get_rolling_average(self.rhr_df, "rhr_bpm", 60)
        avg_90d = self._get_rolling_average(self.rhr_df, "rhr_bpm", 90)
        
        # Calculate 7-day delta
        if len(self.rhr_df) >= 7:
            week_ago_rhr = self.rhr_df["rhr_bpm"].iloc[-7]
            delta_7d = today_rhr - week_ago_rhr
        else:
            delta_7d = 0
        
        # Calculate 30-day delta
        if len(self.rhr_df) >= 30:
            month_ago_rhr = self.rhr_df["rhr_bpm"].iloc[-30]
            delta_30d = today_rhr - month_ago_rhr
        else:
            delta_30d = 0
        
        # Calculate trend slopes for multiple timeframes
        _, trend_slope_7d = self._calculate_trend_slope(self.rhr_df["rhr_bpm"], 7) if not self.rhr_df.empty else (0, "stable")
        _, trend_slope_30d = self._calculate_trend_slope(self.rhr_df["rhr_bpm"], 30) if not self.rhr_df.empty else (0, "stable")
        
        # Sympathetic drive indicator (elevation > 3bpm from 30d low)
        sympathetic_drive = (today_rhr - lowest_30d) > self.RHR_ELEVATION_THRESHOLD_BPM if lowest_30d > 0 else False
        
        return RHRMetrics(
            today_bpm=round(today_rhr, 0),
            lowest_30d=round(lowest_30d, 0),
            lowest_90d=round(lowest_90d, 0),
            avg_7d=round(avg_7d, 1),
            avg_30d=round(avg_30d, 1),
            avg_60d=round(avg_60d, 1),
            avg_90d=round(avg_90d, 1),
            delta_7d=round(delta_7d, 0),
            delta_30d=round(delta_30d, 0),
            trend_slope_7d=trend_slope_7d,
            trend_slope_30d=trend_slope_30d,
            sympathetic_drive=sympathetic_drive
        )
    
    def analyze_sleep(self, target_date: str) -> SleepMetrics:
        """
        Analyze sleep architecture following Huberman's protocol.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to analyze
            
        Returns:
            SleepMetrics dataclass with all calculated values
        """
        target_dt = pd.to_datetime(target_date)
        
        # Get today's sleep data
        today_sleep = {}
        if not self.sleep_df.empty:
            today_row = self.sleep_df[self.sleep_df["date"] == target_dt]
            if not today_row.empty:
                today_sleep = today_row.iloc[0].to_dict()
        
        total_seconds = today_sleep.get("total_seconds", 0)
        total_minutes = total_seconds // 60
        
        deep_seconds = today_sleep.get("deep_seconds", 0)
        deep_minutes = deep_seconds // 60
        deep_percent = (deep_seconds / total_seconds * 100) if total_seconds > 0 else 0
        
        rem_seconds = today_sleep.get("rem_seconds", 0)
        rem_minutes = rem_seconds // 60
        rem_percent = (rem_seconds / total_seconds * 100) if total_seconds > 0 else 0
        
        light_seconds = today_sleep.get("light_seconds", 0)
        light_minutes = light_seconds // 60
        
        awake_seconds = today_sleep.get("awake_seconds", 0)
        awake_minutes = awake_seconds // 60
        
        # Calculate 30-day averages
        deep_30d_avg = self._get_rolling_average(self.sleep_df, "deep_seconds", 30) // 60 if not self.sleep_df.empty else 0
        rem_30d_avg = self._get_rolling_average(self.sleep_df, "rem_seconds", 30) // 60 if not self.sleep_df.empty else 0
        
        # Huberman metrics
        glymphatic_efficiency = deep_percent >= self.DEEP_SLEEP_TARGET_PERCENT
        cognitive_repair = rem_percent >= self.REM_SLEEP_TARGET_PERCENT
        
        # Wake time variance
        wake_time = today_sleep.get("wake_time", "")
        wake_time_variance = 0.0
        circadian_disruption = False
        
        if not self.sleep_df.empty and "wake_time" in self.sleep_df.columns:
            wake_times_mins = self.sleep_df["wake_time"].tail(7).apply(self._parse_wake_time_to_minutes)
            valid_times = wake_times_mins.dropna()
            if len(valid_times) > 1:
                avg_wake_mins = valid_times.mean()
                today_wake_mins = self._parse_wake_time_to_minutes(wake_time)
                if today_wake_mins is not None:
                    wake_time_variance = abs(today_wake_mins - avg_wake_mins)
                    circadian_disruption = wake_time_variance > self.WAKE_TIME_VARIANCE_THRESHOLD_MINS
        
        # Sleep quality classification
        score = today_sleep.get("score", 0)
        if score >= 80:
            quality = "Good"
        elif score >= 60:
            quality = "Fair"
        else:
            quality = "Poor"
        
        return SleepMetrics(
            score=int(score),
            quality=quality,
            total_minutes=int(total_minutes),
            deep_sleep_minutes=int(deep_minutes),
            deep_sleep_percentage=round(deep_percent, 1),
            deep_sleep_30d_avg=int(deep_30d_avg),
            rem_sleep_minutes=int(rem_minutes),
            rem_sleep_percentage=round(rem_percent, 1),
            rem_sleep_30d_avg=int(rem_30d_avg),
            light_sleep_minutes=int(light_minutes),
            awake_minutes=int(awake_minutes),
            interruptions_count=today_sleep.get("interruptions", 0),
            glymphatic_efficiency=glymphatic_efficiency,
            cognitive_repair=cognitive_repair,
            bed_time=today_sleep.get("bed_time", ""),
            wake_time=wake_time,
            wake_time_variance_7d=round(wake_time_variance, 0),
            circadian_disruption=circadian_disruption
        )
    
    def analyze_stress(self, target_date: str) -> StressMetrics:
        """
        Analyze stress and body battery metrics.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to analyze
            
        Returns:
            StressMetrics dataclass with all calculated values
        """
        target_dt = pd.to_datetime(target_date)
        
        # Get today's stress/battery data
        today_stress = {}
        if not self.stress_df.empty:
            today_row = self.stress_df[self.stress_df["date"] == target_dt]
            if not today_row.empty:
                today_stress = today_row.iloc[0].to_dict()
        
        avg_stress = today_stress.get("avg_stress", 0)
        high_stress_min = today_stress.get("high_stress_minutes", 0)
        rest_min = today_stress.get("rest_minutes", 0)
        
        # Stress balance ratio (high stress / total tracked time)
        total_stress_time = high_stress_min + rest_min
        stress_ratio = high_stress_min / total_stress_time if total_stress_time > 0 else 0
        
        # Body battery
        am_charge = today_stress.get("body_battery_high", 0)
        pm_drain = today_stress.get("body_battery_low", 0)
        
        # Calculate overnight recharge (compare to yesterday)
        actual_recharge = 0
        if len(self.stress_df) >= 2:
            yesterday_row = self.stress_df.iloc[-2] if len(self.stress_df) >= 2 else None
            if yesterday_row is not None:
                yesterday_low = yesterday_row.get("body_battery_low", 0)
                actual_recharge = am_charge - yesterday_low
        
        # Resource trend (compare 7-day averages)
        resource_trend = "stable"
        if not self.stress_df.empty and len(self.stress_df) >= 7:
            recent_highs = self.stress_df["body_battery_high"].tail(7).mean()
            older_highs = self.stress_df["body_battery_high"].tail(14).head(7).mean() if len(self.stress_df) >= 14 else recent_highs
            if recent_highs > older_highs + 5:
                resource_trend = "positive"
            elif recent_highs < older_highs - 5:
                resource_trend = "negative"
        
        return StressMetrics(
            daily_stress_avg=round(avg_stress, 0),
            high_stress_duration_min=int(high_stress_min),
            rest_stress_duration_min=int(rest_min),
            stress_balance_ratio=round(stress_ratio, 2),
            am_charge_level=int(am_charge),
            pm_drain_level=int(pm_drain),
            actual_recharge=int(actual_recharge),
            body_battery_floor=int(pm_drain),
            resource_trend=resource_trend
        )
    
    def analyze_activity(self, target_date: str) -> ActivityMetrics:
        """
        Analyze activity metrics following Attia's protocol.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to analyze
            
        Returns:
            ActivityMetrics dataclass with all calculated values
        """
        target_dt = pd.to_datetime(target_date)
        
        # Get today's stats
        today_stats = {}
        if not self.stats_df.empty:
            today_row = self.stats_df[self.stats_df["date"] == target_dt]
            if not today_row.empty:
                today_stats = today_row.iloc[0].to_dict()
        
        steps = today_stats.get("steps", 0)
        active_cal = today_stats.get("active_calories", 0)
        total_cal = today_stats.get("total_calories", 0)
        floors = today_stats.get("floors_climbed", 0)
        zone_2_min = today_stats.get("moderate_intensity_minutes", 0)
        zone_5_min = today_stats.get("vigorous_intensity_minutes", 0)
        
        # Calculate weekly Zone 2 total (last 7 days)
        weekly_zone_2 = 0
        if not self.stats_df.empty:
            weekly_zone_2 = self.stats_df["moderate_intensity_minutes"].tail(7).sum()
        
        # Weekly goal progress
        goal_progress = (weekly_zone_2 / self.WEEKLY_ZONE_2_GOAL_MINUTES * 100) if self.WEEKLY_ZONE_2_GOAL_MINUTES > 0 else 0
        
        # Steps analysis
        steps_30d_avg = self._get_rolling_average(self.stats_df, "steps", 30) if not self.stats_df.empty else 0
        steps_delta = ((steps - steps_30d_avg) / steps_30d_avg * 100) if steps_30d_avg > 0 else 0
        
        # Sedentary flag
        sedentary = steps < self.SEDENTARY_STEPS_THRESHOLD
        
        return ActivityMetrics(
            active_calories=int(active_cal),
            total_calories=int(total_cal),
            zone_2_minutes=int(zone_2_min),
            zone_5_minutes=int(zone_5_min),
            weekly_zone_2_total=int(weekly_zone_2),
            weekly_zone_2_goal_progress=round(goal_progress, 1),
            total_steps=int(steps),
            steps_30d_avg=int(steps_30d_avg),
            steps_delta_percent=round(steps_delta, 1),
            floors_climbed=int(floors),
            sedentary_flag=sedentary
        )
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze cross-metric correlations for deeper insights.
        Correlations help identify relationships between metrics that averages miss.
        
        Returns:
            Dictionary containing correlation coefficients and interpretations
        """
        correlations = {}
        
        # Merge dataframes for correlation analysis
        try:
            # HRV vs Sleep Quality correlation
            if not self.hrv_df.empty and not self.sleep_df.empty:
                merged = pd.merge(
                    self.hrv_df[["date", "hrv_ms"]], 
                    self.sleep_df[["date", "score", "deep_seconds", "rem_seconds"]], 
                    on="date", how="inner"
                )
                if len(merged) >= 7:  # Need at least 7 points for meaningful correlation
                    hrv_sleep_corr = merged["hrv_ms"].corr(merged["score"])
                    hrv_deep_corr = merged["hrv_ms"].corr(merged["deep_seconds"])
                    correlations["hrv_sleep_quality"] = {
                        "coefficient": round(hrv_sleep_corr, 3) if not np.isnan(hrv_sleep_corr) else 0,
                        "strength": self._interpret_correlation(hrv_sleep_corr),
                        "insight": self._generate_correlation_insight("HRV", "sleep quality", hrv_sleep_corr)
                    }
                    correlations["hrv_deep_sleep"] = {
                        "coefficient": round(hrv_deep_corr, 3) if not np.isnan(hrv_deep_corr) else 0,
                        "strength": self._interpret_correlation(hrv_deep_corr),
                        "insight": self._generate_correlation_insight("HRV", "deep sleep", hrv_deep_corr)
                    }
            
            # HRV vs RHR correlation (should be negative - higher HRV with lower RHR)
            if not self.hrv_df.empty and not self.rhr_df.empty:
                merged = pd.merge(
                    self.hrv_df[["date", "hrv_ms"]], 
                    self.rhr_df[["date", "rhr_bpm"]], 
                    on="date", how="inner"
                )
                if len(merged) >= 7:
                    hrv_rhr_corr = merged["hrv_ms"].corr(merged["rhr_bpm"])
                    correlations["hrv_rhr"] = {
                        "coefficient": round(hrv_rhr_corr, 3) if not np.isnan(hrv_rhr_corr) else 0,
                        "strength": self._interpret_correlation(hrv_rhr_corr),
                        "expected_direction": "negative",
                        "insight": self._generate_correlation_insight("HRV", "RHR", hrv_rhr_corr, expected_negative=True)
                    }
            
            # Sleep vs Body Battery correlation
            if not self.sleep_df.empty and not self.stress_df.empty:
                merged = pd.merge(
                    self.sleep_df[["date", "score", "total_seconds"]], 
                    self.stress_df[["date", "body_battery_high", "avg_stress"]], 
                    on="date", how="inner"
                )
                if len(merged) >= 7:
                    sleep_battery_corr = merged["score"].corr(merged["body_battery_high"])
                    sleep_stress_corr = merged["total_seconds"].corr(merged["avg_stress"])
                    correlations["sleep_body_battery"] = {
                        "coefficient": round(sleep_battery_corr, 3) if not np.isnan(sleep_battery_corr) else 0,
                        "strength": self._interpret_correlation(sleep_battery_corr),
                        "insight": self._generate_correlation_insight("sleep quality", "morning energy", sleep_battery_corr)
                    }
                    correlations["sleep_duration_stress"] = {
                        "coefficient": round(sleep_stress_corr, 3) if not np.isnan(sleep_stress_corr) else 0,
                        "strength": self._interpret_correlation(sleep_stress_corr),
                        "insight": self._generate_correlation_insight("sleep duration", "stress", sleep_stress_corr, expected_negative=True)
                    }
            
            # Activity vs HRV (next day) - Training impact
            if not self.stats_df.empty and not self.hrv_df.empty:
                # Shift HRV by 1 day to see how activity affects next day's HRV
                hrv_shifted = self.hrv_df.copy()
                hrv_shifted["date"] = hrv_shifted["date"] - pd.Timedelta(days=1)
                merged = pd.merge(
                    self.stats_df[["date", "moderate_intensity_minutes", "steps"]], 
                    hrv_shifted[["date", "hrv_ms"]], 
                    on="date", how="inner"
                )
                if len(merged) >= 7:
                    activity_hrv_corr = merged["moderate_intensity_minutes"].corr(merged["hrv_ms"])
                    steps_hrv_corr = merged["steps"].corr(merged["hrv_ms"])
                    correlations["zone2_next_day_hrv"] = {
                        "coefficient": round(activity_hrv_corr, 3) if not np.isnan(activity_hrv_corr) else 0,
                        "strength": self._interpret_correlation(activity_hrv_corr),
                        "insight": self._generate_correlation_insight("Zone 2 training", "next-day HRV", activity_hrv_corr)
                    }
                    correlations["steps_next_day_hrv"] = {
                        "coefficient": round(steps_hrv_corr, 3) if not np.isnan(steps_hrv_corr) else 0,
                        "strength": self._interpret_correlation(steps_hrv_corr),
                        "insight": self._generate_correlation_insight("daily steps", "next-day HRV", steps_hrv_corr)
                    }
            
            # Calculate data quality for correlations
            correlations["data_quality"] = {
                "hrv_days": len(self.hrv_df),
                "sleep_days": len(self.sleep_df),
                "rhr_days": len(self.rhr_df),
                "stress_days": len(self.stress_df),
                "activity_days": len(self.stats_df),
                "sufficient_data": all([
                    len(self.hrv_df) >= 30,
                    len(self.sleep_df) >= 30,
                    len(self.rhr_df) >= 30
                ])
            }
            
        except Exception as e:
            correlations["error"] = str(e)
        
        return correlations
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient strength."""
        if np.isnan(corr):
            return "insufficient_data"
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _generate_correlation_insight(self, metric1: str, metric2: str, corr: float, 
                                       expected_negative: bool = False) -> str:
        """Generate human-readable insight from correlation."""
        if np.isnan(corr):
            return f"Insufficient data to determine {metric1}-{metric2} relationship."
        
        strength = self._interpret_correlation(corr)
        direction = "positive" if corr > 0 else "negative"
        
        if strength == "negligible":
            return f"No significant relationship between {metric1} and {metric2} in your data."
        
        relationship = "increases with" if corr > 0 else "decreases with"
        
        # Check if correlation matches expected direction
        if expected_negative:
            if corr < -0.2:
                health_note = " (healthy pattern)"
            elif corr > 0.2:
                health_note = " (unexpected - may indicate dysregulation)"
            else:
                health_note = ""
        else:
            health_note = ""
        
        return f"Your {metric1} {relationship} {metric2} ({strength} {direction} correlation: {corr:.2f}){health_note}"

    def generate_full_analysis(self, target_date: str) -> Dict[str, Any]:
        """
        Generate complete biomarker analysis for a target date.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        hrv = self.analyze_hrv(target_date)
        rhr = self.analyze_rhr(target_date)
        sleep = self.analyze_sleep(target_date)
        stress = self.analyze_stress(target_date)
        activity = self.analyze_activity(target_date)
        correlations = self.analyze_correlations()
        
        return {
            "hrv": hrv,
            "rhr": rhr,
            "sleep": sleep,
            "stress": stress,
            "activity": activity,
            "correlations": correlations
        }
