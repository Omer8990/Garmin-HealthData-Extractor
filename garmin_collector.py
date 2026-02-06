"""
Garmin Data Collector - The Ingestion Layer
Handles authentication and data fetching from Garmin Connect API.
Implements incremental load: 30-day backfill on first run, then daily catch-up.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from garminconnect import Garmin


class GarminCollector:
    """
    Garmin Connect data collector with incremental loading support.
    Supports extended historical data collection (default 90 days) for
    comprehensive trend analysis, correlations, and longitudinal insights.
    """
    
    CACHE_DIR = Path("data_cache")
    LAST_SYNC_FILE = CACHE_DIR / "last_sync.json"
    DEFAULT_BACKFILL_DAYS = 90  # Extended from 30 to 90 days for richer insights
    
    # Configurable backfill periods
    BACKFILL_PRESETS = {
        "minimal": 30,    # Quick analysis
        "standard": 90,   # Default - 3 months
        "extended": 180,  # 6 months - seasonal patterns
        "comprehensive": 365  # 1 year - full annual cycles
    }
    
    def __init__(self, email: Optional[str] = None, password: Optional[str] = None, 
                 backfill_days: Optional[int] = None, history_preset: Optional[str] = None):
        """
        Initialize the Garmin collector.
        
        Args:
            email: Garmin Connect email (or set GARMIN_EMAIL env var)
            password: Garmin Connect password (or set GARMIN_PASSWORD env var)
            backfill_days: Number of days to backfill on first run (default: 90)
            history_preset: Use a preset period ('minimal', 'standard', 'extended', 'comprehensive')
        """
        self.email = email or os.environ.get("GARMIN_EMAIL") or os.environ.get("GARMIN_CONNECT_EMAIL")
        self.password = password or os.environ.get("GARMIN_PASSWORD")
        self.client: Optional[Garmin] = None
        
        # Configure backfill period
        if history_preset and history_preset in self.BACKFILL_PRESETS:
            self.backfill_days = self.BACKFILL_PRESETS[history_preset]
        elif backfill_days:
            self.backfill_days = backfill_days
        else:
            self.backfill_days = self.DEFAULT_BACKFILL_DAYS
        
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def authenticate(self) -> bool:
        """
        Authenticate with Garmin Connect.
        
        Returns:
            True if authentication successful, False otherwise.
        """
        if not self.email or not self.password:
            raise ValueError(
                "Garmin credentials required. Set GARMIN_EMAIL and GARMIN_PASSWORD "
                "environment variables or pass them to the constructor."
            )
        
        try:
            self.client = Garmin(self.email, self.password)
            self.client.login()
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def _get_last_sync_date(self) -> Optional[datetime]:
        """Get the last successful sync date from cache."""
        if self.LAST_SYNC_FILE.exists():
            with open(self.LAST_SYNC_FILE, "r") as f:
                data = json.load(f)
                return datetime.fromisoformat(data.get("last_sync"))
        return None
    
    def _save_last_sync_date(self, date: datetime) -> None:
        """Save the last sync date to cache."""
        with open(self.LAST_SYNC_FILE, "w") as f:
            json.dump({"last_sync": date.isoformat()}, f)
    
    def _get_date_range(self) -> tuple[datetime, datetime]:
        """
        Determine the date range for data fetching.
        Uses configurable backfill period for extended historical analysis.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = datetime.now()
        last_sync = self._get_last_sync_date()
        
        if last_sync is None:
            # First run: backfill configured number of days
            start_date = end_date - timedelta(days=self.backfill_days)
            print(f"First run detected. Backfilling {self.backfill_days} days of data for comprehensive analysis.")
        else:
            # Incremental: from last sync to today
            start_date = last_sync
            print(f"Incremental load from {start_date.date()} to {end_date.date()}")
        
        return start_date, end_date
    
    def _format_date(self, dt: datetime) -> str:
        """Format datetime for Garmin API calls."""
        return dt.strftime("%Y-%m-%d")
    
    def get_sleep_data(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch sleep architecture data for a specific date.
        
        Args:
            date: Date to fetch sleep data for
            
        Returns:
            Sleep data dictionary or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            date_str = self._format_date(date)
            return self.client.get_sleep_data(date_str)
        except Exception as e:
            print(f"Error fetching sleep data for {date}: {e}")
            return None
    
    def get_heart_rate_variability_data(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch HRV status data for a specific date.
        
        Args:
            date: Date to fetch HRV data for
            
        Returns:
            HRV data dictionary or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            date_str = self._format_date(date)
            return self.client.get_hrv_data(date_str)
        except Exception as e:
            print(f"Error fetching HRV data for {date}: {e}")
            return None
    
    def get_rhr_day(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch Resting Heart Rate data for a specific date.
        
        Args:
            date: Date to fetch RHR data for
            
        Returns:
            RHR data dictionary or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            date_str = self._format_date(date)
            return self.client.get_rhr_day(date_str)
        except Exception as e:
            print(f"Error fetching RHR data for {date}: {e}")
            return None
    
    def get_body_battery_events(self, date: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch Body Battery and stress flux data for a specific date.
        
        Args:
            date: Date to fetch body battery data for
            
        Returns:
            Body battery events list or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            date_str = self._format_date(date)
            return self.client.get_body_battery(date_str)
        except Exception as e:
            print(f"Error fetching body battery data for {date}: {e}")
            return None
    
    def get_activities_by_date(self, start_date: datetime, end_date: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch exercise activities within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of activities or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            start_str = self._format_date(start_date)
            end_str = self._format_date(end_date)
            return self.client.get_activities_by_date(start_str, end_str)
        except Exception as e:
            print(f"Error fetching activities from {start_date} to {end_date}: {e}")
            return None
    
    def get_stats(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch daily stats (steps, calories, floors, etc.) for a specific date.
        
        Args:
            date: Date to fetch stats for
            
        Returns:
            Stats dictionary or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            date_str = self._format_date(date)
            return self.client.get_stats(date_str)
        except Exception as e:
            print(f"Error fetching stats for {date}: {e}")
            return None
    
    def get_stress_data(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch stress level data for a specific date.
        
        Args:
            date: Date to fetch stress data for
            
        Returns:
            Stress data dictionary or None if not available
        """
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            date_str = self._format_date(date)
            return self.client.get_stress_data(date_str)
        except Exception as e:
            print(f"Error fetching stress data for {date}: {e}")
            return None
    
    def collect_all_data(self, target_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Collect all required data points for a specific date or date range.
        Merges new data with existing cached data to preserve historical records.
        
        Args:
            target_date: Specific date to collect (defaults to today)
            
        Returns:
            Dictionary containing all collected data
        """
        if target_date is None:
            target_date = datetime.now()
        
        start_date, end_date = self._get_date_range()
        
        # Load existing cached data to merge with new data
        existing_data = self.load_raw_data()
        
        # Collect data for each day in range, merging with existing
        all_data = {
            "collection_date": self._format_date(datetime.now()),
            "date_range": {
                "start": self._format_date(start_date),
                "end": self._format_date(end_date)
            },
            "daily_data": existing_data.get("daily_data", {}) if existing_data else {}
        }
        
        current_date = start_date
        while current_date <= end_date:
            date_str = self._format_date(current_date)
            print(f"Collecting data for {date_str}...")
            
            all_data["daily_data"][date_str] = {
                "sleep": self.get_sleep_data(current_date),
                "hrv": self.get_heart_rate_variability_data(current_date),
                "rhr": self.get_rhr_day(current_date),
                "body_battery": self.get_body_battery_events(current_date),
                "stats": self.get_stats(current_date),
                "stress": self.get_stress_data(current_date)
            }
            
            current_date += timedelta(days=1)
        
        # Fetch activities for entire range
        all_data["activities"] = self.get_activities_by_date(start_date, end_date)
        
        # Update last sync date
        self._save_last_sync_date(end_date)
        
        return all_data
    
    def collect_single_day(self, target_date: datetime) -> Dict[str, Any]:
        """
        Collect all data points for a single specific date.
        
        Args:
            target_date: Date to collect data for
            
        Returns:
            Dictionary containing all collected data for that day
        """
        date_str = self._format_date(target_date)
        print(f"Collecting data for {date_str}...")
        
        return {
            "date": date_str,
            "sleep": self.get_sleep_data(target_date),
            "hrv": self.get_heart_rate_variability_data(target_date),
            "rhr": self.get_rhr_day(target_date),
            "body_battery": self.get_body_battery_events(target_date),
            "stats": self.get_stats(target_date),
            "stress": self.get_stress_data(target_date),
            "activities": self.get_activities_by_date(target_date, target_date)
        }
    
    def save_raw_data(self, data: Dict[str, Any], filename: str = "raw_garmin_data.json") -> Path:
        """
        Save collected raw data to a JSON file.
        
        Args:
            data: Data dictionary to save
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        filepath = self.CACHE_DIR / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Raw data saved to {filepath}")
        return filepath
    
    def load_raw_data(self, filename: str = "raw_garmin_data.json") -> Optional[Dict[str, Any]]:
        """
        Load previously cached raw data.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Data dictionary or None if file doesn't exist
        """
        filepath = self.CACHE_DIR / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return None
