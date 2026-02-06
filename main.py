#!/usr/bin/env python3
"""
Biological ETL Pipeline - Main Orchestrator
A high-signal health data extraction pipeline inspired by:
- Bryan Johnson (Blueprint/HRV)
- Andrew Huberman (Sleep/Circadian)
- Peter Attia (Zone 2/Metabolic Health)

This script ingests Garmin data, applies expert-level filtering,
and outputs a rich JSON context file for AI agent consumption.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from garmin_collector import GarminCollector
from bio_analyzer import BioAnalyzer
from bio_context_generator import BioContextGenerator


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Biological ETL Pipeline - Extract and analyze Garmin health data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for today
  python main.py --run
  
  # Run for a specific date
  python main.py --run --date 2026-02-05
  
  # Use demo mode with sample data
  python main.py --demo
  
  # Backfill 30 days of data
  python main.py --backfill
  
Environment Variables:
  GARMIN_EMAIL    - Your Garmin Connect email
  GARMIN_PASSWORD - Your Garmin Connect password
        """
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full ETL pipeline"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date for analysis (YYYY-MM-DD format, defaults to today)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with sample data (no Garmin credentials required)"
    )
    
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Force a full backfill regardless of last sync"
    )
    
    parser.add_argument(
        "--history-days",
        type=int,
        default=None,
        help="Number of days of historical data to fetch (default: 90)"
    )
    
    parser.add_argument(
        "--history-preset",
        type=str,
        default=None,
        choices=["minimal", "standard", "extended", "comprehensive"],
        help="History period preset: minimal (30d), standard (90d), extended (180d), comprehensive (365d)"
    )
    
    parser.add_argument(
        "--protocol-phase",
        type=str,
        default="Maintenance",
        choices=["Maintenance", "Recovery", "Performance", "Deload"],
        help="Current protocol phase for context generation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="daily_bio_context.json",
        help="Output filename for the context JSON"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def generate_sample_data(days: int = 90) -> dict:
    """Generate sample data for demo mode with extended history."""
    from datetime import datetime, timedelta
    import random
    
    base_date = datetime.now()
    daily_data = {}
    
    # Generate extended days of sample data for comprehensive analysis
    for i in range(days):
        date = base_date - timedelta(days=days-1-i)
        date_str = date.strftime("%Y-%m-%d")
        
        # Simulate realistic health data with some variation
        base_hrv = 45 + random.randint(-10, 15)
        base_rhr = 48 + random.randint(-3, 5)
        base_sleep_score = 75 + random.randint(-15, 20)
        base_steps = 8000 + random.randint(-3000, 6000)
        
        daily_data[date_str] = {
            "hrv": {
                "hrvSummary": {
                    "lastNightAvg": base_hrv + random.uniform(-5, 5)
                }
            },
            "rhr": {
                "restingHeartRate": base_rhr + random.randint(-2, 3)
            },
            "sleep": {
                "dailySleepDTO": {
                    "sleepTimeSeconds": (7 * 3600) + random.randint(-3600, 3600),
                    "deepSleepSeconds": int(1.2 * 3600) + random.randint(-1800, 1800),
                    "remSleepSeconds": int(1.5 * 3600) + random.randint(-1800, 1800),
                    "lightSleepSeconds": int(4 * 3600) + random.randint(-1800, 1800),
                    "awakeSleepSeconds": random.randint(600, 2400),
                    "sleepScore": base_sleep_score,
                    "sleepStartTimestampLocal": "22:30",
                    "sleepEndTimestampLocal": "06:15"
                }
            },
            "stats": {
                "totalSteps": base_steps,
                "activeKilocalories": 400 + random.randint(-100, 300),
                "totalKilocalories": 2200 + random.randint(-200, 400),
                "floorsAscended": random.randint(5, 25),
                "moderateIntensityMinutes": random.randint(0, 60),
                "vigorousIntensityMinutes": random.randint(0, 30)
            },
            "stress": {
                "avgStressLevel": 30 + random.randint(-10, 20),
                "highStressDuration": random.randint(30, 120),
                "restStressDuration": random.randint(180, 360)
            },
            "body_battery": {
                "charged": 70 + random.randint(-20, 25),
                "drained": 20 + random.randint(-10, 30)
            }
        }
    
    return {
        "collection_date": base_date.strftime("%Y-%m-%d"),
        "date_range": {
            "start": (base_date - timedelta(days=days-1)).strftime("%Y-%m-%d"),
            "end": base_date.strftime("%Y-%m-%d")
        },
        "daily_data": daily_data,
        "activities": []
    }


def run_demo_mode(target_date: str, protocol_phase: str, output_file: str, verbose: bool) -> None:
    """Run the pipeline in demo mode with sample data."""
    print("\n" + "="*60)
    print("🧬 BIOLOGICAL ETL PIPELINE - DEMO MODE")
    print("="*60)
    print("\nGenerating sample health data...")
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    if verbose:
        print(f"\nSample data generated for {len(sample_data['daily_data'])} days")
    
    # Create analyzer with sample data
    print("\n📊 Running Bio Analysis...")
    analyzer = BioAnalyzer(sample_data)
    
    # Generate context
    print("🔬 Generating Bio-Twin Context...")
    generator = BioContextGenerator(analyzer, protocol_phase=protocol_phase)
    context = generator.generate_context(target_date)
    
    # Save output
    output_path = generator.save_context(context, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\n📅 Analysis Date: {target_date}")
    print(f"📁 Output File: {output_path}")
    
    # Print key insights
    print("\n🔑 KEY INSIGHTS:")
    print("-" * 40)
    
    ns = context["nervous_system_profile"]
    print(f"\n💓 HRV Status: {ns['hrv_status']['today_ms']}ms "
          f"({ns['hrv_status']['deviation_from_30d']} from 30d baseline)")
    print(f"   Recovery: {ns['hrv_status']['recovery_status']}")
    print(f"   90d Trend: {ns['hrv_status']['trend_analysis']['long_term_90d']}")
    
    sleep = context["sleep_architecture_deep_dive"]
    print(f"\n😴 Sleep Score: {sleep['summary']['score']} ({sleep['summary']['quality']})")
    print(f"   Deep: {sleep['stages']['deep_sleep']['percentage']} | "
          f"REM: {sleep['stages']['rem_sleep']['percentage']}")
    
    metabolic = context["metabolic_engine"]
    print(f"\n🏃 Steps: {metabolic['step_cadence']['total_steps']} "
          f"({metabolic['step_cadence']['delta']} vs 30d avg)")
    print(f"   Zone 2 Progress: {metabolic['intensity_minutes']['weekly_goal_progress']}")
    
    battery = context["recovery_battery"]
    print(f"\n🔋 Body Battery: {battery['am_charge_level']} AM → {battery['pm_drain_level']} PM")
    print(f"   Trend: {battery['resource_trend']}")
    
    # Print recommendations
    recs = context["protocol_recommendations"]
    if any(recs.values()):
        print("\n📋 RECOMMENDATIONS:")
        print("-" * 40)
        for category, items in recs.items():
            if items:
                print(f"\n{category.upper()}:")
                for item in items:
                    print(f"  • {item}")
    
    print("\n" + "="*60)
    if verbose:
        print("\n📄 Full context JSON:")
        print(json.dumps(context, indent=2, cls=NumpyJSONEncoder))


def run_full_pipeline(target_date: str, protocol_phase: str, output_file: str,
                      backfill: bool, verbose: bool, 
                      history_days: Optional[int] = None,
                      history_preset: Optional[str] = None) -> None:
    """Run the full ETL pipeline with real Garmin data."""
    print("\n" + "="*60)
    print("🧬 BIOLOGICAL ETL PIPELINE")
    print("="*60)
    
    # Check for credentials (support both GARMIN_EMAIL and GARMIN_CONNECT_EMAIL)
    email = os.environ.get("GARMIN_EMAIL") or os.environ.get("GARMIN_CONNECT_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")
    
    if not email or not password:
        print("\n❌ ERROR: Garmin credentials not found!")
        print("\nPlease set the following environment variables:")
        print("  export GARMIN_EMAIL='your-email@example.com'")
        print("  export GARMIN_PASSWORD='your-password'")
        print("\nOr run in demo mode: python main.py --demo")
        sys.exit(1)
    
    # Initialize collector with history configuration
    print("\n📡 Initializing Garmin Collector...")
    collector = GarminCollector(
        email, password, 
        backfill_days=history_days,
        history_preset=history_preset
    )
    
    # Display history configuration
    print(f"📅 History period: {collector.backfill_days} days")
    
    # Authenticate
    print("🔐 Authenticating with Garmin Connect...")
    if not collector.authenticate():
        print("❌ Authentication failed!")
        sys.exit(1)
    print("✅ Authentication successful!")
    
    # Handle backfill flag
    if backfill:
        # Remove last sync file to force backfill
        if collector.LAST_SYNC_FILE.exists():
            collector.LAST_SYNC_FILE.unlink()
            print(f"🔄 Forcing {collector.backfill_days}-day backfill...")
    
    # Collect data
    print("\n📥 Collecting Garmin data...")
    raw_data = collector.collect_all_data()
    
    # Save raw data
    collector.save_raw_data(raw_data)
    
    if verbose:
        print(f"\nCollected data for {len(raw_data.get('daily_data', {}))} days")
    
    # Run analysis
    print("\n📊 Running Bio Analysis...")
    analyzer = BioAnalyzer(raw_data)
    
    # Generate context
    print("🔬 Generating Bio-Twin Context...")
    generator = BioContextGenerator(analyzer, protocol_phase=protocol_phase)
    context = generator.generate_context(target_date)
    
    # Save output
    output_path = generator.save_context(context, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\n📅 Analysis Date: {target_date}")
    print(f"📁 Output File: {output_path}")
    print(f"📊 Data Quality: {context['data_quality']['completeness_score']} metrics available")
    print(f"🎯 Confidence: {context['data_quality']['confidence']}")
    
    if verbose:
        print("\n📄 Full context JSON:")
        print(json.dumps(context, indent=2, cls=NumpyJSONEncoder))


def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("   Please use YYYY-MM-DD format (e.g., 2026-02-06)")
            sys.exit(1)
    else:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    # Run appropriate mode
    if args.demo:
        run_demo_mode(
            target_date=target_date,
            protocol_phase=args.protocol_phase,
            output_file=args.output,
            verbose=args.verbose
        )
    elif args.run or args.backfill:
        run_full_pipeline(
            target_date=target_date,
            protocol_phase=args.protocol_phase,
            output_file=args.output,
            backfill=args.backfill,
            verbose=args.verbose,
            history_days=args.history_days,
            history_preset=args.history_preset
        )
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   Demo mode:  python main.py --demo")
        print("   Full run:   python main.py --run")


if __name__ == "__main__":
    main()
