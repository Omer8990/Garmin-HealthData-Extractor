# 🧬 Biological ETL Pipeline

A high-signal health data extraction pipeline that acts as a "Bio-Twin" for AI agents. Inspired by leading health optimization protocols from:

- **Bryan Johnson** (Blueprint/HRV nervous system health)
- **Andrew Huberman** (Sleep architecture/Circadian rhythm)
- **Peter Attia** (Zone 2/Metabolic health)

## Overview

This pipeline ingests raw Garmin Connect data, applies expert-level biomarker filtering, and outputs a rich, hierarchical JSON context file (`daily_bio_context.json`) designed to provide maximum context with zero noise for AI agent consumption.

## Features

### Phase 1: The Collector (Data Ingestion)
- Incremental loading: 30-day backfill on first run, daily catch-up thereafter
- Fetches all critical health endpoints:
  - Sleep architecture data
  - HRV (Heart Rate Variability) status
  - Resting Heart Rate (RHR)
  - Body Battery events
  - Activity/Exercise data
  - Stress levels

### Phase 2: The Expert Filters (Logic Core)

#### A. Blueprint Metric (Nervous System Health)
- HRV baseline deviation with Z-score calculation
- Recovery status classification (Optimal/Strained/Peaking)
- Sympathetic drive detection (RHR elevation >3bpm from 30-day low)

#### B. Huberman Sleep Protocol (Circadian & Glymphatic)
- Glymphatic Efficiency: Deep Sleep > 15% flag
- Cognitive Repair: REM Sleep > 20% flag
- Circadian Anchor: Wake time variance detection (>30 min = disruption)

#### C. Attia Training Engine (Metabolic Health)
- Zone 2 Volume tracking (moderate intensity minutes)
- VO2 Max Work (vigorous intensity minutes)
- Weekly Zone 2 goal progress (target: 180 min/week)
- Sedentary behavior detection

#### D. Bio-Budget (Energy Management)
- Overnight recharge calculation
- Physiological stress load analysis
- Body battery trend tracking

### Phase 3: The Feeder (Bio-Twin Context Object)
Generates `daily_bio_context.json` with three layers per metric:
- **The Now**: High-resolution current day data
- **The Context**: Rolling windows (7d, 14d, 30d) with Z-scores
- **The Vector**: Trend slopes (ascending/descending/stable)

## Installation

```bash
# Clone the repository
cd GarminHealthDataExtraction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your Garmin Connect credentials as environment variables:

```bash
export GARMIN_EMAIL='your-email@example.com'
export GARMIN_PASSWORD='your-password'
```

## Usage

### Demo Mode (No Credentials Required)
Test the pipeline with realistic sample data:

```bash
python main.py --demo
```

### Full Pipeline
Run with real Garmin data:

```bash
# Run for today
python main.py --run

# Run for a specific date
python main.py --run --date 2026-02-05

# Force 30-day backfill
python main.py --backfill

# Specify protocol phase
python main.py --run --protocol-phase Recovery

# Verbose output with full JSON
python main.py --run -v
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--run` | Run the full ETL pipeline |
| `--demo` | Run in demo mode with sample data |
| `--date YYYY-MM-DD` | Target date for analysis (default: today) |
| `--backfill` | Force 30-day data backfill |
| `--protocol-phase` | Current phase: Maintenance, Recovery, Performance, Deload |
| `--output FILENAME` | Output filename (default: daily_bio_context.json) |
| `-v, --verbose` | Enable verbose output with full JSON |

## Output Structure

The generated `daily_bio_context.json` contains:

```json
{
  "meta": {
    "date": "2026-02-06",
    "day_of_week": "Friday",
    "protocol_phase": "Maintenance"
  },
  "nervous_system_profile": {
    "hrv_status": {
      "today_ms": 42,
      "baseline_30d_avg": 55,
      "deviation_from_baseline": "-23.6%",
      "z_score": -1.5,
      "balance_classification": "Unbalanced (Sympathetic Dominance)"
    },
    "resting_heart_rate": {
      "today_bpm": 48,
      "trend_slope_7d": "ascending",
      "sympathetic_drive_elevated": true
    },
    "stress_load": {
      "daily_stress_avg": 35,
      "high_stress_duration_min": 45,
      "stress_balance_ratio": 0.37
    }
  },
  "sleep_architecture_deep_dive": {
    "summary": { "score": 85, "quality": "Good" },
    "stages": {
      "deep_sleep": { "percentage": "10.4%", "status": "Below Optimal" },
      "rem_sleep": { "percentage": "20.8%", "status": "Optimal" }
    },
    "huberman_protocol_compliance": {
      "glymphatic_efficiency_met": false,
      "cognitive_repair_met": true,
      "overall_compliance": false
    }
  },
  "metabolic_engine": {
    "intensity_minutes": {
      "moderate_zone_2": 45,
      "weekly_goal_progress": "85%"
    },
    "step_cadence": { "total_steps": 12500, "delta": "+22%" }
  },
  "recovery_battery": {
    "am_charge_level": 95,
    "actual_recharge": 65,
    "resource_trend": "positive"
  },
  "protocol_recommendations": {
    "training": ["Optimal day for high-intensity training"],
    "recovery": [],
    "sleep": ["Optimize for deep sleep: cool room, no alcohol"],
    "lifestyle": []
  },
  "data_quality": {
    "completeness_score": "5/5",
    "confidence": "High"
  }
}
```

## Project Structure

```
GarminHealthDataExtraction/
├── main.py                    # Main orchestrator with CLI
├── garmin_collector.py        # Data ingestion layer
├── bio_analyzer.py            # Expert filters and biomarker analysis
├── bio_context_generator.py   # JSON context object generator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data_cache/                # Cached raw Garmin data
│   ├── raw_garmin_data.json
│   └── last_sync.json
└── output/                    # Generated context files
    └── daily_bio_context.json
```

## Key Metrics Explained

### HRV Z-Score
Indicates how unusual today's HRV is compared to your baseline:
- **< -1.5**: Significantly below normal (recovery concern)
- **-1.5 to 1.5**: Within normal range
- **> 1.5**: Significantly elevated (peak performance window)

### Glymphatic Efficiency
Deep sleep > 15% of total sleep time enables optimal brain waste clearance (glymphatic system function).

### Zone 2 Training
Moderate intensity exercise (talking pace) builds mitochondrial efficiency. Target: 180 minutes/week per Peter Attia's protocol.

### Body Battery Trend
Tracks whether your energy reserves are accumulating (positive) or depleting (negative) over time.

## Philosophy

> "Maximum Context, Zero Noise"

This pipeline is designed as a **feature store for AI**. Every metric includes:
- Raw values for precision
- Formatted strings for readability
- Z-scores to indicate rarity
- Trend slopes to show direction
- Pre-calculated status flags to reduce AI hallucination

## Acknowledgments

Protocol inspiration from:
- Bryan Johnson's Blueprint Protocol
- Andrew Huberman's Huberman Lab Podcast
- Peter Attia's The Drive Podcast
