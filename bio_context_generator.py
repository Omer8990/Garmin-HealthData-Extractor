"""
Bio-Twin Context Generator - The Feeder
Generates the daily_bio_context.json with rich hierarchical structure.
Philosophy: "Maximum Context, Zero Noise."
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from bio_analyzer import (
    BioAnalyzer,
    HRVMetrics,
    RHRMetrics,
    SleepMetrics,
    StressMetrics,
    ActivityMetrics
)


class BioContextGenerator:
    """
    Generates the Bio-Twin Deep Context Object.
    Outputs rich, hierarchical JSON for AI agent consumption.
    """
    
    OUTPUT_DIR = Path("output")
    
    def __init__(self, analyzer: BioAnalyzer, protocol_phase: str = "Maintenance"):
        """
        Initialize the context generator.
        
        Args:
            analyzer: BioAnalyzer instance with processed data
            protocol_phase: Current protocol phase (e.g., "Maintenance", "Recovery", "Performance")
        """
        self.analyzer = analyzer
        self.protocol_phase = protocol_phase
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def _format_percentage(self, value: float) -> str:
        """Format a float as a percentage string."""
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}%"
    
    def _format_delta(self, value: float, unit: str = "bpm") -> str:
        """Format a delta value with sign and unit."""
        sign = "+" if value > 0 else ""
        return f"{sign}{int(value)} {unit}"
    
    def _get_deep_sleep_status(self, percentage: float, target: float = 15.0) -> str:
        """Generate status string for deep sleep."""
        if percentage >= target:
            return "Optimal"
        else:
            return f"Below Optimal (Huberman Target: >{target}%)"
    
    def _get_rem_sleep_status(self, percentage: float, target: float = 20.0) -> str:
        """Generate status string for REM sleep."""
        if percentage >= target:
            return "Optimal"
        else:
            return f"Below Optimal (Huberman Target: >{target}%)"
    
    def generate_meta(self, target_date: str) -> Dict[str, Any]:
        """Generate metadata section."""
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        return {
            "date": target_date,
            "day_of_week": dt.strftime("%A"),
            "protocol_phase": self.protocol_phase,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_nervous_system_profile(self, hrv: HRVMetrics, rhr: RHRMetrics, 
                                         stress: StressMetrics) -> Dict[str, Any]:
        """Generate nervous system profile section with extended historical context."""
        return {
            "hrv_status": {
                "today_ms": hrv.today_ms,
                "rolling_averages": {
                    "7d": hrv.baseline_7d_avg,
                    "14d": hrv.baseline_14d_avg,
                    "30d": hrv.baseline_30d_avg,
                    "60d": hrv.baseline_60d_avg,
                    "90d": hrv.baseline_90d_avg
                },
                "deviation_from_30d": self._format_percentage(hrv.deviation_from_baseline),
                "deviation_from_90d": self._format_percentage(hrv.deviation_from_90d),
                "deviation_raw_30d": hrv.deviation_from_baseline,
                "deviation_raw_90d": hrv.deviation_from_90d,
                "z_score_30d": hrv.z_score,
                "z_score_90d": hrv.z_score_90d,
                "trend_analysis": {
                    "short_term_7d": hrv.trend_slope_7d,
                    "medium_term_30d": hrv.trend_slope_30d,
                    "long_term_90d": hrv.trend_slope_90d
                },
                "recovery_status": hrv.recovery_status,
                "balance_classification": hrv.balance_classification,
                "insight": self._generate_hrv_insight(hrv)
            },
            "resting_heart_rate": {
                "today_bpm": int(rhr.today_bpm),
                "historical_lows": {
                    "30d": int(rhr.lowest_30d),
                    "90d": int(rhr.lowest_90d)
                },
                "rolling_averages": {
                    "7d": rhr.avg_7d,
                    "30d": rhr.avg_30d,
                    "60d": rhr.avg_60d,
                    "90d": rhr.avg_90d
                },
                "delta_7d": self._format_delta(rhr.delta_7d),
                "delta_30d": self._format_delta(rhr.delta_30d),
                "delta_7d_raw": int(rhr.delta_7d),
                "delta_30d_raw": int(rhr.delta_30d),
                "trend_analysis": {
                    "short_term_7d": rhr.trend_slope_7d,
                    "medium_term_30d": rhr.trend_slope_30d
                },
                "sympathetic_drive_elevated": rhr.sympathetic_drive,
                "insight": self._generate_rhr_insight(rhr)
            },
            "stress_load": {
                "daily_stress_avg": int(stress.daily_stress_avg),
                "high_stress_duration_min": stress.high_stress_duration_min,
                "rest_stress_duration_min": stress.rest_stress_duration_min,
                "stress_balance_ratio": stress.stress_balance_ratio,
                "stress_classification": self._classify_stress(stress),
                "insight": self._generate_stress_insight(stress)
            }
        }
    
    def generate_sleep_architecture(self, sleep: SleepMetrics) -> Dict[str, Any]:
        """Generate sleep architecture deep dive section."""
        return {
            "summary": {
                "score": sleep.score,
                "quality": sleep.quality,
                "total_minutes": sleep.total_minutes,
                "total_hours": round(sleep.total_minutes / 60, 1)
            },
            "stages": {
                "deep_sleep": {
                    "minutes": sleep.deep_sleep_minutes,
                    "percentage": f"{sleep.deep_sleep_percentage}%",
                    "percentage_raw": sleep.deep_sleep_percentage,
                    "30d_avg_minutes": int(sleep.deep_sleep_30d_avg),
                    "status": self._get_deep_sleep_status(sleep.deep_sleep_percentage),
                    "glymphatic_efficiency": sleep.glymphatic_efficiency
                },
                "rem_sleep": {
                    "minutes": sleep.rem_sleep_minutes,
                    "percentage": f"{sleep.rem_sleep_percentage}%",
                    "percentage_raw": sleep.rem_sleep_percentage,
                    "30d_avg_minutes": int(sleep.rem_sleep_30d_avg),
                    "status": self._get_rem_sleep_status(sleep.rem_sleep_percentage),
                    "cognitive_repair": sleep.cognitive_repair
                },
                "light_sleep": {
                    "minutes": sleep.light_sleep_minutes
                },
                "awake_time": {
                    "minutes": sleep.awake_minutes,
                    "interruptions_count": sleep.interruptions_count
                }
            },
            "circadian_alignment": {
                "bed_time": sleep.bed_time,
                "wake_time": sleep.wake_time,
                "wake_time_variance_7d": f"{int(sleep.wake_time_variance_7d)} mins",
                "wake_time_variance_raw_mins": int(sleep.wake_time_variance_7d),
                "circadian_disruption": sleep.circadian_disruption,
                "circadian_anchor_status": "Stable" if not sleep.circadian_disruption else "Disrupted"
            },
            "huberman_protocol_compliance": {
                "glymphatic_efficiency_met": sleep.glymphatic_efficiency,
                "cognitive_repair_met": sleep.cognitive_repair,
                "circadian_anchor_met": not sleep.circadian_disruption,
                "overall_compliance": all([
                    sleep.glymphatic_efficiency,
                    sleep.cognitive_repair,
                    not sleep.circadian_disruption
                ])
            },
            "insight": self._generate_sleep_insight(sleep)
        }
    
    def generate_metabolic_engine(self, activity: ActivityMetrics) -> Dict[str, Any]:
        """Generate metabolic engine section following Attia's protocol."""
        return {
            "active_calories": activity.active_calories,
            "total_calories": activity.total_calories,
            "intensity_minutes": {
                "moderate_zone_2": activity.zone_2_minutes,
                "vigorous_zone_5": activity.zone_5_minutes,
                "weekly_total_zone_2": activity.weekly_zone_2_total,
                "weekly_goal_minutes": 180,
                "weekly_goal_progress": f"{activity.weekly_zone_2_goal_progress}%",
                "weekly_goal_progress_raw": activity.weekly_zone_2_goal_progress,
                "zone_2_status": self._get_zone_2_status(activity.weekly_zone_2_goal_progress)
            },
            "step_cadence": {
                "total_steps": activity.total_steps,
                "avg_30d": activity.steps_30d_avg,
                "delta": self._format_percentage(activity.steps_delta_percent),
                "delta_raw": activity.steps_delta_percent,
                "sedentary_flag": activity.sedentary_flag
            },
            "movement_profile": {
                "floors_climbed": activity.floors_climbed,
                "movement_classification": self._classify_movement(activity)
            },
            "attia_protocol_compliance": {
                "zone_2_weekly_goal_met": activity.weekly_zone_2_goal_progress >= 100,
                "daily_movement_adequate": not activity.sedentary_flag,
                "insight": self._generate_activity_insight(activity)
            }
        }
    
    def generate_recovery_battery(self, stress: StressMetrics) -> Dict[str, Any]:
        """Generate recovery battery section."""
        return {
            "am_charge_level": stress.am_charge_level,
            "pm_drain_level": stress.pm_drain_level,
            "actual_recharge": stress.actual_recharge,
            "body_battery_floor": stress.body_battery_floor,
            "resource_trend": stress.resource_trend,
            "battery_health": {
                "morning_readiness": self._classify_battery_level(stress.am_charge_level),
                "evening_depletion": self._classify_depletion(stress.pm_drain_level),
                "recharge_efficiency": self._classify_recharge(stress.actual_recharge)
            },
            "insight": self._generate_battery_insight(stress)
        }
    
    def _generate_hrv_insight(self, hrv: HRVMetrics) -> str:
        """Generate contextual insight for HRV."""
        insights = []
        
        if hrv.z_score < -1.5:
            insights.append("HRV is significantly below your baseline - consider prioritizing recovery today.")
        elif hrv.z_score > 1.5:
            insights.append("HRV is elevated above baseline - good day for high-intensity training.")
        
        if hrv.recovery_status == "Strained":
            insights.append("Nervous system shows signs of strain. Avoid intense stressors.")
        elif hrv.recovery_status == "Peaking":
            insights.append("Nervous system is well-recovered. Optimal window for performance.")
        
        return " ".join(insights) if insights else "HRV within normal range."
    
    def _generate_rhr_insight(self, rhr: RHRMetrics) -> str:
        """Generate contextual insight for RHR."""
        insights = []
        
        if rhr.sympathetic_drive:
            insights.append(f"RHR elevated {int(rhr.today_bpm - rhr.lowest_30d)}bpm above 30-day low - indicates suppressed recovery.")
        
        if rhr.trend_slope_7d == "ascending":
            insights.append("RHR trending upward over 7 days - monitor for overtraining signs.")
        elif rhr.trend_slope_7d == "descending":
            insights.append("RHR trending downward - positive adaptation signal.")
        
        return " ".join(insights) if insights else "RHR stable and within normal range."
    
    def _generate_stress_insight(self, stress: StressMetrics) -> str:
        """Generate contextual insight for stress."""
        insights = []
        
        if stress.stress_balance_ratio > 0.5:
            insights.append("High stress dominated your day - prioritize parasympathetic activation.")
        elif stress.stress_balance_ratio < 0.2:
            insights.append("Good stress management - rest states dominated.")
        
        if stress.high_stress_duration_min > 120:
            insights.append(f"Extended high-stress exposure ({stress.high_stress_duration_min} mins) - consider stress interventions.")
        
        return " ".join(insights) if insights else "Stress levels balanced."
    
    def _generate_sleep_insight(self, sleep: SleepMetrics) -> str:
        """Generate contextual insight for sleep."""
        insights = []
        
        if not sleep.glymphatic_efficiency:
            insights.append(f"Deep sleep at {sleep.deep_sleep_percentage}% is below the 15% target for optimal brain detox.")
        
        if not sleep.cognitive_repair:
            insights.append(f"REM sleep at {sleep.rem_sleep_percentage}% is below the 20% target for memory consolidation.")
        
        if sleep.circadian_disruption:
            insights.append(f"Wake time variance of {int(sleep.wake_time_variance_7d)} mins exceeds 30-min threshold - circadian rhythm may be disrupted.")
        
        if sleep.interruptions_count > 3:
            insights.append(f"Sleep fragmentation detected with {sleep.interruptions_count} interruptions.")
        
        return " ".join(insights) if insights else "Sleep architecture is optimal for recovery."
    
    def _generate_activity_insight(self, activity: ActivityMetrics) -> str:
        """Generate contextual insight for activity."""
        insights = []
        
        if activity.weekly_zone_2_goal_progress < 50:
            insights.append(f"Zone 2 training at {activity.weekly_zone_2_goal_progress}% of weekly goal - increase aerobic base work.")
        elif activity.weekly_zone_2_goal_progress >= 100:
            insights.append("Weekly Zone 2 goal achieved - mitochondrial efficiency training on track.")
        
        if activity.sedentary_flag:
            insights.append(f"Only {activity.total_steps} steps today - below 5000 sedentary threshold.")
        
        return " ".join(insights) if insights else "Activity levels support metabolic health."
    
    def _generate_battery_insight(self, stress: StressMetrics) -> str:
        """Generate contextual insight for body battery."""
        insights = []
        
        if stress.actual_recharge < 30:
            insights.append(f"Only recharged {stress.actual_recharge} points overnight - sleep quality may be compromised.")
        elif stress.actual_recharge > 60:
            insights.append(f"Excellent overnight recharge of {stress.actual_recharge} points.")
        
        if stress.am_charge_level < 50:
            insights.append("Starting day with low energy reserves - pace activities accordingly.")
        
        if stress.resource_trend == "negative":
            insights.append("Body battery trending downward over past week - accumulating fatigue.")
        elif stress.resource_trend == "positive":
            insights.append("Body battery improving over past week - recovery protocols working.")
        
        return " ".join(insights) if insights else "Energy levels stable."
    
    def _classify_stress(self, stress: StressMetrics) -> str:
        """Classify overall stress level."""
        if stress.daily_stress_avg < 25:
            return "Low"
        elif stress.daily_stress_avg < 50:
            return "Moderate"
        elif stress.daily_stress_avg < 75:
            return "High"
        else:
            return "Very High"
    
    def _classify_battery_level(self, level: int) -> str:
        """Classify body battery level."""
        if level >= 80:
            return "Excellent"
        elif level >= 60:
            return "Good"
        elif level >= 40:
            return "Moderate"
        elif level >= 20:
            return "Low"
        else:
            return "Critical"
    
    def _classify_depletion(self, level: int) -> str:
        """Classify evening depletion."""
        if level >= 40:
            return "Well-managed"
        elif level >= 20:
            return "Moderate depletion"
        else:
            return "Severe depletion"
    
    def _classify_recharge(self, recharge: int) -> str:
        """Classify overnight recharge quality."""
        if recharge >= 60:
            return "Excellent"
        elif recharge >= 40:
            return "Good"
        elif recharge >= 20:
            return "Moderate"
        else:
            return "Poor"
    
    def _get_zone_2_status(self, progress: float) -> str:
        """Get Zone 2 training status."""
        if progress >= 100:
            return "Goal Achieved"
        elif progress >= 75:
            return "On Track"
        elif progress >= 50:
            return "Below Target"
        else:
            return "Needs Attention"
    
    def _classify_movement(self, activity: ActivityMetrics) -> str:
        """Classify daily movement pattern."""
        if activity.sedentary_flag:
            return "Sedentary"
        elif activity.total_steps >= 12000:
            return "Highly Active"
        elif activity.total_steps >= 8000:
            return "Active"
        else:
            return "Lightly Active"
    
    def generate_context(self, target_date: str) -> Dict[str, Any]:
        """
        Generate the complete Bio-Twin context object with extended historical analysis.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to generate context for
            
        Returns:
            Complete hierarchical context dictionary with correlations and trends
        """
        # Get all analysis results
        analysis = self.analyzer.generate_full_analysis(target_date)
        
        hrv = analysis["hrv"]
        rhr = analysis["rhr"]
        sleep = analysis["sleep"]
        stress = analysis["stress"]
        activity = analysis["activity"]
        correlations = analysis.get("correlations", {})
        
        # Build the complete context object
        context = {
            "meta": self.generate_meta(target_date),
            "nervous_system_profile": self.generate_nervous_system_profile(hrv, rhr, stress),
            "sleep_architecture_deep_dive": self.generate_sleep_architecture(sleep),
            "metabolic_engine": self.generate_metabolic_engine(activity),
            "recovery_battery": self.generate_recovery_battery(stress),
            "cross_metric_correlations": self._format_correlations(correlations),
            "longitudinal_insights": self._generate_longitudinal_insights(hrv, rhr, sleep),
            "protocol_recommendations": self._generate_recommendations(hrv, rhr, sleep, stress, activity),
            "data_quality": self._assess_data_quality(hrv, rhr, sleep, stress, activity)
        }
        
        return context
    
    def _format_correlations(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Format correlation data for JSON output."""
        formatted = {
            "description": "Cross-metric correlations reveal hidden patterns in your health data that simple averages miss.",
            "interpretation_guide": {
                "strong": "coefficient >= 0.7: reliable relationship",
                "moderate": "coefficient 0.4-0.7: notable pattern",
                "weak": "coefficient 0.2-0.4: slight tendency",
                "negligible": "coefficient < 0.2: no meaningful relationship"
            }
        }
        
        # Add each correlation with context
        if "hrv_sleep_quality" in correlations:
            formatted["hrv_sleep_quality"] = correlations["hrv_sleep_quality"]
        if "hrv_deep_sleep" in correlations:
            formatted["hrv_deep_sleep"] = correlations["hrv_deep_sleep"]
        if "hrv_rhr" in correlations:
            formatted["hrv_rhr"] = correlations["hrv_rhr"]
        if "sleep_body_battery" in correlations:
            formatted["sleep_body_battery"] = correlations["sleep_body_battery"]
        if "sleep_duration_stress" in correlations:
            formatted["sleep_duration_stress"] = correlations["sleep_duration_stress"]
        if "zone2_next_day_hrv" in correlations:
            formatted["zone2_next_day_hrv"] = correlations["zone2_next_day_hrv"]
        if "steps_next_day_hrv" in correlations:
            formatted["steps_next_day_hrv"] = correlations["steps_next_day_hrv"]
        
        if "data_quality" in correlations:
            formatted["correlation_data_quality"] = correlations["data_quality"]
        
        return formatted
    
    def _generate_longitudinal_insights(self, hrv: HRVMetrics, rhr: RHRMetrics, 
                                         sleep: SleepMetrics) -> Dict[str, Any]:
        """Generate insights from longitudinal trend analysis."""
        insights = {
            "description": "Multi-timeframe analysis reveals whether short-term changes are noise or meaningful shifts.",
            "hrv_trajectory": {
                "short_term": hrv.trend_slope_7d,
                "medium_term": hrv.trend_slope_30d,
                "long_term": hrv.trend_slope_90d,
                "interpretation": self._interpret_trajectory(
                    hrv.trend_slope_7d, hrv.trend_slope_30d, hrv.trend_slope_90d, "HRV"
                )
            },
            "rhr_trajectory": {
                "short_term": rhr.trend_slope_7d,
                "medium_term": rhr.trend_slope_30d,
                "interpretation": self._interpret_trajectory(
                    rhr.trend_slope_7d, rhr.trend_slope_30d, "stable", "RHR"
                )
            },
            "comparison_to_baseline": {
                "hrv_vs_90d_baseline": {
                    "deviation": f"{hrv.deviation_from_90d:+.1f}%",
                    "z_score": hrv.z_score_90d,
                    "status": "within_normal" if abs(hrv.z_score_90d) < 1.5 else "notable_deviation"
                },
                "rhr_vs_90d_low": {
                    "current": int(rhr.today_bpm),
                    "90d_low": int(rhr.lowest_90d),
                    "elevation": f"+{int(rhr.today_bpm - rhr.lowest_90d)} bpm"
                }
            }
        }
        return insights
    
    def _interpret_trajectory(self, short: str, medium: str, long: str, metric: str) -> str:
        """Interpret multi-timeframe trajectory."""
        trends = [short, medium, long]
        ascending_count = trends.count("ascending")
        descending_count = trends.count("descending")
        
        if metric == "HRV":
            # For HRV, ascending is generally positive
            if ascending_count >= 2:
                return "Positive trend - nervous system adapting well"
            elif descending_count >= 2:
                return "Declining trend - consider recovery focus"
            elif short == "ascending" and medium == "descending":
                return "Recent improvement after medium-term decline - recovery in progress"
            elif short == "descending" and medium == "ascending":
                return "Recent dip in longer-term uptrend - likely temporary"
            else:
                return "Stable with normal fluctuations"
        else:  # RHR
            # For RHR, descending is generally positive
            if descending_count >= 2:
                return "Positive trend - improving cardiovascular efficiency"
            elif ascending_count >= 2:
                return "Rising trend - may indicate accumulated stress or detraining"
            elif short == "descending" and medium == "ascending":
                return "Recent improvement after medium-term rise - recovery responding"
            else:
                return "Stable with normal fluctuations"
    
    def _generate_recommendations(self, hrv: HRVMetrics, rhr: RHRMetrics, 
                                   sleep: SleepMetrics, stress: StressMetrics,
                                   activity: ActivityMetrics) -> Dict[str, Any]:
        """Generate protocol-specific recommendations."""
        recommendations = {
            "training": [],
            "recovery": [],
            "sleep": [],
            "lifestyle": []
        }
        
        # Training recommendations
        if hrv.recovery_status == "Strained" or rhr.sympathetic_drive:
            recommendations["training"].append("Reduce training intensity - system under stress")
        elif hrv.recovery_status == "Peaking":
            recommendations["training"].append("Optimal day for high-intensity or strength training")
        
        if activity.weekly_zone_2_goal_progress < 75:
            recommendations["training"].append(f"Add {180 - activity.weekly_zone_2_total} more Zone 2 minutes this week")
        
        # Recovery recommendations
        if stress.stress_balance_ratio > 0.4:
            recommendations["recovery"].append("Schedule parasympathetic activation (breathing, meditation)")
        
        if stress.am_charge_level < 50:
            recommendations["recovery"].append("Consider a recovery day - starting with depleted reserves")
        
        # Sleep recommendations
        if not sleep.glymphatic_efficiency:
            recommendations["sleep"].append("Optimize for deep sleep: cool room, no alcohol, early dinner")
        
        if not sleep.cognitive_repair:
            recommendations["sleep"].append("Protect REM: avoid late caffeine, maintain consistent wake time")
        
        if sleep.circadian_disruption:
            recommendations["sleep"].append("Anchor wake time within 30-minute window for circadian stability")
        
        # Lifestyle recommendations
        if activity.sedentary_flag:
            recommendations["lifestyle"].append("Increase NEAT: walking meetings, hourly movement breaks")
        
        if stress.high_stress_duration_min > 90:
            recommendations["lifestyle"].append("Implement stress interrupts throughout the day")
        
        return recommendations
    
    def _assess_data_quality(self, hrv: HRVMetrics, rhr: RHRMetrics,
                             sleep: SleepMetrics, stress: StressMetrics,
                             activity: ActivityMetrics) -> Dict[str, Any]:
        """Assess the quality and completeness of the data."""
        quality_flags = {
            "hrv_available": hrv.today_ms > 0,
            "rhr_available": rhr.today_bpm > 0,
            "sleep_available": sleep.total_minutes > 0,
            "stress_available": stress.daily_stress_avg > 0 or stress.am_charge_level > 0,
            "activity_available": activity.total_steps > 0
        }
        
        available_count = sum(quality_flags.values())
        total_count = len(quality_flags)
        
        return {
            "completeness_score": f"{available_count}/{total_count}",
            "completeness_percentage": round(available_count / total_count * 100, 0),
            "data_flags": quality_flags,
            "confidence": "High" if available_count >= 4 else "Medium" if available_count >= 2 else "Low"
        }
    
    def _convert_bools(self, obj: Any) -> Any:
        """Recursively convert numpy/pandas bools to Python bools for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_bools(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif str(type(obj).__name__) in ('bool_', 'np.bool_'):
            return bool(obj)
        return obj
    
    def save_context(self, context: Dict[str, Any], 
                     filename: str = "daily_bio_context.json") -> Path:
        """
        Save the context to a JSON file.
        
        Args:
            context: Context dictionary to save
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        filepath = self.OUTPUT_DIR / filename
        # Convert any numpy bools to Python bools
        clean_context = self._convert_bools(context)
        with open(filepath, "w") as f:
            json.dump(clean_context, f, indent=2, default=str)
        print(f"Bio context saved to {filepath}")
        return filepath
    
    def generate_and_save(self, target_date: str, 
                          filename: str = "daily_bio_context.json") -> Path:
        """
        Generate context and save to file in one step.
        
        Args:
            target_date: Date string (YYYY-MM-DD) to generate context for
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        context = self.generate_context(target_date)
        return self.save_context(context, filename)
