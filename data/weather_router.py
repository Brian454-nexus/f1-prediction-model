"""
F1 APEX — Weather Router
========================
Queries OpenMeteo for a race location and determines which prediction
branch to use for that weekend.

Decision rule (from PRD §11):
    If P(rain) > RAIN_PROBABILITY_THRESHOLD at T-3 h pre-race
    → route to WetWeatherModel
    Otherwise
    → route to MetaEnsemble (standard dry model)

OpenMeteo is chosen because it is:
  - Free with no API key for the resolution we need (hourly, 3-day)
  - Accurate for circuit-level lat/lon queries
  - Stable — used by weather.com as a data source

The router is intentionally stateless: it is called once per race weekend
by the Prefect orchestration DAG and its result is passed as a flag to
subsequent pipeline tasks.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

from utils.logger import get_logger

logger = get_logger("APEX.WeatherRouter")

# ── Constants ─────────────────────────────────────────────────────────────────

OPENMETEO_BASE = os.getenv(
    "WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1/forecast"
)
RAIN_THRESHOLD: float = float(os.getenv("RAIN_PROBABILITY_THRESHOLD", "0.30"))

# Amount of precipitation (mm/hr) that also flags wet conditions regardless of
# rain probability — catches light drizzle underrepresented in probability metrics
DRIZZLE_MM_THRESHOLD: float = 0.5


class WeatherDecision:
    """Value object returned by the router."""

    def __init__(
        self,
        rain_probability: float,
        precipitation_mm: float,
        temperature_c: float,
        wind_kmh: float,
        route_to_wet_model: bool,
        queried_at: str,
    ) -> None:
        self.rain_probability = rain_probability
        self.precipitation_mm = precipitation_mm
        self.temperature_c = temperature_c
        self.wind_kmh = wind_kmh
        self.route_to_wet_model = route_to_wet_model
        self.queried_at = queried_at
        self.branch = "WET_WEATHER" if route_to_wet_model else "DRY_STANDARD"

    def __repr__(self) -> str:
        return (
            f"WeatherDecision(branch={self.branch!r}, "
            f"rain_prob={self.rain_probability:.1%}, "
            f"precip={self.precipitation_mm:.1f}mm/hr)"
        )


class WeatherRouter:
    """
    Queries OpenMeteo and determines whether to trigger the wet-weather
    prediction branch.

    Args:
        rain_threshold: Probability (0–1) above which wet branch is activated.
    """

    def __init__(self, rain_threshold: float = RAIN_THRESHOLD) -> None:
        self.rain_threshold = rain_threshold

    def get_race_weather_decision(
        self,
        lat: float,
        lon: float,
        race_utc_iso: str,
    ) -> WeatherDecision:
        """
        Fetch hourly weather for a race location and evaluate routing decision.

        Args:
            lat:           Circuit latitude.
            lon:           Circuit longitude.
            race_utc_iso:  Race start time in ISO-8601 UTC (e.g. '2026-03-29T14:00:00Z').

        Returns:
            WeatherDecision object with branch routing flag.
        """
        logger.info(
            f"Querying weather at ({lat:.4f}, {lon:.4f}) for race time {race_utc_iso}"
        )

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation_probability,precipitation,temperature_2m,windspeed_10m",
            "timezone": "UTC",
            "forecast_days": 4,
        }

        try:
            response = requests.get(OPENMETEO_BASE, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error(f"Weather API call failed: {exc}. Defaulting to DRY branch.")
            # Fail-safe: on API error, run the standard model rather than blocking
            return self._failed_decision()

        decision = self._extract_decision(data, race_utc_iso)
        logger.info(f"Weather decision: {decision}")
        return decision

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_decision(self, data: dict, race_utc_iso: str) -> WeatherDecision:
        """
        Finds the hourly bucket closest to the race start time and reads
        precipitation probability and amount for that hour.
        """
        race_dt = datetime.fromisoformat(race_utc_iso.replace("Z", "+00:00"))
        times = data["hourly"]["time"]

        # Find index of the closest hour to race start
        best_idx = 0
        best_delta = float("inf")
        for i, t in enumerate(times):
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            delta = abs((dt - race_dt).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_idx = i

        rain_prob = (data["hourly"]["precipitation_probability"][best_idx] or 0) / 100
        precip_mm = data["hourly"]["precipitation"][best_idx] or 0.0
        temp_c    = data["hourly"]["temperature_2m"][best_idx] or 20.0
        wind_kmh  = data["hourly"]["windspeed_10m"][best_idx] or 0.0

        route_wet = (rain_prob > self.rain_threshold) or (precip_mm > DRIZZLE_MM_THRESHOLD)

        return WeatherDecision(
            rain_probability=rain_prob,
            precipitation_mm=precip_mm,
            temperature_c=temp_c,
            wind_kmh=wind_kmh,
            route_to_wet_model=route_wet,
            queried_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _failed_decision() -> WeatherDecision:
        return WeatherDecision(
            rain_probability=0.0,
            precipitation_mm=0.0,
            temperature_c=20.0,
            wind_kmh=0.0,
            route_to_wet_model=False,
            queried_at=datetime.now(timezone.utc).isoformat(),
        )


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 APEX — Weather Router CLI")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--race-time", type=str, required=True,
                        help="Race UTC start in ISO-8601 e.g. 2026-03-29T14:00:00Z")
    args = parser.parse_args()

    router = WeatherRouter()
    decision = router.get_race_weather_decision(args.lat, args.lon, args.race_time)
    print(f"Branch: {decision.branch} | Rain prob: {decision.rain_probability:.1%} | "
          f"Precip: {decision.precipitation_mm:.1f} mm/hr")
