"""Geocode a human-supplied city + country into coordinates.

The training-plan calendar forecasts weather from a home location, but asking a
user for raw latitude/longitude is unfriendly and error-prone. Instead we let
them type a city and country and resolve it to coordinates via the free
Open-Meteo geocoding API (no key required), then store the resolved lat/lon (and
the normalised city/country) alongside.

The API is best-effort and read-only; a failure raises ``GeocodeError`` so the
config save can surface a clear message rather than silently storing garbage.
"""

from __future__ import annotations

from typing import Any

import httpx

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT_S = 10.0


class GeocodeError(ValueError):
    """Raised when a city/country cannot be resolved to coordinates."""


def geocode(city: str, country: str) -> dict[str, Any]:
    """Resolve ``city`` (+ optional ``country``) into geographical coordinates.

    Returns ``{city, country, lat, lon}`` using the first best match. ``country``
    may be a name (e.g. ``"Slovakia"``) or an ISO-3166 code (``"SK"``); when
    provided we prefer a result whose country matches. Raises ``GeocodeError``
    when nothing can be resolved or the request fails.
    """
    city = (city or "").strip()
    country = (country or "").strip()
    if not city:
        raise GeocodeError("enter a city name")

    params: dict[str, Any] = {
        "name": city,
        "count": 10,
        "language": "en",
        "format": "json",
    }
    try:
        resp = httpx.get(_GEOCODE_URL, params=params, timeout=_TIMEOUT_S)
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
    except (httpx.HTTPError, ValueError) as exc:
        raise GeocodeError(f"could not reach the geocoding service: {exc}") from exc

    if not results:
        raise GeocodeError(f"no match found for {city!r}")

    match = _best_match(results, country) if country else results[0]
    return {
        "city": match.get("name") or city,
        "country": match.get("country") or match.get("country_code") or country,
        "lat": float(match["latitude"]),
        "lon": float(match["longitude"]),
    }


def _best_match(results: list[dict[str, Any]], country: str) -> dict[str, Any]:
    """Return the result whose country best matches ``country`` (name or code)."""
    needle = country.lower()
    # First pass: exact match on the country name or the ISO-3166 code.
    for r in results:
        name = (r.get("country") or "").lower()
        code = (r.get("country_code") or "").lower()
        if name == needle or code == needle:
            return r
    # Second pass: substring match (e.g. "Slovak Republic" matched by "slovak").
    for r in results:
        name = (r.get("country") or "").lower()
        if needle in name:
            return r
    # Fall back to the top result (the country check was advisory only).
    return results[0]
