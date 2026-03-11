"""Client helpers for the remote RSM measurement API.

This module isolates the network and filesystem side effects required to:

- Fetch paginated sensor measurements from the remote REST API.
- Persist the payloads to CSV files under ``data/`` using the repository's
  existing acquisition schema.
- Reload those CSV files into pandas data frames with parsed PSD arrays.

The intent is to keep notebooks thin while making the boundary behavior
testable without embedding raw HTTP logic inside notebook cells.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning


DEFAULT_API_BASE_URL = "https://rsm.ane.gov.co:12443/api"
DEFAULT_DATA_DOWNLOAD_DIR = Path("data") / "api_downloads"
DEFAULT_CAMPAIGN_IDS: dict[str, int] = {
    "no dc, no iq": 202,
    "no dc, yes iq": 203,
    "yes dc, yes iq": 204,
    "test-calibration": 205,
    "FM original": 176,
}
DEFAULT_SENSOR_MAC_BY_LABEL: dict[str, str] = {
    "Node1": "d8:3a:dd:f7:1d:f2",
    "Node2": "d8:3a:dd:f4:4e:26",
    "Node3": "d8:3a:dd:f7:22:87",
    "Node5": "d8:3a:dd:f7:21:52",
    "Node9": "d8:3a:dd:f4:4e:d1",
    "Node10": "d8:3a:dd:f7:1d:90",
}

CSV_FIELDNAMES: tuple[str, ...] = (
    "id",
    "mac",
    "campaign_id",
    "pxx",
    "start_freq_hz",
    "end_freq_hz",
    "timestamp",
    "lat",
    "lng",
    "excursion_peak_to_peak_hz",
    "excursion_peak_deviation_hz",
    "excursion_rms_deviation_hz",
    "depth_peak_to_peak",
    "depth_peak_deviation",
    "depth_rms_deviation",
    "created_at",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "id",
    "campaign_id",
    "start_freq_hz",
    "end_freq_hz",
    "timestamp",
    "lat",
    "lng",
    "excursion_peak_to_peak_hz",
    "excursion_peak_deviation_hz",
    "excursion_rms_deviation_hz",
    "depth_peak_to_peak",
    "depth_peak_deviation",
    "depth_rms_deviation",
    "created_at",
)


class MeasurementApiError(RuntimeError):
    """Raised when the remote API response or payload contract is invalid."""


@dataclass(frozen=True)
class MeasurementApiConfig:
    """Configuration for the measurement API HTTP boundary.

    Parameters
    ----------
    base_url:
        Base REST URL without a trailing slash.
    verify_tls:
        Whether HTTPS certificates should be verified. The deployed API used by
        the exploratory notebook currently requires ``False`` because it serves
        a certificate chain that is not trusted in this environment.
    timeout_s:
        Request timeout applied to each HTTP page fetch [s].
    page_size:
        Number of measurements requested per paginated response.
    """

    base_url: str = DEFAULT_API_BASE_URL
    verify_tls: bool = False
    timeout_s: float = 30.0
    page_size: int = 5_000


class MeasurementApiClient:
    """Thin adapter for the paginated sensor measurement API."""

    def __init__(
        self,
        config: MeasurementApiConfig | None = None,  # HTTP boundary settings
        session: requests.Session | None = None,  # Optional injected session
    ) -> None:
        """Initialize the client with explicit boundary dependencies."""

        resolved_config = MeasurementApiConfig() if config is None else config
        if resolved_config.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if resolved_config.page_size <= 0:
            raise ValueError("page_size must be positive")

        self._config = resolved_config
        self._session = requests.Session() if session is None else session

        # The current API endpoint uses an untrusted certificate chain, so keep
        # the warning suppression local to this adapter when TLS verification is
        # intentionally disabled.
        if not resolved_config.verify_tls:
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    @property
    def config(self) -> MeasurementApiConfig:
        """Return the immutable HTTP configuration used by this client."""

        return self._config

    def fetch_sensor_measurements(
        self,
        mac_address: str,  # Sensor MAC address accepted by the API
        campaign_id: int,  # Campaign identifier used as query parameter
    ) -> list[dict[str, Any]]:  # Normalized measurements across all pages
        """Fetch every measurement page for one sensor and campaign.

        Parameters
        ----------
        mac_address:
            Sensor MAC address used in the endpoint path.
        campaign_id:
            Campaign identifier used in the paginated query string.

        Returns
        -------
        list[dict[str, Any]]
            Raw measurement dictionaries as returned by the API.

        Raises
        ------
        ValueError
            If the sensor MAC or campaign identifier is invalid.
        MeasurementApiError
            If the HTTP request fails or the JSON payload does not contain the
            expected pagination contract.
        """

        if not mac_address.strip():
            raise ValueError("mac_address must be a non-empty string")
        if campaign_id <= 0:
            raise ValueError("campaign_id must be positive")

        page = 1
        measurements: list[dict[str, Any]] = []
        url = f"{self._config.base_url.rstrip('/')}/campaigns/sensor/{mac_address}/signals"

        while True:
            params = {
                "campaign_id": campaign_id,
                "page": page,
                "page_size": self._config.page_size,
            }
            payload = self._request_json(url=url, params=params)
            page_measurements = payload.get("measurements")
            pagination = payload.get("pagination")
            if not isinstance(page_measurements, list):
                raise MeasurementApiError(
                    "API payload is missing a list-valued 'measurements' field"
                )
            if not isinstance(pagination, Mapping) or "has_next" not in pagination:
                raise MeasurementApiError(
                    "API payload is missing the 'pagination.has_next' contract"
                )

            for measurement in page_measurements:
                if not isinstance(measurement, Mapping):
                    raise MeasurementApiError(
                        "Every item in 'measurements' must be a JSON object"
                    )
                measurements.append(dict(measurement))
            if not bool(pagination["has_next"]):
                break

            page += 1

        return measurements

    def download_campaign_csvs(
        self,
        campaign_label: str,  # Human-readable name used for the output folder
        campaign_id: int,  # Campaign identifier used for every sensor request
        sensor_mac_by_label: Mapping[str, str],  # Output CSV name -> sensor MAC
        output_root: Path = DEFAULT_DATA_DOWNLOAD_DIR,  # Root under data/
        drop_missing_pxx: bool = True,  # Skip rows without PSD payloads
    ) -> dict[str, Path]:  # Saved CSV path per sensor label
        """Download one campaign and persist each sensor payload as a CSV file.

        Parameters
        ----------
        campaign_label:
            Human-readable campaign name. It is sanitized into a directory name.
        campaign_id:
            Numeric campaign identifier.
        sensor_mac_by_label:
            Mapping from a sensor label, for example ``"Node1"``, to the
            corresponding MAC address expected by the API.
        output_root:
            Root directory under which a campaign subdirectory is created.
        drop_missing_pxx:
            Whether rows with missing ``pxx`` arrays should be discarded before
            writing the CSV. This keeps the output directly usable by the
            plotting notebook and by the calibration loaders.

        Returns
        -------
        dict[str, Path]
            Saved CSV path for each requested sensor label.
        """

        if not campaign_label.strip():
            raise ValueError("campaign_label must be a non-empty string")
        if not sensor_mac_by_label:
            raise ValueError("sensor_mac_by_label must contain at least one sensor")

        output_dir = Path(output_root) / _sanitize_path_component(campaign_label)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        for sensor_label, mac_address in sensor_mac_by_label.items():
            measurements = self.fetch_sensor_measurements(
                mac_address=mac_address,
                campaign_id=campaign_id,
            )
            if drop_missing_pxx:
                measurements = [
                    measurement
                    for measurement in measurements
                    if measurement.get("pxx") not in ("", None)
                ]

            output_path = output_dir / f"{_sanitize_path_component(sensor_label)}.csv"
            save_measurements_csv(
                measurements=measurements,
                output_path=output_path,
                mac_address=mac_address,
                campaign_id=campaign_id,
            )
            saved_paths[sensor_label] = output_path

        return saved_paths

    def _request_json(
        self,
        url: str,  # Fully-qualified endpoint URL
        params: Mapping[str, Any],  # Query string parameters for the request
    ) -> dict[str, Any]:  # Parsed JSON object payload
        """Execute one HTTP GET request and validate that the payload is JSON."""

        try:
            response = self._session.get(
                url,
                params=params,
                verify=self._config.verify_tls,
                timeout=self._config.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise MeasurementApiError(
                f"Request to {url} failed with params {dict(params)!r}"
            ) from exc
        except ValueError as exc:
            raise MeasurementApiError(
                f"Endpoint {url} returned a non-JSON payload"
            ) from exc

        if not isinstance(payload, dict):
            raise MeasurementApiError("API payload must be a JSON object")
        return payload


def save_measurements_csv(
    measurements: Sequence[Mapping[str, Any]],  # Raw API measurements to persist
    output_path: Path,  # CSV destination path
    mac_address: str,  # Sensor MAC used as a fallback field value
    campaign_id: int,  # Campaign identifier used as a fallback field value
) -> Path:  # The written CSV path for fluent call chains
    """Persist measurement payloads using the repository's acquisition schema."""

    csv.field_size_limit(sys.maxsize)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize every payload row before touching the filesystem so contract
    # errors fail fast and do not leave partially-written files behind.
    normalized_rows = [
        _normalize_measurement_row(
            measurement=measurement,
            mac_address=mac_address,
            campaign_id=campaign_id,
        )
        for measurement in measurements
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return output_path


def load_measurement_dataframe(
    csv_path: Path,  # CSV path produced by save_measurements_csv
) -> pd.DataFrame:  # DataFrame with parsed PSD arrays and numeric metadata
    """Load one saved measurement CSV into a typed pandas data frame.

    The returned frame keeps ``pxx`` as ``numpy.float64`` arrays so notebooks
    can plot directly without re-parsing JSON manually.
    """

    csv.field_size_limit(sys.maxsize)
    frame = pd.read_csv(csv_path)

    # Parse the PSD column explicitly because pandas does not understand the
    # JSON-encoded array representation used by the acquisition CSV schema.
    frame["pxx"] = frame["pxx"].apply(_parse_pxx_array)

    # Coerce numeric metadata columns consistently so plotting and comparisons
    # do not depend on pandas' heuristic mixed-type inference.
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def load_measurement_frames(
    csv_paths_by_label: Mapping[str, Path],  # Sensor label -> saved CSV path
) -> dict[str, pd.DataFrame]:  # Loaded frame per sensor label
    """Load multiple saved sensor CSVs into pandas data frames."""

    return {
        sensor_label: load_measurement_dataframe(csv_path)
        for sensor_label, csv_path in csv_paths_by_label.items()
    }


def _normalize_measurement_row(
    measurement: Mapping[str, Any],  # One API measurement payload row
    mac_address: str,  # Fallback MAC address when payload omits it
    campaign_id: int,  # Fallback campaign identifier when payload omits it
) -> dict[str, str]:  # CSV-ready row matching CSV_FIELDNAMES
    """Normalize one API payload row into the acquisition CSV schema."""

    normalized_row: dict[str, str] = {}
    for field_name in CSV_FIELDNAMES:
        raw_value = measurement.get(field_name, "")
        if field_name == "mac" and raw_value in ("", None):
            raw_value = mac_address
        elif field_name == "campaign_id" and raw_value in ("", None):
            raw_value = campaign_id
        elif field_name == "pxx":
            normalized_row[field_name] = _serialize_pxx(raw_value)
            continue

        normalized_row[field_name] = "" if raw_value is None else str(raw_value)

    return normalized_row


def _serialize_pxx(
    raw_pxx: Any,  # Raw pxx field from the API payload
) -> str:  # Compact JSON array string accepted by the existing CSV loaders
    """Serialize the PSD payload into a compact JSON array string."""

    if raw_pxx in ("", None):
        return ""
    if isinstance(raw_pxx, str):
        try:
            parsed_pxx = json.loads(raw_pxx)
        except json.JSONDecodeError as exc:
            raise MeasurementApiError("pxx string payload is not valid JSON") from exc
    elif isinstance(raw_pxx, Sequence):
        parsed_pxx = list(raw_pxx)
    else:
        raise MeasurementApiError("pxx payload must be a JSON array or sequence")

    if not isinstance(parsed_pxx, list):
        raise MeasurementApiError("pxx payload must decode to a JSON array")

    try:
        numeric_pxx = [float(value) for value in parsed_pxx]
    except (TypeError, ValueError) as exc:
        raise MeasurementApiError("pxx payload contains non-numeric values") from exc

    return json.dumps(numeric_pxx, separators=(",", ":"))


def _parse_pxx_array(
    raw_value: Any,  # Serialized PSD JSON array from the CSV
) -> np.ndarray:  # PSD array in floating-point dB units
    """Parse one serialized PSD array from the saved CSV representation."""

    if pd.isna(raw_value) or raw_value == "":
        return np.asarray([], dtype=np.float64)

    parsed = json.loads(str(raw_value))
    return np.asarray(parsed, dtype=np.float64)


def _sanitize_path_component(
    value: str,  # Human-readable label or campaign name
) -> str:  # Filesystem-safe component
    """Convert a label into a stable, filesystem-safe path component."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        raise ValueError("Path component cannot be empty after sanitization")
    return sanitized


__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_CAMPAIGN_IDS",
    "DEFAULT_DATA_DOWNLOAD_DIR",
    "DEFAULT_SENSOR_MAC_BY_LABEL",
    "MeasurementApiClient",
    "MeasurementApiConfig",
    "MeasurementApiError",
    "load_measurement_dataframe",
    "load_measurement_frames",
    "save_measurements_csv",
]
