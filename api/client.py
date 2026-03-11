"""Client helpers for the remote RSM measurement API.

This module isolates the network and filesystem side effects required to:

- Fetch paginated sensor measurements from the remote REST API.
- Resolve campaign downloads against the full Node1..Node10 sensor network.
- Persist campaign payloads to CSV files under ``data/campaigns/`` using the
  repository's acquisition schema.
- Reload those CSV files into pandas data frames with parsed PSD arrays.

The client does not own any campaign registry. Callers must provide the
campaign label and numeric identifier explicitly for each download request.
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


API_BASE_URL = "https://rsm.ane.gov.co:12443/api"
CAMPAIGNS_DATA_DIR = Path("data") / "campaigns"

# Keep a single source of truth for the deployed Node1..Node10 network.
SENSOR_NETWORK_MAC_BY_LABEL: dict[str, str] = {
    "Node1": "d8:3a:dd:f7:1d:f2",
    "Node2": "d8:3a:dd:f4:4e:26",
    "Node3": "d8:3a:dd:f7:22:87",
    "Node4": "d8:3a:dd:f6:fc:be",
    "Node5": "d8:3a:dd:f7:21:52",
    "Node6": "d8:3a:dd:f7:1a:cc",
    "Node7": "d8:3a:dd:f7:1d:b6",
    "Node8": "d8:3a:dd:f7:1b:20",
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


class MeasurementApiRequestError(MeasurementApiError):
    """Raised when the remote API request fails at the HTTP boundary."""

    def __init__(
        self,
        message: str,  # Human-readable request failure description
        *,
        status_code: int | None = None,  # HTTP status code when available
    ) -> None:
        """Store the request failure with the optional HTTP status code."""

        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class CampaignDownloadResult:
    """Materialized result of one campaign download request.

    Parameters
    ----------
    campaign_label:
        Human-readable campaign label provided by the caller.
    campaign_id:
        Numeric campaign identifier provided by the caller.
    output_dir:
        Directory where campaign CSV files were written.
    requested_sensor_mac_by_label:
        Effective sensor mapping used for this request after resolving any
        caller-provided labels or overrides.
    saved_csv_paths:
        CSV path written for each sensor that produced retained measurements.
        skipped_sensors:
            Sensors that were skipped, together with the reason. Typical causes are
            API 404 responses or empty datasets after filtering.
    """

    campaign_label: str
    campaign_id: int
    output_dir: Path
    requested_sensor_mac_by_label: dict[str, str]
    saved_csv_paths: dict[str, Path]
    skipped_sensors: dict[str, str]


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

    base_url: str = API_BASE_URL
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
        campaign_label: str,  # Human-readable campaign name for the output path
        campaign_id: int,  # Numeric campaign identifier supplied by the caller
        sensor_labels: Sequence[str] | None = None,  # Optional subset of nodes
        sensor_mac_by_label: Mapping[str, str] | None = None,  # Optional MAC overrides
        output_root: Path = CAMPAIGNS_DATA_DIR,  # Root under data/campaigns/
        drop_missing_pxx: bool = True,  # Skip rows without PSD payloads
        skip_missing_sensors: bool = True,  # Adapt to partial campaign coverage
    ) -> CampaignDownloadResult:
        """Download one campaign and persist each available sensor payload as CSV.

        Parameters
        ----------
        campaign_label:
            Human-readable campaign label. It is sanitized into a directory
            component, but it is otherwise owned by the caller.
        campaign_id:
            Numeric campaign identifier supplied by the caller.
        sensor_labels:
            Optional subset of sensor labels, for example ``("Node1", "Node10")``.
            When omitted, the full ``SENSOR_NETWORK_MAC_BY_LABEL`` constant is used.
        sensor_mac_by_label:
            Optional explicit sensor mapping. This is useful when the caller
            wants to override the repository sensor network or provide an ad-hoc
            mapping for a campaign. It is mutually exclusive with
            ``sensor_labels``.
        output_root:
            Root directory under which a campaign subdirectory is created.
        drop_missing_pxx:
            Whether rows with missing ``pxx`` arrays should be discarded before
            writing the CSV. This keeps the output directly usable by the
            plotting notebook and by the calibration loaders.
        skip_missing_sensors:
            Whether sensors that are unavailable for the campaign should be
            recorded in ``skipped_sensors`` instead of aborting the whole
            request. This enables one campaign request to adapt dynamically to
            partial sensor coverage.

        Returns
        -------
        CampaignDownloadResult
            Structured summary of the materialized campaign download.

        Side Effects
        ------------
        Issues one or more HTTPS GET requests per resolved sensor and writes one
        CSV file per retained sensor dataset under ``output_root``.
        """

        if not campaign_label.strip():
            raise ValueError("campaign_label must be a non-empty string")
        if campaign_id <= 0:
            raise ValueError("campaign_id must be positive")

        resolved_sensor_mac_by_label = resolve_sensor_mac_by_label(
            sensor_labels=sensor_labels,
            sensor_mac_by_label=sensor_mac_by_label,
        )
        output_dir = build_campaign_output_dir(
            campaign_label=campaign_label,
            output_root=output_root,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        skipped_sensors: dict[str, str] = {}

        for sensor_label, mac_address in resolved_sensor_mac_by_label.items():
            try:
                measurements = self.fetch_sensor_measurements(
                    mac_address=mac_address,
                    campaign_id=campaign_id,
                )
            except MeasurementApiRequestError as exc:
                if skip_missing_sensors and exc.status_code == 404:
                    skipped_sensors[sensor_label] = (
                        f"Remote API returned HTTP 404 for campaign_id={campaign_id}"
                    )
                    continue
                raise

            if drop_missing_pxx:
                measurements = [
                    measurement
                    for measurement in measurements
                    if measurement.get("pxx") not in ("", None)
                ]

            if not measurements:
                message = "No measurements remained after applying the download filter"
                if skip_missing_sensors:
                    skipped_sensors[sensor_label] = message
                    continue
                raise MeasurementApiError(
                    f"{sensor_label} returned no retained measurements for campaign_id={campaign_id}"
                )

            output_path = output_dir / f"{_sanitize_path_component(sensor_label)}.csv"
            save_measurements_csv(
                measurements=measurements,
                output_path=output_path,
                mac_address=mac_address,
                campaign_id=campaign_id,
            )
            saved_paths[sensor_label] = output_path

        return CampaignDownloadResult(
            campaign_label=campaign_label,
            campaign_id=campaign_id,
            output_dir=output_dir,
            requested_sensor_mac_by_label=resolved_sensor_mac_by_label,
            saved_csv_paths=saved_paths,
            skipped_sensors=skipped_sensors,
        )

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
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            raise MeasurementApiRequestError(
                f"Request to {url} failed with HTTP {status_code} and params {dict(params)!r}",
                status_code=status_code,
            ) from exc
        except requests.RequestException as exc:
            raise MeasurementApiRequestError(
                f"Request to {url} failed with params {dict(params)!r}",
                status_code=None,
            ) from exc
        except ValueError as exc:
            raise MeasurementApiError(
                f"Endpoint {url} returned a non-JSON payload"
            ) from exc

        if not isinstance(payload, dict):
            raise MeasurementApiError("API payload must be a JSON object")
        return payload


def build_campaign_output_dir(
    campaign_label: str,  # Human-readable campaign name provided by the caller
    output_root: Path = CAMPAIGNS_DATA_DIR,  # Root under data/campaigns/
) -> Path:  # Campaign-specific directory path
    """Build the output directory used for one materialized campaign download."""

    if not campaign_label.strip():
        raise ValueError("campaign_label must be a non-empty string")
    return Path(output_root) / _sanitize_path_component(campaign_label)


def resolve_sensor_mac_by_label(
    sensor_labels: Sequence[str] | None = None,  # Optional subset of node labels
    sensor_mac_by_label: Mapping[str, str] | None = None,  # Optional explicit overrides
) -> dict[str, str]:  # Validated sensor mapping for one request
    """Resolve a sensor selection against the repository sensor-network constants.

    Parameters
    ----------
    sensor_labels:
        Optional subset of labels chosen from ``SENSOR_NETWORK_MAC_BY_LABEL``.
        When omitted, the full sensor network constant is used.
    sensor_mac_by_label:
        Optional explicit mapping supplied by the caller. It is mutually
        exclusive with ``sensor_labels`` and bypasses the repository constants.

    Returns
    -------
    dict[str, str]
        Validated sensor mapping in the iteration order requested by the caller.
    """

    if sensor_labels is not None and sensor_mac_by_label is not None:
        raise ValueError(
            "sensor_labels and sensor_mac_by_label are mutually exclusive"
        )

    if sensor_mac_by_label is not None:
        resolved_mapping = dict(sensor_mac_by_label)
    elif sensor_labels is None:
        resolved_mapping = dict(SENSOR_NETWORK_MAC_BY_LABEL)
    else:
        resolved_mapping = {}
        seen_labels: set[str] = set()
        for raw_label in sensor_labels:
            sensor_label = raw_label.strip()
            if not sensor_label:
                raise ValueError("sensor_labels cannot contain empty labels")
            if sensor_label in seen_labels:
                raise ValueError(
                    f"sensor_labels contains the duplicated label {sensor_label!r}"
                )
            if sensor_label not in SENSOR_NETWORK_MAC_BY_LABEL:
                raise KeyError(
                    f"Unknown sensor label {sensor_label!r}; choose from {tuple(SENSOR_NETWORK_MAC_BY_LABEL)}"
                )
            seen_labels.add(sensor_label)
            resolved_mapping[sensor_label] = SENSOR_NETWORK_MAC_BY_LABEL[sensor_label]

    if not resolved_mapping:
        raise ValueError("At least one sensor must be resolved for a campaign request")

    # Validate the mapping at the boundary so downstream download logic can
    # reason about a single well-formed representation.
    normalized_mapping: dict[str, str] = {}
    for raw_sensor_label, raw_mac_address in resolved_mapping.items():
        sensor_label = raw_sensor_label.strip()
        if not sensor_label:
            raise ValueError("sensor labels must be non-empty strings")
        mac_address = raw_mac_address.strip()
        if not mac_address:
            raise ValueError(f"{sensor_label} has an empty MAC address")
        normalized_mapping[sensor_label] = mac_address

    return normalized_mapping


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
    "API_BASE_URL",
    "CAMPAIGNS_DATA_DIR",
    "CSV_FIELDNAMES",
    "CampaignDownloadResult",
    "MeasurementApiClient",
    "MeasurementApiConfig",
    "MeasurementApiError",
    "MeasurementApiRequestError",
    "NUMERIC_COLUMNS",
    "SENSOR_NETWORK_MAC_BY_LABEL",
    "build_campaign_output_dir",
    "load_measurement_dataframe",
    "load_measurement_frames",
    "resolve_sensor_mac_by_label",
    "save_measurements_csv",
]
