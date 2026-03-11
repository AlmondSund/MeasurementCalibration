"""Tests for the notebook-facing remote API client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import requests

from api.client import (
    SENSOR_NETWORK_MAC_BY_LABEL,
    MeasurementApiClient,
    MeasurementApiConfig,
    MeasurementApiError,
    build_campaign_output_dir,
    load_measurement_dataframe,
    resolve_sensor_mac_by_label,
)


@dataclass
class _FakeResponse:
    """Minimal fake response object compatible with the client adapter."""

    payload: dict[str, Any] | None = None
    status_code: int = 200
    url: str = "https://example.test/api"

    def raise_for_status(self) -> None:
        """Raise ``requests.HTTPError`` when the configured status is failing."""

        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response.url = self.url
            raise requests.HTTPError(
                f"{self.status_code} Client Error",
                response=response,
            )

    def json(self) -> dict[str, Any]:
        """Return the preconfigured JSON payload."""

        if self.payload is None:
            raise ValueError("No JSON payload configured for the fake response")
        return self.payload


class _FakeSession:
    """Record HTTP calls and replay deterministic responses keyed by URL."""

    def __init__(self, responses_by_url: Mapping[str, list[_FakeResponse]]) -> None:
        """Store the response queue used by successive ``get`` calls."""

        self._responses_by_url = {
            url: list(responses)
            for url, responses in responses_by_url.items()
        }
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any],
        verify: bool,
        timeout: float,
    ) -> _FakeResponse:
        """Return the next fake response while recording the request contract."""

        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "verify": verify,
                "timeout": timeout,
            }
        )
        if url not in self._responses_by_url or not self._responses_by_url[url]:
            raise AssertionError(f"Unexpected HTTP request in test: {url!r}")

        response = self._responses_by_url[url].pop(0)
        response.url = url
        return response


def _sensor_url(mac_address: str) -> str:
    """Build the sensor endpoint URL used by the fake-session tests."""

    return f"https://rsm.ane.gov.co:12443/api/campaigns/sensor/{mac_address}/signals"


def test_fetch_sensor_measurements_paginates_until_has_next_is_false() -> None:
    """The client should accumulate rows across every paginated response."""

    mac_address = SENSOR_NETWORK_MAC_BY_LABEL["Node1"]
    session = _FakeSession(
        responses_by_url={
            _sensor_url(mac_address): [
                _FakeResponse(
                    payload={
                        "measurements": [{"id": 1, "pxx": [-60.0, -59.0]}],
                        "pagination": {"has_next": True},
                    }
                ),
                _FakeResponse(
                    payload={
                        "measurements": [{"id": 2, "pxx": [-58.0, -57.5]}],
                        "pagination": {"has_next": False},
                    }
                ),
            ]
        }
    )
    client = MeasurementApiClient(
        config=MeasurementApiConfig(page_size=2, verify_tls=False, timeout_s=12.0),
        session=session,
    )

    measurements = client.fetch_sensor_measurements(
        mac_address=mac_address,
        campaign_id=176,
    )

    assert [measurement["id"] for measurement in measurements] == [1, 2]
    assert len(session.calls) == 2
    assert session.calls[0]["params"]["page"] == 1
    assert session.calls[1]["params"]["page"] == 2
    assert session.calls[0]["params"]["page_size"] == 2
    assert session.calls[0]["verify"] is False
    assert session.calls[0]["timeout"] == pytest.approx(12.0)


def test_download_campaign_csvs_writes_campaign_data_and_skips_404_sensors(
    tmp_path: Path,
) -> None:
    """One campaign request should adapt to partial sensor availability."""

    node1_mac = SENSOR_NETWORK_MAC_BY_LABEL["Node1"]
    node2_mac = SENSOR_NETWORK_MAC_BY_LABEL["Node2"]
    session = _FakeSession(
        responses_by_url={
            _sensor_url(node1_mac): [
                _FakeResponse(
                    payload={
                        "measurements": [
                            {
                                "id": 208061,
                                "pxx": [-21.09, -21.24, -21.19],
                                "start_freq_hz": 88_000_000,
                                "end_freq_hz": 108_000_000,
                                "timestamp": 1_771_929_017_886,
                                "created_at": 1_771_947_018_592,
                            },
                            {
                                "id": 208062,
                                "pxx": None,
                                "start_freq_hz": 88_000_000,
                                "end_freq_hz": 108_000_000,
                                "timestamp": 1_771_929_117_886,
                                "created_at": 1_771_947_118_592,
                            },
                        ],
                        "pagination": {"has_next": False},
                    }
                )
            ],
            _sensor_url(node2_mac): [_FakeResponse(status_code=404)],
        }
    )
    client = MeasurementApiClient(session=session)

    result = client.download_campaign_csvs(
        campaign_label="FM original",
        campaign_id=176,
        sensor_labels=("Node1", "Node2"),
        output_root=tmp_path / "campaigns",
        drop_missing_pxx=True,
        skip_missing_sensors=True,
    )

    saved_path = result.saved_csv_paths["Node1"]
    frame = load_measurement_dataframe(saved_path)

    assert result.output_dir == build_campaign_output_dir(
        campaign_label="FM original",
        output_root=tmp_path / "campaigns",
    )
    assert saved_path == tmp_path / "campaigns" / "FM_original" / "Node1.csv"
    assert saved_path.exists()
    assert "Node2" not in result.saved_csv_paths
    assert "Node2" in result.skipped_sensors
    assert "404" in result.skipped_sensors["Node2"]
    assert len(frame) == 1
    assert frame["campaign_id"].iloc[0] == 176
    assert frame["mac"].iloc[0] == node1_mac
    assert frame["timestamp"].iloc[0] == 1_771_929_017_886
    assert np.allclose(frame["pxx"].iloc[0], np.asarray([-21.09, -21.24, -21.19]))
def test_fetch_sensor_measurements_rejects_non_mapping_rows() -> None:
    """Malformed payload rows should fail fast at the HTTP boundary."""

    mac_address = SENSOR_NETWORK_MAC_BY_LABEL["Node1"]
    session = _FakeSession(
        responses_by_url={
            _sensor_url(mac_address): [
                _FakeResponse(
                    payload={
                        "measurements": ["not-a-dict"],
                        "pagination": {"has_next": False},
                    }
                )
            ]
        }
    )
    client = MeasurementApiClient(session=session)

    with pytest.raises(MeasurementApiError, match="Every item"):
        client.fetch_sensor_measurements(
            mac_address=mac_address,
            campaign_id=176,
        )


def test_resolve_sensor_mac_by_label_supports_full_network_and_subset_labels() -> None:
    """Sensor resolution should expose the full network and stable subsets."""

    full_network = resolve_sensor_mac_by_label()
    subset_network = resolve_sensor_mac_by_label(sensor_labels=("Node3", "Node10"))

    assert tuple(full_network) == tuple(f"Node{index}" for index in range(1, 11))
    assert full_network["Node10"] == "d8:3a:dd:f7:1d:90"
    assert subset_network == {
        "Node3": "d8:3a:dd:f7:22:87",
        "Node10": "d8:3a:dd:f7:1d:90",
    }
