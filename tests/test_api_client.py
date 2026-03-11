"""Tests for the notebook-facing remote API client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from api.client import (
    MeasurementApiClient,
    MeasurementApiConfig,
    MeasurementApiError,
    load_measurement_dataframe,
)


@dataclass
class _FakeResponse:
    """Minimal fake response object compatible with the client adapter."""

    payload: dict[str, Any]

    def raise_for_status(self) -> None:
        """Pretend that the HTTP status code was successful."""

    def json(self) -> dict[str, Any]:
        """Return the preconfigured JSON payload."""

        return self.payload


class _FakeSession:
    """Record HTTP calls and replay a deterministic list of JSON payloads."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        """Store the payload queue used by successive ``get`` calls."""

        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any],
        verify: bool,
        timeout: float,
    ) -> _FakeResponse:
        """Return the next fake payload while recording the request contract."""

        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "verify": verify,
                "timeout": timeout,
            }
        )
        if not self._payloads:
            raise AssertionError("Unexpected extra HTTP request in test")
        return _FakeResponse(self._payloads.pop(0))


def test_fetch_sensor_measurements_paginates_until_has_next_is_false() -> None:
    """The client should accumulate rows across every paginated response."""

    session = _FakeSession(
        payloads=[
            {
                "measurements": [{"id": 1, "pxx": [-60.0, -59.0]}],
                "pagination": {"has_next": True},
            },
            {
                "measurements": [{"id": 2, "pxx": [-58.0, -57.5]}],
                "pagination": {"has_next": False},
            },
        ]
    )
    client = MeasurementApiClient(
        config=MeasurementApiConfig(page_size=2, verify_tls=False, timeout_s=12.0),
        session=session,
    )

    measurements = client.fetch_sensor_measurements(
        mac_address="d8:3a:dd:f7:1d:f2",
        campaign_id=176,
    )

    assert [measurement["id"] for measurement in measurements] == [1, 2]
    assert len(session.calls) == 2
    assert session.calls[0]["params"]["page"] == 1
    assert session.calls[1]["params"]["page"] == 2
    assert session.calls[0]["params"]["page_size"] == 2
    assert session.calls[0]["verify"] is False
    assert session.calls[0]["timeout"] == pytest.approx(12.0)


def test_download_campaign_csvs_writes_compatible_csv_and_loads_frame(
    tmp_path: Path,
) -> None:
    """Downloaded API payloads should round-trip through the CSV helper."""

    session = _FakeSession(
        payloads=[
            {
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
        ]
    )
    client = MeasurementApiClient(session=session)

    saved_paths = client.download_campaign_csvs(
        campaign_label="FM original",
        campaign_id=176,
        sensor_mac_by_label={"Node1": "d8:3a:dd:f7:1d:f2"},
        output_root=tmp_path / "data",
        drop_missing_pxx=True,
    )

    saved_path = saved_paths["Node1"]
    frame = load_measurement_dataframe(saved_path)

    assert saved_path == tmp_path / "data" / "FM_original" / "Node1.csv"
    assert saved_path.exists()
    assert len(frame) == 1
    assert frame["campaign_id"].iloc[0] == 176
    assert frame["mac"].iloc[0] == "d8:3a:dd:f7:1d:f2"
    assert frame["timestamp"].iloc[0] == 1_771_929_017_886
    assert np.allclose(frame["pxx"].iloc[0], np.asarray([-21.09, -21.24, -21.19]))


def test_fetch_sensor_measurements_rejects_non_mapping_rows() -> None:
    """Malformed payload rows should fail fast at the HTTP boundary."""

    session = _FakeSession(
        payloads=[
            {
                "measurements": ["not-a-dict"],
                "pagination": {"has_next": False},
            }
        ]
    )
    client = MeasurementApiClient(session=session)

    with pytest.raises(MeasurementApiError, match="Every item"):
        client.fetch_sensor_measurements(
            mac_address="d8:3a:dd:f7:1d:f2",
            campaign_id=176,
        )
