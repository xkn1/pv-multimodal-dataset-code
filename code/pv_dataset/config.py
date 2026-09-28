from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "site_config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    site = config.get("site", {})
    power = config.get("power", {})
    ramp = config.get("ramp", {})
    required = {
        "site.installed_dc_kwp": site.get("installed_dc_kwp"),
        "site.timezone": site.get("timezone"),
        "power.expected_interval_minutes": power.get("expected_interval_minutes"),
        "ramp.horizons_minutes": ramp.get("horizons_minutes"),
    }
    missing = [name for name, value in required.items() if value in (None, "", [])]
    if missing:
        raise ValueError("Missing required configuration: " + ", ".join(missing))
    if float(site["installed_dc_kwp"]) <= 0:
        raise ValueError("site.installed_dc_kwp must be positive")
    cadence = int(power["expected_interval_minutes"])
    if cadence <= 0:
        raise ValueError("power.expected_interval_minutes must be positive")
    if any(int(h) <= 0 or int(h) % cadence for h in ramp["horizons_minutes"]):
        raise ValueError("Every ramp horizon must be a positive multiple of the power cadence")
