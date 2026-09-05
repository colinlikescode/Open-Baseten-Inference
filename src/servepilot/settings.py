"""Configuration: environment settings and the YAML configuration file model.

Two layers exist on purpose:

* :class:`ServePilotSettings` – process-wide environment settings (``SERVEPILOT_*``) such as cache
  and state directories, port ranges, and testing switches.
* :class:`ServePilotConfig` – a validated ``servepilot.yaml`` document describing *what* to serve.
  CLI flags override file values through :func:`ServePilotConfig.merged`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from servepilot.constants import (
    CACHE_DIR_NAME,
    DEFAULT_BACKEND_PORT_END,
    DEFAULT_BACKEND_PORT_START,
    DEFAULT_BENCHMARK_SEED,
    DEFAULT_FINAL_CONFIRMATION_MULTIPLIER,
    DEFAULT_MAX_QUEUE_DEPTH,
    DEFAULT_MAX_STRUCTURAL_CANDIDATES,
    DEFAULT_MEMORY_HEADROOM_FRACTION,
    DEFAULT_PUBLIC_HOST,
    DEFAULT_PUBLIC_PORT,
    DEFAULT_STAGE_A_REQUESTS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_SWEEP_REQUESTS_PER_POINT,
    DEFAULT_TOP_K_FOR_CONCURRENCY_SWEEP,
)
from servepilot.exceptions import ConfigurationError
from servepilot.schemas.workload import (
    PRESETS,
    LatencyConstraints,
    Objective,
    WorkloadProfile,
    workload_from_preset,
)


def default_cache_dir() -> Path:
    """XDG-compatible cache directory (``$XDG_CACHE_HOME/servepilot`` or ``~/.cache/servepilot``)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / CACHE_DIR_NAME


def default_state_dir() -> Path:
    """XDG-compatible state directory (``$XDG_STATE_HOME/servepilot`` or ``~/.local/state/servepilot``)."""
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / CACHE_DIR_NAME


class ServePilotSettings(BaseSettings):
    """Environment-driven process settings (prefix ``SERVEPILOT_``)."""

    model_config = SettingsConfigDict(env_prefix="SERVEPILOT_", extra="ignore")

    cache_dir: Path = Field(default_factory=default_cache_dir)
    state_dir: Path = Field(default_factory=default_state_dir)
    backend_port_start: int = DEFAULT_BACKEND_PORT_START
    backend_port_end: int = DEFAULT_BACKEND_PORT_END
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS
    # Testing switches. ``fake_hardware`` names a fixture from ``servepilot.testing.fake_hardware``;
    # ``enable_fake_engine`` exposes the ``fake`` engine adapter.
    fake_hardware: str | None = None
    enable_fake_engine: bool = False
    fake_engine_behavior: Path | None = None


class ProfileConfig(BaseModel):
    """Workload section of the YAML file."""

    name: str | None = None
    input_tokens_p50: int | None = None
    input_tokens_p95: int | None = None
    output_tokens_p50: int | None = None
    output_tokens_p95: int | None = None
    max_context_tokens: int | None = None
    expected_concurrency: int | None = None
    target_request_rate: float | None = None
    shared_prefix_fraction: float | None = None
    streaming: bool | None = None
    max_p95_ttft_ms: float | None = None
    max_p95_latency_ms: float | None = None
    max_p95_tpot_ms: float | None = None

    @field_validator("name")
    @classmethod
    def _check_preset(cls, v: str | None) -> str | None:
        if v is not None and v != "custom" and v not in PRESETS:
            raise ValueError(f"unknown profile preset {v!r}; choose from {sorted(PRESETS)}")
        return v


class HardwareConfig(BaseModel):
    gpus: Literal["auto"] | list[int] = "auto"
    memory_headroom: float = Field(default=DEFAULT_MEMORY_HEADROOM_FRACTION, ge=0.0, lt=1.0)
    allow_busy_gpus: bool = False


class ServerConfig(BaseModel):
    host: str = DEFAULT_PUBLIC_HOST
    port: int = Field(default=DEFAULT_PUBLIC_PORT, ge=1, le=65535)
    served_model_name: str | None = None
    max_queue_depth: int = Field(default=DEFAULT_MAX_QUEUE_DEPTH, ge=0)


class TuningConfig(BaseModel):
    enabled: bool = True
    use_cache: bool = True
    max_structural_candidates: int = Field(default=DEFAULT_MAX_STRUCTURAL_CANDIDATES, ge=1)
    top_k: int = Field(default=DEFAULT_TOP_K_FOR_CONCURRENCY_SWEEP, ge=1)
    stage_a_requests: int = Field(default=DEFAULT_STAGE_A_REQUESTS, ge=1)
    sweep_requests: int = Field(default=DEFAULT_SWEEP_REQUESTS_PER_POINT, ge=1)
    final_multiplier: int = Field(default=DEFAULT_FINAL_CONFIRMATION_MULTIPLIER, ge=1)
    memory_tuning: bool = True
    startup_timeout_seconds: float = Field(default=DEFAULT_STARTUP_TIMEOUT_SECONDS, gt=0)
    seed: int = DEFAULT_BENCHMARK_SEED
    resume: bool = False


class ModelOptionsConfig(BaseModel):
    revision: str | None = None
    trust_remote_code: bool = False
    kv_cache_dtype: str | None = None


class ConstraintsConfig(BaseModel):
    tp: int | None = Field(default=None, ge=1)
    replicas: int | None = Field(default=None, ge=1)
    context_length: int | None = Field(default=None, ge=16)
    max_concurrency: int | None = Field(default=None, ge=1)
    memory_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    allow_context_override: bool = False
    engine_args: dict[str, Any] = Field(default_factory=dict)


class ServePilotConfig(BaseModel):
    """Validated ``servepilot.yaml``."""

    model: str | None = None
    engine: Literal["auto", "vllm", "sglang", "fake"] = "auto"
    objective: Objective = Objective.THROUGHPUT
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    model_options: ModelOptionsConfig = Field(default_factory=ModelOptionsConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_top_level(cls, data: Any) -> Any:
        if isinstance(data, dict):
            unknown = set(data) - set(cls.model_fields)
            if unknown:
                raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return data

    def merged(self, overrides: dict[str, Any]) -> ServePilotConfig:
        """Return a copy with nested ``overrides`` applied (``None`` values are ignored).

        ``overrides`` uses dotted keys such as ``"server.port"``.
        """
        data = self.model_dump(mode="python")
        for dotted, value in overrides.items():
            if value is None:
                continue
            target = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        try:
            return ServePilotConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigurationError(_format_validation_error(exc)) from exc

    def build_workload(self) -> WorkloadProfile:
        """Materialize a :class:`WorkloadProfile` from the profile section and objective."""
        p = self.profile
        slo = LatencyConstraints(
            max_p95_ttft_ms=p.max_p95_ttft_ms,
            max_p95_latency_ms=p.max_p95_latency_ms,
            max_p95_tpot_ms=p.max_p95_tpot_ms,
        )
        constraints = None if slo.is_empty else slo
        preset_name = p.name or "chat"
        try:
            if preset_name == "custom":
                base = workload_from_preset("chat")
            else:
                base = workload_from_preset(preset_name)
            values = base.model_dump()
            values["name"] = preset_name
            # Changing the token distribution without naming a preset makes the profile custom;
            # context, concurrency, request rate and streaming leave the preset name alone.
            distribution_keys = (
                "input_tokens_p50",
                "input_tokens_p95",
                "output_tokens_p50",
                "output_tokens_p95",
                "shared_prefix_fraction",
            )
            for key in (
                *distribution_keys,
                "max_context_tokens",
                "expected_concurrency",
                "target_request_rate",
                "streaming",
            ):
                v = getattr(p, key)
                if v is not None:
                    values[key] = v
                    if key in distribution_keys and not p.name:
                        values["name"] = "custom"
            values["objective"] = self.objective
            values["latency_constraints"] = constraints
            # Grow the context automatically when the user asked for longer prompts than the
            # preset context permits, rather than failing validation.
            needed = values["input_tokens_p95"] + values["output_tokens_p95"]
            if values["max_context_tokens"] < needed and p.max_context_tokens is None:
                values["max_context_tokens"] = needed
            return WorkloadProfile.model_validate(values)
        except ValidationError as exc:
            raise ConfigurationError(
                _format_validation_error(exc, header="workload profile is invalid:"),
                hints=[
                    "Raise --context-length, or lower --input-tokens-p95 / --output-tokens-p95.",
                    "Presets: chat, long-context, decode-heavy (see `servepilot plan --help`).",
                ],
            ) from exc
        except ValueError as exc:
            raise ConfigurationError(f"workload profile is invalid: {exc}") from exc


def _format_validation_error(
    exc: ValidationError, *, header: str = "configuration is invalid:"
) -> str:
    lines = [header]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        lines.append(f"  {loc}: {msg}" if loc else f"  {msg}")
    return "\n".join(lines)


def load_config(path: Path | None) -> ServePilotConfig:
    """Load and validate a YAML configuration file; a missing path yields defaults."""
    if path is None:
        return ServePilotConfig()
    if not path.exists():
        raise ConfigurationError(
            f"configuration file not found: {path}",
            hints=["Check the path passed to --config."],
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"could not parse YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration file {path} must contain a mapping at top level")
    try:
        return ServePilotConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc)) from exc
