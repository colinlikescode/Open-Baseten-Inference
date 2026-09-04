"""ServePilot exception hierarchy.

Every expected failure raised by ServePilot derives from :class:`ServePilotError` and carries an
exit code and an optional list of actionable hints that the CLI renders for the user.
"""

from __future__ import annotations

from collections.abc import Sequence

from servepilot.constants import ExitCode


class ServePilotError(Exception):
    """Base class for all expected ServePilot failures."""

    exit_code: ExitCode = ExitCode.UNEXPECTED_ERROR

    def __init__(self, message: str, *, hints: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hints: list[str] = list(hints or [])

    def __str__(self) -> str:
        return self.message

    def render(self) -> str:
        """Return the message followed by numbered hints, suitable for a terminal."""
        if not self.hints:
            return self.message
        lines = [self.message, "", "Try:"]
        lines.extend(f"  {i}. {hint}" for i, hint in enumerate(self.hints, start=1))
        return "\n".join(lines)


class ConfigurationError(ServePilotError):
    """Invalid user configuration, CLI flags or YAML file."""

    exit_code = ExitCode.CONFIGURATION_ERROR


class HardwareError(ServePilotError):
    """Hardware could not be inspected or is unsuitable."""

    exit_code = ExitCode.ENVIRONMENT_ERROR


class ModelInspectionError(ServePilotError):
    """The model could not be inspected (missing, gated, unreadable, ...)."""

    exit_code = ExitCode.MODEL_ERROR


class EngineUnavailableError(ServePilotError):
    """No usable inference engine is installed or the requested engine is missing."""

    exit_code = ExitCode.ENGINE_UNAVAILABLE


class NoViablePlanError(ServePilotError):
    """The planner could not find any candidate that fits the hardware and constraints."""

    exit_code = ExitCode.NO_VIABLE_PLAN


class LaunchError(ServePilotError):
    """An engine process failed to launch or become ready."""

    exit_code = ExitCode.RUNTIME_FAILURE


class BenchmarkError(ServePilotError):
    """A benchmark could not be executed."""

    exit_code = ExitCode.BENCHMARK_ERROR


class CacheError(ServePilotError):
    """The tuning cache is unreadable or incompatible."""

    exit_code = ExitCode.CACHE_ERROR


class RuntimeStateError(ServePilotError):
    """Runtime state (PID files, ports) is inconsistent."""

    exit_code = ExitCode.RUNTIME_FAILURE
