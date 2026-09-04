"""Hardware discovery (NVML) and abstractions used by the planner."""

from servepilot.hardware.base import HardwareProvider, get_hardware_provider

__all__ = ["HardwareProvider", "get_hardware_provider"]
