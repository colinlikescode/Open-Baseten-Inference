"""Testing utilities shipped with ServePilot.

These are real implementations of the hardware/engine abstractions that do not need NVIDIA
hardware: fake hardware fixtures, a fake inference engine adapter and a fake OpenAI-compatible
backend. They power ServePilot's own test-suite and let contributors exercise the full tuning
and serving loop on a laptop.
"""
