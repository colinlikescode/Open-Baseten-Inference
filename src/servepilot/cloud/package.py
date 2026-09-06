"""Prepare the installed ServePilot source for reproducible cloud deployment."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from servepilot.exceptions import ConfigurationError

SOURCE_PACKAGE = "git+https://github.com/colinlikescode/Open-Baseten-Inference.git@main"


def prepare_package(
    package: str | None, cache_dir: Path, *, source_dir: Path | None = None
) -> tuple[str, dict[str, str]]:
    """Build a checkout into a wheel, or use an explicitly selected pip requirement.

    Only the built wheel is uploaded: credentials, virtual environments and other workspace
    files never become file mounts. Content-addressed output also includes uncommitted fixes.
    """
    if package is not None:
        local = Path(package).expanduser()
        if local.is_file() and local.suffix == ".whl":
            remote = f"/tmp/servepilot-package/{local.name}"
            return remote, {remote: str(local.resolve())}
        return package, {}

    root = source_dir or Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / "src/servepilot").is_dir():
        return SOURCE_PACKAGE, {}

    files = sorted((root / "src/servepilot").rglob("*.py"))
    files.extend(root / name for name in ("pyproject.toml", "README.md", "LICENSE"))
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    output = cache_dir / "packages" / digest.hexdigest()
    wheels = list(output.glob("*.whl"))
    if not wheels:
        output.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--wheel-dir",
                    str(output),
                    str(root),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConfigurationError(
                f"could not build the ServePilot deployment wheel: {exc}"
            ) from exc
        if result.returncode != 0:
            raise ConfigurationError(
                "could not build the ServePilot deployment wheel",
                hints=[
                    (result.stderr or result.stdout)[-2000:],
                    "Install pip in this environment, or use --package with a wheel or pip requirement.",
                ],
            )
        wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ConfigurationError(f"expected one deployment wheel in {output}, found {len(wheels)}")
    wheel = wheels[0]
    remote = f"/tmp/servepilot-package/{wheel.name}"
    return remote, {remote: str(wheel.resolve())}
