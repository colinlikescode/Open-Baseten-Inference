"""Cloud launches package current source without uploading the surrounding workspace."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from servepilot.cloud.package import SOURCE_PACKAGE, prepare_package
from servepilot.cloud.skypilot import LaunchRequest, render_task
from servepilot.exceptions import ConfigurationError


def test_explicit_requirement_is_preserved(tmp_path: Path) -> None:
    requirement = "git+https://example.com/servepilot.git@123456"
    assert prepare_package(requirement, tmp_path) == (requirement, {})


def test_local_wheel_is_uploaded(tmp_path: Path) -> None:
    wheel = tmp_path / "servepilot-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    package, mounts = prepare_package(str(wheel), tmp_path)
    assert package == f"/tmp/servepilot-package/{wheel.name}"
    assert mounts == {package: str(wheel)}
    task = render_task(
        LaunchRequest(
            model="m", cloud="gcp", accelerators="B200:8", package=package, file_mounts=mounts
        )
    )
    assert task["envs"]["SERVEPILOT_PACKAGE"] == package and task["file_mounts"] == mounts


def test_installation_without_checkout_uses_source(tmp_path: Path) -> None:
    assert prepare_package(None, tmp_path, source_dir=tmp_path) == (SOURCE_PACKAGE, {})


def test_build_is_cached_by_source_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkout"
    source = root / "src/servepilot"
    source.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'servepilot'\n")
    code = source / "__init__.py"
    code.write_text("VERSION = 1\n")
    (root / "credentials.json").write_text("never upload")
    builds: list[list[str]] = []

    def build(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        builds.append(args)
        output = Path(args[args.index("--wheel-dir") + 1])
        (output / "servepilot-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
        return subprocess.CompletedProcess(args, 0, "built", "")

    monkeypatch.setattr("servepilot.cloud.package.subprocess.run", build)
    first = prepare_package(None, tmp_path / "cache", source_dir=root)
    assert len(builds) == 1 and builds[0][-1] == str(root)
    assert len(first[1]) == 1 and next(iter(first[1].values())).endswith(".whl")
    assert prepare_package(None, tmp_path / "cache", source_dir=root) == first
    assert len(builds) == 1
    code.write_text("VERSION = 2\n")
    updated = prepare_package(None, tmp_path / "cache", source_dir=root)
    assert len(builds) == 2 and updated[1] != first[1]


def test_build_failure_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "src/servepilot").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.setattr(
        "servepilot.cloud.package.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "No module named pip"),
    )
    with pytest.raises(ConfigurationError) as exc:
        prepare_package(None, tmp_path / "cache", source_dir=tmp_path)
    assert "No module named pip" in exc.value.render() and "--package" in exc.value.render()
