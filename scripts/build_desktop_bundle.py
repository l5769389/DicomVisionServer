from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


PYINSTALLER_REQUIREMENT = "pyinstaller>=6.11.0,<7.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DicomVisionServer desktop bundle with PyInstaller.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory that receives the PyInstaller onedir bundle. Defaults to ./dist.",
    )
    parser.add_argument("--bundle-name", default="DicomVisionServer", help="PyInstaller bundle name.")
    parser.add_argument(
        "--target-arch",
        choices=["x86_64", "arm64", "universal2"],
        default=None,
        help="macOS-only PyInstaller target architecture.",
    )
    return parser.parse_args()


def has_module(python_executable: Path, module_name: str) -> bool:
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module_name!r}) else 1)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def find_venv_python(server_root: Path) -> Path | None:
    candidates = [
        server_root / ".venv" / "Scripts" / "python.exe",
        server_root / ".venv" / "bin" / "python",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def resolve_pyinstaller_command(server_root: Path) -> list[str]:
    venv_python = find_venv_python(server_root)
    if venv_python and has_module(venv_python, "PyInstaller"):
        return [str(venv_python), "-m", "PyInstaller"]

    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "run", "--with", PYINSTALLER_REQUIREMENT, "python", "-m", "PyInstaller"]

    if venv_python:
        raise RuntimeError(
            f"PyInstaller is not installed in {venv_python} and uv was not found. "
            f"Install it with: uv pip install '{PYINSTALLER_REQUIREMENT}'"
        )

    current_python = Path(sys.executable)
    if has_module(current_python, "PyInstaller"):
        return [str(current_python), "-m", "PyInstaller"]

    raise RuntimeError(
        "PyInstaller is not available and uv was not found. "
        f"Install uv or install PyInstaller with: python -m pip install '{PYINSTALLER_REQUIREMENT}'"
    )


def resolve_entry_path(server_root: Path) -> Path:
    for candidate in (server_root / "desktop_entry.py", server_root / "run.py"):
        if candidate.exists():
            return candidate
    raise RuntimeError("Server entry not found. Checked desktop_entry.py and run.py.")


def build_bundle(args: argparse.Namespace) -> Path:
    server_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve() if args.output_root else server_root / "dist"
    work_root = server_root / "build" / "pyinstaller"
    bundle_root = output_root / args.bundle_name
    executable_name = f"{args.bundle_name}.exe" if sys.platform == "win32" else args.bundle_name
    bundle_executable = bundle_root / executable_name

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if bundle_root.exists():
        shutil.rmtree(bundle_root)

    entry_path = resolve_entry_path(server_root)
    pyinstaller_args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        args.bundle_name,
        "--distpath",
        str(output_root),
        "--workpath",
        str(work_root),
        "--specpath",
        str(work_root),
        "--hidden-import",
        "scipy",
        "--hidden-import",
        "scipy.ndimage",
        "--hidden-import",
        "scipy.ndimage._nd_image",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "scipy.conftest",
        "--exclude-module",
        "scipy.tests",
        "--hidden-import",
        "vtkmodules.util.numpy_support",
        "--hidden-import",
        "vtkmodules.util.vtkConstants",
        "--hidden-import",
        "vtkmodules.vtkCommonCore",
        "--hidden-import",
        "vtkmodules.vtkCommonDataModel",
        "--hidden-import",
        "vtkmodules.vtkRenderingCore",
        "--hidden-import",
        "vtkmodules.vtkRenderingOpenGL2",
        "--hidden-import",
        "vtkmodules.vtkRenderingVolumeOpenGL2",
        "--collect-binaries",
        "vtk",
        "--collect-data",
        "vtk",
    ]

    if args.target_arch:
        if sys.platform != "darwin":
            raise RuntimeError("--target-arch is only supported on macOS.")
        pyinstaller_args.extend(["--target-arch", args.target_arch])

    pyinstaller_args.append(str(entry_path))

    command = resolve_pyinstaller_command(server_root)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    result = subprocess.run(command + pyinstaller_args, cwd=server_root, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"PyInstaller failed with exit code {result.returncode}")

    if not bundle_executable.exists():
        raise RuntimeError(f"Bundle build completed without expected executable: {bundle_executable}")

    if sys.platform != "win32":
        bundle_executable.chmod(bundle_executable.stat().st_mode | 0o755)

    return bundle_root


def main() -> int:
    try:
        bundle_root = build_bundle(parse_args())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Desktop bundle built at: {bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
