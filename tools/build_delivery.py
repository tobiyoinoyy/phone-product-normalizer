#!/usr/bin/env python3
"""Build the two small hand-off bundles for this project.

The model weights, virtual environment, generated outputs, and source photos
are deliberately excluded.  The command writes ZIP files and unpacked
directories to ``../交付包`` by default, so release artifacts do not get mixed
into the source tree that will later be committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import stat
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT.parent / "交付包"


def _project_version() -> str:
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else "0.1.0"


DEVELOPER_FILES = (
    ".gitignore",
    "README.md",
    "DELIVERY.md",
    "开发交付说明.md",
    "设计师使用说明.md",
    "SKILL.md",
    "pyproject.toml",
    "requirements.txt",
    # Keep the macOS launcher with the full source/test bundle as well.  The
    # developer package ships ``tests/test_macos_entry.py``; omitting the
    # launcher would make that otherwise valid package-level smoke test fail
    # even though the source checkout is complete.
    "手机图片一键处理.command",
    "agents/openai.yaml",
    "phone_normalizer",
    "scripts",
    "tests",
    "tools",
)

DESIGNER_FILES = (
    ".gitignore",
    "设计师使用说明.md",
    "requirements.txt",
    "手机图片一键处理.command",
    "scripts/__init__.py",
    "scripts/batch_normalize.py",
    "scripts/geometry_postprocess.py",
    "scripts/segmentation.py",
    "scripts/phone_normalizer_gui.py",
    "scripts/phone_normalizer_macos.applescript",
)


def _copy_entry(source: Path, destination: Path) -> None:
    if source.is_dir():
        for child in sorted(source.iterdir(), key=lambda item: item.name.casefold()):
            # Never carry generated or personal material into a hand-off.
            if (
                child.name in {".venv", "outputs", "inputs", "__pycache__", ".DS_Store"}
                or child.name.endswith(".egg-info")
            ):
                continue
            _copy_entry(child, destination / child.name)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_manifest(entries: tuple[str, ...], staging: Path) -> None:
    for relative in entries:
        source = PROJECT_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"delivery source is missing: {source}")
        _copy_entry(source, staging / relative)


def _zip_directory(directory: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        files = sorted(directory.rglob("*"), key=lambda item: str(item.relative_to(directory)).casefold())
        for path in files:
            relative = path.relative_to(directory)
            # Put the bundle directory itself at the top level of the archive.
            arcname = f"{directory.name}/{relative.as_posix()}"
            if path.is_dir():
                info = zipfile.ZipInfo(arcname.rstrip("/") + "/")
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                handle.writestr(info, b"")
                continue
            info = zipfile.ZipInfo(arcname)
            mode = stat.S_IMODE(path.stat().st_mode)
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, path.read_bytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(destination: Path, *, force: bool = False) -> list[Path]:
    version = _project_version()
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bundles = [
        (f"phone-normalizer-developer-{version}", DEVELOPER_FILES),
        (f"phone-normalizer-designer-macos-{version}", DESIGNER_FILES),
    ]
    archives: list[Path] = []
    with TemporaryDirectory(prefix="phone-normalizer-release-") as temporary:
        temporary_root = Path(temporary)
        for name, entries in bundles:
            staging = temporary_root / name
            _copy_manifest(entries, staging)

            # Keep an unpacked copy alongside the ZIP for teams that prefer to
            # inspect or copy a folder directly.
            unpacked = destination / name
            if unpacked.exists():
                marker = unpacked / ".delivery-bundle-marker"
                if not force and not marker.is_file():
                    raise FileExistsError(
                        f"refusing to replace an unmarked directory: {unpacked}; "
                        "use --force only when it is a previous generated bundle"
                    )
                shutil.rmtree(unpacked)
            shutil.copytree(staging, unpacked)
            (unpacked / ".delivery-bundle-marker").write_text(
                "Generated by tools/build_delivery.py; safe to replace.\n",
                encoding="utf-8",
            )

            archive = destination / f"{name}.zip"
            _zip_directory(staging, archive)
            archives.append(archive)

    checksum_file = destination / "SHA256SUMS.txt"
    lines = [f"{_sha256(path)}  {path.name}" for path in archives]
    checksum_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"输出目录（默认：{DEFAULT_DESTINATION}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许替换没有交付标记的同名目录（仅在确认其为旧交付包时使用）",
    )
    args = parser.parse_args()
    archives = build(args.destination, force=args.force)
    print(f"交付包目录：{Path(args.destination).expanduser().resolve()}")
    for archive in archives:
        print(f"已生成：{archive}")
    print(f"校验文件：{Path(args.destination).expanduser().resolve() / 'SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
