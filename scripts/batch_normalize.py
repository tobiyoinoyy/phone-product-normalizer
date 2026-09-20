#!/usr/bin/env python3
"""批量处理手机商品图的设计师友好入口。

这个入口把多个图片交给 ``segmentation.py`` 的正式流程处理，默认使用
``ZhengPeng7/BiRefNet_dynamic``，并把透明 PNG 和可选的 JSON 记录放到一个
独立的结果文件夹。

可以直接在终端运行：

    python scripts/batch_normalize.py photo.jpg another.jpg

也可以把图片或文件夹拖到项目根目录的 ``手机图片一键处理.command`` 上。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

try:
    from .segmentation import (
        DEFAULT_MAX_GEOMETRY_PIXELS,
        DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
        DEFAULT_BIREFNET_MAX_SIDE,
        NormalizationInfo,
        create_segmentation_backend,
        process_file,
        parse_birefnet_max_side,
        parse_max_geometry_pixels,
        _clear_exception_traceback,
        _release_torch_memory,
    )
except ImportError:  # direct ``python scripts/batch_normalize.py`` execution
    from segmentation import (  # type: ignore
        DEFAULT_MAX_GEOMETRY_PIXELS,
        DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
        DEFAULT_BIREFNET_MAX_SIDE,
        NormalizationInfo,
        create_segmentation_backend,
        process_file,
        parse_birefnet_max_side,
        parse_max_geometry_pixels,
        _clear_exception_traceback,
        _release_torch_memory,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "结果"
DEFAULT_MODEL = "ZhengPeng7/BiRefNet_dynamic"
DEFAULT_BACKEND = "birefnet"
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        # The project root contains many historical PNGs under ``outputs``.
        # If a designer drags the root folder itself, re-ingesting those
        # artifacts would waste time and can recreate the original memory
        # spike.  Explicitly supplied files are still accepted by
        # ``collect_image_paths``.
        "outputs",
        "结果",
    }
)


@dataclass(frozen=True)
class BatchResult:
    """One source file's result, suitable for the GUI and CLI summary."""

    source: Path
    output: Optional[Path] = None
    metadata: Optional[Path] = None
    info: Optional[NormalizationInfo] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.output is not None


ProgressCallback = Callable[[int, int, Path], None]
# ``progress`` predates the designer GUI and intentionally remains a small,
# three-argument callback.  ``status`` is an optional richer channel for a
# front-end: it reports model loading, the start/end of each item, and the
# result object as soon as it is available.  Keeping this as a second callback
# means existing scripts and integrations do not need to change their
# callback signature.
StatusCallback = Callable[
    [str, int, int, Optional[Path], Optional[BatchResult]],
    None,
]
StopCallback = Callable[[], bool]

# Lifecycle values sent through ``status``.  The callback is deliberately
# string-based for backwards compatibility with the first GUI prototype, but
# keeping the contract here makes it clear which values a front-end should
# handle.  ``cancelled`` is emitted only after the current image (if any) has
# returned; inference is not force-killed in the middle of a model call.
STATUS_EVENTS = frozenset(
    {
        "scanning",
        "queued",
        "loading",
        "load_failed",
        "ready",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "finished",
    }
)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _path_is_within(path: Path, directory: Path) -> bool:
    """Return whether ``path`` is inside ``directory`` safely."""

    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
    return True


def _iter_directory(directory: Path, recursive: bool) -> Iterable[Path]:
    """Yield image files while avoiding generated results and environments."""

    if recursive:
        # ``os.walk`` lets us prune directories before descending into them.
        for root, dirnames, filenames in os.walk(directory):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in EXCLUDED_DIRECTORY_NAMES and not name.startswith(".")
            )
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                path = Path(root) / name
                if _is_image(path):
                    yield path
    else:
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if not path.name.startswith(".") and _is_image(path):
                yield path


def collect_image_paths(
    inputs: Sequence[Path],
    *,
    recursive: bool = True,
) -> List[Path]:
    """Expand files/folders into a stable, de-duplicated image list.

    Missing paths and non-image files are ignored here so a drag operation can
    include a mixture of files.  The CLI reports a useful error when no image
    remains; the GUI can show the same message without aborting the app.
    """

    found: List[Path] = []
    seen = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        candidates: Iterable[Path]
        if resolved.is_dir():
            candidates = _iter_directory(resolved, recursive)
        elif _is_image(resolved):
            candidates = (resolved,)
        else:
            candidates = ()
        for candidate in candidates:
            try:
                normalized = candidate.resolve()
            except OSError:
                continue
            key = str(normalized)
            if key not in seen:
                seen.add(key)
                found.append(normalized)
    return sorted(found, key=lambda item: str(item).lower())


def _output_paths(
    sources: Sequence[Path],
    output_dir: Path,
) -> List[Tuple[Path, Path]]:
    """Allocate non-conflicting PNG/JSON names for a batch.

    The first file normally keeps its original stem.  Duplicate stems (or
    names already present in the result folder) receive ``-2``, ``-3`` and so
    on.  Existing source paths are reserved too, so a batch containing both
    ``phone.jpg`` and ``phone.png`` can never overwrite the PNG source while
    creating the JPG result.
    """

    allocated = {}
    reserved = set()
    for source in sources:
        try:
            reserved.add(str(source.resolve()).casefold())
        except OSError:
            reserved.add(str(source).casefold())
    result: List[Tuple[Path, Path]] = []
    for source in sources:
        stem = source.stem or "image"
        key = stem.casefold()
        count = int(allocated.get(key, 0))
        while True:
            count += 1
            suffix = "" if count == 1 else f"-{count}"
            output = output_dir / f"{stem}{suffix}.png"
            metadata = output_dir / f"{stem}{suffix}.json"
            try:
                output_key = str(output.resolve()).casefold()
                metadata_key = str(metadata.resolve()).casefold()
            except OSError:
                output_key = str(output).casefold()
                metadata_key = str(metadata).casefold()
            if (
                output_key not in reserved
                and metadata_key not in reserved
                and not output.exists()
                and not metadata.exists()
            ):
                break
        allocated[key] = count
        reserved.update({output_key, metadata_key})
        result.append((output, metadata))
    return result


def process_files(
    inputs: Sequence[Path],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    recursive: bool = True,
    save_metadata: bool = True,
    progress: Optional[ProgressCallback] = None,
    status: Optional[StatusCallback] = None,
    should_stop: Optional[StopCallback] = None,
    birefnet_max_side: Any = DEFAULT_BIREFNET_MAX_SIDE,
    birefnet_cpu_fallback_max_side: Any = DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
    max_geometry_pixels: Any = DEFAULT_MAX_GEOMETRY_PIXELS,
) -> List[BatchResult]:
    """Process all images and continue after an individual file error.

    The segmentation backend is constructed once, so a batch does not reload
    the (large) model for every image.  A failed source is represented by a
    ``BatchResult(error=...)`` and does not prevent the remaining files from
    being attempted.  When supplied, ``status`` receives lifecycle events from
    :data:`STATUS_EVENTS`; ``should_stop`` is polled before each image and
    after model loading.  A stop request is cooperative: the image currently
    inside ``process_file`` is allowed to finish, then ``cancelled`` and
    ``finished`` are emitted and no later image is started.
    """

    def emit_status(
        event: str,
        index: int,
        total: int,
        source: Optional[Path] = None,
        result: Optional[BatchResult] = None,
    ) -> None:
        # A UI callback must never be able to interrupt image processing.  In
        # particular, a window can be closed while its worker is unwinding.
        if status is not None:
            try:
                status(event, index, total, source, result)
            except Exception:
                pass

    def stop_requested() -> bool:
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:
            return False

    birefnet_max_side = parse_birefnet_max_side(birefnet_max_side)
    birefnet_cpu_fallback_max_side = parse_birefnet_max_side(
        birefnet_cpu_fallback_max_side
    )
    max_geometry_pixels = parse_max_geometry_pixels(max_geometry_pixels)

    if status is not None:
        # The total is not known until directory inputs have been expanded.
        # This early event lets a GUI show that the request was received rather
        # than appearing frozen while a large folder is scanned.
        emit_status("scanning", 0, 0)

    destination = Path(output_dir).expanduser().resolve()
    # A directory scan must exclude a result subtree, otherwise a rerun can
    # feed yesterday's generated PNGs back into the model.  An explicitly
    # supplied image remains valid even when it lives in that subtree (or in
    # the destination itself); ``_output_paths`` will allocate a non-conflicting
    # suffix and never overwrite the source.  Keep the distinction here rather
    # than changing ``collect_image_paths``'s long-standing return contract.
    explicit_keys: set[str] = set()
    for raw_path in inputs:
        raw = Path(raw_path).expanduser()
        try:
            resolved = raw.resolve()
        except OSError:
            continue
        if _is_image(resolved):
            explicit_keys.add(str(resolved).casefold())
    sources = []
    for source in collect_image_paths(inputs, recursive=recursive):
        source_key = str(source).casefold()
        if source_key in explicit_keys or not _path_is_within(source, destination):
            sources.append(source)
    if not sources:
        raise ValueError("没有找到可处理的图片（支持 JPG、PNG、WEBP、TIFF、BMP）。")

    total = len(sources)
    emit_status("queued", 0, total)

    if stop_requested():
        emit_status("cancelled", 0, total)
        emit_status("finished", 0, total)
        return []

    destination.mkdir(parents=True, exist_ok=True)
    allocated_paths = _output_paths(sources, destination)

    # Loading is intentionally outside the loop: this is both faster and less
    # memory-fragmenting on MPS/CUDA.  The BiRefNet backend exposes its model
    # through a lazy property, so touch that property here rather than during
    # the first ``processing`` event.  This keeps the GUI's loading/ready state
    # truthful and surfaces dependency/download errors before any output file
    # is attempted.
    emit_status("loading", 0, total)
    backend = None
    try:
        backend = create_segmentation_backend(
            DEFAULT_BACKEND,
            model_name=model_name,
            device=device,
            birefnet_resolution=None,
            birefnet_max_side=birefnet_max_side,
            birefnet_cpu_fallback_max_side=birefnet_cpu_fallback_max_side,
        )
        loaded_model = getattr(backend, "model", None)
        # The local reference is not needed after construction; dropping it
        # prevents a model returned by a custom backend from being retained in
        # this frame in addition to the backend's own reference.
        del loaded_model
        _release_torch_memory(str(getattr(backend, "device", device)))
    except Exception as exc:
        # A failed model load can leave an allocator reservation behind (most
        # notably after a partially downloaded/initialised MPS module).  Trim
        # it before propagating the error to the GUI.
        loaded_device = str(getattr(backend, "device", device))
        # Do not let the traceback retain the backend/model through this
        # frame while the GUI waits to display the error.
        backend = None
        _clear_exception_traceback(exc)
        _release_torch_memory(loaded_device)
        emit_status("load_failed", 0, total)
        raise
    emit_status("ready", 0, total)

    if stop_requested():
        loaded_device = str(getattr(backend, "device", device))
        close = getattr(backend, "close", None) if backend is not None else None
        if callable(close):
            try:
                close()
            except Exception:
                pass
        del close
        backend = None
        _release_torch_memory(loaded_device)
        emit_status("cancelled", 0, total)
        emit_status("finished", 0, total)
        return []

    results: List[BatchResult] = []
    stopped = False
    try:
        for index, (source, paths) in enumerate(zip(sources, allocated_paths), start=1):
            if stop_requested():
                stopped = True
                break
            if progress is not None:
                progress(index, total, source)
            emit_status("processing", index, total, source)
            output, metadata = paths
            try:
                info = process_file(
                    source,
                    output,
                    backend=DEFAULT_BACKEND,
                    model_name=model_name,
                    device=device,
                    birefnet_resolution=None,
                    birefnet_max_side=birefnet_max_side,
                    birefnet_cpu_fallback_max_side=birefnet_cpu_fallback_max_side,
                    max_geometry_pixels=max_geometry_pixels,
                    metadata_path=(metadata if save_metadata else None),
                    backend_instance=backend,
                )
                result = BatchResult(
                    source=source,
                    output=output,
                    metadata=(metadata if save_metadata else None),
                    info=info,
                )
                results.append(result)
                emit_status("completed", index, total, source, result)
            except Exception as exc:  # keep a bad photo from stopping a batch
                error_text = f"{type(exc).__name__}: {exc}"
                loaded_device = str(getattr(backend, "device", device))
                # A failed geometry/model call may leave large temporary
                # arrays in traceback frames.  Detach them before retaining
                # only the short error string in the result list.
                _clear_exception_traceback(exc)
                _release_torch_memory(loaded_device)
                result = BatchResult(source=source, error=error_text)
                results.append(result)
                emit_status("failed", index, total, source, result)
    finally:
        # Also release the backend if a progress/status callback or an
        # unexpected BaseException interrupts the loop.  Without this finally
        # a traceback held by the caller could retain the full model and its
        # allocator reservation.
        loaded_device = str(getattr(backend, "device", device))
        close = getattr(backend, "close", None) if backend is not None else None
        if callable(close):
            try:
                close()
            except Exception:
                pass
        del close
        backend = None
        _release_torch_memory(loaded_device)
    if stopped:
        # A stop request is intentionally reported separately from a model or
        # per-file error, so the front-end can say exactly what happened.
        emit_status("cancelled", len(results), total)
    # ``process_files`` owns the backend it creates.  Release that ownership
    # before notifying the front-end that the batch is finished; otherwise a
    # completed GUI worker can keep the full BiRefNet module and its allocator
    # reservation alive until a later garbage-collection cycle.
    emit_status("finished", len(results), total)
    return results


def _format_result_line(result: BatchResult) -> str:
    if not result.succeeded:
        return f"失败  {result.source.name}  —  {result.error}"
    info = result.info
    if info is None:
        return f"完成  {result.source.name}  →  {result.output}"
    return (
        f"完成  {result.source.name}  →  {result.output.name}"
        f"（{info.output_size[0]}×{info.output_size[1]}，"
        f"旋转 {info.rotation_degrees:.2f}°）"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量抠图、去支架、摆正并输出透明 PNG（默认 BiRefNet_dynamic）。"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="图片或文件夹路径；可一次拖入多个项目",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"结果文件夹（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="推理设备（默认 auto）",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="拖入文件夹时只处理该文件夹第一层",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="不写入同名 JSON 处理记录，只输出 PNG",
    )
    parser.add_argument(
        "--birefnet-max-side",
        default=str(DEFAULT_BIREFNET_MAX_SIDE),
        metavar="PIXELS",
        help=f"dynamic BiRefNet 最长边上限（默认 {DEFAULT_BIREFNET_MAX_SIDE}；可用 none 关闭）",
    )
    parser.add_argument(
        "--birefnet-cpu-fallback-max-side",
        default=str(DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE),
        metavar="PIXELS",
        help=f"MPS 内存不足切 CPU 时的最长边上限（默认 {DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE}）",
    )
    parser.add_argument(
        "--max-geometry-pixels",
        default=str(DEFAULT_MAX_GEOMETRY_PIXELS),
        metavar="PIXELS",
        help=f"几何后处理像素上限（默认 {DEFAULT_MAX_GEOMETRY_PIXELS}；可用 none 关闭）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"正在处理 {len(args.inputs)} 个拖入项目…")
    print(f"模型：{DEFAULT_MODEL}    设备：{args.device}")
    print(f"结果：{Path(args.output_dir).expanduser().resolve()}")
    try:
        max_side = parse_birefnet_max_side(args.birefnet_max_side)
        cpu_fallback_max_side = parse_birefnet_max_side(args.birefnet_cpu_fallback_max_side)
        max_geometry_pixels = parse_max_geometry_pixels(args.max_geometry_pixels)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    print(
        f"内存保护：最长边 ≤ {max_side or '不限制'} px；"
        f"CPU 回退 ≤ {cpu_fallback_max_side or '不限制'} px"
        f"；几何像素 ≤ {max_geometry_pixels or '不限制'}"
    )

    def report(index: int, total: int, source: Path) -> None:
        print(f"[{index}/{total}] {source}", flush=True)

    try:
        results = process_files(
            args.inputs,
            output_dir=args.output_dir,
            model_name=DEFAULT_MODEL,
            device=args.device,
            recursive=not args.no_recursive,
            save_metadata=not args.no_metadata,
            progress=report,
            birefnet_max_side=max_side,
            birefnet_cpu_fallback_max_side=cpu_fallback_max_side,
            max_geometry_pixels=max_geometry_pixels,
        )
    except Exception as exc:
        print(f"\n无法开始处理：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "请确认已完成首次安装，并保持网络可用以下载 BiRefNet 权重。"
            "可双击“手机图片一键处理.command”重新安装。",
            file=sys.stderr,
        )
        return 2

    succeeded = sum(item.succeeded for item in results)
    failed = len(results) - succeeded
    print("\n处理结果：")
    for item in results:
        print(_format_result_line(item))
    print(f"\n完成 {succeeded} 张，失败 {failed} 张。")
    print(f"透明 PNG 位于：{Path(args.output_dir).expanduser().resolve()}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
