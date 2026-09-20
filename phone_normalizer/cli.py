"""Command-line entry point for the installable phone normalizer package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .api import DEFAULT_BACKEND, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, Normalizer, NormalizerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phone-normalizer",
        description=(
            "批量抠图、去支架、摆正并输出透明 PNG；仅使用 "
            "BiRefNet_dynamic。"
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="图片或文件夹路径，可一次传入多个")
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
        "--birefnet-resolution",
        default="auto",
        metavar="RESOLUTION",
        help="BiRefNet 输入尺寸，例如 1024、1024x1024；dynamic 默认 auto",
    )
    parser.add_argument(
        "--birefnet-max-side",
        default="auto",
        metavar="PIXELS",
        help="dynamic 模型最长边安全上限（默认 1024；可用 none 关闭）",
    )
    parser.add_argument(
        "--birefnet-cpu-fallback-max-side",
        default="1024",
        metavar="PIXELS",
        help="MPS 内存不足切到 CPU 时的最长边上限（默认 1024）",
    )
    parser.add_argument(
        "--max-geometry-pixels",
        default="auto",
        metavar="PIXELS",
        help=(
            "几何后处理的源图像像素上限（默认 64MP；可用 none 关闭，"
            "仅超大图触发）"
        ),
    )
    parser.add_argument(
        "--no-memory-cleanup",
        action="store_true",
        help="不在每张图片后清理 Python/加速器缓存（不建议）",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="MPS 内存不足时不自动切换 CPU",
    )
    parser.add_argument("--geometry-threshold", type=int, default=32)
    parser.add_argument("--crop-threshold", type=int, default=8)
    parser.add_argument("--max-deskew-degrees", type=float, default=45.0)
    parser.add_argument(
        "--confidence-floor",
        "--support-min-confidence",
        dest="confidence_floor",
        type=float,
        default=0.55,
        help="自动去支架的最低置信度（默认 0.55）",
    )
    parser.add_argument("--require-support-border", action="store_true")
    parser.add_argument("--no-support-removal", action="store_true")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-tight-crop", action="store_true")
    parser.add_argument("--no-recursive", action="store_true", help="文件夹只处理第一层")
    parser.add_argument("--no-metadata", action="store_true", help="不写每张图片的 JSON 记录")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="可选的批处理汇总 JSON 路径",
    )
    parser.add_argument("--fail-fast", action="store_true", help="遇到一张失败图片立即停止")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _config_from_args(args: argparse.Namespace) -> NormalizerConfig:
    return NormalizerConfig(
        device=args.device,
        birefnet_resolution=args.birefnet_resolution,
        geometry_threshold=args.geometry_threshold,
        crop_threshold=args.crop_threshold,
        max_deskew_degrees=args.max_deskew_degrees,
        confidence_floor=args.confidence_floor,
        remove_support=not args.no_support_removal,
        deskew=not args.no_deskew,
        tight_crop=not args.no_tight_crop,
        require_support_border=args.require_support_border,
        birefnet_max_side=args.birefnet_max_side,
        birefnet_cpu_fallback_max_side=args.birefnet_cpu_fallback_max_side,
        max_geometry_pixels=args.max_geometry_pixels,
        birefnet_memory_cleanup=not args.no_memory_cleanup,
        birefnet_fallback_to_cpu=not args.no_cpu_fallback,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    normalizer: Optional[Normalizer] = None
    try:
        config = _config_from_args(args)
        normalizer = Normalizer(config)
        print(f"模型：{config.effective_model}    设备：{config.device}")
        print(f"结果：{Path(args.output_dir).expanduser().resolve()}")
        if config.backend == "birefnet":
            max_side = config.parsed_birefnet_max_side
            fallback_side = config.parsed_birefnet_cpu_fallback_max_side
            print(
                "内存保护：动态最长边 ≤ "
                f"{max_side or '不限制'} px；CPU 回退 ≤ "
                f"{fallback_side or '不限制'} px；逐图清理="
                f"{'开' if config.birefnet_memory_cleanup else '关'}；"
                f"几何像素 ≤ {config.max_geometry_pixels or '不限制'}"
            )

        def progress(index: int, total: int, source: Path) -> None:
            print(f"[{index}/{total}] {source}", flush=True)

        results = normalizer.process_many(
            args.inputs,
            output_dir=args.output_dir,
            recursive=not args.no_recursive,
            save_metadata=not args.no_metadata,
            manifest_path=args.manifest,
            progress=progress,
            continue_on_error=not args.fail_fast,
        )
    except Exception as exc:
        print(f"无法开始处理：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        # ``main`` can be called repeatedly by an embedding process (and a
        # failed batch can return before the normalizer's caller gets a chance
        # to clean it up).  Release the cached model and MPS/CUDA allocator in
        # every path; a one-shot CLI process would eventually reclaim it, but
        # a GUI/plugin host must not retain a multi-gigabyte backend between
        # invocations.
        if normalizer is not None:
            normalizer.close()

    succeeded = sum(item.succeeded for item in results)
    failed = len(results) - succeeded
    for item in results:
        if item.succeeded:
            info = item.info
            detail = ""
            if info is not None:
                detail = f"（{info.output_size[0]}×{info.output_size[1]}，旋转 {info.rotation_degrees:.2f}°）"
            print(f"完成  {item.source.name} → {item.output}{detail}")
        else:
            print(f"失败  {item.source.name} — {item.error}", file=sys.stderr)
    print(f"完成 {succeeded} 张，失败 {failed} 张。")
    return 0 if failed == 0 else 1


__all__ = ["build_parser", "main"]
