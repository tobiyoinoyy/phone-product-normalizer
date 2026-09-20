#!/usr/bin/env python3
"""Offline checks for the designer batch entry point."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.batch_normalize import (
    DEFAULT_MAX_GEOMETRY_PIXELS,
    DEFAULT_MODEL,
    STATUS_EVENTS,
    _build_parser,
    process_files,
    _output_paths,
    collect_image_paths,
)
from scripts.normalize_ben2 import NormalizationInfo


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        nested = root / "nested"
        nested.mkdir()
        first = root / "phone.jpg"
        second = nested / "phone.jpg"
        ignored = root / "notes.txt"
        generated = root / "结果"
        generated.mkdir()
        generated_image = generated / "old.png"
        generated_root = root / "outputs"
        generated_root.mkdir()
        Image.new("RGB", (8, 8), "orange").save(generated_root / "old-output.png")

        Image.new("RGB", (8, 8), "white").save(first)
        Image.new("RGB", (8, 8), "black").save(second)
        Image.new("RGB", (8, 8), "red").save(generated_image)
        ignored.write_text("not an image", encoding="utf-8")

        found = collect_image_paths([root])
        assert found == sorted([first.resolve(), second.resolve()], key=lambda item: str(item).lower())
        assert collect_image_paths([root], recursive=False) == [first.resolve()]

        outputs = _output_paths(found, root / "out")
        assert [output.name for output, _ in outputs] == ["phone.png", "phone-2.png"]

        output_dir = root / "out"
        output_dir.mkdir()
        Image.new("RGBA", (8, 8), "blue").save(output_dir / "phone.png")
        collision_safe = _output_paths([first.resolve()], output_dir)
        assert collision_safe[0][0].name == "phone-2.png"

        png_source = root / "original.png"
        Image.new("RGBA", (8, 8), "green").save(png_source)
        same_folder = _output_paths([png_source.resolve()], root)
        assert same_folder[0][0].name == "original-2.png"
        assert DEFAULT_MODEL == "ZhengPeng7/BiRefNet_dynamic"
        parser_args = _build_parser().parse_args(
            [
                "image.jpg",
                "--birefnet-max-side",
                "768",
                "--birefnet-cpu-fallback-max-side",
                "512",
                "--max-geometry-pixels",
                "1000000",
            ]
        )
        assert parser_args.birefnet_max_side == "768"
        assert parser_args.birefnet_cpu_fallback_max_side == "512"
        assert parser_args.max_geometry_pixels == "1000000"
        default_parser_args = _build_parser().parse_args(["image.jpg"])
        assert default_parser_args.max_geometry_pixels == str(DEFAULT_MAX_GEOMETRY_PIXELS)

        # The designer batch implementation also guards a custom result
        # directory nested below a recursively scanned source folder.
        # (The actual inference path is covered by the package API smoke test;
        # this assertion documents the safe path predicate used by the wrapper.)
        from scripts.batch_normalize import _path_is_within

        assert _path_is_within(generated_image, generated)
        assert not _path_is_within(first, generated)

        # Explicit file inputs remain valid when the chosen result directory
        # is the same directory.  Directory discovery still excludes that
        # subtree, so generated PNGs cannot be fed back into a recursive run.
        explicit_output = root / "explicit-output"
        explicit_output.mkdir()
        explicit_source = explicit_output / "already.png"
        Image.new("RGB", (8, 8), "purple").save(explicit_source)
        fake_info = NormalizationInfo(
            support_removed=False,
            support_confidence=0.0,
            rotation_degrees=0.0,
            crop_box=(0, 0, 8, 8),
            input_size=(8, 8),
            output_size=(8, 8),
            backend="fake",
            model="fake-model",
            device="cpu",
        )

        def fake_backend_factory(*args, **kwargs):
            return object()

        def fake_process_file(source_path, output_path, **kwargs):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.open(source_path).convert("RGBA").save(output_path)
            return fake_info

        with patch("scripts.batch_normalize.create_segmentation_backend", fake_backend_factory), patch(
            "scripts.batch_normalize.process_file", fake_process_file
        ):
            explicit_results = process_files(
                [explicit_source],
                output_dir=explicit_output,
                save_metadata=False,
            )
        assert len(explicit_results) == 1 and explicit_results[0].succeeded
        assert explicit_results[0].output is not None
        assert explicit_results[0].output.name == "already-2.png"
        assert explicit_source.exists()

        # The richer lifecycle callback has a deterministic order and a stop
        # request takes effect between images.  This is deliberately offline:
        # patching the model factory/file operation verifies GUI-facing state
        # and cancellation without downloading weights or allocating MPS
        # tensors.
        batch_root = root / "batch-input"
        batch_root.mkdir()
        batch_sources = []
        for number in range(3):
            path = batch_root / f"image-{number}.jpg"
            Image.new("RGB", (8, 8), (number * 40, 20, 80)).save(path)
            batch_sources.append(path)
        lifecycle = []
        processed = {"count": 0}

        def record_status(event, index, total, source, result):
            assert event in STATUS_EVENTS
            lifecycle.append(event)

        def stop_after_one():
            return processed["count"] >= 1

        def fake_process_then_stop(source_path, output_path, **kwargs):
            processed["count"] += 1
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.open(source_path).convert("RGBA").save(output_path)
            return fake_info

        with patch("scripts.batch_normalize.create_segmentation_backend", fake_backend_factory), patch(
            "scripts.batch_normalize.process_file", fake_process_then_stop
        ):
            stopped_results = process_files(
                batch_sources,
                output_dir=root / "batch-output",
                save_metadata=False,
                status=record_status,
                should_stop=stop_after_one,
            )
        assert len(stopped_results) == 1
        assert stopped_results[0].succeeded
        assert lifecycle == [
            "scanning",
            "queued",
            "loading",
            "ready",
            "processing",
            "completed",
            "cancelled",
            "finished",
        ]

        # Model loading is deliberately surfaced before ``ready``.  A lazy
        # backend property must be touched during the loading phase rather
        # than blocking the first ``processing`` event while downloading.
        preload_root = root / "preload-input"
        preload_root.mkdir()
        preload_source = preload_root / "one.jpg"
        Image.new("RGB", (8, 8), "white").save(preload_source)
        preload_events = []

        class PreloadBackend:
            device = "cpu"

            @property
            def model(self):
                preload_events.append("model")
                return object()

        def preload_factory(*args, **kwargs):
            preload_events.append("factory")
            return PreloadBackend()

        def preload_process(source_path, output_path, **kwargs):
            preload_events.append("process")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.open(source_path).convert("RGBA").save(output_path)
            return fake_info

        with patch("scripts.batch_normalize.create_segmentation_backend", preload_factory), patch(
            "scripts.batch_normalize.process_file", preload_process
        ):
            preload_results = process_files(
                [preload_source],
                output_dir=root / "preload-output",
                save_metadata=False,
            )
        assert preload_results[0].succeeded
        assert preload_events == ["factory", "model", "process"]

    print("designer batch entry smoke test passed")


if __name__ == "__main__":
    main()
