#!/usr/bin/env python3
"""Offline smoke tests for the installable delivery API and CLI."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from phone_normalizer import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_MAX_GEOMETRY_PIXELS,
    DEFAULT_MODEL,
    Normalizer,
    NormalizerConfig,
    collect_image_paths,
    parse_max_geometry_pixels,
    validate_geometry_size,
)
from phone_normalizer.cli import build_parser  # noqa: E402


class FakeBackend:
    name = "fake"
    model_name = "fake-model"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: Image.Image) -> Image.Image:
        self.calls += 1
        source = image.convert("RGB")
        alpha = Image.new("L", source.size, 0)
        # A simple non-support rectangle keeps this test independent of model
        # weights while exercising crop/rotation/metadata plumbing.
        ImageDraw.Draw(alpha).rounded_rectangle(
            (10, 12, source.width - 11, source.height - 13), radius=8, fill=255
        )
        return Image.merge("RGBA", (*source.split(), alpha))


def main() -> None:
    defaults = NormalizerConfig()
    assert DEFAULT_BACKEND == "birefnet"
    assert DEFAULT_MODEL.endswith("BiRefNet_dynamic")
    assert defaults.backend == DEFAULT_BACKEND
    assert defaults.effective_model == DEFAULT_MODEL
    assert defaults.parsed_birefnet_resolution is None
    assert defaults.max_geometry_pixels == DEFAULT_MAX_GEOMETRY_PIXELS
    assert parse_max_geometry_pixels("auto") == DEFAULT_MAX_GEOMETRY_PIXELS
    assert validate_geometry_size((2524, 2524)) == (2524, 2524)

    parser = build_parser()
    args = parser.parse_args(["photos", "--output-dir", "out", "--manifest", "out/manifest.json"])
    assert args.backend == "birefnet"
    assert args.model is None
    assert args.birefnet_resolution == "auto"
    assert args.max_geometry_pixels == "auto"
    assert args.manifest == Path("out/manifest.json")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_dir = root / "source"
        nested = source_dir / "nested"
        nested.mkdir(parents=True)
        generated_dir = source_dir / "custom-results"
        generated_dir.mkdir()
        image = Image.new("RGB", (96, 72), (30, 60, 90))
        first = source_dir / "phone.jpg"
        second = nested / "phone.jpg"
        generated = generated_dir / "old-output.png"
        image.save(first)
        image.save(second)
        image.save(generated)
        (source_dir / "ignore.txt").write_text("not an image", encoding="utf-8")
        assert collect_image_paths([source_dir]) == sorted(
            [first.resolve(), second.resolve(), generated.resolve()],
            key=lambda item: str(item).casefold(),
        )
        # Arbitrary result folder names can be excluded by their actual path;
        # process_many uses this to prevent recursive output ingestion.
        assert collect_image_paths(
            [source_dir], excluded_directories=[generated_dir]
        ) == sorted(
            [first.resolve(), second.resolve()], key=lambda item: str(item).casefold()
        )
        assert collect_image_paths([source_dir], recursive=False) == [first.resolve()]

        fake = FakeBackend()
        normalizer = Normalizer(
            NormalizerConfig(backend="fake", model="fake-model", device="cpu"),
            backend_instance=fake,
        )
        # The wrapper can be used as a context manager and detaches an
        # injected backend without taking ownership of its lifecycle.
        with normalizer as active:
            assert active is normalizer
            assert normalizer._backend is fake
        assert normalizer._backend is None
        # Recreate a wrapper for the batch assertions below.
        normalizer = Normalizer(
            NormalizerConfig(backend="fake", model="fake-model", device="cpu"),
            backend_instance=fake,
        )
        # Put the destination below the input root.  Its pre-existing PNG must
        # not be treated as a third source by the recursive batch scan.
        output_dir = generated_dir
        manifest = output_dir / "manifest.json"
        results = normalizer.process_many(
            [source_dir],
            output_dir=output_dir,
            manifest_path=manifest,
        )
        assert len(results) == 2
        assert all(item.succeeded for item in results)
        assert fake.calls == 2  # one injected backend reused for the batch
        assert results[0].output is not None and results[0].output.exists()
        assert results[1].output is not None and results[1].output.exists()
        assert results[0].output != results[1].output
        assert results[0].metadata is not None and results[0].metadata.exists()
        metadata = json.loads(results[0].metadata.read_text(encoding="utf-8"))
        assert metadata["success"] is True
        assert metadata["backend"] == "fake"
        assert metadata["support_removed"] is False
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_data["success"] is True
        assert len(manifest_data["items"]) == 2

        # An explicitly supplied PNG inside the output folder remains valid,
        # but neither it nor an older processed result may be overwritten.
        inline = output_dir / "inline.png"
        prior = output_dir / "inline-processed.png"
        image.save(inline)
        image.save(prior)
        inline_results = normalizer.process_many(
            [inline],
            output_dir=output_dir,
            save_metadata=False,
        )
        assert len(inline_results) == 1 and inline_results[0].succeeded
        assert inline_results[0].output is not None
        assert inline_results[0].output.name == "inline-processed-2.png"
        assert inline.exists() and prior.exists()

        # Validate manifest paths before inference or writing any artifact.
        calls_before_manifest_checks = fake.calls
        try:
            normalizer.process_many(
                [first],
                output_dir=root / "manifest-input-test",
                manifest_path=first,
            )
        except ValueError as exc:
            assert "input image" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("manifest must not overwrite an input image")

        # The same protection applies to an image below a directory input that
        # is excluded from the source list because it is the output subtree.
        hidden_input = output_dir / "hidden-input.jpg"
        image.save(hidden_input)
        try:
            normalizer.process_many(
                [source_dir],
                output_dir=output_dir,
                manifest_path=hidden_input,
            )
        except ValueError as exc:
            assert "input image" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("manifest must not overwrite an excluded input image")

        manifest_output_dir = root / "manifest-output-test"
        try:
            normalizer.process_many(
                [first],
                output_dir=manifest_output_dir,
                manifest_path=manifest_output_dir / "phone.json",
            )
        except ValueError as exc:
            assert "per-image outputs" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("manifest must not overwrite an item sidecar")
        assert fake.calls == calls_before_manifest_checks

        # The package's safety contract rejects attempts to overwrite input.
        try:
            normalizer.process_file(first, first)
        except ValueError as exc:
            assert "different paths" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("input overwrite must be rejected")

    print("delivery API smoke test passed")


if __name__ == "__main__":
    main()
