#!/usr/bin/env python3
"""Offline tests for the selectable segmentation entry point and backends."""

from __future__ import annotations

import io
import gc
import sys
import tempfile
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.normalize_ben2 import (  # noqa: E402
    DEFAULT_MAX_GEOMETRY_PIXELS,
    DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
    DEFAULT_BIREFNET_MAX_SIDE,
    BiRefNetBackend,
    Ben2Backend,
    DEFAULT_BIREFNET_MODEL,
    RembgBackend,
    SegmentationBackend,
    _build_parser,
    available_backends,
    create_segmentation_backend,
    normalize_rgba,
    infer_ben2_alpha,
    normalize_image,
    process_file,
    parse_birefnet_resolution,
    parse_birefnet_max_side,
    _compatible_mps_low_watermark,
)


def synthetic_phone_with_stand() -> tuple[Image.Image, np.ndarray]:
    """Create a phone-like RGB image and a continuous-alpha stand mask."""

    height, width = 500, 400
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (80, 50), (320, 300), 255, -1)
    for center in ((100, 70), (300, 70), (100, 280), (300, 280)):
        cv2.circle(mask, center, 20, 255, -1)
    # The support reaches the image boundary, as in the supplied photos.
    cv2.rectangle(mask, (165, 300), (235, height - 1), 255, -1)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.2)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 45
    rgb[..., 1] = 70
    rgb[..., 2] = 120
    return Image.fromarray(rgb), alpha


class FakeBackend(SegmentationBackend):
    name = "fake"

    def __init__(self, alpha: np.ndarray):
        super().__init__(model_name="fake-model", device="cpu")
        self.alpha = alpha
        self.calls = 0

    def segment(self, image: Image.Image) -> Image.Image:
        self.calls += 1
        return Image.merge("RGBA", (*image.convert("RGB").split(), Image.fromarray(self.alpha)))


class FakeBen2Model:
    def __init__(self, alpha: np.ndarray):
        self.alpha = alpha
        self.refine_values: list[bool] = []

    def inference(self, image: Image.Image, refine_foreground: bool = True) -> Image.Image:
        self.refine_values.append(refine_foreground)
        return Image.merge("RGBA", (*image.convert("RGB").split(), Image.fromarray(self.alpha)))


class FakeBiRefNetModel:
    """Small logits-only model matching the official BiRefNet eval contract."""

    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self.calls = 0
        self.last_shape = None
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, inputs):
        import torch

        self.calls += 1
        self.last_shape = tuple(inputs.shape)
        height, width = inputs.shape[-2:]
        return [torch.full((1, 1, height, width), self.value, device=inputs.device)]


def test_backend_registry_and_default() -> None:
    assert "ben2" in available_backends()
    assert "rembg" in available_backends()
    backend = create_segmentation_backend("ben2", model_name="local/test", device="cpu")
    assert isinstance(backend, Ben2Backend)
    assert backend.model_name == "local/test"
    assert backend.device == "cpu"
    try:
        create_segmentation_backend("not-a-real-backend")
    except ValueError as exc:
        assert "unknown segmentation backend" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unknown backends must fail explicitly")

    class FutureBackend(SegmentationBackend):
        name = "future-test"

        def __init__(self, device="auto", **kwargs):
            super().__init__(model_name="future-default", device=device)

    from scripts.normalize_ben2 import register_backend

    register_backend("future-test", FutureBackend)
    future = create_segmentation_backend("future-test", device="cpu")
    assert future.model_name == "future-default"

    birefnet = create_segmentation_backend(
        "birefnet",
        model_name="ZhengPeng7/BiRefNet_dynamic",
        device="cpu",
        birefnet_resolution=(512, 768),
    )
    assert isinstance(birefnet, BiRefNetBackend)
    assert birefnet.model_name.endswith("BiRefNet_dynamic")
    assert birefnet.resolution == (512, 768)
    assert not birefnet.dynamic_resolution

    hr = create_segmentation_backend(
        "birefnet", model_name="ZhengPeng7/BiRefNet_HR", device="cpu"
    )
    assert isinstance(hr, BiRefNetBackend)
    assert hr.resolution == (2048, 2048)
    lite = create_segmentation_backend(
        "birefnet", model_name="ZhengPeng7/BiRefNet_lite-2K", device="cpu"
    )
    assert isinstance(lite, BiRefNetBackend)
    assert lite.resolution == (1440, 2560)


def test_ben2_backend_uses_alpha_only() -> None:
    source, alpha = synthetic_phone_with_stand()
    fake_model = FakeBen2Model(alpha)
    backend = Ben2Backend(model=fake_model, device="cpu")
    result = backend.segment(source)
    assert result.mode == "RGBA"
    assert result.size == source.size
    assert fake_model.refine_values == [False]
    assert np.array_equal(np.asarray(result.getchannel("A")), alpha)
    assert np.array_equal(np.asarray(result.convert("RGB")), np.asarray(source))
    assert np.array_equal(infer_ben2_alpha(fake_model, source), alpha)


def test_birefnet_backend_decodes_logits_and_preserves_rgb() -> None:
    source = Image.new("RGB", (73, 51), (31, 67, 109))
    fake_model = FakeBiRefNetModel(value=0.0)
    backend = BiRefNetBackend(
        model_name=DEFAULT_BIREFNET_MODEL,
        model=fake_model,
        device="cpu",
        resolution=(64, 64),
    )
    result = backend.segment(source)
    assert result.mode == "RGBA"
    assert result.size == source.size
    assert fake_model.calls == 1
    assert fake_model.last_shape == (1, 3, 64, 64)
    assert fake_model.eval_calls >= 1
    assert np.array_equal(np.asarray(result.convert("RGB")), np.asarray(source))
    # sigmoid(0) is 0.5; PIL's uint8 conversion may round a few interpolated
    # pixels down by one, so both neighboring values are valid.
    alpha = np.asarray(result.getchannel("A"))
    assert set(np.unique(alpha)).issubset({127, 128})


def test_birefnet_backend_rejects_ambiguous_channels() -> None:
    import torch

    class BadModel(FakeBiRefNetModel):
        def __call__(self, inputs):
            return [torch.zeros((1, 2, inputs.shape[-2], inputs.shape[-1]), device=inputs.device)]

    source = Image.new("RGB", (32, 32), (1, 2, 3))
    try:
        BiRefNetBackend(model=BadModel(), device="cpu").segment(source)
    except RuntimeError as exc:
        assert "one channel" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("ambiguous multi-channel predictions must fail explicitly")


def test_birefnet_dynamic_resolution_preserves_aspect_and_pads() -> None:
    source = Image.new("RGB", (73, 51), (31, 67, 109))
    fake_model = FakeBiRefNetModel(value=0.0)
    backend = BiRefNetBackend(
        model_name="ZhengPeng7/BiRefNet_dynamic",
        model=fake_model,
        device="cpu",
    )
    result = backend.segment(source)
    assert result.size == source.size
    assert backend.dynamic_resolution
    assert backend.last_resolution == (64, 96)
    assert fake_model.last_shape == (1, 3, 64, 96)
    assert set(np.unique(np.asarray(result.getchannel("A")))).issubset({127, 128})
    assert np.array_equal(np.asarray(result.convert("RGB")), np.asarray(source))
    assert parse_birefnet_resolution("1024x1536") == (1024, 1536)
    assert parse_birefnet_resolution("auto") is None
    try:
        BiRefNetBackend(
            model_name="ZhengPeng7/BiRefNet_dynamic",
            model=fake_model,
            device="cpu",
            resolution=(1000, 1000),
        )
    except ValueError as exc:
        assert "divisible by 32" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("dynamic fixed resolutions must be divisible by 32")


def test_birefnet_dynamic_resolution_has_safe_long_side_cap() -> None:
    """Large camera frames are downscaled before model activations are built."""

    source = Image.new("RGB", (2721, 1800), (31, 67, 109))
    fake_model = FakeBiRefNetModel(value=0.0)
    backend = BiRefNetBackend(
        model_name="ZhengPeng7/BiRefNet_dynamic",
        model=fake_model,
        device="cpu",
        # Keep the test explicit even if the project default is tuned later.
        max_inference_side=1024,
    )
    result = backend.segment(source)
    assert result.size == source.size
    # The padded input is block aligned and never exceeds the configured cap.
    assert backend.last_resolution == (704, 1024)
    assert fake_model.last_shape == (1, 3, 704, 1024)
    assert backend.last_content_size == (677, 1024)
    assert parse_birefnet_max_side("auto") == DEFAULT_BIREFNET_MAX_SIDE
    assert parse_birefnet_max_side("none") is None
    assert parse_birefnet_max_side(0) is None
    assert DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE == 1024


def test_birefnet_dynamic_resolution_can_disable_cap() -> None:
    source = Image.new("RGB", (73, 51), (31, 67, 109))
    fake_model = FakeBiRefNetModel(value=0.0)
    backend = BiRefNetBackend(
        model_name="ZhengPeng7/BiRefNet_dynamic",
        model=fake_model,
        device="cpu",
        max_inference_side=None,
    )
    backend.segment(source)
    assert backend.max_inference_side is None
    assert backend.last_resolution == (64, 96)


def test_mps_watermark_defaults_stay_ordered() -> None:
    """Lowering HIGH must not leave PyTorch's older LOW default above it."""

    assert _compatible_mps_low_watermark("0.7") == "0.6"
    assert _compatible_mps_low_watermark("0.3") == "0.24"
    assert _compatible_mps_low_watermark("0.6") == "0.48"
    assert _compatible_mps_low_watermark("0") == "0.0"


def test_cpu_retry_cap_never_grows_requested_cap() -> None:
    backend = BiRefNetBackend(
        model_name="ZhengPeng7/BiRefNet_dynamic",
        model=FakeBiRefNetModel(),
        device="cpu",
        max_inference_side=512,
        cpu_fallback_max_side=1024,
    )
    assert backend._cpu_retry_max_side() == 512


def test_birefnet_auxiliary_outputs_are_released_before_decode() -> None:
    """Only the selected logits tensor should survive model forward."""

    import torch

    class ResultBox:
        def __init__(self, values):
            self.predictions = values

    class AuxiliaryModel(FakeBiRefNetModel):
        def __init__(self):
            super().__init__(value=0.0)
            self.result_ref = None

        def __call__(self, inputs):
            self.calls += 1
            height, width = inputs.shape[-2:]
            low = torch.zeros((1, 1, max(1, height // 2), max(1, width // 2)), device=inputs.device)
            high = torch.zeros((1, 1, height, width), device=inputs.device)
            box = ResultBox([low, high])
            self.result_ref = weakref.ref(box)
            return box

    model = AuxiliaryModel()
    backend = BiRefNetBackend(model=model, device="cpu", resolution=(64, 64))
    backend.segment(Image.new("RGB", (48, 40), (20, 30, 40)))
    gc.collect()
    assert model.result_ref is not None
    assert model.result_ref() is None


def test_birefnet_selects_highest_resolution_prediction() -> None:
    import torch

    from scripts.normalize_ben2 import _birefnet_output_tensor

    low = torch.zeros((1, 1, 8, 8))
    high = torch.zeros((1, 1, 16, 16))
    # Deliberately put the low-resolution result last; selection must follow
    # shape rather than relying on a checkpoint-specific list order.
    assert _birefnet_output_tensor([high, low]) is high


def test_birefnet_static_oom_fallback_rebuilds_with_safe_cap() -> None:
    """Fixed HR/lite checkpoints must not retry their huge shape on CPU."""

    import torch

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or not mps.is_available():
        # The test is meaningful only where an MPS tensor can be created; the
        # regular CPU path remains covered by the tests above.
        return

    class OOMOnce(FakeBiRefNetModel):
        def __init__(self):
            super().__init__(value=0.0)
            self.shapes = []

        def to(self, device):
            return self

        def __call__(self, inputs):
            self.calls += 1
            self.shapes.append(tuple(int(v) for v in inputs.shape))
            if self.calls == 1:
                raise RuntimeError("MPS backend out of memory")
            height, width = inputs.shape[-2:]
            return [torch.zeros((1, 1, height, width), device=inputs.device)]

    model = OOMOnce()
    backend = BiRefNetBackend(
        model_name="ZhengPeng7/BiRefNet_HR",
        model=model,
        device="cpu",
        cpu_fallback_max_side=1024,
    )
    # Inject the accelerator label while keeping model loading offline.
    backend.device = "mps"
    result = backend.segment(Image.new("RGB", (48, 40), (20, 30, 40)))
    # The bounded CPU mode persists for subsequent batch items; it must not
    # silently return to the HR checkpoint's 2048² default.
    backend.segment(Image.new("RGB", (48, 40), (20, 30, 40)))
    assert result.size == (48, 40)
    assert model.shapes == [
        (1, 3, 2048, 2048),
        (1, 3, 1024, 1024),
        (1, 3, 1024, 1024),
    ]
    assert backend.device == "cpu"
    assert backend.fallback_count == 1


def test_birefnet_model_load_oom_falls_back_to_cpu() -> None:
    """An OOM while moving weights must be recoverable before first image."""

    class FakeLoadedModel:
        def eval(self):
            return self

    calls = []

    def fake_loader(model_name, device):
        calls.append(device)
        if device == "mps":
            raise RuntimeError("wrapper") from RuntimeError("MPS backend out of memory")
        return FakeLoadedModel()

    backend = BiRefNetBackend(model_name="local/test", device="cpu")
    # Keep construction offline while exercising the accelerator-labelled
    # property path.  ``_release_torch_memory`` is best effort on machines
    # without MPS, so this remains a portable test.
    backend.device = "mps"
    with patch("scripts.normalize_ben2._load_birefnet_model", fake_loader):
        assert isinstance(backend.model, FakeLoadedModel)
    assert calls == ["mps", "cpu"]
    assert backend.device == "cpu"
    assert backend.fallback_count == 1


def test_normalize_image_calls_geometry_and_preserves_source_rgb() -> None:
    source, alpha = synthetic_phone_with_stand()
    backend = FakeBackend(alpha)
    result, info = normalize_image(source, backend=backend)
    assert backend.calls == 1
    assert result.mode == "RGBA"
    assert info.backend == "fake"
    assert info.model == "fake-model"
    assert info.support_removed
    assert result.size[1] < source.height
    values = np.asarray(result)
    assert len(np.unique(values[..., 3])) > 2
    # All fully transparent pixels are RGB-clean after alpha-safe processing.
    assert np.all(values[..., :3][values[..., 3] == 0] == 0)


def test_stage_switches_keep_size_and_report_no_support() -> None:
    source, alpha = synthetic_phone_with_stand()
    result, info = normalize_image(
        source,
        backend=FakeBackend(alpha),
        remove_support=False,
        deskew=False,
        tight_crop=False,
    )
    assert result.size == source.size
    assert info.output_size == source.size
    assert not info.support_removed
    assert info.rotation_degrees == 0.0


def test_geometry_guard_runs_before_backend_inference() -> None:
    """An oversized header is rejected without touching a lazy backend."""

    # Construct a header-only PIL object; validation should happen from its
    # dimensions before any pixel buffer (or model input) is requested.
    oversized = Image.Image()
    oversized._mode = "RGB"
    oversized._size = (9000, 9000)

    class CountingBackend:
        calls = 0

        def segment(self, image: Image.Image) -> Image.Image:
            del image
            self.calls += 1
            raise AssertionError("the backend must not run for an oversized source")

    backend = CountingBackend()
    try:
        normalize_image(oversized, backend=backend)
    except ValueError as exc:
        assert "geometry memory guard" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("the default geometry guard must reject 81 MP")
    assert backend.calls == 0

    # The same early check must precede construction of a named/lazy backend.
    with patch("scripts.normalize_ben2.create_segmentation_backend") as factory:
        try:
            normalize_image(oversized, backend_name="fake")
        except ValueError as exc:
            assert "geometry memory guard" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("factory construction must be skipped")
    factory.assert_not_called()


def test_standalone_backend_is_closed_after_success_or_error() -> None:
    """One-shot normalize/process calls must not retain their model backend."""

    source, alpha = synthetic_phone_with_stand()

    class ClosableBackend(FakeBackend):
        def __init__(self, mask: np.ndarray, *, fail: bool = False):
            super().__init__(mask)
            self.close_calls = 0
            self.fail = fail

        def segment(self, image: Image.Image) -> Image.Image:
            if self.fail:
                self.calls += 1
                raise RuntimeError("synthetic inference failure")
            return super().segment(image)

        def close(self) -> None:
            self.close_calls += 1

    owned = ClosableBackend(alpha)
    with patch(
        "scripts.normalize_ben2.create_segmentation_backend",
        return_value=owned,
    ) as factory:
        result, info = normalize_image(
            source,
            backend_name="fake",
            remove_support=False,
            deskew=False,
            tight_crop=False,
        )
    assert factory.call_count == 1
    assert owned.close_calls == 1
    assert result.size == source.size
    assert info.backend == "fake"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.jpg"
        output_path = root / "output.png"
        source.save(input_path)
        file_owned = ClosableBackend(alpha)
        with patch(
            "scripts.normalize_ben2.create_segmentation_backend",
            return_value=file_owned,
        ):
            file_info = process_file(
                input_path,
                output_path,
                backend="fake",
                no_support_removal=True,
                no_deskew=True,
                no_tight_crop=True,
            )
        assert output_path.exists()
        assert file_info.backend == "fake"
        assert file_owned.close_calls == 1

    failing = ClosableBackend(alpha, fail=True)
    with patch(
        "scripts.normalize_ben2.create_segmentation_backend",
        return_value=failing,
    ):
        try:
            normalize_image(
                source,
                backend_name="fake",
                remove_support=False,
                deskew=False,
                tight_crop=False,
            )
        except RuntimeError as exc:
            assert "synthetic inference failure" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("the synthetic backend must fail")
    assert failing.close_calls == 1


def test_alpha_only_pil_result_and_legacy_alpha_helper() -> None:
    source, alpha = synthetic_phone_with_stand()

    class AlphaBackend(SegmentationBackend):
        name = "alpha-test"

        def segment(self, image: Image.Image) -> Image.Image:
            del image
            return Image.fromarray(alpha)

    result, info = normalize_image(
        source,
        backend=AlphaBackend("alpha-model", "cpu"),
        remove_support=False,
        deskew=False,
        tight_crop=False,
    )
    assert result.mode == "RGBA"
    assert result.size == source.size
    assert info.backend == "alpha-test"
    assert np.array_equal(np.asarray(result.getchannel("A")), alpha)

    # The compatibility helper must use the same geometry implementation.
    normalized, alpha_info = normalize_rgba(
        source,
        alpha,
        deskew=False,
        tight_crop=False,
    )
    assert normalized.mode == "RGBA"
    assert alpha_info.backend == "alpha-only"


def test_process_file_metadata_and_parser_flags() -> None:
    source, alpha = synthetic_phone_with_stand()
    backend = FakeBackend(alpha)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "input.jpg"
        output_path = root / "nested" / "output.png"
        metadata_path = root / "nested" / "output.json"
        source.save(source_path, quality=95)
        info = process_file(
            source_path,
            output_path,
            backend_instance=backend,
            no_support_removal=True,
            no_deskew=True,
            no_tight_crop=True,
            metadata_path=metadata_path,
        )
        assert output_path.exists()
        assert metadata_path.exists()
        metadata = metadata_path.read_text(encoding="utf-8")
        assert '"backend": "fake"' in metadata
        assert '"support_removed": false' in metadata
        assert '"support_detected": false' in metadata
        assert info.output_size == source.size

        try:
            process_file(source_path, source_path, backend_instance=backend)
        except ValueError as exc:
            assert "different paths" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("the input must never be overwritten")

    parser = _build_parser()
    args = parser.parse_args(
        [
            "in.jpg",
            "out.png",
            "--backend",
            "birefnet",
            "--model",
            "isnet-general-use",
            "--birefnet-resolution",
            "1024x1536",
            "--device",
            "cpu",
            "--max-deskew-degrees",
            "12",
            "--crop-threshold",
            "20",
            "--confidence-floor",
            "0.7",
            "--no-support-removal",
            "--no-deskew",
            "--no-tight-crop",
            "--no-alpha-matting",
            "--max-geometry-pixels",
            "1000000",
        ]
    )
    assert args.backend == "birefnet"
    assert args.model == "isnet-general-use"
    assert args.birefnet_resolution == "1024x1536"
    assert args.max_deskew_degrees == 12
    assert args.crop_threshold == 20
    assert args.support_min_confidence == 0.7
    assert args.no_support_removal and args.no_deskew and args.no_tight_crop
    assert args.no_alpha_matting
    assert args.max_geometry_pixels == "1000000"
    defaults = _build_parser().parse_args(["in.jpg", "out.png"])
    assert defaults.max_geometry_pixels == str(DEFAULT_MAX_GEOMETRY_PIXELS)


def test_optional_rembg_backend_with_fake_module() -> None:
    source, alpha = synthetic_phone_with_stand()
    rgba = Image.merge("RGBA", (*source.split(), Image.fromarray(alpha)))
    payload = io.BytesIO()
    rgba.save(payload, format="PNG")
    calls: list[tuple[str, object]] = []

    def fake_new_session(model: str) -> object:
        calls.append(("session", model))
        return {"model": model}

    def fake_remove(*args: object, **kwargs: object) -> bytes:
        calls.append(("remove", kwargs.get("session")))
        return payload.getvalue()

    fake_rembg = SimpleNamespace(new_session=fake_new_session, remove=fake_remove)
    previous = sys.modules.get("rembg")
    sys.modules["rembg"] = fake_rembg
    try:
        backend = RembgBackend(model_name="u2net", device="cpu", alpha_matting=False)
        result = backend.segment(source)
    finally:
        if previous is None:
            sys.modules.pop("rembg", None)
        else:
            sys.modules["rembg"] = previous
    assert result.size == source.size
    assert np.array_equal(np.asarray(result.getchannel("A")), alpha)
    assert calls == [("session", "u2net"), ("remove", {"model": "u2net"})]


def main() -> None:
    test_backend_registry_and_default()
    test_ben2_backend_uses_alpha_only()
    test_birefnet_backend_decodes_logits_and_preserves_rgb()
    test_birefnet_backend_rejects_ambiguous_channels()
    test_birefnet_dynamic_resolution_preserves_aspect_and_pads()
    test_birefnet_dynamic_resolution_has_safe_long_side_cap()
    test_birefnet_dynamic_resolution_can_disable_cap()
    test_birefnet_auxiliary_outputs_are_released_before_decode()
    test_birefnet_selects_highest_resolution_prediction()
    test_birefnet_static_oom_fallback_rebuilds_with_safe_cap()
    test_birefnet_model_load_oom_falls_back_to_cpu()
    test_normalize_image_calls_geometry_and_preserves_source_rgb()
    test_stage_switches_keep_size_and_report_no_support()
    test_geometry_guard_runs_before_backend_inference()
    test_alpha_only_pil_result_and_legacy_alpha_helper()
    test_process_file_metadata_and_parser_flags()
    test_optional_rembg_backend_with_fake_module()
    print("segmentation backend pipeline smoke tests passed")


if __name__ == "__main__":
    main()
