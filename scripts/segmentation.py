#!/usr/bin/env python3
"""BiRefNet_dynamic segmentation with phone stand removal, deskew and alpha-safe crop."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


# PyTorch reads these variables when its MPS backend is initialized.  Set
# conservative defaults here as well as in the macOS launcher so direct API or
# command-line use receives the same safety net.  Non-empty environment values
# deliberately remain untouched (for example a developer running a controlled
# high-memory benchmark); an empty value is treated as “not configured”.
def _compatible_mps_low_watermark(high_value: Any) -> str:
    """Choose a valid low watermark when the caller only lowers ``high``.

    Some PyTorch versions default the low watermark to ``1.4``.  That is valid
    with the stock high watermark (``1.7``), but not when this project lowers
    the high watermark to ``0.7``.  If a deployment explicitly chooses an even
    smaller high value, derive a proportionally smaller low value instead of
    creating the invalid ``low > high`` pair.  An explicitly supplied LOW
    variable is never changed.
    """

    try:
        high = float(high_value)
    except (TypeError, ValueError):
        # Let PyTorch report a malformed explicit HIGH value; a zero LOW does
        # not introduce a second, misleading ordering error.
        return "0.0"
    if not math.isfinite(high) or high <= 0.0:
        return "0.0"
    if high <= 0.6:
        # Leave a small gap below HIGH so older allocators that require a
        # strictly smaller low watermark are also accepted.
        return f"{high * 0.8:.6g}"
    return "0.6"


if sys.platform == "darwin":
    high_watermark = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if not high_watermark:
        high_watermark = "0.7"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = high_watermark
    # Older PyTorch releases default the low watermark to 1.4.  That value is
    # valid with the stock high watermark (1.7), but becomes invalid when the
    # conservative high watermark above is enabled (low must be <= high).
    # Keep a little headroom below the high limit while preserving an explicit
    # deployment override.
    if not os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"):
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = _compatible_mps_low_watermark(
            high_watermark
        )
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from .geometry_postprocess import (
        DEFAULT_MAX_GEOMETRY_PIXELS,
        GeometryConfig,
        SupportCut,
        clip_support,
        estimate_long_angle,
        find_support_cut,
        process_array,
        rotate_alpha_safe,
        parse_max_geometry_pixels,
        tight_crop,
        validate_geometry_size,
    )
except ImportError:  # direct ``python scripts/segmentation.py`` execution
    from geometry_postprocess import (
        DEFAULT_MAX_GEOMETRY_PIXELS,
        GeometryConfig,
        SupportCut,
        clip_support,
        estimate_long_angle,
        find_support_cut,
        process_array,
        rotate_alpha_safe,
        parse_max_geometry_pixels,
        tight_crop,
        validate_geometry_size,
    )


DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet_dynamic"
DEFAULT_BIREFNET_RESOLUTION = (1024, 1024)
# ``BiRefNet_dynamic`` accepts arbitrary image shapes, but its decoder keeps
# several full-resolution feature maps alive.  Feeding a 2.7K camera frame
# directly to it can therefore consume tens of gigabytes on Apple's unified
# memory.  The model was trained around 1K inputs; a 1024-pixel long side is
# the conservative default and can be raised explicitly by callers.
DEFAULT_BIREFNET_MAX_SIDE = 1024
DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE = 1024
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")


def parse_birefnet_resolution(value: Any) -> Optional[Tuple[int, int]]:
    """Parse a BiRefNet ``HxW`` resolution, or ``auto`` for dynamic models."""

    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("BiRefNet resolution must contain height and width")
        height, width = int(value[0]), int(value[1])
    else:
        text = str(value).strip().lower().replace("×", "x")
        if text in {"", "auto", "original", "dynamic"}:
            return None
        parts = text.split("x")
        if len(parts) == 1 and parts[0].isdigit():
            height = width = int(parts[0])
        elif len(parts) == 2 and all(part.strip().isdigit() for part in parts):
            height, width = (int(part.strip()) for part in parts)
        else:
            raise ValueError(
                "BiRefNet resolution must be an integer, HxW, or 'auto'"
            )
    if height < 32 or width < 32:
        raise ValueError("BiRefNet resolution must be at least 32×32")
    return (height, width)


def parse_birefnet_max_side(value: Any) -> Optional[int]:
    """Parse the dynamic-model long-side safety cap.

    ``"auto"``/``"default"`` (and an empty value) select the conservative
    project default.  ``"none"``/``"off"``/``0`` disable the cap for callers
    that explicitly accept the memory risk.  ``None`` has the same disabling
    meaning for the programmatic API, which keeps the usual Python idiom
    useful while the command line can spell it out unambiguously.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("BiRefNet max side must be an integer, 'auto', or 'none'")
    if isinstance(value, (int, np.integer)):
        side = int(value)
    else:
        text = str(value).strip().lower().replace("×", "x")
        if text in {"", "auto", "default"}:
            return DEFAULT_BIREFNET_MAX_SIDE
        if text in {"none", "off", "unlimited", "0"}:
            return None
        if not text.isdigit():
            raise ValueError(
                "BiRefNet max side must be a positive integer, 'auto', or 'none'"
            )
        side = int(text)
    if side == 0:
        return None
    if side < 32:
        raise ValueError("BiRefNet max side must be at least 32 pixels")
    return side


def _dynamic_birefnet_sizes(
    image_size: Tuple[int, int],
    max_side: Optional[int] = DEFAULT_BIREFNET_MAX_SIDE,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return ``(content_hw, padded_hw)`` for a dynamic checkpoint.

    The content dimensions are resized down only when a safety cap is
    exceeded; their aspect ratio is preserved.  The padded dimensions are
    rounded up to BiRefNet's required 32-pixel blocks.  Keeping these values
    separate lets the caller crop logits before upsampling, so replicated
    padding never becomes a visible halo at the source edge.
    """

    source_width, source_height = (int(image_size[0]), int(image_size[1]))
    if source_width < 1 or source_height < 1:
        raise ValueError("image dimensions must be positive")
    cap = parse_birefnet_max_side(max_side)
    content_width, content_height = source_width, source_height
    if cap is not None:
        # Use a block-aligned cap so the longest padded side never exceeds the
        # advertised limit.  A caller may provide a non-multiple (e.g. 1300);
        # rounding down by at most 31 pixels is preferable to an unexpected
        # allocation above the safety budget.
        block_cap = max(32, (int(cap) // 32) * 32)
        longest = max(source_width, source_height)
        if longest > block_cap:
            scale = block_cap / float(longest)
            content_width = max(1, int(round(source_width * scale)))
            content_height = max(1, int(round(source_height * scale)))
    padded_width = max(32, ((content_width + 31) // 32) * 32)
    padded_height = max(32, ((content_height + 31) // 32) * 32)
    # Rounding can push a dimension one block above a very small cap when the
    # source is highly elongated.  In that case reduce the content dimension
    # by one block and retain the aspect-preserving resize guarantee.
    if cap is not None:
        block_cap = max(32, (int(cap) // 32) * 32)
        if padded_width > block_cap or padded_height > block_cap:
            scale = min(
                block_cap / float(max(padded_width, 1)),
                block_cap / float(max(padded_height, 1)),
            )
            content_width = max(1, int(round(content_width * scale)))
            content_height = max(1, int(round(content_height * scale)))
            padded_width = max(32, ((content_width + 31) // 32) * 32)
            padded_height = max(32, ((content_height + 31) // 32) * 32)
    return (content_height, content_width), (padded_height, padded_width)


def _dynamic_birefnet_resolution(
    image_size: Tuple[int, int],
    max_side: Optional[int] = DEFAULT_BIREFNET_MAX_SIDE,
) -> Tuple[int, int]:
    """Choose a safe dynamic model input shape divisible by 32.

    This compatibility helper retains its historical ``(height, width)``
    return shape while adding an optional long-side cap.
    """

    _content, padded = _dynamic_birefnet_sizes(image_size, max_side=max_side)
    return padded


def _bounded_birefnet_resolution(
    resolution: Tuple[int, int],
    max_side: Optional[int],
) -> Tuple[int, int]:
    """Return a fixed-model resolution bounded by a long-side cap.

    On an accelerator OOM retry, cap an explicitly requested fixed input
    size to avoid repeating the same large allocation on the host.
    Passing ``None`` preserves the requested shape.
    """

    height, width = (int(resolution[0]), int(resolution[1]))
    if height < 1 or width < 1:
        raise ValueError("BiRefNet resolution dimensions must be positive")
    cap = parse_birefnet_max_side(max_side)
    if cap is None:
        return height, width
    _content, padded = _dynamic_birefnet_sizes(
        (width, height),
        max_side=cap,
    )
    return tuple(int(value) for value in padded)


def _prepare_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation once and return an RGB image.

    The pipeline is read-only with respect to the source pixels.  Reusing an
    already-normalized RGB image avoids making another full-resolution copy at
    every backend boundary (which is noticeable for multi-megapixel photos).
    Images with a non-trivial EXIF orientation or a different mode are still
    materialized as a correctly oriented RGB image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if image.mode == "RGB":
        try:
            orientation = image.getexif().get(274, 1)
        except Exception:
            orientation = 1
        if orientation in (None, 1):
            return image
    transposed = ImageOps.exif_transpose(image)
    return transposed if transposed.mode == "RGB" else transposed.convert("RGB")


def _alpha_array(value: Any, expected_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert a PIL/NumPy alpha value to a validated uint8 2-D array."""

    if isinstance(value, Image.Image):
        if "A" in value.getbands():
            value = value.getchannel("A")
        else:
            value = value.convert("L")
        array = np.asarray(value)
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] >= 4:
            array = array[..., 3]
        else:
            raise ValueError("alpha must be 2-D or an RGBA value")
    if array.ndim != 2:
        raise ValueError(f"alpha must be 2-D, got shape {array.shape!r}")
    if expected_size is not None and (array.shape[1], array.shape[0]) != expected_size:
        raise ValueError(
            f"alpha size {(array.shape[1], array.shape[0])} does not match "
            f"image size {expected_size}"
        )
    if array.dtype != np.uint8:
        values = array.astype(np.float32, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.max()) <= 1.0:
            values = values * 255.0
        array = np.rint(np.nan_to_num(values, nan=0.0)).clip(0, 255).astype(np.uint8)
    else:
        array = array.copy()
    return array


def _rgba_with_alpha(image: Image.Image, alpha: Any) -> Image.Image:
    """Combine original RGB pixels with a backend's continuous alpha."""

    source = _prepare_image(image)
    # ``Image.merge("RGBA", (*source.split(), alpha))`` materializes three
    # full-size channel images *and* a fourth full-size result.  That transient
    # fan-out is surprisingly expensive for camera originals and can coexist
    # with the model's allocator reservation.  Converting once to RGBA and
    # replacing its alpha keeps one destination buffer instead of three RGB
    # channel buffers.  ``putalpha`` does not resample the matte, so the
    # continuous edge values are preserved byte-for-byte.
    alpha_array = _alpha_array(alpha, source.size)
    # A 2-D uint8 array is inferred as ``L`` by Pillow.  Omitting the explicit
    # mode keeps this compatible with Pillow 13+, where the ``mode=`` override
    # is removed, without changing the matte bytes or allocating a conversion.
    alpha_image = Image.fromarray(alpha_array)
    rgba = source.convert("RGBA")
    rgba.putalpha(alpha_image)
    del alpha_array, alpha_image
    return rgba


def _foreground_alpha(result: Any, expected_size: Tuple[int, int]) -> np.ndarray:
    """Extract alpha from a backend result and validate its dimensions."""

    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("segmentation backend returned an empty result")
        result = result[0]
    if isinstance(result, Image.Image):
        if "A" not in result.getbands():
            raise RuntimeError("segmentation backend did not return an alpha channel")
        return _alpha_array(result.getchannel("A"), expected_size)
    array = np.asarray(result)
    if array.ndim == 3 and array.shape[-1] < 4:
        raise RuntimeError("segmentation backend returned RGB without alpha")
    return _alpha_array(array, expected_size)


def resolve_device(device: str = "auto") -> str:
    """Resolve and validate a torch device lazily."""

    value = str(device).strip().lower()
    if value not in SUPPORTED_DEVICES:
        raise ValueError(
            f"unsupported device {device!r}; choose one of {', '.join(SUPPORTED_DEVICES)}"
        )
    try:
        import torch
    except ImportError:
        if value == "auto":
            return "auto"
        raise RuntimeError("torch is required for an explicit device")
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return value


def _release_torch_memory(device: str = "") -> None:
    """Release Python and accelerator-side inference temporaries.

    MPS uses the Mac's unified memory, while CUDA maintains a device cache; an
    apparently idle allocator reservation can still starve the rest of the
    desktop/process.
    ``empty_cache`` is intentionally best-effort: older torch builds and test
    doubles may not expose every method.  The helper never masks the original
    inference error.
    """

    # Run collection first so tensors that only became unreachable in a
    # ``finally`` block are returned to the allocator before it is trimmed.
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
    except ImportError:
        return
    value = str(device).strip().lower()
    try:
        if value.startswith("mps"):
            mps = getattr(torch, "mps", None)
            synchronize = getattr(mps, "synchronize", None)
            if callable(synchronize):
                synchronize()
            # MPS work is asynchronous on some torch builds.  A second GC
            # pass after synchronization can now see tensors whose stream
            # references were keeping them alive during the first pass.
            try:
                gc.collect()
            except Exception:
                pass
            empty_cache = getattr(mps, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        elif value.startswith("cuda"):
            cuda = getattr(torch, "cuda", None)
            synchronize = getattr(cuda, "synchronize", None)
            if callable(synchronize):
                synchronize()
            try:
                gc.collect()
            except Exception:
                pass
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        # Cache trimming is an optimization only.  An accelerator can report a
        # transient synchronization error while recovering from OOM; callers
        # still need the original exception/fallback path.
        pass


def _is_out_of_memory_error(error: BaseException) -> bool:
    """Whether an exception looks like an accelerator out-of-memory error."""

    # Hugging Face/remote-code loaders often wrap the original torch OOM in a
    # generic ``RuntimeError``.  Walk the explicit cause/context chain so a
    # model-load fallback is not skipped merely because that wrapper changed
    # the top-level message.
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    markers = (
        "out of memory",
        "outofmemory",
        "mps backend out of memory",
        "resource limit",
        "memory allocation",
        "cuda error: out of memory",
    )
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        text = str(current).lower()
        error_name = type(current).__name__.lower()
        if (
            "outofmemory" in error_name
            or "out_of_memory" in error_name
            or error_name == "memoryerror"
            or any(marker in text for marker in markers)
        ):
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)
    return False


def _clear_exception_traceback(error: BaseException) -> None:
    """Detach traceback frames that may retain failed accelerator tensors.

    Python keeps the active exception (and its traceback) alive for the whole
    ``except`` suite.  Clearing the traceback before an OOM retry lets the
    failed model-forward frame release its activation references immediately,
    instead of competing with the CPU retry allocation.
    """

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            current.__traceback__ = None
        except Exception:
            pass
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)


class SegmentationBackend:
    """Minimal backend contract for an alpha-producing segmenter."""

    name = "backend"

    def __init__(self, model_name: str, device: str = "auto", **_: Any) -> None:
        self.model_name = str(model_name)
        self.device = str(device)

    def segment(self, image: Image.Image) -> Image.Image:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.segment(image)

    def close(self) -> None:
        """Release common model/session references and accelerator cache."""

        device = self.device
        for attribute in ("_model", "_session"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)
        _release_torch_memory(device)


def _load_birefnet_model(model_name: str, device: str) -> Any:
    """Load a BiRefNet checkpoint through its official HF remote code.

    BiRefNet publishes a Transformers-compatible model whose implementation is
    stored alongside the checkpoint.  Importing Transformers lazily keeps the
    geometry-only operations lightweight, while inference
    gives a clear dependency error when the inference dependencies are not installed.
    """

    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:  # pragma: no cover - dependency-specific path
        raise RuntimeError(
            "BiRefNet requires transformers, accelerate and the model dependencies; "
            "install requirements.txt"
        ) from exc

    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load BiRefNet model {model_name!r}; "
            "check the Hugging Face model id, network access, and installed dependencies"
        ) from exc
    if model is None:
        raise RuntimeError(f"BiRefNet returned no model for {model_name!r}")
    if hasattr(model, "to"):
        moved = model.to(torch.device(device))
        if moved is not None:
            model = moved
    if hasattr(model, "eval"):
        model.eval()
    # This backend is inference-only.  Disabling parameter gradients avoids
    # accidentally constructing autograd metadata if a caller invokes the
    # model outside ``_run_model`` and is effectively free for the loaded
    # checkpoint (the normal path also uses ``inference_mode``).
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    return model


def _birefnet_output_tensor(result: Any) -> Any:
    """Find the highest-resolution prediction tensor in a model result.

    The official BiRefNet remote model returns a list containing the final
    logits, but accepting common ``ModelOutput``/mapping forms makes the
    backend usable with local checkpoints and lightweight test doubles too.
    The returned object is deliberately left as a torch tensor (or array) so
    the caller can apply the model's required sigmoid exactly once.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - BiRefNet itself needs torch
        torch = None  # type: ignore[assignment]

    def is_tensor(value: Any) -> bool:
        return bool(torch is not None and isinstance(value, torch.Tensor)) or isinstance(
            value, np.ndarray
        )

    def spatial_area(value: Any) -> int:
        """Return a tensor-like prediction's spatial area for ranking."""

        try:
            shape = tuple(int(v) for v in value.shape)
            if len(shape) >= 2:
                return max(0, shape[-2]) * max(0, shape[-1])
        except Exception:
            pass
        return -1

    def choose(candidates: list[Any]) -> Any:
        if not candidates:
            return None
        # Auxiliary predictions are often returned in low→high order, but
        # local checkpoints/test doubles are not required to preserve that
        # convention.  Ranking by actual spatial area keeps the decoder on
        # the finest mask while retaining only Python references (not copies)
        # during traversal.
        return max(candidates, key=spatial_area)

    if is_tensor(result):
        return result
    if isinstance(result, dict):
        # Prefer names used by Transformers and by the upstream implementation.
        preferred: list[Any] = []
        for key in ("logits", "predictions", "pred", "mask", "masks"):
            if key in result:
                found = _birefnet_output_tensor(result[key])
                if found is not None:
                    preferred.append(found)
        if preferred:
            # A named prediction is the strongest semantic signal; if more
            # than one exists, still select the highest-resolution one.
            return choose(preferred)
        candidates: list[Any] = []
        for value in reversed(tuple(result.values())):
            found = _birefnet_output_tensor(value)
            if found is not None:
                candidates.append(found)
        return choose(candidates)
    if isinstance(result, (list, tuple)):
        candidates = []
        for value in reversed(result):
            found = _birefnet_output_tensor(value)
            if found is not None:
                candidates.append(found)
        return choose(candidates)
    candidates = []
    for key in ("logits", "predictions", "pred", "mask", "masks"):
        if hasattr(result, key):
            found = _birefnet_output_tensor(getattr(result, key))
            if found is not None:
                candidates.append(found)
    return choose(candidates)


def _birefnet_alpha(
    result: Any,
    output_size: Tuple[int, int],
    *,
    crop_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert BiRefNet logits to a same-size continuous uint8 alpha mask.

    ``crop_size`` is used by the dynamic model when its input tensor was
    replicate-padded to a multiple of 32.  Cropping logits before the final
    resize prevents the artificial padded border from influencing the alpha.
    """

    logits = _birefnet_output_tensor(result)
    if logits is None:
        raise RuntimeError("BiRefNet returned no prediction tensor")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - BiRefNet needs torch
        raise RuntimeError("torch is required to decode BiRefNet predictions") from exc

    tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if tensor.ndim == 4:
        if tensor.shape[0] < 1 or tensor.shape[1] < 1:
            raise RuntimeError(f"BiRefNet prediction has an empty batch/channel: {tuple(tensor.shape)}")
        if tensor.shape[1] != 1:
            raise RuntimeError(
                "BiRefNet prediction must have one channel; "
                f"got shape {tuple(tensor.shape)}"
            )
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        # Accept [B,H,W] and [C,H,W] forms, but reject an ambiguous multi-
        # channel result rather than silently selecting the wrong class.
        if tensor.shape[0] != 1:
            raise RuntimeError(
                "BiRefNet prediction must be [B,1,H,W], [B,H,W], or [H,W]; "
                f"got shape {tuple(tensor.shape)}"
            )
        tensor = tensor[0]
    elif tensor.ndim == 2:
        pass
    else:
        raise RuntimeError(
            "BiRefNet prediction must be 2-D/3-D/4-D logits; "
            f"got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[-2] < 1 or tensor.shape[-1] < 1:
        raise RuntimeError(
            "BiRefNet prediction has an empty spatial dimension: "
            f"{tuple(tensor.shape)}"
        )
    # Decode on CPU.  The prediction is only one channel, and transferring it
    # before sigmoid/upsampling avoids allocating a full source-sized mask on
    # MPS/CUDA (particularly important for 2K+ source images).
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    # ``logits`` still points at the original accelerator tensor after the
    # assignment above.  Drop it before CPU sigmoid/interpolation so the
    # device allocation can be reclaimed as early as possible.
    del logits
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("BiRefNet prediction contains NaN or infinite values")
    tensor = torch.sigmoid(tensor)
    if crop_size is not None:
        crop_height, crop_width = (int(crop_size[0]), int(crop_size[1]))
        if crop_height < 1 or crop_width < 1:
            raise ValueError("BiRefNet crop size must be positive")
        if tensor.shape[-2] < crop_height or tensor.shape[-1] < crop_width:
            raise RuntimeError(
                "BiRefNet prediction is smaller than the unpadded source: "
                f"prediction={tuple(tensor.shape[-2:])}, crop={(crop_height, crop_width)}"
            )
        tensor = tensor[..., :crop_height, :crop_width]
    target_hw = (int(output_size[1]), int(output_size[0]))
    if tuple(int(v) for v in tensor.shape[-2:]) != target_hw:
        tensor = F.interpolate(
            tensor[None, None], size=target_hw, mode="bilinear", align_corners=False
        )[0, 0]
    # Materialize an owning NumPy array before returning.  Calling
    # ``numpy()`` directly on a CPU view can keep the underlying torch storage
    # (and, on some builds, the original device tensor) alive through the
    # returned array.  The explicit cast/copy breaks that retention chain.
    cpu_alpha = tensor.clamp(0.0, 1.0)
    alpha = np.rint(cpu_alpha.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    result_array = np.array(alpha, dtype=np.uint8, copy=True)
    del cpu_alpha, alpha, tensor
    return result_array


class BiRefNetBackend(SegmentationBackend):
    """BiRefNet general-purpose foreground segmentation backend.

    The dynamic inference path preserves aspect ratio and normalizes
    with ImageNet statistics, runs the final logits through sigmoid, and then
    restores the mask to the source dimensions.  We retain the source RGB
    pixels and use only this continuous mask as alpha, for the geometry stage.
    """

    name = "birefnet"

    def __init__(
        self,
        model_name: str = DEFAULT_BIREFNET_MODEL,
        device: str = "auto",
        model: Any = None,
        resolution: Optional[Tuple[int, int]] = None,
        max_inference_side: Any = DEFAULT_BIREFNET_MAX_SIDE,
        cpu_fallback_max_side: Any = DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
        memory_cleanup: bool = True,
        fallback_to_cpu: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if model_name != DEFAULT_BIREFNET_MODEL:
            raise ValueError("Only BiRefNet_dynamic is supported")
        self.requested_device = str(device)
        super().__init__(model_name=model_name, device=resolve_device(device))
        self.max_inference_side = parse_birefnet_max_side(max_inference_side)
        self.cpu_fallback_max_side = parse_birefnet_max_side(cpu_fallback_max_side)
        self.memory_cleanup = bool(memory_cleanup)
        self.fallback_to_cpu = bool(fallback_to_cpu)
        self.fallback_count = 0
        parsed_resolution = parse_birefnet_resolution(resolution)
        if (
            parsed_resolution is not None
            and any(int(value) % 32 for value in parsed_resolution)
        ):
            raise ValueError(
                "an explicit resolution for BiRefNet_dynamic must be divisible by 32; "
                "use --birefnet-resolution auto to pad arbitrary source dimensions"
            )
        self.dynamic_resolution = parsed_resolution is None
        self.resolution = (
            None
            if self.dynamic_resolution
            else parsed_resolution
        )
        self._model = model
        # A backend owns one large model/allocator context.  Serializing calls
        # prevents two API threads from lazily loading duplicate weights or
        # running concurrent MPS forwards that multiply activation peaks.
        self._inference_lock = threading.RLock()
        self._transform_cache: Dict[Tuple[int, int], Any] = {}
        self._dynamic_transform: Any = None
        self.last_resolution: Optional[Tuple[int, int]] = None
        self.last_content_size: Optional[Tuple[int, int]] = None
        # Once an accelerator OOM has forced a CPU retry, keep using the CPU
        # fallback cap for the rest of this backend's lifetime.  Otherwise the
        # next image in a batch would silently return to an unbounded dynamic
        # shape (or an explicit fixed shape) and recreate the memory spike.
        self._cpu_fallback_active = False
        self._failed_reason: Optional[str] = None

    def _cpu_retry_max_side(self) -> Optional[int]:
        """Return the most conservative cap for an accelerator→CPU retry."""

        requested = self.max_inference_side
        fallback = self.cpu_fallback_max_side
        if requested is None:
            return fallback
        if fallback is None:
            return requested
        return min(int(requested), int(fallback))

    @property
    def model(self) -> Any:
        if self._failed_reason is not None:
            raise RuntimeError(
                "BiRefNet backend is unavailable after a previous failure: "
                f"{self._failed_reason}"
            )
        if self._model is None:
            try:
                self._model = _load_birefnet_model(self.model_name, self.device)
            except Exception as first_error:
                # Loading can itself exhaust MPS/CUDA (for example when the
                # model is moved from its temporary CPU state dict).  The
                # inference-time fallback in ``segment`` cannot help in that
                # case because no model object exists yet, so retry directly
                # on CPU here.  Keep this path exception-chain free to avoid
                # retaining a partially allocated model through traceback
                # frames after the caller catches the error.
                can_fallback = (
                    self.device in {"mps", "cuda"}
                    and self.fallback_to_cpu
                    and _is_out_of_memory_error(first_error)
                )
                if not can_fallback:
                    raise
                error_message = str(first_error)
                accelerator_device = self.device
                _clear_exception_traceback(first_error)
                del first_error
                self._model = None
                _release_torch_memory(accelerator_device)
                try:
                    self.device = "cpu"
                    self._cpu_fallback_active = True
                    self._model = _load_birefnet_model(self.model_name, "cpu")
                    self.fallback_count += 1
                except Exception as fallback_error:
                    fallback_message = str(fallback_error)
                    _clear_exception_traceback(fallback_error)
                    del fallback_error
                    self._model = None
                    self._transform_cache.clear()
                    self._dynamic_transform = None
                    self._failed_reason = fallback_message
                    _release_torch_memory("cpu")
                    raise RuntimeError(
                        "BiRefNet accelerator model loading ran out of memory and "
                        f"CPU fallback also failed: {fallback_message} "
                        f"(original: {error_message})"
                    ) from None
        return self._model

    def _transform_for(self, resolution: Tuple[int, int]) -> Any:
        key = (int(resolution[0]), int(resolution[1]))
        if key not in self._transform_cache:
            try:
                from torchvision import transforms
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "BiRefNet requires torchvision; install requirements.txt"
                ) from exc
            transform = transforms.Compose(
                [
                    transforms.Resize(key),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )
            self._transform_cache[key] = transform
        return self._transform_cache[key]

    def _normalize_transform(self) -> Any:
        """Build the no-resize transform used by dynamic checkpoints."""

        if self._dynamic_transform is not None:
            return self._dynamic_transform
        try:
            from torchvision import transforms
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BiRefNet requires torchvision; install requirements.txt"
            ) from exc
        self._dynamic_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        return self._dynamic_transform

    @property
    def transform(self) -> Any:
        """Return the active preprocessing transform for introspection."""

        if self.dynamic_resolution:
            return self._normalize_transform()
        resolution = self.resolution or DEFAULT_BIREFNET_RESOLUTION
        return self._transform_for(resolution)

    def _inference_resolution(self, source_size: Tuple[int, int]) -> Tuple[int, int]:
        if self.resolution is not None:
            if self._cpu_fallback_active:
                return _bounded_birefnet_resolution(
                    self.resolution,
                    self._cpu_retry_max_side(),
                )
            return self.resolution
        return _dynamic_birefnet_resolution(
            source_size,
            max_side=(
                self._cpu_retry_max_side()
                if self._cpu_fallback_active
                else self.max_inference_side
            ),
        )

    def _dynamic_inputs(
        self,
        source: Image.Image,
        *,
        max_side: Optional[int],
    ) -> tuple[Any, Tuple[int, int], Tuple[int, int], Image.Image]:
        """Build a padded dynamic input and retain its unpadded content size."""

        content_hw, padded_hw = _dynamic_birefnet_sizes(source.size, max_side=max_side)
        content_height, content_width = content_hw
        if (content_width, content_height) != source.size:
            # LANCZOS is materially cleaner around phone glass and chrome than
            # letting a tensor interpolation implicitly resize the RGB image.
            model_source = source.resize(
                (content_width, content_height),
                resample=Image.Resampling.LANCZOS,
            )
        else:
            model_source = source
        inputs = self._normalize_transform()(model_source).unsqueeze(0)
        pad_height = max(0, padded_hw[0] - content_height)
        pad_width = max(0, padded_hw[1] - content_width)
        if pad_height or pad_width:
            import torch.nn.functional as F

            inputs = F.pad(inputs, (0, pad_width, 0, pad_height), mode="replicate")
        return inputs, padded_hw, content_hw, model_source

    def _move_model_to(self, device: str) -> Any:
        """Move a loaded model and update metadata, tolerating test doubles."""

        try:
            import torch
        except ImportError:  # pragma: no cover - model loading needs torch
            torch = None  # type: ignore[assignment]
        model = self.model
        if hasattr(model, "to") and torch is not None:
            moved = model.to(torch.device(device))
            if moved is not None:
                model = moved
                self._model = moved
        self.device = str(device)
        return model

    def _run_model(self, model: Any, inputs: Any, torch: Any) -> Any:
        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            return model(inputs)

    def segment(self, image: Image.Image) -> Image.Image:
        source = _prepare_image(image)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - model loading already needs it
            raise RuntimeError("BiRefNet requires torch; install requirements.txt") from exc
        # Keep all accelerator tensors scoped to this method.  In particular,
        # a list of auxiliary BiRefNet predictions can otherwise survive until
        # Python's next GC cycle and make an accelerator memory footprint climb
        # across a batch.
        inputs = None
        model_inputs = None
        result = None
        prediction = None
        alpha = None
        # Keep a local model reference only for the duration of this call.  In
        # particular, explicitly clearing it in ``finally`` prevents an
        # exception traceback from retaining a second reference to a large
        # accelerator-backed module while a batch continues with the next
        # image.
        model = None
        model_source = source
        self._inference_lock.acquire()
        try:
            # Resolve/load the model before constructing the input tensor.  A
            # lazy load may itself OOM and switch the backend to CPU; loading
            # first lets the active fallback cap govern the very first input
            # instead of briefly allocating the accelerator-sized tensor.
            model = self.model
            if self.dynamic_resolution:
                inputs, resolution, crop_size, model_source = self._dynamic_inputs(
                    source,
                    max_side=(
                        self._cpu_retry_max_side()
                        if self._cpu_fallback_active
                        else self.max_inference_side
                    ),
                )
            else:
                resolution = self._inference_resolution(source.size)
                inputs = self._transform_for(resolution)(source).unsqueeze(0)
                crop_size = None
            self.last_resolution = tuple(int(v) for v in resolution)
            self.last_content_size = tuple(int(v) for v in crop_size) if crop_size else source.size[::-1]
            try:
                model_inputs = inputs.to(torch.device(self.device))
                # Once the device copy succeeds, the CPU preprocessing tensor
                # is redundant.  Drop it before the forward pass so a large
                # source cannot briefly occupy both host and accelerator
                # buffers.
                inputs = None
                result = self._run_model(model, model_inputs, torch)
            except (RuntimeError, MemoryError) as first_error:
                # MPS/CUDA report allocator exhaustion as RuntimeError.  Move
                # the already-loaded model to CPU and retry at a bounded shape;
                # this avoids terminating a long designer batch while
                # preserving the original resolution whenever it fits.
                can_fallback = (
                    self.device in {"mps", "cuda"}
                    and self.fallback_to_cpu
                    and _is_out_of_memory_error(first_error)
                )
                if not can_fallback:
                    raise
                # Drop references before asking the allocator to trim.  A
                # caught exception's traceback can otherwise keep the failed
                # accelerator input alive until the end of this method.
                error_message = str(first_error)
                accelerator_device = self.device
                _clear_exception_traceback(first_error)
                del first_error
                result = None
                model_inputs = None
                inputs = None
                _release_torch_memory(self.device)
                try:
                    model = self._move_model_to("cpu")
                    self._cpu_fallback_active = True
                    # Moving parameters off an accelerator does not
                    # necessarily trim its allocator immediately; clear the
                    # old device after the move while its identity is known.
                    _release_torch_memory(accelerator_device)
                    if self.dynamic_resolution:
                        (
                            inputs,
                            resolution,
                            crop_size,
                            model_source,
                        ) = self._dynamic_inputs(
                            source,
                            max_side=self._cpu_retry_max_side(),
                        )
                        self.last_resolution = tuple(int(v) for v in resolution)
                        self.last_content_size = tuple(int(v) for v in crop_size)
                    else:
                        # Explicit input sizes can be large. If the accelerator cannot
                        # hold it, honor the CPU fallback cap instead of
                        # recreating the same 2K/4K tensor on the host.
                        resolution = _bounded_birefnet_resolution(
                            resolution,
                            self._cpu_retry_max_side(),
                        )
                        self.last_resolution = tuple(int(v) for v in resolution)
                        # ``inputs`` was deliberately dropped above before
                        # trimming the failed accelerator allocation.  The
                        # old path forgot to rebuild it for fixed-resolution
                        # checkpoints, so every static-model OOM fallback
                        # failed with ``NoneType has no attribute 'to'``.
                        # Recreate the small CPU tensor from the original RGB
                        # source just as in the first attempt.
                        inputs = self._transform_for(resolution)(source).unsqueeze(0)
                        crop_size = None
                    model_inputs = inputs.to(torch.device("cpu"))
                    inputs = None
                    result = self._run_model(model, model_inputs, torch)
                    self.fallback_count += 1
                except Exception as fallback_error:
                    # Do not chain the fallback exception: its traceback frame
                    # contains ``model``/tensor locals and would keep those
                    # allocations alive until the caller releases the wrapped
                    # exception.  The textual cause is retained for the user.
                    fallback_message = str(fallback_error)
                    _clear_exception_traceback(fallback_error)
                    del fallback_error
                    self._model = None
                    self._transform_cache.clear()
                    self._dynamic_transform = None
                    self._failed_reason = fallback_message
                    _release_torch_memory("cpu")
                    raise RuntimeError(
                        "BiRefNet accelerator inference ran out of memory and CPU "
                        f"fallback also failed: {fallback_message} (original: {error_message})"
                    ) from None
            # The model no longer needs its input once forward has returned;
            # release it before allocating the decoded/upsampled alpha.
            model_inputs = None
            inputs = None
            # Extract just the highest-resolution prediction before decoding.
            # BiRefNet may return a nested list/tuple containing auxiliary
            # predictions; dropping the container immediately releases those
            # extra accelerator tensors before CPU interpolation allocates its
            # output mask.
            prediction = _birefnet_output_tensor(result)
            result = None
            alpha = _birefnet_alpha(prediction, source.size, crop_size=crop_size)
            # The CPU-owned NumPy mask is now independent of the logits; drop
            # the selected tensor before constructing the full RGBA image.
            prediction = None
            rgba = _rgba_with_alpha(source, alpha)
            alpha = None
            return rgba
        finally:
            # Explicit deletes matter on MPS/CUDA because the allocator can
            # retain storage for tensors that remain reachable through a result
            # list.
            del result, prediction, model_inputs, inputs, alpha, model_source, model
            if self.memory_cleanup:
                _release_torch_memory(self.device)
            self._inference_lock.release()

    def close(self) -> None:
        """Release model/transform references held by this backend."""

        self._inference_lock.acquire()
        try:
            device = self.device
            self._model = None
            self._transform_cache.clear()
            self._dynamic_transform = None
            self.last_resolution = None
            self.last_content_size = None
            self._failed_reason = None
            self._cpu_fallback_active = False
            _release_torch_memory(device)
        finally:
            self._inference_lock.release()


def create_segmentation_backend(
    backend: str = "birefnet",
    *,
    model_name: Optional[str] = None,
    device: str = "auto",
    birefnet_resolution: Optional[Tuple[int, int]] = None,
    birefnet_max_side: Any = DEFAULT_BIREFNET_MAX_SIDE,
    birefnet_cpu_fallback_max_side: Any = DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
    birefnet_memory_cleanup: bool = True,
    birefnet_fallback_to_cpu: bool = True,
) -> SegmentationBackend:
    """Construct the sole supported model without running inference."""

    if backend != "birefnet":
        raise ValueError("Only BiRefNet_dynamic is supported")
    if model_name not in (None, DEFAULT_BIREFNET_MODEL):
        raise ValueError("Only BiRefNet_dynamic is supported")
    return BiRefNetBackend(
        device=device, resolution=birefnet_resolution,
        max_inference_side=birefnet_max_side,
        cpu_fallback_max_side=birefnet_cpu_fallback_max_side,
        memory_cleanup=birefnet_memory_cleanup,
        fallback_to_cpu=birefnet_fallback_to_cpu,
    )


@dataclass(frozen=True)
class NormalizationInfo:
    """Flattened pipeline diagnostics suitable for JSON metadata."""

    support_removed: bool
    support_confidence: float
    rotation_degrees: float
    crop_box: Tuple[int, int, int, int]
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    backend: str = "birefnet"
    model: str = DEFAULT_BIREFNET_MODEL
    device: str = "auto"
    angle_degrees: Optional[float] = None
    support_threshold_hits: int = 0
    support_detected: bool = False
    support_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    support_edge_reach: float = 0.0
    support_axis_residual: float = 0.0
    model_input_size: Optional[Tuple[int, int]] = None


# Compatibility helpers ----------------------------------------------------
#
# Earlier prototypes exposed these names from this module.  Keep them as thin
# adapters around geometry_postprocess so existing notebooks/scripts continue
# to work while there remains only one implementation of the geometry logic.


def apply_support_cut(
    alpha: Any,
    cut: Optional[SupportCut],
    min_confidence: float = 0.35,
) -> tuple[np.ndarray, bool]:
    """Return ``(clipped_alpha, applied)`` for the legacy API."""

    clipped = clip_support(alpha, cut, confidence_floor=min_confidence)
    applied = bool(cut is not None and cut.confidence >= float(min_confidence))
    return clipped, applied


def estimate_deskew_angle(mask: Any, max_angle: float = 45.0) -> float:
    """Return the correction angle used by the legacy API."""

    angle = estimate_long_angle(_alpha_array(mask))
    if angle is None:
        return 0.0
    target = 0.0 if abs(float(angle)) <= 45.0 else (90.0 if angle >= 0 else -90.0)
    correction = float(angle) - target
    return correction if abs(correction) <= float(max_angle) else 0.0


def rotate_premultiplied(
    rgb: Any,
    alpha: Any,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separately rotated RGB/alpha for the legacy API."""

    rgb_array = np.asarray(rgb)
    if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
        raise ValueError("rgb must have shape HxWx3")
    rgba = np.dstack((rgb_array, _alpha_array(alpha, (rgb_array.shape[1], rgb_array.shape[0]))))
    rotated = rotate_alpha_safe(rgba, angle_degrees)
    return rotated[..., :3], rotated[..., 3]


def normalize_rgba(
    image: Image.Image,
    alpha: Any,
    *,
    geometry_threshold: int = 32,
    crop_threshold: int = 8,
    remove_support: bool = True,
    deskew: bool = True,
    tight_crop: bool = True,
    support_min_confidence: float = 0.55,
    max_deskew_degrees: float = 45.0,
) -> tuple[Image.Image, NormalizationInfo]:
    """Normalize an already-segmented alpha without running a model.

    This helper is useful when comparing BiRefNet_dynamic masks or writing a custom
    backend.  It deliberately uses the same geometry path as ``process_file``.
    """

    source = _prepare_image(image)
    input_size = tuple(int(v) for v in source.size)
    # Materialize one owned RGBA buffer and hand ownership to ``process_array``
    # so it does not immediately make a second full-size copy.
    rgba = np.array(_rgba_with_alpha(source, alpha), dtype=np.uint8, copy=True)
    # The owned NumPy buffer is now the only full-resolution representation
    # needed by the geometry stage.  Drop the PIL source promptly; retaining
    # both RGB and RGBA copies is avoidable on large inputs.
    del source
    config = GeometryConfig(
        geometry_threshold=int(geometry_threshold),
        crop_threshold=int(crop_threshold),
        max_deskew_degrees=float(max_deskew_degrees),
    )
    output_array, geometry = process_array(
        rgba,
        config,
        confidence_floor=float(support_min_confidence),
        remove_support=remove_support,
        deskew=deskew,
        tight_crop_enabled=tight_crop,
        copy=False,
    )
    del rgba
    output = Image.fromarray(output_array)
    support = geometry.support
    return output, NormalizationInfo(
        support_removed=bool(geometry.support_applied),
        support_confidence=float(support.confidence if support is not None else 0.0),
        rotation_degrees=float(geometry.rotation_degrees),
        crop_box=tuple(int(v) for v in geometry.crop_box),
        input_size=input_size,
        output_size=tuple(int(v) for v in output.size),
        backend="alpha-only",
        model="",
        device="",
        model_input_size=None,
        angle_degrees=(None if geometry.angle_degrees is None else float(geometry.angle_degrees)),
        support_threshold_hits=int(support.threshold_hits if support is not None else 0),
        support_detected=bool(support is not None),
        support_line=(
            None
            if support is None
            else (tuple(float(v) for v in support.p), tuple(float(v) for v in support.q))
        ),
        support_edge_reach=float(support.edge_reach if support is not None else 0.0),
        support_axis_residual=float(support.axis_residual if support is not None else 0.0),
    )


def _backend_metadata(backend: Any) -> tuple[str, str, str]:
    return (
        str(getattr(backend, "name", backend.__class__.__name__.lower())),
        str(getattr(backend, "model_name", "")),
        str(getattr(backend, "device", "auto")),
    )


def _close_backend_safely(backend: Any, device: str = "") -> None:
    """Close an internally-owned backend and trim accelerator caches.

    ``normalize_image`` accepts an injected backend for callers that want to
    reuse a model across a batch.  When it creates a backend itself, however,
    that backend is one-shot and must not keep a model/allocator reservation
    alive after segmentation returns (or after an inference error).  Cleanup
    is best-effort and deliberately never masks the original exception.
    """

    if backend is None:
        _release_torch_memory(device)
        return
    active_device = str(getattr(backend, "device", device))
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    del close
    _release_torch_memory(active_device)


def normalize_image(
    image: Image.Image,
    *,
    backend: Optional[Any] = None,
    backend_name: str = "birefnet",
    model_name: Optional[str] = None,
    device: str = "auto",
    birefnet_resolution: Optional[Tuple[int, int]] = None,
    birefnet_max_side: Any = DEFAULT_BIREFNET_MAX_SIDE,
    birefnet_cpu_fallback_max_side: Any = DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
    birefnet_memory_cleanup: bool = True,
    birefnet_fallback_to_cpu: bool = True,
    geometry_config: Optional[GeometryConfig] = None,
    max_geometry_pixels: Any = DEFAULT_MAX_GEOMETRY_PIXELS,
    confidence_floor: float = 0.55,
    remove_support: bool = True,
    deskew: bool = True,
    tight_crop: bool = True,
) -> tuple[Image.Image, NormalizationInfo]:
    """Segment and geometrically normalize one in-memory image.

    Passing a backend instance transfers no ownership; the caller can reuse
    it for a batch and is responsible for calling ``close`` when finished.
    If a backend name is used instead, this one-shot helper creates and closes
    the backend around the segmentation call so model/allocator state does not
    leak across repeated standalone invocations.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    # Build/validate the geometry configuration before touching the backend or
    # applying EXIF orientation.  ``Image.open`` exposes ``size`` without
    # decoding pixels, so an oversized source is rejected before even the
    # first full-resolution RGB copy can be created.
    # Dynamic BiRefNet already caps its model input, but restoring a mask and
    # running contour/rotation analysis at an accidental panorama's native
    # size can still allocate many full-frame arrays.  Reject such inputs
    # before lazy model loading; callers can pass a custom GeometryConfig or
    # ``max_geometry_pixels=None`` when they explicitly accept that risk.
    config = geometry_config or GeometryConfig(max_geometry_pixels=max_geometry_pixels)
    if remove_support or deskew or tight_crop:
        validate_geometry_size(image.size, config.max_geometry_pixels)
    source = _prepare_image(image)
    input_size = tuple(int(v) for v in source.size)
    # Accept either a backend instance or the convenient string form.  The
    # latter keeps the programmatic API aligned with the command line while
    # preserving dependency injection for tests and custom models.
    if isinstance(backend, str):
        backend_name = backend
        backend = None
    owns_backend = backend is None
    active_backend: Any = None
    try:
        active_backend = (
            backend
            if backend is not None
            else create_segmentation_backend(
                backend_name,
                model_name=model_name,
                device=device,
                birefnet_resolution=birefnet_resolution,
                birefnet_max_side=birefnet_max_side,
                birefnet_cpu_fallback_max_side=birefnet_cpu_fallback_max_side,
                birefnet_memory_cleanup=birefnet_memory_cleanup,
                birefnet_fallback_to_cpu=birefnet_fallback_to_cpu,
            )
        )
        if hasattr(active_backend, "segment"):
            segmented = active_backend.segment(source)
        elif callable(active_backend):
            segmented = active_backend(source)
        else:
            raise TypeError("backend must provide segment(image) or be callable")
        # Capture diagnostics while the backend is still open.  BiRefNet's
        # ``close`` method clears ``last_resolution`` along with its model.
        backend_label, backend_model, backend_device = _backend_metadata(active_backend)
        model_input_size = getattr(active_backend, "last_resolution", None)
        if model_input_size is not None:
            model_input_size = tuple(int(v) for v in model_input_size)
    except BaseException:
        if owns_backend:
            _close_backend_safely(active_backend, device)
            active_backend = None
        raise
    if owns_backend:
        # The model is no longer needed once the alpha-producing call has
        # returned.  Close it before allocating full-resolution RGBA/geometry
        # buffers, which also makes repeated standalone calls bounded.
        _close_backend_safely(active_backend, backend_device)
        active_backend = None
    # Extract only the alpha channel and compose it with the source RGB once.
    # The previous implementation materialized an intermediate full-size RGBA
    # image for array results and then rebuilt another RGBA image to restore the
    # original RGB pixels.  On multi-megapixel inputs that transient double
    # allocation is avoidable and can compete with the model's allocator.
    alpha_value: Any
    if isinstance(segmented, Image.Image):
        if segmented.size != source.size:
            raise RuntimeError(
                f"backend output size {segmented.size} does not match input size {source.size}"
            )
        if "A" in segmented.getbands():
            alpha_value = segmented.getchannel("A")
        elif segmented.mode == "L":
            # A PIL L image is a valid alpha-only backend result.  Treating it
            # as RGB and converting to RGBA would incorrectly make every
            # pixel opaque.
            alpha_value = segmented
        else:
            raise RuntimeError("segmentation backend returned PIL RGB without alpha")
    else:
        array = np.asarray(segmented)
        if array.ndim == 2:
            alpha_value = np.array(array, copy=True)
        elif array.ndim == 3 and array.shape[-1] >= 4:
            if (int(array.shape[1]), int(array.shape[0])) != source.size:
                raise RuntimeError(
                    f"backend output size {(array.shape[1], array.shape[0])} "
                    f"does not match input size {source.size}"
                )
            # Own just the single alpha plane before releasing a potentially
            # large RGBA backend result.  This keeps the model output and the
            # geometry buffer from being live at the same time for longer than
            # necessary on multi-megapixel inputs.
            alpha_value = np.array(array[..., 3], copy=True)
        else:
            raise RuntimeError("segmentation backend must return RGBA or alpha")
        del array
    rgba = _rgba_with_alpha(source, alpha_value)
    # ``segmented`` can be a full-resolution RGBA image returned by a custom
    # backend.  It is no longer needed once its alpha has been copied into the
    # new image.
    del segmented, alpha_value
    # Take ownership of a single contiguous RGBA buffer.  ``process_array``
    # can then operate in place and avoid a redundant full-frame copy while
    # retaining its safe copying default for external callers.
    rgba_array = np.array(rgba, dtype=np.uint8, copy=True)
    del rgba
    output_array, geometry = process_array(
        rgba_array,
        config,
        confidence_floor=confidence_floor,
        remove_support=remove_support,
        deskew=deskew,
        tight_crop_enabled=tight_crop,
        copy=False,
    )
    del rgba_array
    output = Image.fromarray(output_array)
    support = geometry.support
    info = NormalizationInfo(
        support_removed=bool(geometry.support_applied),
        support_confidence=float(support.confidence if support is not None else 0.0),
        rotation_degrees=float(geometry.rotation_degrees),
        crop_box=tuple(int(v) for v in geometry.crop_box),
        input_size=input_size,
        output_size=tuple(int(v) for v in output.size),
        backend=backend_label,
        model=backend_model,
        device=backend_device,
        model_input_size=model_input_size,
        angle_degrees=(None if geometry.angle_degrees is None else float(geometry.angle_degrees)),
        support_threshold_hits=int(support.threshold_hits if support is not None else 0),
        support_detected=bool(support is not None),
        support_line=(
            None
            if support is None
            else (tuple(float(v) for v in support.p), tuple(float(v) for v in support.q))
        ),
        support_edge_reach=float(support.edge_reach if support is not None else 0.0),
        support_axis_residual=float(support.axis_residual if support is not None else 0.0),
    )
    return output, info


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    backend: str = "birefnet",
    model_name: Optional[str] = None,
    device: str = "auto",
    birefnet_resolution: Optional[Tuple[int, int]] = None,
    birefnet_max_side: Any = DEFAULT_BIREFNET_MAX_SIDE,
    birefnet_cpu_fallback_max_side: Any = DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
    birefnet_memory_cleanup: bool = True,
    birefnet_fallback_to_cpu: bool = True,
    max_geometry_pixels: Any = DEFAULT_MAX_GEOMETRY_PIXELS,
    geometry_threshold: int = 32,
    crop_threshold: int = 8,
    max_deskew_degrees: float = 45.0,
    support_min_confidence: float = 0.55,
    no_support_removal: bool = False,
    no_deskew: bool = False,
    no_tight_crop: bool = False,
    require_support_border: bool = False,
    metadata_path: Optional[Path] = None,
    backend_instance: Optional[Any] = None,
) -> NormalizationInfo:
    """Process one file, saving a transparent PNG and optional metadata."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different paths; source files are never overwritten")
    if output_path.suffix and output_path.suffix.lower() != ".png":
        raise ValueError("output must be a PNG path ending in .png")
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if metadata_path.resolve() == output_path.resolve():
            raise ValueError("metadata and output must be different paths")
        if metadata_path.resolve() == input_path.resolve():
            raise ValueError("metadata and input must be different paths; source files are never overwritten")
    config = GeometryConfig(
        geometry_threshold=int(geometry_threshold),
        crop_threshold=int(crop_threshold),
        max_deskew_degrees=float(max_deskew_degrees),
        require_support_border=bool(require_support_border),
        max_geometry_pixels=max_geometry_pixels,
    )
    # Keep the decoder's lazy source image inside the file context and let
    # ``normalize_image`` apply EXIF orientation exactly once.  The previous
    # eager ``exif_transpose(...).convert('RGB')`` created a second full-size
    # RGB copy before the backend could downscale it for BiRefNet.
    with Image.open(input_path) as opened:
        output, info = normalize_image(
            opened,
            backend=backend_instance,
            backend_name=backend,
            model_name=model_name,
            device=device,
            geometry_config=config,
            birefnet_resolution=birefnet_resolution,
            birefnet_max_side=birefnet_max_side,
            birefnet_cpu_fallback_max_side=birefnet_cpu_fallback_max_side,
            birefnet_memory_cleanup=birefnet_memory_cleanup,
            birefnet_fallback_to_cpu=birefnet_fallback_to_cpu,
            max_geometry_pixels=max_geometry_pixels,
            confidence_floor=float(support_min_confidence),
            remove_support=not no_support_removal,
            deskew=not no_deskew,
            tight_crop=not no_tight_crop,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, format="PNG")
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(asdict(info), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return info
