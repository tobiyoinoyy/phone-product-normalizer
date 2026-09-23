"""Precise phone crop geometry; never invokes a segmentation model.

Use BiRefNet_dynamic RGBA as input. The positioning mask is deliberately
separate from display alpha: a higher threshold determines the crop box but
does not erase soft pixels inside that box. This remains crop/deskew geometry,
not camera calibration or UV-template registration.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


from .geometry_postprocess import (
    DEFAULT_MAX_GEOMETRY_PIXELS,
    GeometryConfig,
    clip_support,
    estimate_long_angle,
    find_support_cut,
    rotate_alpha_safe,
    validate_geometry_size,
)


def _threshold(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer between 1 and 254")
    if not 1 <= int(value) <= 254:
        raise ValueError(f"{name} must be between 1 and 254")
    return int(value)


def _largest(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _keep_connected_soft_alpha(
    rgba: np.ndarray, seed_threshold: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove disconnected alpha islands; keep all soft alpha on the subject.

    The principal high-alpha component selects the subject. Flood connectivity
    is then evaluated at alpha > 0, so a legitimate soft edge is retained at
    its original value. A faint bridge can retain adjacent noise: this is a
    deliberate conservative limit and is not solved by eroding the phone.
    """
    result = rgba.copy()
    alpha = result[..., 3]
    principal = _largest(alpha >= seed_threshold)
    if not np.any(principal):
        raise ValueError(
            f"No subject reaches alpha {seed_threshold}; cannot locate a reliable crop"
        )
    _, labels = cv2.connectedComponents((alpha > 0).astype(np.uint8), connectivity=8)
    # Every high-threshold seed belongs to the same >0 component.
    subject_label = int(labels[principal][0])
    removed = (alpha > 0) & (labels != subject_label)
    removed_count = int(np.count_nonzero(removed))
    removed_alpha_sum = int(alpha[removed].sum(dtype=np.uint64))
    alpha[removed] = 0
    result[..., :3][alpha == 0] = 0
    return result, {
        "removed_pixels": removed_count,
        "removed_alpha_sum": removed_alpha_sum,
        "preserved_soft_pixels": int(
            np.count_nonzero((alpha > 0) & (alpha < seed_threshold))
        ),
    }


def normalize_candidate(
    rgba: np.ndarray,
    *,
    crop_threshold: int = 128,
    geometry_threshold: int = 32,
    confidence_floor: float = 0.55,
    remove_support: bool = True,
    deskew: bool = True,
    max_deskew_degrees: float = 45.0,
    max_geometry_pixels: Any = DEFAULT_MAX_GEOMETRY_PIXELS,
    require_support_border: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a cropped RGBA candidate and JSON-compatible diagnostics.

    The caller retains ownership of its input. No resize, perspective warp,
    content synthesis, or model inference is performed. Output alpha can change
    through the existing conservative support cut, disconnected-island cleanup,
    interpolated rotation, and loss of pixels *outside* the chosen crop box.
    Surviving pixels inside the box are not hard-thresholded.
    """
    crop_threshold = _threshold(crop_threshold, "crop_threshold")
    geometry_threshold = _threshold(geometry_threshold, "geometry_threshold")
    if not math.isfinite(float(confidence_floor)) or not 0 <= confidence_floor <= 1:
        raise ValueError("confidence_floor must be finite and between 0 and 1")
    if not math.isfinite(float(max_deskew_degrees)) or max_deskew_degrees < 0:
        raise ValueError("max_deskew_degrees must be finite and nonnegative")
    source = np.asarray(rgba)
    if source.ndim != 3 or source.shape[2] != 4 or source.dtype != np.uint8:
        raise ValueError("rgba must be a uint8 array with shape (height, width, 4)")
    # Parse the source-pixel guard before any owning image/geometry allocation.
    config = GeometryConfig(
        geometry_threshold=geometry_threshold,
        crop_threshold=crop_threshold,
        max_deskew_degrees=float(max_deskew_degrees),
        max_geometry_pixels=max_geometry_pixels,
        require_support_border=bool(require_support_border),
    )
    validate_geometry_size(
        (int(source.shape[1]), int(source.shape[0])), config.max_geometry_pixels
    )
    work = source.copy()
    alpha_sum_before_support = int(work[..., 3].sum(dtype=np.uint64))
    support = find_support_cut(work[..., 3], config) if remove_support else None
    work[..., 3] = clip_support(
        work[..., 3], support, confidence_floor=confidence_floor,
        margin_fraction=config.contact_margin_fraction,
        min_margin=config.min_contact_margin,
    )
    support_applied = bool(
        support is not None and support.confidence >= confidence_floor
    )
    support_alpha_removed = alpha_sum_before_support - int(
        work[..., 3].sum(dtype=np.uint64)
    )
    work, initial_cleanup = _keep_connected_soft_alpha(work, geometry_threshold)
    angle = estimate_long_angle(work[..., 3], geometry_threshold)
    rotation = 0.0
    if deskew and angle is not None:
        target = 0.0 if abs(angle) <= 45.0 else (90.0 if angle >= 0 else -90.0)
        candidate_rotation = float(angle - target)
        if abs(candidate_rotation) <= max_deskew_degrees:
            rotation = candidate_rotation
    rotated = rotate_alpha_safe(work, rotation)
    rotated, rotated_cleanup = _keep_connected_soft_alpha(rotated, geometry_threshold)

    positioning = _largest(rotated[..., 3] >= crop_threshold)
    ys, xs = np.where(positioning)
    if not len(xs):
        raise ValueError(f"No subject reaches crop alpha {crop_threshold}")
    box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    x0, y0, x1, y1 = box
    # Pure slicing is essential: do NOT apply the positioning mask to alpha.
    output = rotated[y0:y1, x0:x1].copy()
    full_alpha = rotated[..., 3]
    output_alpha = output[..., 3]
    removed_by_crop = int(np.count_nonzero(full_alpha)) - int(
        np.count_nonzero(output_alpha)
    )
    total_alpha = int(full_alpha.sum(dtype=np.uint64))
    crop_alpha_removed = total_alpha - int(output_alpha.sum(dtype=np.uint64))
    notes = [
        "Precise cropping is not UV-template registration; output dimensions can vary.",
        "The higher-alpha crop can omit legitimate faint pixels outside its box.",
        "Connected faint noise can remain; it is not removed by shrinking the phone.",
        "Perspective, camera/side protrusions, and occlusion are not corrected.",
    ]
    if support_applied:
        notes.append("The original heuristic support cut was applied and needs visual review.")
    return output, {
        "algorithm_version": "precise-geometry-v1",
        "input_size": [int(source.shape[1]), int(source.shape[0])],
        "output_size": [int(output.shape[1]), int(output.shape[0])],
        "rotated_size": [int(rotated.shape[1]), int(rotated.shape[0])],
        "crop_box": list(box),
        "crop_box_coordinate_space": "expanded rotated image; x1/y1 exclusive",
        "crop_threshold": crop_threshold,
        "geometry_threshold": geometry_threshold,
        "max_geometry_pixels": config.max_geometry_pixels,
        "require_support_border": config.require_support_border,
        "angle_degrees": None if angle is None else float(angle),
        "rotation_degrees": rotation,
        "support_detected": support is not None,
        "support_applied": support_applied,
        "support_confidence": float(support.confidence) if support is not None else 0.0,
        "support_alpha_removed": support_alpha_removed,
        "support_line": None if support is None else [list(support.p), list(support.q)],
        "cleanup_before_rotation": initial_cleanup,
        "cleanup_after_rotation": rotated_cleanup,
        "nonzero_pixels_outside_crop": removed_by_crop,
        "alpha_sum_outside_crop": crop_alpha_removed,
        "alpha_coverage_outside_crop_percent": 100.0 * crop_alpha_removed / max(total_alpha, 1),
        "soft_pixels_preserved_inside_crop": int(
            np.count_nonzero((output_alpha > 0) & (output_alpha < crop_threshold))
        ),
        "alpha_inside_crop_matches_rotated_image": bool(
            np.array_equal(output_alpha, full_alpha[y0:y1, x0:x1])
        ),
        "notes": notes,
    }
