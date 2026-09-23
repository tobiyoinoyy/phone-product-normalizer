"""Fixed per-face phone coordinates using body geometry and an approved crop.

Profiles contain numeric geometry only. This stage does not identify semantic
landmarks, modify local defects, or validate a texture against a 3D model's UVs.
Images supplied to ``align_to_profile`` must already have the profile orientation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
from PIL import Image

from scripts.geometry_postprocess import DEFAULT_MAX_GEOMETRY_PIXELS, validate_geometry_size


FACES = ("front", "back", "left", "right", "top", "bottom")
BUILTIN_PROFILE = "iphone12pro-user-crop-v1"
BODY_BBOX_METHOD = "alpha128_largest_component_median_central70pct_long_edges"
_ROOT_KEYS = {"schema_version", "profile_id", "body_bbox_method", "faces"}
_FACE_KEYS = {"reference_size", "reference_body_bbox", "crop_xyxy", "orientation_quarter_turns"}
_MAX_EXTENT_GAP = 1
_MAX_EXTENT_SCALE_CHANGE = .01


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _sequence(value: Any, length: int, label: str) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    return list(value)


def _validate_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and copy; reject unknown fields and incomplete face profiles."""
    if not isinstance(profile, Mapping) or set(profile) != _ROOT_KEYS:
        raise ValueError(f"profile fields must be exactly {sorted(_ROOT_KEYS)}")
    if _integer(profile["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported UV profile schema_version; expected 1")
    profile_id = profile["profile_id"]
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a nonempty string")
    if profile["body_bbox_method"] != BODY_BBOX_METHOD:
        raise ValueError(f"body_bbox_method must be {BODY_BBOX_METHOD!r}")
    faces = profile["faces"]
    if not isinstance(faces, Mapping) or set(faces) != set(FACES):
        raise ValueError(f"profile must contain exactly these six faces: {', '.join(FACES)}")
    cleaned = {}
    for face in FACES:
        rec = faces[face]
        if not isinstance(rec, Mapping) or set(rec) != _FACE_KEYS:
            raise ValueError(f"{face} fields must be exactly {sorted(_FACE_KEYS)}")
        size = [_integer(v, f"{face}.reference_size") for v in
                _sequence(rec["reference_size"], 2, f"{face}.reference_size")]
        if min(size) <= 0:
            raise ValueError(f"{face}.reference_size must be positive")
        box = [_finite(v, f"{face}.reference_body_bbox") for v in
               _sequence(rec["reference_body_bbox"], 4, f"{face}.reference_body_bbox")]
        crop = [_integer(v, f"{face}.crop_xyxy") for v in
                _sequence(rec["crop_xyxy"], 4, f"{face}.crop_xyxy")]
        for name, bounds in (("reference_body_bbox", box), ("crop_xyxy", crop)):
            if not (0 <= bounds[0] < bounds[2] <= size[0]
                    and 0 <= bounds[1] < bounds[3] <= size[1]):
                raise ValueError(f"{face}.{name} must be a positive rectangle inside reference_size")
        turns = _integer(rec["orientation_quarter_turns"], f"{face}.orientation_quarter_turns")
        if turns not in (-3, -2, -1, 0, 1, 2, 3):
            raise ValueError(f"{face}.orientation_quarter_turns must be between -3 and 3")
        cleaned[face] = {"reference_size": size, "reference_body_bbox": box,
                         "crop_xyxy": crop, "orientation_quarter_turns": turns}
    return {"schema_version": 1, "profile_id": profile_id,
            "body_bbox_method": BODY_BBOX_METHOD, "faces": cleaned}


def load_profile(name_or_path: Union[str, os.PathLike], *,
                 max_geometry_pixels: Any = DEFAULT_MAX_GEOMETRY_PIXELS) -> Dict[str, Any]:
    """Load the builtin name or a JSON file; return a strictly validated copy."""
    value = os.fspath(name_or_path)
    path = (Path(__file__).with_name("profiles") / f"{BUILTIN_PROFILE}.json"
            if value == BUILTIN_PROFILE else Path(value).expanduser())
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    validated = _validate_profile(profile)
    for rec in validated["faces"].values():
        validate_geometry_size(tuple(rec["reference_size"]), max_geometry_pixels)
    return validated


def _face_record(face: str, profile: Mapping[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(face, str) or face.strip().lower() not in FACES:
        raise ValueError(f"face must be one of {', '.join(FACES)}")
    key = face.strip().lower()
    validated = _validate_profile(profile)
    return key, validated, validated["faces"][key]


def orient_candidate(image: Image.Image, face: str, profile: Mapping[str, Any]) -> Image.Image:
    """Apply whole quarter turns without interpolation; positive means CCW."""
    _, _, rec = _face_record(face, profile)
    rgba = image.convert("RGBA")
    turns = rec["orientation_quarter_turns"] % 4
    if turns == 0:
        return rgba.copy()
    operation = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                 3: Image.Transpose.ROTATE_270}[turns]
    return rgba.transpose(operation)


def body_bbox(image: Image.Image) -> list:
    """Return the alpha128 main body's median long-edge bbox (exclusive max).

    Use the largest 4-connected alpha component and the central 70% of its long
    axis. Median short-axis edges reduce the influence of buttons/camera bumps.
    This geometric heuristic is not semantic body or landmark recognition.
    """
    from scipy import ndimage

    mask = np.asarray(image.convert("RGBA"))[:, :, 3] >= 128
    labels, count = ndimage.label(mask)
    if not count:
        raise ValueError("no foreground at alpha >= 128")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(sizes.argmax())
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if y1 - y0 >= x1 - x0:
        rows = mask[y0 + int(.15 * (y1-y0)):y0 + int(.85 * (y1-y0))]
        rows = rows[rows.any(axis=1)]
        if len(rows):  # A 1-pixel body can have an empty central interval.
            x0 = float(np.median(np.argmax(rows, axis=1)))
            x1 = float(mask.shape[1] - np.median(np.argmax(rows[:, ::-1], axis=1)))
    else:
        cols = mask[:, x0 + int(.15 * (x1-x0)):x0 + int(.85 * (x1-x0))].T
        cols = cols[cols.any(axis=1)]
        if len(cols):
            y0 = float(np.median(np.argmax(cols, axis=1)))
            y1 = float(mask.shape[0] - np.median(np.argmax(cols[:, ::-1], axis=1)))
    return [float(x0), float(y0), float(x1), float(y1)]


def _bbox_matrix(source: list, target: list) -> np.ndarray:
    sx = (target[2] - target[0]) / (source[2] - source[0])
    sy = (target[3] - target[1]) / (source[3] - source[1])
    return np.array([[sx, 0., target[0] - sx * source[0]],
                     [0., sy, target[1] - sy * source[1]], [0., 0., 1.]])


def _warp_rgba(image: Image.Image, matrix: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Bilinear premultiplied-alpha sampling, matching the accepted experiment."""
    from scipy.ndimage import affine_transform

    src = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.
    src[:, :, :3] *= src[:, :, 3:4]
    inv = np.linalg.inv(matrix)
    linear = inv[:2, :2][::-1, ::-1]
    offset = inv[:2, 2][::-1]
    channels = [affine_transform(src[:, :, k], linear, offset=offset,
                output_shape=(size[1], size[0]), order=1, mode="constant",
                cval=0., prefilter=False) for k in range(4)]
    out = np.stack(channels, axis=-1)
    out[:, :, :3] = np.divide(out[:, :, :3], out[:, :, 3:4],
                             out=np.zeros_like(out[:, :, :3]),
                             where=out[:, :, 3:4] > 1e-7)
    return Image.fromarray(np.round(np.clip(out, 0., 1.) * 255).astype(np.uint8))


def _render(image: Image.Image, matrix: np.ndarray, size: Tuple[int, int]) -> Tuple[Image.Image, str]:
    if (np.array_equal(matrix[:2, :2], np.eye(2))
            and np.array_equal(matrix[2], [0., 0., 1.])
            and all(float(v).is_integer() for v in matrix[:2, 2])):
        out = Image.new("RGBA", size)
        # No alpha-composite mask: copy every original RGBA byte, including matte.
        out.paste(image, tuple(int(v) for v in matrix[:2, 2]))
        return out, "none_integer_copy"
    return _warp_rgba(image, matrix, size), "bilinear_premultiplied_alpha"


def _alpha_bbox(image: Image.Image, threshold: int) -> Any:
    ys, xs = np.nonzero(np.asarray(image)[:, :, 3] >= threshold)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]


def _border_gaps(box: Any, size: Tuple[int, int]) -> Dict[str, int]:
    if box is None:
        return {"left": size[0], "top": size[1], "right": size[0], "bottom": size[1]}
    return dict(zip(("left", "top", "right", "bottom"),
                    (box[0], box[1], size[0]-box[2], size[1]-box[3])))


def _extent_correction(image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Only correct an actual transparent border totaling <=1 px per axis."""
    size = image.size
    solid = _alpha_bbox(image, 128)
    visible = _alpha_bbox(image, 1)
    gaps = _border_gaps(visible, size)
    correction = np.eye(3)
    info = {"applied": False, "maximum_gap_px_per_axis": _MAX_EXTENT_GAP,
            "maximum_scale_change": _MAX_EXTENT_SCALE_CHANGE,
            "transparent_border_before": gaps, "axes": {}}
    if solid is None:
        info["reason"] = "no_alpha128_foreground_after_crop"
        return correction, info
    for axis, names in enumerate((("left", "right"), ("top", "bottom"))):
        label = "x" if axis == 0 else "y"
        actual_gap = sum(gaps[name] for name in names)
        if not actual_gap:
            continue
        low, high = solid[axis], solid[axis+2]-1
        solid_gap = low + size[axis]-1-high
        if actual_gap > _MAX_EXTENT_GAP or solid_gap > _MAX_EXTENT_GAP or high <= low:
            info["axes"][label] = {"applied": False, "reason": "gap_exceeds_one_pixel_or_degenerate_extent"}
            continue
        scale = (size[axis]-1) / (high-low)
        if scale-1 > _MAX_EXTENT_SCALE_CHANGE + 1e-12:
            info["axes"][label] = {"applied": False, "reason": "scale_change_exceeds_one_percent"}
            continue
        correction[axis, axis] = scale
        correction[axis, 2] = -scale * low
        info["axes"][label] = {"applied": True, "scale": float(scale),
                                "translation": float(correction[axis, 2])}
        info["applied"] = True
    info["matrix"] = correction.tolist()
    return correction, info


def align_to_profile(oriented_candidate: Image.Image, face: str,
                     profile: Mapping[str, Any], *,
                     max_geometry_pixels: Any = DEFAULT_MAX_GEOMETRY_PIXELS) -> Tuple[Image.Image, Dict[str, Any]]:
    """Fit body edges in reference coordinates, then apply the approved crop.

    A detected one-pixel transparent outer border may receive a <=1% global
    correction. Its final image is sampled directly from the original candidate,
    never from the already warped probe. Larger gaps remain flagged for review.
    """
    key, validated, rec = _face_record(face, profile)
    # Check dimensions before convert/copy or reference-canvas allocation.
    validate_geometry_size(oriented_candidate.size, max_geometry_pixels)
    validate_geometry_size(tuple(rec["reference_size"]), max_geometry_pixels)
    candidate = oriented_candidate.convert("RGBA")
    source_box = body_bbox(candidate)
    reference_size = tuple(rec["reference_size"])
    initial_matrix = _bbox_matrix(source_box, rec["reference_body_bbox"])
    reference_canvas, resampling = _render(candidate, initial_matrix, reference_size)
    crop = rec["crop_xyxy"]
    result = reference_canvas.crop(crop)
    crop_matrix = np.array([[1., 0., -crop[0]], [0., 1., -crop[1]], [0., 0., 1.]])
    initial_final_matrix = crop_matrix @ initial_matrix
    final_matrix = initial_final_matrix
    correction, extent_info = _extent_correction(result)
    if extent_info["applied"]:
        proposed_matrix = correction @ initial_final_matrix
        corrected, proposed_sampling = _render(candidate, proposed_matrix, result.size)
        before = sum(_border_gaps(_alpha_bbox(result, 1), result.size).values())
        after = sum(_border_gaps(_alpha_bbox(corrected, 1), corrected.size).values())
        if after < before:
            result, final_matrix, resampling = corrected, proposed_matrix, proposed_sampling
        else:
            extent_info["applied"] = False
            extent_info["reason"] = "bounded_correction_did_not_reduce_transparent_border"
    remaining = _border_gaps(_alpha_bbox(result, 1), result.size)
    metadata = {
        "profile_id": validated["profile_id"], "face": key,
        "method": "body_bbox_profile", "input_size": list(candidate.size),
        "reference_size": list(reference_size), "output_size": list(result.size),
        "candidate_body_bbox": source_box, "reference_body_bbox": rec["reference_body_bbox"],
        "body_bbox_method": BODY_BBOX_METHOD, "crop_xyxy": crop,
        "orientation": "caller_supplied_profile_orientation",
        "orientation_quarter_turns": rec["orientation_quarter_turns"],
        "candidate_to_reference_matrix": initial_matrix.tolist(),
        "candidate_to_final_matrix_before_extent_correction": initial_final_matrix.tolist(),
        "candidate_to_final_matrix": final_matrix.tolist(),
        "outer_extent_correction": extent_info,
        "final_alpha_bbox_128": _alpha_bbox(result, 128),
        "final_alpha_bbox_nonzero": _alpha_bbox(result, 1),
        "remaining_transparent_border_px": remaining,
        "resampling": resampling,
        "source_resampling_passes_in_final_image": 0 if resampling == "none_integer_copy" else 1,
        "reference_pixels_painted": False, "content_reconstructed": False,
        "semantic_anchor_validation": "not_performed", "uv_fit_validation": "not_performed",
        "needs_manual_review": True,
        "review_reasons": ["body_geometry_only_no_semantic_anchor_or_uv_fit_validation"],
    }
    if any(remaining.values()):
        metadata["review_reasons"].append("transparent_outer_border_remains")
    if metadata["final_alpha_bbox_128"] is None:
        metadata["review_reasons"].append("no_alpha128_foreground_after_crop")
    return result, metadata
