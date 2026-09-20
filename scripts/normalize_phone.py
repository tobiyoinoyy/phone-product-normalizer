#!/usr/bin/env python3
"""Create a transparent cutout using rembg only."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image


def remove_background(
    input_path: Path,
    output_path: Path,
    model: str = "u2net",
    alpha_matting: bool = True,
) -> None:
    """Run rembg and save its RGBA result without any additional processing."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError("rembg is required; install dependencies from requirements.txt") from exc

    image = Image.open(input_path).convert("RGB")
    source = io.BytesIO()
    image.save(source, format="PNG")

    session = new_session(model)
    result = remove(
        source.getvalue(),
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=6,
    )

    # Save the model output as-is. In particular, do not threshold, crop,
    # rotate, trim supports, or composite onto a new background here.
    rgba = Image.open(io.BytesIO(result)).convert("RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="u2net", help="rembg model name (default: u2net)")
    parser.add_argument("--no-alpha-matting", action="store_true", help="Disable rembg alpha matting")
    args = parser.parse_args()

    remove_background(
        args.input,
        args.output,
        model=args.model,
        alpha_matting=not args.no_alpha_matting,
    )
    print(f"saved {args.output} (rembg model={args.model})")


if __name__ == "__main__":
    main()
