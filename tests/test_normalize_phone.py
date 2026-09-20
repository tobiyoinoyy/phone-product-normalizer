#!/usr/bin/env python3
"""Offline smoke test for the rembg-only pipeline."""

from pathlib import Path
import io
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

# Allow the smoke test to be run exactly as documented from the project
# directory (``python tests/test_normalize_phone.py``).
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from PIL import Image, ImageDraw, ImageFilter

from scripts.normalize_phone import remove_background


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "synthetic.jpg"
        output = root / "result.png"

        image = Image.new("RGB", (320, 480), (242, 242, 242))
        ImageDraw.Draw(image).rounded_rectangle((100, 55, 220, 385), radius=18, fill=(35, 40, 45))
        image.save(source, quality=95)

        # Fake rembg output with a fractional alpha edge. The production code
        # must preserve this alpha and the original image dimensions.
        mocked = image.convert("RGBA")
        alpha = Image.new("L", image.size, 0)
        ImageDraw.Draw(alpha).rounded_rectangle((100, 55, 220, 385), radius=18, fill=255)
        alpha = alpha.filter(ImageFilter.GaussianBlur(1.0))
        mocked.putalpha(alpha)
        payload = io.BytesIO()
        mocked.save(payload, format="PNG")

        fake_rembg = SimpleNamespace(
            new_session=lambda model: {"model": model},
            remove=lambda *args, **kwargs: payload.getvalue(),
        )
        with patch.dict(sys.modules, {"rembg": fake_rembg}):
            remove_background(source, output)

        result = Image.open(output)
        alpha_values = result.getchannel("A").getextrema()
        assert result.mode == "RGBA"
        assert result.size == image.size
        assert alpha_values[0] < alpha_values[1]
        assert len(set(result.getchannel("A").getdata())) > 2

    print("rembg-only smoke test passed")


if __name__ == "__main__":
    main()
