"""Side-by-side strip of step-count images with labels, small enough to track (JPEG, 384 px tiles).
Usage: uv run python side_by_side.py out.jpg label1=img1.png label2=img2.png ..."""
import sys

from PIL import Image, ImageDraw

out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
tile, pad = 384, 28
strip = Image.new("RGB", (tile * len(items), tile + pad), "white")
draw = ImageDraw.Draw(strip)
for i, (label, path) in enumerate(items):
    strip.paste(Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), (i * tile, pad))
    draw.text((i * tile + 8, 6), label, fill="black")
strip.save(out, quality=82)
