# Cuts the before and after images out of the example chat screenshots in docs/screenshots/, for the
# homepage's before/after cards. Run again after the screenshots change; check the boxes still fit.
#   python3 web/scripts/crop-examples.py
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[2]
out = root / 'web/src/assets/examples'
# Top edge of image 1 and image 2 in each screenshot; both are 414 px squares starting at x = 10.
tops = {
    'use-remove-object': (99, 665),
    'use-remove-person': (99, 665),
    'use-remove-text': (113, 679),
}
inset, side = 5, 414  # stay clear of the rounded corners
for name, (before, after) in tops.items():
    shot = Image.open(root / 'docs/screenshots' / f'{name}.jpg').convert('RGB')
    for label, top in (('before', before), ('after', after)):
        box = (10 + inset, top + inset, 10 + side - inset, top + side - inset)
        shot.crop(box).save(out / f'{name}-{label}.jpg', quality=92)
# The single-image cards: just the picture, so it fills the card (the chat around it made it tiny).
# Boxes are (left, top, right, bottom) in the screenshot; versions keeps its v1/v2 button row.
pictures = {
    'use-poster': (10, 114, 422, 717),
    'use-banner': (9, 113, 425, 397),
    'versions': (13, 42, 584, 658),
}
for name, (l, t, r, b) in pictures.items():
    shot = Image.open(root / 'docs/screenshots' / f'{name}.jpg').convert('RGB')
    shot.crop((l + 3, t + 3, r - 3, b - 3)).save(out / f'{name}-image.jpg', quality=92)
print('wrote', out)
