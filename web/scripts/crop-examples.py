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
print('wrote', out)
