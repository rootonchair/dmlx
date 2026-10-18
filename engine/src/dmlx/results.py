"""Result store: every generated image gets an id, a PNG on disk, and its full request embedded."""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from PIL import Image, PngImagePlugin

from dmlx import paths


class ResultStore:
    def __init__(self, root: Path | None = None):
        self.root = root or paths.results_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_id(prefix: str = "img") -> str:
        return f"{prefix}_{time.strftime('%Y%m%d')}_{secrets.token_hex(6)}"

    def path(self, result_id: str) -> Path:
        if not result_id.replace("_", "").isalnum():
            raise KeyError(result_id)
        path = self.root / f"{result_id}.png"
        if not path.exists():
            raise KeyError(result_id)
        return path

    def exists(self, result_id: str) -> bool:
        try:
            self.path(result_id)
            return True
        except KeyError:
            return False

    def save(self, image: Image.Image, meta: dict, result_id: str | None = None) -> str:
        result_id = result_id or self.new_id()
        info = PngImagePlugin.PngInfo()
        info.add_text("dmlx", json.dumps(meta))
        info.add_text("parameters", meta.get("prompt", ""))
        image.save(self.root / f"{result_id}.png", pnginfo=info)
        (self.root / f"{result_id}.json").write_text(json.dumps(meta, indent=2))
        return result_id

    def save_upload(self, data: bytes) -> str:
        """Store an uploaded source/reference image as a result so later turns can reference it."""
        import io

        image = Image.open(io.BytesIO(data))
        image.load()
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        return self.save(image, {"kind": "upload"}, self.new_id("up"))

    def meta(self, result_id: str) -> dict:
        path = self.path(result_id).with_suffix(".json")
        return json.loads(path.read_text()) if path.exists() else {}
