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
        self.workflow_root = self.root / "workflows"
        self.workflow_root.mkdir(parents=True, exist_ok=True)

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
        return self.save_upload_image(self.decode_upload(data))

    @staticmethod
    def decode_upload(data: bytes) -> Image.Image:
        """The uploaded image, upright and in RGB(A). Raises Pillow's errors when it is not an image."""
        import io
        from PIL import ImageOps

        image = Image.open(io.BytesIO(data))
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        return image

    def save_upload_image(self, image: Image.Image) -> str:
        return self.save(image, {"kind": "upload"}, self.new_id("up"))

    def meta(self, result_id: str) -> dict:
        path = self.path(result_id).with_suffix(".json")
        return json.loads(path.read_text()) if path.exists() else {}

    def save_workflow_job(self, value: dict) -> None:
        """Atomically persist a multi-step job checkpoint for crash recovery."""
        job_id = str(value.get("id", ""))
        if not job_id.replace("_", "").isalnum():
            raise ValueError("Invalid workflow job id.")
        target = self.workflow_root / f"{job_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(target)

    def workflow_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.workflow_root.glob("job_*.json")):
            try:
                value = json.loads(path.read_text())
                if isinstance(value, dict):
                    jobs.append(value)
            except (OSError, json.JSONDecodeError):
                continue
        return jobs

    def delete_workflow_job(self, job_id: str) -> None:
        if not job_id.replace("_", "").isalnum():
            raise ValueError("Invalid workflow job id.")
        path = self.workflow_root / f"{job_id}.json"
        if path.exists():
            path.unlink()
