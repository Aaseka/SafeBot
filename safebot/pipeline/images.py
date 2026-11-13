"""Image stage: NSFW model, OCR (text in images goes through the text checks),
QR code decoding, and a perceptual hash so mod decisions on an image stick.

Everything heavy runs in a worker thread so the event loop stays free.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs
MAX_BYTES = int(os.getenv("SAFEBOT_IMAGE_MAX_BYTES", str(8 * 1024 * 1024)))
NSFW_LABELS = {"nsfw", "porn", "hentai", "sexy", "explicit"}
OCR_MAX_SIDE = 1600

try:
    import cv2

    cv2.setNumThreads(1)
except ImportError:  # QR decoding is optional
    cv2 = None


@dataclass
class ImageReport:
    dhash: str
    nsfw: float | None = None
    ocr_text: str = ""
    qr_payloads: list[str] = field(default_factory=list)


def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash: survives resizing, recompression and small edits."""
    gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] < px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _open(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.seek(0)  # first frame of GIFs / animated WebP
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)


class ImageModerator:
    def __init__(self, nsfw_model: str | None, ocr: bool = True, concurrency: int = 1):
        self.nsfw_model = nsfw_model or None
        self.ocr = ocr and shutil.which("tesseract") is not None
        self.qr = cv2 is not None
        self._pipe = None
        self._sem = asyncio.Semaphore(concurrency)
        if ocr and not self.ocr:
            log.warning("tesseract not found: OCR on images is disabled")

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("image-classification", model=self.nsfw_model)
        return self._pipe

    def _nsfw(self, img: Image.Image) -> float:
        out = self._load()(_downscale(img, 512))
        return max((float(d["score"]) for d in out if d["label"].lower() in NSFW_LABELS), default=0.0)

    def _ocr(self, img: Image.Image) -> str:
        import pytesseract

        try:
            text = pytesseract.image_to_string(_downscale(img, OCR_MAX_SIDE), timeout=10)
        except RuntimeError:  # tesseract timed out
            return ""
        return " ".join(text.split())[:2000]

    def _qr(self, img: Image.Image) -> list[str]:
        arr = np.asarray(_downscale(img, 1200))[:, :, ::-1]  # RGB -> BGR
        detector = cv2.QRCodeDetector()
        try:
            ok, decoded, _, _ = detector.detectAndDecodeMulti(arr)
        except cv2.error:
            return []
        return [d for d in decoded if d] if ok else []

    def _analyze(self, data: bytes, want_nsfw: bool) -> ImageReport:
        img = _open(data)
        report = ImageReport(dhash(img))
        if want_nsfw and self.nsfw_model:
            report.nsfw = self._nsfw(img)
        if self.ocr:
            report.ocr_text = self._ocr(img)
        if self.qr:
            report.qr_payloads = self._qr(img)
        return report

    async def analyze(self, data: bytes, want_nsfw: bool = True) -> ImageReport | None:
        if len(data) > MAX_BYTES:
            return None
        async with self._sem:
            try:
                return await asyncio.to_thread(self._analyze, data, want_nsfw)
            except (OSError, Image.DecompressionBombError, ValueError):
                log.info("Skipping unreadable or oversized image")
                return None

    async def warmup(self) -> None:
        if self.nsfw_model:
            await asyncio.to_thread(self._nsfw, Image.new("RGB", (64, 64)))
