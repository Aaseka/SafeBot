import io
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from safebot.pipeline.heuristics import check_qr, check_scam_text
from safebot.pipeline.images import ImageModerator, dhash, hamming

cv2 = pytest.importorskip("cv2")


def _png(img: Image.Image, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _qr_image(payload: str) -> Image.Image:
    qr = cv2.QRCodeEncoder.create().encode(payload)
    qr = cv2.resize(qr, (qr.shape[1] * 8, qr.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
    qr = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(qr).convert("RGB")


def _text_image(text: str) -> Image.Image:
    img = Image.new("RGB", (1100, 160), "white")
    try:
        font = ImageFont.load_default(size=40)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((20, 50), text, fill="black", font=font)
    return img


def _photo_like(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((400, 300), Image.Resampling.BICUBIC)


def test_dhash_survives_resize_and_jpeg():
    img = _photo_like(1)
    copy = Image.open(io.BytesIO(_png(img.resize((180, 135)), "JPEG", quality=60)))
    assert hamming(dhash(img), dhash(copy)) <= 6
    assert hamming(dhash(img), dhash(_photo_like(2))) > 12


@pytest.mark.asyncio
async def test_discord_login_qr_detected():
    mod = ImageModerator(nsfw_model=None, ocr=False)
    report = await mod.analyze(_png(_qr_image("https://discord.com/ra/AbCdEf123456")))
    assert report.qr_payloads == ["https://discord.com/ra/AbCdEf123456"]
    assert max(s.score for s in check_qr(report.qr_payloads)) >= 0.95


def test_qr_lookalike_and_benign():
    assert max(s.score for s in check_qr(["https://dlscord.gift/claim"])) >= 0.8
    assert check_qr(["https://github.com/Aaseka/SafeBot"]) == []


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
@pytest.mark.asyncio
async def test_ocr_scam_screenshot():
    mod = ImageModerator(nsfw_model=None, ocr=True)
    report = await mod.analyze(_png(_text_image("GIVEAWAY: send 0.1 BTC, get 0.2 BTC back")))
    assert "btc" in report.ocr_text.lower()
    assert check_scam_text(report.ocr_text) is not None


def test_scam_text_negative():
    assert check_scam_text("screenshot of my minecraft base lol") is None


@pytest.mark.asyncio
async def test_oversized_and_garbage_rejected():
    mod = ImageModerator(nsfw_model=None, ocr=False)
    assert await mod.analyze(b"not an image") is None
    assert await mod.analyze(b"0" * (9 * 1024 * 1024)) is None
