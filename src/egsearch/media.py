from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# EmbeddingGemma 2 shares one 8,192-token context across modalities.
# Video defaults to 140 tokens per frame at 1 fps (about 58 frames max).
# Audio defaults to 25 tokens per second (about 327 seconds max).
VIDEO_FPS = 1.0
VIDEO_WINDOW_SECONDS = 32.0
MAX_VIDEO_WINDOW = 48.0
AUDIO_WINDOW_SECONDS = 120.0
MAX_AUDIO_WINDOW = 300.0


def split_png_blobs(blob: bytes) -> list[bytes]:
    parts = blob.split(PNG_MAGIC)
    return [PNG_MAGIC + part for part in parts[1:] if part]


def validate_windows(audio_window: float, video_window: float) -> None:
    if not 0 < audio_window <= MAX_AUDIO_WINDOW:
        raise ValueError(
            f"audio window must be between 0 and {MAX_AUDIO_WINDOW:g} seconds "
            "so a chunk stays inside the 8192-token context"
        )
    if not 0 < video_window <= MAX_VIDEO_WINDOW:
        raise ValueError(
            f"video window must be between 0 and {MAX_VIDEO_WINDOW:g} seconds at 1 fps "
            "so a chunk stays inside the 8192-token context"
        )


def time_windows(duration: float, window: float, max_chunks: int) -> list[tuple[float, float]]:
    if window <= 0:
        raise ValueError("window must be positive")
    if duration <= 0:
        spans = [(0.0, window)]
    else:
        spans = []
        start = 0.0
        while start < duration - 1.0:
            spans.append((round(start, 3), round(min(window, duration - start), 3)))
            start += window
        if not spans:
            spans = [(0.0, round(min(window, duration), 3))]
    if max_chunks > 0:
        return spans[:max_chunks]
    return spans


def format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def media_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return max(0.0, float(proc.stdout.strip()))
    except ValueError:
        return 0.0


def ffmpeg_to_wav(path: Path, *, start: float = 0.0, seconds: float = AUDIO_WINDOW_SECONDS) -> bytes:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "-c:a",
            "pcm_s16le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode audio {path}: {detail or 'empty output'}")
    return proc.stdout


def wav_bytes_to_f32(blob: bytes):
    import numpy as np
    import wave

    with wave.open(io.BytesIO(blob), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise RuntimeError(f"expected 16-bit PCM wav, got sample width {width}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def load_audio(path: Path, *, start: float = 0.0, seconds: float = AUDIO_WINDOW_SECONDS):
    return wav_bytes_to_f32(ffmpeg_to_wav(path, start=start, seconds=seconds))


def extract_video_frames(
    path: Path,
    *,
    start: float = 0.0,
    duration: float = VIDEO_WINDOW_SECONDS,
    max_frames: int | None = None,
    fps: float = VIDEO_FPS,
):
    from PIL import Image

    frame_budget = max_frames if max_frames is not None else max(1, int(duration * fps))
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(path),
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(frame_budget),
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode video {path}: {detail or 'no frames'}")
    frames = []
    for blob in split_png_blobs(proc.stdout):
        with Image.open(io.BytesIO(blob)) as image:
            frames.append(image.convert("RGB"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {path}")
    return frames


def looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"%PDF-" in handle.read(1024)
    except OSError:
        return False


def pdf_page_count(path: Path) -> int:
    if not looks_like_pdf(path):
        raise RuntimeError(f"{path.name} is not a PDF")
    proc = subprocess.run(
        ["pdfinfo", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"could not read {path.name}")
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise RuntimeError(f"could not read {path.name}")


def pdf_page_text(path: Path, page: int) -> str:
    proc = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-q", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def render_pdf_page(path: Path, page: int):
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "page")
        proc = subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-png",
                "-r",
                "100",
                "-singlefile",
                str(path),
                prefix,
            ],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"could not render {path.name} page {page}")
        with Image.open(prefix + ".png") as image:
            return image.convert("RGB")
