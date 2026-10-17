from __future__ import annotations

import os
import sys
from pathlib import Path

from egsearch.media import (
    AUDIO_WINDOW_SECONDS,
    VIDEO_WINDOW_SECONDS,
    extract_video_frames,
    load_audio,
)
from egsearch.paths import MODEL_ID, NATIVE_DIM, model_cache
from egsearch.quiet import install as install_quiet

TASK_PROMPTS = {
    "search": "SearchQuery",
    "code": "CodeRetrieval",
    "qa": "QuestionAnswering",
}

TASK_PREFIXES = {
    "search": "task: search result | query: ",
    "code": "task: code retrieval | query: ",
    "qa": "task: question answering | query: ",
}

_MAX_IMAGE_EDGE = 1280


def _gb(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.1f}GB"


def _mps_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "mps allocated" in text


def _fit_image(source):
    from PIL import Image

    if isinstance(source, Image.Image):
        image = source.convert("RGB")
    else:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
    width, height = image.size
    edge = max(width, height)
    if edge <= _MAX_IMAGE_EDGE:
        return image
    scale = _MAX_IMAGE_EDGE / edge
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


def choose_device(torch) -> tuple[str, object]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return "cuda", dtype
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


class GemmaEmbedder:
    def __init__(self, root: Path) -> None:
        install_quiet()
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(6)
        device, dtype = choose_device(torch)
        if device == "mps":
            recommended = torch.mps.recommended_max_memory()
            cap = 28 * 1024**3
            fraction = min(1.0, cap / recommended) if recommended else 0.4
            torch.mps.set_per_process_memory_fraction(float(fraction))
            print(
                f"mps memory cap {cap / 1024**3:.0f}GB "
                f"({fraction:.2f} of {recommended / 1024**3:.1f}GB recommended)",
                file=sys.stderr,
                flush=True,
            )
        cache = model_cache(root)
        cache.mkdir(parents=True, exist_ok=True)
        cached = _weights_cached(cache)
        if cached:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            print(f"Downloading {MODEL_ID} into {cache} (about 1.5 GB).", file=sys.stderr)
        self.model = SentenceTransformer(
            MODEL_ID,
            device=device,
            cache_folder=str(cache),
            local_files_only=cached,
            model_kwargs={"torch_dtype": dtype},
        )
        self.dim = NATIVE_DIM

    def _encode(self, payload, **kwargs):
        import numpy as np
        import torch

        with torch.inference_mode():
            raw = self.model.encode(
                payload,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs,
            )
        vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim == 2:
            if vector.shape[0] != 1:
                raise RuntimeError(f"expected one embedding, got shape {tuple(vector.shape)}")
            vector = vector[0]
        if vector.ndim != 1:
            raise RuntimeError(f"unexpected embedding shape {tuple(vector.shape)}")
        if not np.isfinite(vector).all():
            raise RuntimeError(
                "EmbeddingGemma 2 returned NaN or Inf. It must run in bfloat16 or float32, never float16."
            )
        self._release_gpu()
        return vector

    def encode(self, payload, **kwargs):
        try:
            return self._encode(payload, **kwargs)
        except RuntimeError as exc:
            if not _mps_oom(exc):
                raise
            print("MPS out of memory. Released the cache and retrying.", file=sys.stderr, flush=True)
            self._release_gpu()
            return self._encode(payload, **kwargs)

    def encode_many(self, payloads, **kwargs):
        try:
            return self._encode_many(payloads, **kwargs)
        except RuntimeError as exc:
            if not _mps_oom(exc):
                raise
            self._release_gpu()
            if len(payloads) <= 1:
                print("MPS out of memory. Released the cache and retrying.", file=sys.stderr, flush=True)
                return self._encode_many(payloads, **kwargs)
            mid = max(1, len(payloads) // 2)
            print(
                f"MPS out of memory on {len(payloads)}. Splitting into {mid} and {len(payloads) - mid}.",
                file=sys.stderr,
                flush=True,
            )
            return self.encode_many(payloads[:mid], **kwargs) + self.encode_many(payloads[mid:], **kwargs)

    def _encode_many(self, payloads, **kwargs):
        import numpy as np
        import torch

        with torch.inference_mode():
            raw = self.model.encode(
                payloads,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs,
            )
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(payloads):
            raise RuntimeError(f"expected {len(payloads)} embeddings, got {array.shape[0]}")
        if not np.isfinite(array).all():
            raise RuntimeError(
                "EmbeddingGemma 2 returned NaN or Inf. It must run in bfloat16 or float32, never float16."
            )
        self._release_gpu()
        return [row for row in array]

    def embed_batch(self, items: list[dict]) -> list:
        counts: dict[str, int] = {}
        for item in items:
            kind = str(item.get("kind") or "?")
            counts[kind] = counts.get(kind, 0) + 1
        before = self._gpu_bytes()
        vectors: list = []
        start = 0
        try:
            while start < len(items):
                kind = items[start]["kind"]
                limit = _BATCH_LIMITS[kind]
                end = start + 1
                while end < len(items) and items[end]["kind"] == kind and end - start < limit:
                    end += 1
                vectors.extend(self._embed_kind(kind, items[start:end]))
                start = end
            return vectors
        finally:
            self._release_gpu()
            after = self._gpu_bytes()
            summary = " ".join(f"{kind}x{count}" for kind, count in counts.items())
            print(
                f"embed {summary} gpu {_gb(before)} -> {_gb(after)}",
                file=sys.stderr,
                flush=True,
            )

    def _embed_kind(self, kind: str, items: list[dict]):
        if kind == "text":
            payloads = [f"title: {item['title']} | text: {item['body']}" for item in items]
            return self.encode_many(payloads, batch_size=len(payloads))
        if kind == "image":
            return self._encode_images([{"image": _fit_image(item["path"])} for item in items])
        if kind == "pdf":
            payloads = []
            for item in items:
                body = " ".join(str(item.get("excerpt") or "").split())[:1500]
                image = _fit_image(item["image"])
                if body:
                    payloads.append({"text": f"title: {item['title']} | text: {body} <|image|>", "image": image})
                else:
                    payloads.append({"image": image})
            return self._encode_images(payloads)
        if kind == "video":
            if len(items) != 1:
                raise RuntimeError("video batches are one clip at a time")
            frames = [_fit_image(frame_path) for frame_path in items[0]["frames"]]
            return [self.embed_video_frames(frames)]
        if kind == "audio":
            if len(items) != 1:
                raise RuntimeError("audio batches are one clip at a time")
            return [self.embed_audio_array(load_audio(Path(items[0]["wav"])))]
        raise RuntimeError(f"unsupported batch kind {kind}")

    def _encode_images(self, payloads: list):
        try:
            return self.encode_many(payloads, batch_size=len(payloads))
        except Exception:
            return [self.encode(payload) for payload in payloads]

    def _gpu_bytes(self) -> int:
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return 0
        return int(torch.mps.driver_allocated_memory())

    def _release_gpu(self) -> None:
        import gc
        import torch

        gc.collect()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def embed_text_document(self, title: str, body: str):
        return self.encode(f"title: {title} | text: {body}")

    def embed_text_query(self, text: str, task: str):
        prompt = TASK_PROMPTS[task]
        return self.encode(text, prompt_name=prompt)

    def embed_image(self, path: Path):
        return self.encode({"image": _fit_image(path)})

    def embed_page(self, title: str, text: str, image):
        body = " ".join(text.split())
        if len(body) > 1500:
            body = body[:1500]
        fitted = _fit_image(image)
        if not body:
            return self.encode({"image": fitted})
        return self.encode({"text": f"title: {title} | text: {body} <|image|>", "image": fitted})

    def embed_video_frames(self, frames):
        return self.encode(
            {"video": frames},
            processing_kwargs={
                "video": {
                    "max_frames": len(frames),
                    "fps": 1,
                    "overflow_strategy": "truncate",
                }
            },
        )

    def embed_audio_array(self, samples):
        return self.encode({"audio": samples})

    def embed_video(self, path: Path):
        frames = extract_video_frames(path, start=0, duration=VIDEO_WINDOW_SECONDS)
        return self.embed_video_frames(frames)

    def embed_audio(self, path: Path):
        return self.embed_audio_array(load_audio(path, start=0, seconds=AUDIO_WINDOW_SECONDS))

    def embed_query(
        self,
        *,
        text: str = "",
        image: Path | None = None,
        video: Path | None = None,
        audio: Path | None = None,
        task: str = "search",
    ):
        media = {}
        if image is not None:
            media["image"] = str(image)
        if video is not None:
            media["video"] = extract_video_frames(video)
        if audio is not None:
            media["audio"] = load_audio(audio)
        if not media:
            if not text.strip():
                raise ValueError("a search needs text, an image, a video, or audio")
            return self.embed_text_query(text, task)
        pieces = []
        if text.strip():
            pieces.append(TASK_PREFIXES[task] + text.strip())
        if image is not None:
            pieces.append("<|image|>")
        if video is not None:
            pieces.append("<|video|>")
        if audio is not None:
            pieces.append("<|audio|>")
        if pieces:
            media["text"] = " ".join(pieces)
        return self.encode(media)


_BATCH_LIMITS = {
    "text": 32,
    "image": 1,
    "pdf": 1,
    "video": 1,
    "audio": 1,
}


def _weights_cached(cache: Path) -> bool:
    snapshots = cache / "models--google--embeddinggemma-2" / "snapshots"
    return snapshots.is_dir() and any(snapshots.glob("*/model.safetensors"))
