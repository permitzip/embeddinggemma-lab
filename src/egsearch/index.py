from __future__ import annotations

import hashlib
import queue
import shutil
import sys
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from egsearch.media import (
    AUDIO_WINDOW_SECONDS,
    VIDEO_WINDOW_SECONDS,
    extract_video_frames,
    format_clock,
    load_audio,
    media_duration,
    pdf_page_count,
    pdf_page_text,
    render_pdf_page,
    time_windows,
    validate_windows,
)
from egsearch.store import IndexStore, pack_f32

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".csv",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".yml",
    ".yaml",
    ".toml",
    ".sql",
    ".rst",
    ".sh",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
PDF_EXTENSIONS = {".pdf"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "models",
    ".egsearch",
    "__pycache__",
    "node_modules",
    ".hf-cache",
}
MAX_TEXT_CHARS = 4000
TEXT_OVERLAP = 400
SIZE_LIMITS = {
    "text": None,
    "image": 80 * 1024 * 1024,
    "pdf": None,
    "audio": None,
    "video": None,
}


@dataclass(frozen=True)
class ChunkOptions:
    max_chunks: int = 8
    audio_window: float = AUDIO_WINDOW_SECONDS
    video_window: float = VIDEO_WINDOW_SECONDS


@dataclass(frozen=True)
class FoundFile:
    path: Path
    modality: str


def modality_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    return None


def discover(paths: list[Path]) -> list[FoundFile]:
    found: list[FoundFile] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_file():
            modality = modality_for(path)
            if modality is not None:
                found.append(FoundFile(path, modality))
            continue
        if not path.is_dir():
            print(f"skip missing path {path}", file=sys.stderr)
            continue
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            if any(part in SKIP_DIRS for part in child.parts):
                continue
            modality = modality_for(child)
            if modality is not None:
                found.append(FoundFile(child, modality))
    found.sort(key=lambda item: str(item.path))
    return found


def chunk_text(text: str) -> list[str]:
    body = text.strip()
    if not body:
        return []
    if len(body) <= MAX_TEXT_CHARS:
        return [body]
    parts: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + MAX_TEXT_CHARS)
        parts.append(body[start:end])
        if end == len(body):
            break
        start = end - TEXT_OVERLAP
    return parts


def sha256_file(path: Path) -> str:
    """Hash the whole file up to 512 MB.

    Larger media is identified by size, mtime, and both ends, so an in-place
    edit in the middle of a huge file still changes the key.
    """
    digest = hashlib.sha256()
    stat = path.stat()
    size = stat.st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        if size <= 512 * 1024 * 1024:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        else:
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(handle.read(8 * 1024 * 1024))
            handle.seek(max(0, size - 8 * 1024 * 1024))
            digest.update(handle.read(8 * 1024 * 1024))
    return digest.hexdigest()


def content_key(digest: str, options: ChunkOptions) -> str:
    return (
        f"{digest}|chunks={options.max_chunks}"
        f"|audio={options.audio_window:g}|video={options.video_window:g}"
    )


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        raise ValueError("file looks binary")
    return data.decode("utf-8", errors="replace")


def excerpt_for(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= 160:
        return collapsed
    return collapsed[:157] + "..."


def index_files(
    files: list[FoundFile],
    store: IndexStore,
    embedder,
    *,
    max_files: int = 2000,
    options: ChunkOptions | None = None,
    workers: int = 1,
) -> tuple[int, int]:
    chunking = options or ChunkOptions()
    validate_windows(chunking.audio_window, chunking.video_window)
    if len(files) > max_files:
        raise RuntimeError(
            f"found {len(files)} files, above --max-files {max_files}. Pass a narrower folder or raise the limit."
        )
    skipped = 0
    jobs: list[tuple[int, FoundFile, str, int, int]] = []
    total = len(files)
    store_lock = threading.Lock()
    for number, item in enumerate(files, start=1):
        path = item.path
        size = path.stat().st_size
        limit = SIZE_LIMITS[item.modality]
        if limit is not None and size > limit:
            print(f"[{number}/{total}] skip {path} ({size} bytes exceeds {item.modality} limit)", file=sys.stderr)
            skipped += 1
            continue
        digest = content_key(sha256_file(path), chunking)
        try:
            used, available = chunk_plan(item, chunking)
        except Exception as exc:
            print(f"[{number}/{total}] skip {path}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if used == 0:
            print(f"[{number}/{total}] skip empty {path}", file=sys.stderr)
            skipped += 1
            continue
        with store_lock:
            current = store.is_current(str(path), digest, used)
        if current:
            print(f"[{number}/{total}] unchanged {item.modality} {path}", file=sys.stderr)
            skipped += 1
            continue
        if used < available:
            print(
                f"  {path.name}: embedding {used} of {available} chunks. "
                "Raise --max-chunks, or pass 0 for the whole file.",
                file=sys.stderr,
            )
        jobs.append((number, item, digest, size, used))

    indexed = 0
    pool_size = max(1, workers)
    if hasattr(embedder, "embed_batch"):
        done, failed = _pipeline_embed(jobs, embedder, chunking, store, store_lock, total, pool_size)
        return done, skipped + failed

    def embed_job(job: tuple[int, FoundFile, str, int, int]) -> str:
        number, item, digest, size, used = job
        print(
            f"[{number}/{total}] embed {item.modality} {item.path} ({used} chunk{'s' if used != 1 else ''})",
            file=sys.stderr,
        )
        if hasattr(embedder, "embed_batch"):
            pieces = _pieces_batched(item, embedder, chunking)
        else:
            pieces = _pieces(item, embedder, chunking)
        if not pieces:
            return "empty"
        rows = []
        for index, (title, body, vector) in enumerate(pieces):
            values = [float(value) for value in vector]
            rows.append(
                {
                    "path": str(item.path),
                    "chunk_index": index,
                    "modality": item.modality,
                    "title": title,
                    "sha256": digest,
                    "mtime": item.path.stat().st_mtime,
                    "size_bytes": size,
                    "excerpt": excerpt_for(body),
                    "dim": len(values),
                    "embedding": pack_f32(values),
                }
            )
        with store_lock:
            store.replace_file(rows)
        return "indexed"

    if pool_size == 1:
        for job in jobs:
            try:
                indexed += _count_job(embed_job(job))
            except Exception as exc:
                print(f"skip {job[1].path}: {exc}", file=sys.stderr)
                skipped += 1
        return indexed, skipped

    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = [pool.submit(embed_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                indexed += _count_job(future.result())
            except Exception as exc:
                print(f"skip failed file: {exc}", file=sys.stderr)
                skipped += 1
    return indexed, skipped


_PREP_PRIORITY = {"text": 0, "image": 1, "pdf": 2, "audio": 3, "video": 4}


def _pipeline_embed(jobs, embedder, options: ChunkOptions, store: IndexStore, store_lock, total: int, workers: int):
    """Prepare files on CPU and embed whatever is already ready, smallest work first."""
    import heapq

    ready: list = []
    ready_lock = threading.Lock()
    ready_cv = threading.Condition(ready_lock)
    sequence = 0
    indexed = 0
    skipped = 0
    pending: dict[str, dict] = {}
    # Small files are planned first so a large video does not occupy every decoder.
    ordered = sorted(jobs, key=lambda job: (job[3], _PREP_PRIORITY.get(job[1].modality, 9)))

    def note_ready(item) -> None:
        nonlocal sequence
        priority = 0
        if item[0] == "slice" and item[3]:
            priority = _PREP_PRIORITY.get(str(item[3][0].get("kind")), 9)
        with ready_cv:
            heapq.heappush(ready, (priority, sequence, item))
            sequence += 1
            ready_cv.notify()

    def expand_and_prepare(job: tuple[int, FoundFile, str, int, int]) -> None:
        number, item, _digest, _size, used = job
        print(
            f"[{number}/{total}] embed {item.modality} {item.path} ({used} chunk{'s' if used != 1 else ''})",
            file=sys.stderr,
        )
        try:
            plans = _slice_plans(item, options)
        except Exception as exc:
            note_ready(("error", job, 0, None, None, exc))
            return
        if not plans:
            note_ready(("empty", job, 0, None, None, None))
            return
        for plan in plans:
            try:
                specs, cleanup = _prepare_plan(plan)
                note_ready(("slice", job, plan.get("ordinal", 0), specs, cleanup, None))
            except Exception as exc:
                note_ready(("error", job, plan.get("ordinal", 0), None, None, exc))

    def produce() -> None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(expand_and_prepare, ordered))
        with ready_cv:
            heapq.heappush(ready, (999, 0, None))
            ready_cv.notify()

    def consume(message) -> None:
        nonlocal indexed, skipped
        tag, job, _ordinal, specs, cleanup, error = message
        _number, item, digest, size, used = job
        state = pending.setdefault(
            str(item.path),
            {"job": job, "pieces": [], "failed": False, "used": used, "saved": False, "accounted": False},
        )
        if tag in {"error", "empty"} or error is not None or state["failed"]:
            if cleanup is not None:
                cleanup()
            if not state["accounted"]:
                state["failed"] = True
                state["accounted"] = True
                if tag != "empty":
                    print(f"skip {item.path}: {error}", file=sys.stderr)
                skipped += 1
            return
        try:
            vectors = embedder.embed_batch(specs)
        except Exception as exc:
            if cleanup is not None:
                cleanup()
            if not state["accounted"]:
                state["failed"] = True
                state["accounted"] = True
                print(f"skip {item.path}: {exc}", file=sys.stderr)
                skipped += 1
            return
        if cleanup is not None:
            cleanup()
        if len(vectors) != len(specs):
            if not state["accounted"]:
                state["failed"] = True
                state["accounted"] = True
                print(f"skip {item.path}: batch size mismatch", file=sys.stderr)
                skipped += 1
            return
        batch_rows = []
        for spec, vector in zip(specs, vectors):
            index = int(spec["index"])
            title = str(spec["title"])
            excerpt = str(spec["excerpt"])
            state["pieces"].append((index, title, excerpt, vector))
            values = [float(value) for value in vector]
            batch_rows.append(
                {
                    "path": str(item.path),
                    "chunk_index": index,
                    "modality": item.modality,
                    "title": title,
                    "sha256": digest,
                    "mtime": item.path.stat().st_mtime,
                    "size_bytes": size,
                    "excerpt": excerpt_for(excerpt),
                    "dim": len(values),
                    "embedding": pack_f32(values),
                }
            )
        with store_lock:
            store.append_chunks(str(item.path), batch_rows, replace=not state["saved"])
        state["saved"] = True
        print(f"  saved {len(state['pieces'])}/{used} {item.path.name}", file=sys.stderr)
        if len(state["pieces"]) == used and not state["failed"]:
            state["accounted"] = True
            indexed += 1
            del pending[str(item.path)]

    def save_vectors(message, vectors) -> None:
        nonlocal indexed, skipped
        _tag, job, _ordinal, specs, _cleanup, _error = message
        _number, item, digest, size, used = job
        state = pending.setdefault(
            str(item.path),
            {"job": job, "pieces": [], "failed": False, "used": used, "saved": False, "accounted": False},
        )
        if state["failed"] or state.get("accounted"):
            return
        if len(vectors) != len(specs):
            state["failed"] = True
            state["accounted"] = True
            print(f"skip {item.path}: batch size mismatch", file=sys.stderr)
            skipped += 1
            return
        batch_rows = []
        for spec, vector in zip(specs, vectors):
            index = int(spec["index"])
            title = str(spec["title"])
            excerpt = str(spec["excerpt"])
            state["pieces"].append((index, title, excerpt, vector))
            values = [float(value) for value in vector]
            batch_rows.append(
                {
                    "path": str(item.path),
                    "chunk_index": index,
                    "modality": item.modality,
                    "title": title,
                    "sha256": digest,
                    "mtime": item.path.stat().st_mtime,
                    "size_bytes": size,
                    "excerpt": excerpt_for(excerpt),
                    "dim": len(values),
                    "embedding": pack_f32(values),
                }
            )
        with store_lock:
            store.append_chunks(str(item.path), batch_rows, replace=not state["saved"])
        state["saved"] = True
        print(f"  saved {len(state['pieces'])}/{used} {item.path.name}", file=sys.stderr)
        if len(state["pieces"]) == used:
            state["accounted"] = True
            indexed += 1
            del pending[str(item.path)]

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        while True:
            with ready_cv:
                while not ready:
                    ready_cv.wait()
                priority, seq, first = heapq.heappop(ready)
            if first is None:
                break
            batch = [first]
            if first[0] == "slice" and first[3]:
                kind = str(first[3][0].get("kind"))
                limit = {"text": 32, "image": 8, "pdf": 4, "video": 1, "audio": 1}.get(kind, 1)
                count = len(first[3])
                with ready_cv:
                    while ready and ready[0][2] is not None and count < limit:
                        nxt = ready[0][2]
                        if nxt[0] != "slice" or not nxt[3] or str(nxt[3][0].get("kind")) != kind:
                            break
                        extra = len(nxt[3])
                        if count + extra > limit:
                            break
                        heapq.heappop(ready)
                        batch.append(nxt)
                        count += extra
            if len(batch) == 1 or batch[0][0] != "slice":
                consume(batch[0])
                continue
            merged = [spec for message in batch for spec in message[3]]
            try:
                vectors = embedder.embed_batch(merged)
            except Exception as exc:
                for message in batch:
                    consume(("error", message[1], message[2], None, message[4], exc))
                continue
            for message in batch:
                if message[4] is not None:
                    message[4]()
            offset = 0
            for message in batch:
                width = len(message[3])
                save_vectors(message, vectors[offset : offset + width])
                offset += len(message[3])
    finally:
        producer.join()
    for state in pending.values():
        if state.get("accounted"):
            continue
        if not state["failed"]:
            print(f"skip {state['job'][1].path}: incomplete embed", file=sys.stderr)
        skipped += 1
    return indexed, skipped


def _write_pieces(store: IndexStore, store_lock, state: dict) -> None:
    number, item, digest, size, _used = state["job"]
    del number
    rows = []
    for index, title, excerpt, vector in sorted(state["pieces"], key=lambda piece: piece[0]):
        values = [float(value) for value in vector]
        rows.append(
            {
                "path": str(item.path),
                "chunk_index": index,
                "modality": item.modality,
                "title": title,
                "sha256": digest,
                "mtime": item.path.stat().st_mtime,
                "size_bytes": size,
                "excerpt": excerpt_for(excerpt),
                "dim": len(values),
                "embedding": pack_f32(values),
            }
        )
    with store_lock:
        store.replace_file(rows)


def _slice_plans(item: FoundFile, options: ChunkOptions) -> list[dict]:
    path = item.path
    if item.modality == "text":
        chunks = _cap(chunk_text(read_text(path)), options.max_chunks)
        plans = []
        for start in range(0, len(chunks), 32):
            specs = []
            for offset, chunk in enumerate(chunks[start : start + 32]):
                index = start + offset
                title = path.name if len(chunks) == 1 else f"{path.name} #{index + 1}"
                specs.append({"kind": "text", "title": title, "body": chunk, "excerpt": chunk, "index": index})
            plans.append({"kind": "text", "specs": specs})
        return plans
    if item.modality == "image":
        return [
            {
                "kind": "image",
                "specs": [
                    {
                        "kind": "image",
                        "title": path.name,
                        "excerpt": f"image file {path.name}",
                        "path": str(path),
                        "index": 0,
                    }
                ],
            }
        ]
    if item.modality == "pdf":
        pages = _cap(list(range(1, pdf_page_count(path) + 1)), options.max_chunks)
        return [
            {"kind": "pdf", "path": str(path), "pages": pages[start : start + 4], "ordinal": start}
            for start in range(0, len(pages), 4)
        ]
    if item.modality in {"video", "audio"}:
        window = options.video_window if item.modality == "video" else options.audio_window
        spans = _cap(time_windows(media_duration(path), window, 0), options.max_chunks)
        single = len(spans) == 1
        return [
            {
                "kind": item.modality,
                "path": str(path),
                "start": start,
                "length": length,
                "ordinal": index,
                "single": single,
            }
            for index, (start, length) in enumerate(spans)
        ]
    raise RuntimeError(f"unsupported modality {item.modality}")


def _prepare_plan(plan: dict):
    if plan["kind"] in {"text", "image"}:
        return plan["specs"], _noop
    if plan["kind"] == "pdf":
        specs, cleanup = _prepare_pdf_group(Path(plan["path"]), plan["pages"])
        for offset, spec in enumerate(specs):
            spec["index"] = plan["ordinal"] + offset
        return specs, cleanup
    path = Path(plan["path"])
    specs, cleanup = _prepare_media(path, plan["kind"], plan["start"], plan["length"], plan["single"])
    specs[0]["index"] = plan["ordinal"]
    return specs, cleanup


def _count_job(result: str) -> int:
    if result == "indexed":
        return 1
    return 0


def _pieces_batched(item: FoundFile, embedder, options: ChunkOptions) -> list[tuple[str, str, object]]:
    pieces: list[tuple[str, str, object]] = []
    for specs, cleanup in _prepared_groups(item, options):
        try:
            vectors = embedder.embed_batch(specs)
        finally:
            cleanup()
        if len(vectors) != len(specs):
            raise RuntimeError(f"expected {len(specs)} vectors, got {len(vectors)}")
        for spec, vector in zip(specs, vectors):
            pieces.append((str(spec["title"]), str(spec["excerpt"]), vector))
    return pieces


def _prepared_groups(item: FoundFile, options: ChunkOptions):
    path = item.path
    if item.modality == "text":
        chunks = _cap(chunk_text(read_text(path)), options.max_chunks)
        for start in range(0, len(chunks), 16):
            specs = []
            for offset, chunk in enumerate(chunks[start : start + 16]):
                index = start + offset
                title = path.name if len(chunks) == 1 else f"{path.name} #{index + 1}"
                specs.append({"kind": "text", "title": title, "body": chunk, "excerpt": chunk})
            if specs:
                yield specs, _noop
        return
    if item.modality == "image":
        yield [
            {
                "kind": "image",
                "title": path.name,
                "excerpt": f"image file {path.name}",
                "path": str(path),
            }
        ], _noop
        return
    if item.modality == "pdf":
        pages = _cap(list(range(1, pdf_page_count(path) + 1)), options.max_chunks)
        for start in range(0, len(pages), 4):
            yield _prepare_pdf_group(path, pages[start : start + 4])
        return
    if item.modality in {"video", "audio"}:
        window = options.video_window if item.modality == "video" else options.audio_window
        spans = _cap(time_windows(media_duration(path), window, 0), options.max_chunks)
        single = len(spans) == 1
        for start, length in spans:
            yield _prepare_media(path, item.modality, start, length, single)
        return
    raise RuntimeError(f"unsupported modality {item.modality}")


def _noop() -> None:
    return None


def _prepare_pdf_group(path: Path, pages: list[int]):
    folder = Path(tempfile.mkdtemp(prefix="egsearch-"))

    def cleanup() -> None:
        shutil.rmtree(folder, ignore_errors=True)

    specs = []
    try:
        for page in pages:
            text = pdf_page_text(path, page)
            image = render_pdf_page(path, page)
            image_path = folder / f"page-{page}.png"
            image.save(image_path)
            title = f"{path.name} page {page}"
            excerpt = text.strip() or f"pdf page {page} of {path.name}"
            specs.append({"kind": "pdf", "title": title, "excerpt": excerpt, "image": str(image_path)})
    except Exception:
        cleanup()
        raise
    return specs, cleanup


def _prepare_media(path: Path, kind: str, start: float, length: float, single: bool):
    folder = Path(tempfile.mkdtemp(prefix="egsearch-"))
    label = f"{format_clock(start)}-{format_clock(start + length)}"
    title = path.name if single else f"{path.name} {label}"
    excerpt = f"{kind} {label} {path.name}"

    def cleanup() -> None:
        shutil.rmtree(folder, ignore_errors=True)

    try:
        if kind == "video":
            frames = extract_video_frames(path, start=start, duration=length, fps=1)
            frame_paths = []
            for index, frame in enumerate(frames):
                frame_path = folder / f"{index}.jpg"
                frame.save(frame_path, quality=85)
                frame_paths.append(str(frame_path))
            spec = {"kind": "video", "title": title, "excerpt": excerpt, "frames": frame_paths}
        else:
            samples = load_audio(path, start=start, seconds=length)
            wav_path = folder / "audio.wav"
            _write_wav(wav_path, samples)
            spec = {"kind": "audio", "title": title, "excerpt": excerpt, "wav": str(wav_path)}
    except Exception:
        cleanup()
        raise
    return [spec], cleanup


def _write_wav(path: Path, samples) -> None:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())


def _pieces(item: FoundFile, embedder, options: ChunkOptions) -> list[tuple[str, str, object]]:
    path = item.path
    title = path.name
    if item.modality == "text":
        chunks = _cap(chunk_text(read_text(path)), options.max_chunks)
        pieces = []
        for index, chunk in enumerate(chunks):
            chunk_title = title if len(chunks) == 1 else f"{title} #{index + 1}"
            pieces.append((chunk_title, chunk, embedder.embed_text_document(chunk_title, chunk)))
        return pieces
    if item.modality == "image":
        return [(title, f"image file {title}", embedder.embed_image(path))]
    if item.modality == "video":
        return _media_windows(path, "video", options.video_window, options, embedder)
    if item.modality == "audio":
        return _media_windows(path, "audio", options.audio_window, options, embedder)
    if item.modality == "pdf":
        return _pdf_pages(path, options, embedder)
    raise RuntimeError(f"unsupported modality {item.modality}")


def chunk_plan(item: FoundFile, options: ChunkOptions) -> tuple[int, int]:
    """Return (chunks that will be embedded, chunks available in the file)."""
    if item.modality == "image":
        return 1, 1
    if item.modality == "text":
        available = len(chunk_text(read_text(item.path)))
    elif item.modality == "video":
        available = len(time_windows(media_duration(item.path), options.video_window, 0))
    elif item.modality == "audio":
        available = len(time_windows(media_duration(item.path), options.audio_window, 0))
    elif item.modality == "pdf":
        available = pdf_page_count(item.path)
    else:
        raise RuntimeError(f"unsupported modality {item.modality}")
    if options.max_chunks > 0:
        return min(available, options.max_chunks), available
    return available, available


def _embed_video(embedder, path: Path, start: float, length: float):
    window = getattr(embedder, "embed_video_window", None)
    if window is not None:
        return window(path, start, length)
    frames = extract_video_frames(path, start=start, duration=length, fps=1)
    return embedder.embed_video_frames(frames)


def _embed_audio(embedder, path: Path, start: float, length: float):
    window = getattr(embedder, "embed_audio_window", None)
    if window is not None:
        return window(path, start, length)
    return embedder.embed_audio_array(load_audio(path, start=start, seconds=length))


def _embed_pdf(embedder, path: Path, page: int, title: str, excerpt: str):
    remote = getattr(embedder, "embed_pdf_page", None)
    if remote is not None:
        return remote(path, page, title, excerpt)
    image = render_pdf_page(path, page)
    return embedder.embed_page(title, excerpt, image)


def _cap(items: list, max_chunks: int) -> list:
    if max_chunks > 0:
        return items[:max_chunks]
    return items


def _media_windows(path: Path, kind: str, window: float, options: ChunkOptions, embedder):
    duration = media_duration(path)
    available = time_windows(duration, window, 0)
    spans = _cap(available, options.max_chunks)
    pieces = []
    for start, length in spans:
        label = f"{format_clock(start)}-{format_clock(start + length)}"
        title = path.name if len(spans) == 1 else f"{path.name} {label}"
        excerpt = f"{kind} {label} {path.name}"
        if kind == "video":
            vector = _embed_video(embedder, path, start, length)
        else:
            vector = _embed_audio(embedder, path, start, length)
        pieces.append((title, excerpt, vector))
    return pieces


def _pdf_pages(path: Path, options: ChunkOptions, embedder):
    page_count = pdf_page_count(path)
    pages = _cap(list(range(1, page_count + 1)), options.max_chunks)
    pieces = []
    for page in pages:
        text = pdf_page_text(path, page)
        title = f"{path.name} page {page}"
        excerpt = text.strip() or f"pdf page {page} of {path.name}"
        pieces.append((title, excerpt, _embed_pdf(embedder, path, page, title, excerpt)))
    return pieces
