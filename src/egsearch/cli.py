from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from egsearch.index import ChunkOptions, discover, index_files
from egsearch.media import AUDIO_WINDOW_SECONDS, VIDEO_WINDOW_SECONDS, validate_windows
from egsearch.paths import (
    EMBEDDER_VERSION,
    MODEL_ID,
    NATIVE_DIM,
    TRUNCATION_DIMS,
    default_corpus,
    index_path,
    project_root,
)
from egsearch.procname import adopt_process_name, process_name
from egsearch.quiet import install as install_quiet
from egsearch.quiet import prefix_logs
from egsearch.resident import IndexEmbedder, RemoteEmbedder, ensure_worker, running, stop
from egsearch.store import Hit, IndexStore

_RANGE = re.compile(r"^(?:video|audio) (\d{2}:\d{2}-\d{2}:\d{2})\b")
_PAGE = re.compile(r" page (\d+)$")
_KIND_COLOR = {
    "text": "32",
    "image": "33",
    "video": "36",
    "audio": "35",
    "pdf": "34",
}


def main(argv: list[str] | None = None) -> int:
    install_quiet()
    parser = build_parser()
    args = parser.parse_args(argv)
    root = project_root(args.root)
    paths = list(getattr(args, "paths", None) or [])
    if args.command == "index" and not paths:
        paths = [default_corpus(root)]
    name = process_name(args.command, role=str(getattr(args, "role", "search")), paths=paths)
    adopt_process_name(root, name)
    prefix_logs(name)
    try:
        return args.func(args, root)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="egsearch",
        description="Search local text, images, video, and audio with EmbeddingGemma 2.",
    )
    parser.add_argument("--root", type=Path, help="Project directory. Defaults to this checkout.")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download google/embeddinggemma-2 into ./models")
    download.set_defaults(func=cmd_download)

    doctor = sub.add_parser("doctor", help="Load the model and check a tiny text similarity")
    doctor.set_defaults(func=cmd_doctor)

    index = sub.add_parser("index", help="Embed files into the local index")
    index.add_argument("paths", nargs="*", type=Path, help="Files or directories. Default: ./corpus")
    index.add_argument("--max-files", type=int, default=2000)
    index.add_argument(
        "--max-chunks",
        type=int,
        default=8,
        help="Chunks per file. 0 embeds every window or PDF page.",
    )
    index.add_argument("--audio-window", type=float, default=AUDIO_WINDOW_SECONDS)
    index.add_argument("--video-window", type=float, default=VIDEO_WINDOW_SECONDS)
    index.add_argument(
        "--workers",
        type=int,
        default=12,
        help="CPU threads that decode the next batches while the GPU embeds the current one.",
    )
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="Search the local index")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--image", type=Path)
    search.add_argument("--video", type=Path)
    search.add_argument("--audio", type=Path)
    search.add_argument("--top", type=int, default=5)
    search.add_argument("--dim", type=int, choices=TRUNCATION_DIMS, default=NATIVE_DIM)
    search.add_argument("--task", choices=("search", "code", "qa"), default="search")
    search.add_argument("--modality", choices=("text", "image", "video", "audio", "pdf"))
    search.add_argument("--json", action="store_true")
    search.add_argument(
        "--chunks",
        action="store_true",
        help="Show every matching window. The default keeps the best window of each file.",
    )
    search.set_defaults(func=cmd_search)

    listing = sub.add_parser("ls", help="List indexed files")
    listing.set_defaults(func=cmd_ls)

    demo = sub.add_parser("demo", help="Index ./corpus and run the sample queries")
    demo.set_defaults(func=cmd_demo)

    serve = sub.add_parser("serve", help="Keep a search model loaded")
    serve.add_argument("--background", action="store_true", help="Detach and return")
    serve.add_argument("--role", default="search", help="Worker name, such as search or index-0")
    serve.set_defaults(func=cmd_serve)

    status = sub.add_parser("status", help="Show whether the model process is loaded")
    status.set_defaults(func=cmd_status)

    stop_parser = sub.add_parser("stop", help="Unload the background model process")
    stop_parser.set_defaults(func=cmd_stop)
    return parser


def cmd_download(args, root: Path) -> int:
    embedder = load_embedder(root)
    count = sum(parameter.numel() for parameter in embedder.model.parameters())
    print(f"ready: {MODEL_ID} ({count:,} parameters, dim {embedder.dim})")
    if count < 700_000_000:
        print("expected the full 740M multimodal model, including vision and audio", file=sys.stderr)
        return 1
    return 0


def cmd_doctor(args, root: Path) -> int:
    embedder = load_embedder(root)
    words = ["king", "queen", "car"]
    vectors = [embedder.encode(word, prompt_name="SentenceSimilarity") for word in words]
    king_queen = float(sum(a * b for a, b in zip(vectors[0], vectors[1])))
    king_car = float(sum(a * b for a, b in zip(vectors[0], vectors[2])))
    device = getattr(embedder.model, "device", "unknown")
    dtype = getattr(embedder.model, "dtype", "unknown")
    count = sum(parameter.numel() for parameter in embedder.model.parameters())
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"parameters: {count:,}")
    if count < 700_000_000:
        print("expected the full 740M multimodal model, including vision and audio", file=sys.stderr)
        return 1
    print(f"similarity king/queen: {king_queen:.3f}")
    print(f"similarity king/car: {king_car:.3f}")
    if king_queen <= king_car:
        print("doctor: king/queen should outrank king/car", file=sys.stderr)
        return 1
    print("doctor: ok")
    return 0


def cmd_index(args, root: Path) -> int:
    targets = list(args.paths) or [default_corpus(root)]
    files = discover(targets)
    if not files:
        raise RuntimeError(f"no supported files under {', '.join(str(path) for path in targets)}")
    options = chunk_options(args)
    ensure_worker(root, "search")
    print(
        f"Indexing with {args.workers} CPU threads and 1 GPU model process.",
        file=sys.stderr,
    )
    embedder = IndexEmbedder(root, args.workers)
    store = open_store(root, embedder.dim)
    indexed, skipped = index_files(
        files,
        store,
        embedder,
        max_files=args.max_files,
        options=options,
        workers=args.workers,
    )
    store.close()
    print(f"indexed {indexed}, skipped {skipped}, total files seen {len(files)}")
    return 0


def cmd_search(args, root: Path) -> int:
    store = IndexStore(index_path(root))
    if store.count() == 0:
        store.close()
        raise RuntimeError("index is empty. Run: egsearch index")
    embedder = resident_embedder(root)
    vector = embedder.embed_query(
        text=args.query,
        image=args.image,
        video=args.video,
        audio=args.audio,
        task=args.task,
    )
    fetch = args.top if args.chunks or args.json else max(args.top * 6, 24)
    hits = store.search(vector, top=fetch, dim=args.dim, modality=args.modality)
    store.close()
    shown = hits if args.chunks or args.json else best_per_file(hits, args.top)
    if args.json:
        print(json.dumps({"query": args.query, "hits": [hit_record(hit) for hit in shown]}, indent=2))
        return 0
    if not shown:
        print("no hits")
        return 0
    print(format_hits(shown, color=sys.stdout.isatty() and not os.environ.get("NO_COLOR")))
    return 0


def cmd_ls(args, root: Path) -> int:
    store = IndexStore(index_path(root))
    rows = store.list_rows()
    store.close()
    if not rows:
        print("index is empty")
        return 0
    for row in rows:
        suffix = "" if int(row["chunk_index"]) == 0 else f"  chunk {int(row['chunk_index']) + 1}"
        print(f"{row['modality']:<5}  {display_path(str(row['path']))}{suffix}")
    print(f"{len(rows)} chunks")
    return 0


def cmd_demo(args, root: Path) -> int:
    corpus = default_corpus(root)
    missing = [
        corpus / "images" / "golden_gate.png",
        corpus / "images" / "kitchen_painting.png",
        corpus / "video" / "cat_dozing.mp4",
        corpus / "audio" / "roses_are.wav",
    ]
    absent = [path for path in missing if not path.is_file()]
    if absent:
        raise RuntimeError(
            "sample media is missing. Run: python scripts/fetch_samples.py\n"
            + "\n".join(str(path) for path in absent)
        )
    files = discover([corpus])
    embedder = IndexEmbedder(root, 1)
    store = open_store(root, embedder.dim)
    indexed, skipped = index_files(files, store, embedder, max_files=2000, workers=1)
    print(f"indexed {indexed}, skipped {skipped}", file=sys.stderr)
    cases = [
        ("a photograph of the Golden Gate Bridge", "search", "golden_gate"),
        ("a cat dozing", "search", "cat_dozing"),
        ("Roses are red, violets are blue.", "search", "roses_are"),
        ("milk, eggs, and bread", "search", "grocery_list"),
        ("what causes the northern lights", "search", "northern_lights"),
        ("rooftop fan external static pressure", "code", "static_pressure"),
    ]
    failed = 0
    for query, task, needle in cases:
        vector = embedder.embed_text_query(query, task)
        hits = store.search(vector, top=3, dim=NATIVE_DIM)
        names = [Path(hit.path).name for hit in hits]
        ok = any(needle in name for name in names)
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        top = ", ".join(f"{Path(hit.path).name} ({hit.score:.3f})" for hit in hits)
        print(f"{mark}  {query!r} -> {top}")
    store.close()
    if failed:
        print(f"demo: {failed} query missed its expected file in the top 3", file=sys.stderr)
        return 1
    print("demo: ok")
    return 0


def chunk_options(args) -> ChunkOptions:
    validate_windows(args.audio_window, args.video_window)
    return ChunkOptions(
        max_chunks=args.max_chunks,
        audio_window=args.audio_window,
        video_window=args.video_window,
    )


def cmd_serve(args, root: Path) -> int:
    if args.background:
        info = ensure_worker(root, args.role)
        print(f"loaded {info['role']} pid {info['pid']} on {info['device']}")
        return 0
    embedder = load_embedder(root)
    from egsearch.resident import run

    run(root, embedder, role=args.role)
    return 0


def cmd_status(args, root: Path) -> int:
    from egsearch.resident import _known_roles, call

    roles = _known_roles(root)
    if not roles:
        print("stopped")
        return 1
    shown = 0
    for role in roles:
        if not running(root, role):
            continue
        try:
            info = call(root, {"op": "ping"}, timeout=5, role=role)
        except Exception as exc:
            print(f"{role}: busy or not answering ({exc})")
            shown += 1
            continue
        print(f"{info['role']} pid {info['pid']} on {info['device']} ({info['parameters']:,} parameters)")
        shown += 1
    if shown == 0:
        print("stopped")
        return 1
    return 0


def cmd_stop(args, root: Path) -> int:
    if stop(root):
        print("stopped")
        return 0
    print("stopped")
    return 0


def resident_embedder(root: Path) -> RemoteEmbedder:
    return RemoteEmbedder(root)


def load_embedder(root: Path):
    from egsearch.embed import GemmaEmbedder

    return GemmaEmbedder(root)


def open_store(root: Path, dim: int) -> IndexStore:
    store = IndexStore(index_path(root))
    wiped = store.prepare_for_model(MODEL_ID, EMBEDDER_VERSION, dim)
    if wiped:
        print("index was cleared because the model version changed", file=sys.stderr)
    return store


def display_path(path: str) -> str:
    resolved = Path(path).resolve()
    candidates = [str(resolved)]
    try:
        candidates.append(resolved.relative_to(Path.cwd().resolve()).as_posix())
    except ValueError:
        pass
    try:
        candidates.append("~/" + resolved.relative_to(Path.home()).as_posix())
    except ValueError:
        pass
    return min(candidates, key=len)


def best_per_file(hits: list[Hit], limit: int) -> list[Hit]:
    chosen: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.path in seen:
            continue
        seen.add(hit.path)
        chosen.append(hit)
        if len(chosen) == limit:
            break
    return chosen


def hit_record(hit: Hit) -> dict[str, object]:
    return {
        "score": round(hit.score, 4),
        "modality": hit.modality,
        "path": display_path(hit.path),
        "where": where_label(hit),
        "title": hit.title,
        "excerpt": hit.excerpt,
    }


def where_label(hit: Hit) -> str:
    match = _RANGE.match(hit.excerpt)
    if match:
        return match.group(1).replace("-", "–")
    page = _PAGE.search(hit.title)
    if page:
        return f"page {page.group(1)}"
    if hit.chunk_index:
        return f"part {hit.chunk_index + 1}"
    return ""


def format_hits(hits: list[Hit], *, color: bool = False, width: int | None = None) -> str:
    columns = width or shutil.get_terminal_size(fallback=(100, 24)).columns
    where_width = 11
    kind_width = 5
    lines: list[str] = []
    for hit in hits:
        kind = _paint(hit.modality.ljust(kind_width), _KIND_COLOR.get(hit.modality, ""), enabled=color)
        where = where_label(hit).ljust(where_width)
        score = _paint(f"{hit.score:0.3f}", "1", enabled=color)
        used = 2 + 5 + 2 + kind_width + 2 + where_width + 2
        path = shorten(display_path(hit.path), max(20, columns - used))
        lines.append(f"  {score}  {kind}  {where}  {path}")
        detail = snippet(hit)
        if detail:
            lines.append(f"         {_paint(detail, '2', enabled=color)}")
    return "\n".join(lines)


def snippet(hit: Hit) -> str:
    if hit.modality in {"video", "audio", "image"}:
        return ""
    text = " ".join(hit.excerpt.split())
    if len(text) > 110:
        return text[:107] + "..."
    return text


def shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width < 8:
        return text[:width]
    if "/" not in text:
        head = max(1, width // 2 - 1)
        return text[:head] + "…" + text[-(width - head - 1) :]
    parent, name = text.rsplit("/", 1)
    if len(name) >= width - 2:
        return "…" + name[-(width - 1) :]
    budget = width - len(name) - 1
    if len(parent) <= budget:
        return text
    if budget < 2:
        return "…/" + name
    return parent[: budget - 1] + "…/" + name


def _paint(text: str, code: str, *, enabled: bool) -> str:
    if not enabled or not code:
        return text
    return f"\033[{code}m{text}\033[0m"


if __name__ == "__main__":
    raise SystemExit(main())
