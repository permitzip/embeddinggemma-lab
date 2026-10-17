"""Download the public EmbeddingGemma 2 sample media into ./corpus."""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

SAMPLES = (
    (
        "images/golden_gate.png",
        "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
    ),
    (
        "images/kitchen_painting.png",
        "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/kitchen_painting.jpg",
    ),
    (
        "audio/roses_are.wav",
        "https://ai.google.dev/gemma/docs/audio/roses-are.wav",
    ),
    (
        "video/cat_dozing.mp4",
        "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/cat_dozing.mp4",
    ),
)


def main() -> None:
    for relative, url in SAMPLES:
        destination = CORPUS / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"keep {destination.relative_to(ROOT)} ({destination.stat().st_size} bytes)")
            continue
        print(f"fetch {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"wrote {destination.relative_to(ROOT)} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
