# EmbeddingGemma 2 local search

Small command-line lab for [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`google/embeddinggemma-2`). The model is Apache 2.0, about 740M parameters, and maps text, images, video, and audio into one 768-dimensional space. Weights download into `models/` (gitignored) and stay on this machine.

The code is MIT licensed (see `LICENSE`). The text and code notes in `corpus/` are original. The sample media comes from Google's public Gemma examples and keeps its own license:

- `corpus/images/golden_gate.png` and `corpus/images/kitchen_painting.png` — [google-gemma/cookbook](https://github.com/google-gemma/cookbook), Apache 2.0
- `corpus/audio/roses_are.wav` — [Gemma audio docs](https://ai.google.dev/gemma/docs/audio), CC BY 4.0
- `corpus/video/cat_dozing.mp4` — not checked in; `scripts/fetch_samples.py` downloads it from the original source

Run `python scripts/fetch_samples.py` to fill in anything missing from `corpus/`.

## Setup

```bash
cd embeddinggemma-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/fetch_samples.py
egsearch download
```

The first download is about 1.5 GB. On Apple Silicon the model runs in float32 on Metal; on CUDA it uses bfloat16 when supported. EmbeddingGemma 2 is not compatible with float16.

## Search

Search keeps its own model process. Indexing uses one model process on the GPU and several CPU threads to decode the next files while that model is busy. `--workers` is the CPU thread count, not extra copies of the model.

```bash
egsearch serve --background   # optional; search starts this on its own
egsearch status
egsearch stop
egsearch index                 # embeds ./corpus
egsearch index --max-chunks 4 ~/Downloads/some-clip.mp4 ~/Downloads/notes.pdf
egsearch search "a cat dozing"
egsearch search "milk, eggs, and bread"
egsearch search "rooftop fan external static pressure" --task code
egsearch search --image corpus/images/kitchen_painting.png
egsearch demo                  # index plus a few expected-hit checks
```

`egsearch index ~/some/folder` walks that folder. It picks up text and code, images, video, audio, and PDFs. Files stay where they are; the tool does not copy them into this repo.

EmbeddingGemma 2 puts every input in one 8,192-token window. A chunk is sized to fit that window:

- video: 32 seconds at 1 frame per second (32 frames, about 140 tokens each)
- audio: 2 minutes, resampled to 16 kHz mono (about 25 tokens per second)
- PDFs: one page at a time, as the page image plus that page's text
- long text: about 4,000 characters, with a short overlap

`--max-chunks` defaults to 8 so a long recording or a drawing set does not embed every window on the first run. `--max-chunks 0` indexes the whole file. `--video-window` can go up to 48 seconds and `--audio-window` up to 300 seconds; past that a single chunk would overflow the context window.

The index is `.egsearch/index.sqlite`. Unchanged files are skipped. Passing `--dim 256` truncates the stored 768-d vectors and re-normalizes them (Matryoshka sizes are 128, 256, 512, and 768).

```bash
python -m unittest tests/test_lab.py
```

## License

MIT. See `LICENSE`. EmbeddingGemma 2 weights are distributed by Google under Apache 2.0 and are not included in this repository.
