from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from egsearch.index import ChunkOptions, FoundFile, chunk_plan, chunk_text, discover, index_files, modality_for
from egsearch.media import split_png_blobs, time_windows, validate_windows
from egsearch.cli import best_per_file, format_hits
from egsearch.procname import fit, folder_label, process_name
from egsearch.quiet import _is_noise
from egsearch.store import Hit, IndexStore, dot, pack_f32, project, unpack_f32

MINI_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0000003010100c9fe92e40000000049454e44ae426082"
)


class HashEmbedder:
    def embed_text_document(self, title: str, body: str):
        return _bag(f"{title} {body}")

    def embed_image(self, path: Path):
        return _bag(path.name)

    def embed_video(self, path: Path):
        return _bag(path.name)

    def embed_audio(self, path: Path):
        return _bag(path.name)


def _bag(text: str) -> list[float]:
    vector = [0.0] * 8
    for token in text.lower().replace("_", " ").replace(".", " ").split():
        vector[hash(token) % 8] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class StoreTests(unittest.TestCase):
    def test_pack_roundtrip(self) -> None:
        values = [0.25, -1.5, 0.0, 3.0]
        self.assertEqual(unpack_f32(pack_f32(values)), values)

    def test_project_renormalizes(self) -> None:
        projected = project([3.0, 4.0, 100.0], 2)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in projected)), 1.0)
        self.assertAlmostEqual(dot(projected, projected), 1.0)

    def test_search_orders_by_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "index.sqlite")
            store.prepare_for_model("test", "1", 2)
            store.replace_file(
                [
                    _row("/tmp/near.txt", [1.0, 0.0], "near"),
                    _row("/tmp/far.txt", [0.0, 1.0], "far"),
                ]
            )
            hits = store.search([1.0, 0.0], top=2, dim=2)
            self.assertEqual([hit.path for hit in hits], ["/tmp/near.txt", "/tmp/far.txt"])
            self.assertGreater(hits[0].score, hits[1].score)
            store.close()

    def test_model_change_wipes_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "index.sqlite")
            store.prepare_for_model("model-a", "1", 2)
            store.replace_file([_row("/tmp/a.txt", [1.0, 0.0], "a")])
            wiped = store.prepare_for_model("model-b", "1", 2)
            self.assertTrue(wiped)
            self.assertEqual(store.count(), 0)
            store.close()


class CorpusTests(unittest.TestCase):
    def test_modality(self) -> None:
        self.assertEqual(modality_for(Path("note.txt")), "text")
        self.assertEqual(modality_for(Path("photo.JPEG")), "image")
        self.assertEqual(modality_for(Path("clip.mp4")), "video")
        self.assertEqual(modality_for(Path("speech.wav")), "audio")
        self.assertEqual(modality_for(Path("sheet.pdf")), "pdf")
        self.assertIsNone(modality_for(Path("archive.zip")))

    def test_long_media_is_windowed(self) -> None:
        spans = time_windows(70, 32, 0)
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0], (0.0, 32.0))
        self.assertEqual(spans[1][0], 32.0)
        self.assertAlmostEqual(spans[2][0], 64.0)
        self.assertEqual(time_windows(70, 32, 2), spans[:2])
        with self.assertRaises(ValueError):
            validate_windows(400, 32)

    def test_chunk_plan_caps_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "long.txt"
            note.write_text("alpha " * 5000, encoding="utf-8")
            used, available = chunk_plan(FoundFile(note, "text"), ChunkOptions(max_chunks=2))
            self.assertGreater(available, 2)
            self.assertEqual(used, 2)

    def test_chunk_overlap(self) -> None:
        parts = chunk_text("a" * 9000)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 4000 for part in parts))

    def test_discover_and_index_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "notes" / "grocery_list.txt"
            note.parent.mkdir()
            note.write_text("milk eggs and bread\n", encoding="utf-8")
            (root / "skip.zip").write_bytes(b"not indexed")
            found = discover([root])
            self.assertEqual([item.path.name for item in found], ["grocery_list.txt"])
            store = IndexStore(root / "index.sqlite")
            indexed, skipped = index_files(found, store, HashEmbedder(), max_files=10)
            self.assertEqual((indexed, skipped), (1, 0))
            again, unchanged = index_files(found, store, HashEmbedder(), max_files=10)
            self.assertEqual((again, unchanged), (0, 1))
            hits = store.search(_bag("grocery list milk eggs"), top=1, dim=8)
            self.assertEqual(Path(hits[0].path).name, "grocery_list.txt")
            store.close()


class ResidentTests(unittest.TestCase):
    def test_ping_and_stop(self) -> None:
        import threading
        import time

        from egsearch.resident import call, run, running, stop

        class Fake:
            dim = 4
            device = "cpu"
            model = None

            def embed_query(self, **kwargs):
                return [1.0, 0.0, 0.0, float(len(kwargs.get("text") or ""))]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thread = threading.Thread(target=run, args=(root, Fake()), daemon=True)
            thread.start()
            deadline = time.monotonic() + 5
            while not running(root):
                if time.monotonic() > deadline:
                    self.fail("resident process did not start")
                time.sleep(0.05)
            info = call(root, {"op": "ping"})
            self.assertEqual(info["dim"], 4)
            vector = call(root, {"op": "embed_query", "text": "cat", "task": "search"})["vector"]
            self.assertEqual(vector[-1], 3.0)
            self.assertTrue(stop(root))
            self.assertFalse(running(root))
            thread.join(timeout=2)


class PipelineTests(unittest.TestCase):
    def test_cpu_prep_overlaps_gpu_batches(self) -> None:
        class BatchEmbedder:
            def embed_batch(self, items):
                return [[1.0, 0.0] for _ in items]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "one.txt"
            second = root / "two.txt"
            first.write_text("alpha " * 3000, encoding="utf-8")
            second.write_text("beta bricks\n", encoding="utf-8")
            files = discover([root])
            store = IndexStore(root / "index.sqlite")
            indexed, skipped = index_files(
                files,
                store,
                BatchEmbedder(),
                max_files=10,
                options=ChunkOptions(max_chunks=1000),
                workers=4,
            )
            self.assertEqual(skipped, 0)
            self.assertEqual(indexed, 2)
            self.assertGreater(store.count(), 2)
            store.close()


class DisplayTests(unittest.TestCase):
    def test_best_hit_per_file(self) -> None:
        hits = [
            _hit("/tmp/cat.mp4", 0.70, "video", "video 00:00-00:29 cat.mp4"),
            _hit("/tmp/music.m4a", 0.62, "audio", "audio 02:00-04:00 music.m4a"),
            _hit("/tmp/music.m4a", 0.61, "audio", "audio 00:00-02:00 music.m4a"),
            _hit("/tmp/note.txt", 0.55, "text", "milk and bread"),
        ]
        chosen = best_per_file(hits, 2)
        self.assertEqual([hit.path for hit in chosen], ["/tmp/cat.mp4", "/tmp/music.m4a"])

    def test_format_hides_redundant_media_excerpt(self) -> None:
        rendered = format_hits(
            [_hit("/tmp/cat.mp4", 0.698, "video", "video 00:00-00:29 cat.mp4")],
            color=False,
            width=100,
        )
        self.assertIn("0.698", rendered)
        self.assertIn("00:00–00:29", rendered)
        self.assertNotIn("video 00:00", rendered)

    def test_noise_filter_matches_hub_warning(self) -> None:
        self.assertTrue(_is_noise("Warning: You are sending unauthenticated requests to the HF Hub."))
        self.assertFalse(_is_noise("[1/3] embed video clip.mp4"))


class ProcessNameTests(unittest.TestCase):
    def test_long_running_roles_get_short_names(self) -> None:
        self.assertEqual(process_name("serve", role="search"), "eg-search")
        self.assertEqual(process_name("serve", role="index-0"), "eg-index")
        self.assertEqual(process_name("search"), "eg-query")
        self.assertEqual(process_name("index", paths=[Path("/home/user/Downloads")]), "eg-downloads")
        self.assertEqual(process_name("index", paths=[Path("/home/user/Documents")]), "eg-documents")

    def test_names_fit_activity_monitor(self) -> None:
        label = folder_label(Path("/tmp/Very Long Folder Name"))
        self.assertEqual(process_name("index", paths=[Path("/tmp/Very Long Folder Name")]), fit(f"eg-{label}"))
        self.assertLessEqual(len(fit("eg-" + "x" * 40)), 16)


def _hit(path: str, score: float, modality: str, excerpt: str) -> Hit:
    return Hit(score=score, modality=modality, path=path, title=path, excerpt=excerpt, chunk_index=0)


class MediaTests(unittest.TestCase):
    def test_split_png_blobs(self) -> None:
        blobs = split_png_blobs(MINI_PNG + MINI_PNG)
        self.assertEqual(len(blobs), 2)
        self.assertTrue(all(blob.startswith(b"\x89PNG") for blob in blobs))

    def test_ffmpeg_audio_and_frames(self) -> None:
        import shutil

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("numpy and pillow are installed with the project")
        from egsearch.media import extract_video_frames, load_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_path = root / "tone.wav"
            _write_tone(wav_path)
            audio = load_audio(wav_path)
            self.assertGreater(len(audio), 1000)
            self.assertLess(float(audio.max()), 1.01)
            video_path = root / "color.mp4"
            import subprocess

            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:s=160x120:d=2",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=False,
                capture_output=True,
            )
            if proc.returncode != 0 or not video_path.is_file():
                self.skipTest("ffmpeg lavfi is unavailable")
            frames = extract_video_frames(video_path, max_frames=2, fps=1)
            self.assertGreaterEqual(len(frames), 1)
            self.assertEqual(frames[0].size, (160, 120))


def _row(path: str, vector: list[float], title: str) -> dict[str, object]:
    return {
        "path": path,
        "chunk_index": 0,
        "modality": "text",
        "title": title,
        "sha256": "abc",
        "mtime": 0.0,
        "size_bytes": 1,
        "excerpt": title,
        "dim": len(vector),
        "embedding": pack_f32(vector),
    }


def _write_tone(path: Path) -> None:
    rate = 16000
    frames = [int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(rate)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(int(frame).to_bytes(2, "little", signed=True) for frame in frames))


if __name__ == "__main__":
    unittest.main()
