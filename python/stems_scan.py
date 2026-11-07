"""Split the library into stems (drums / bass / other / vocals) and measure what stem-based transitions need.

Keeps the stems (FLAC in /root/autodj-stems inside WSL, ~60 MB a song) and stores in MongoDB:
    stemBands   per bar: instrumental bass / mids / highs in dB against the song's loud bars, to find where a song
                has cooled down and where two songs clash in a band
    vocalBars   per bar and per beat vocal level (0-1), from the vocal stem, so vocal fades land between words
    vocalBeats
Songs that share their tempo with the most other songs go first (they can join the most mixes). Resumable, with a
time budget. Runs inside WSL with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/stems_scan.py 60"
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from demucs.pretrained import get_model
from pymongo import MongoClient

import auto_dj as dj
import render_transition as rt

SR = rt.SAMPLE_RATE
TEMPO_NEIGHBOURS = 0.015  # songs within 1.5% of each other's tempo can share a mix
AHEAD = 2  # songs loaded ahead and saves still running, so the GPU isn't waiting on the disk


def separate(model, audio):
    """All four stems of a (2, samples) song, same Demucs settings as render_transition.load_stems."""
    import torch
    from demucs.apply import apply_model

    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cuda", split=True, overlap=0.25, progress=False)[0]
    sources = (sources * ref.std() + ref.mean()).cpu().numpy()
    return {name: source.astype(np.float32) for name, source in zip(model.sources, sources)}


def band_bars(song, stems):
    """Per bar: the instrumental's bass / mids / highs in dB against the song's loud (95th percentile) bars."""
    low, mid, high = rt.split_bands((stems["drums"] + stems["bass"] + stems["other"]).mean(0, keepdims=True))
    edges = (np.append(song["downbeats"], song["duration"]) * SR).astype(int)
    rms = np.array([[np.sqrt(np.mean(band[0, a:b] ** 2)) + 1e-9 if b > a else 1e-9 for band in (low, mid, high)]
                    for a, b in zip(edges[:-1], edges[1:])])
    return np.round(20 * np.log10(rms / np.percentile(rms, 95, axis=0)), 1)


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    collection = MongoClient(os.environ["MONGO_URL"], socketTimeoutMS=30000, connectTimeoutMS=20000,
                             serverSelectionTimeoutMS=30000).get_default_database("autodj")["songs"]
    songs = list(collection.find({"analysisVersion": 2, "stemBands": {"$exists": False}},
                                 {"title": 1, "artist": 1, "bpm": 1, "filePath": 1, "downbeats": 1, "beats": 1, "duration": 1}))
    tempos = np.array([s["bpm"] for s in collection.find({"analysisVersion": 2}, {"bpm": 1})])
    neighbours = lambda bpm: int(np.sum(np.abs(tempos - bpm) / bpm <= TEMPO_NEIGHBOURS))
    todo = sorted(songs, key=lambda s: -neighbours(s["bpm"]))
    if sys.argv[2:]:  # only these songs (start of the title)
        todo = [s for s in todo if any(s["title"].lower().startswith(name.lower()) for name in sys.argv[2:])]
    print(f"{len(todo)} songs to split")

    os.makedirs(rt.LIBRARY_STEM_DIR, exist_ok=True)
    model = get_model("htdemucs")
    model.eval()
    started, done = time.time(), 0

    def load(song):
        """(stems, None) if the song was split before, else (None, audio)."""
        if all(os.path.exists(p) for p in rt.stem_paths(song).values()):
            return rt.load_stems(song), None  # already split (the hand-picked sets)
        return None, rt.load_stereo(song)

    def save(song, stems, fresh, t):
        try:
            write(song, stems, fresh, t)
        except Exception as err:
            print(f"FAILED saving {song['title']}: {err}")

    def write(song, stems, fresh, t):
        if fresh:
            paths = rt.stem_paths(song)
            os.makedirs(os.path.dirname(paths["vocals"]), exist_ok=True)
            for name, audio in stems.items():
                with rt.AudioFile(paths[name], "w", SR, num_channels=2, bit_depth=16) as f:
                    f.write(audio)
        vocals = stems["vocals"].mean(axis=0)
        bars = dj.vocal_level(song, song["downbeats"], vocals)
        beats = dj.vocal_level(song, song["beats"], vocals)
        collection.update_one({"_id": song["_id"]}, {"$set": {
            "stemBands": band_bars(song, stems).tolist(),
            "vocalBars": [round(float(v), 3) for v in bars], "vocalBeats": [round(float(v), 3) for v in beats]}})
        print(f"{song['title'][:40]:<40} {song['bpm']:6.1f} BPM ({neighbours(song['bpm'])} songs at this tempo)  {time.time() - t:.0f}s")
        sys.stdout.flush()

    with ThreadPoolExecutor(AHEAD) as loader, ThreadPoolExecutor(AHEAD) as saver:
        loading = [loader.submit(load, s) for s in todo[:AHEAD]]
        saving = []
        for i, song in enumerate(todo):
            if time.time() - started > minutes * 60:
                print(f"time budget of {minutes:.0f} min used; {len(todo) - i} songs left, run again to continue")
                break
            t = time.time()
            if i + AHEAD < len(todo):
                loading.append(loader.submit(load, todo[i + AHEAD]))
            try:
                stems, audio = loading[i].result()
                fresh = stems is None
                if fresh:
                    stems = separate(model, audio)
                saving = [f for f in saving if not f.done()]
                while len(saving) >= AHEAD:  # don't pile up stems in memory (~300 MB a song)
                    saving.pop(0).result()
                saving.append(saver.submit(save, song, stems, fresh, t))
            except Exception as err:
                print(f"FAILED {song['title']}: {err}")
                continue
            done += 1
        for f in saving:
            f.result()
    print(f"{done} songs split")


if __name__ == "__main__":
    main()
