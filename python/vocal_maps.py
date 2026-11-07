"""Where each song has singing: per-bar and per-beat vocal levels from Demucs, stored in MongoDB (vocalBars, vocalBeats).

Only the small maps are kept, not the stems (all of them would be ~20 GB). Songs closest to 112 BPM go first, matching
vibe_test_set.py's library mode; resumable, with a time budget. Runs inside WSL with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/vocal_maps.py 60"
"""

import os
import sys
import time

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pymongo import MongoClient

import auto_dj as dj
import render_transition as rt


def separate_vocals(model, audio):
    """Mono vocal track of a (2, samples) song, same Demucs settings as render_transition.load_stems."""
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cuda", split=True, overlap=0.25, progress=False)[0]
    vocals = sources[model.sources.index("vocals")] * ref.std() + ref.mean()
    return vocals.mean(0).cpu().numpy()


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    collection = MongoClient(os.environ["MONGO_URL"], socketTimeoutMS=30000, connectTimeoutMS=20000,
                             serverSelectionTimeoutMS=30000).get_default_database("autodj")["songs"]
    todo = sorted(collection.find({"analysisVersion": 2, "vocalBars": {"$exists": False}}),
                  key=lambda s: abs(s["bpm"] - 112))
    model = get_model("htdemucs")
    model.eval()
    started, done = time.time(), 0

    for song in todo:
        if time.time() - started > minutes * 60:
            print(f"time budget of {minutes:.0f} min used; {len(todo) - done} songs left, run again to continue")
            break
        t = time.time()
        try:
            paths = rt.stem_paths(song)
            if all(os.path.exists(p) for p in paths.values()):
                vocals = rt.load_stems(song)["vocals"].mean(axis=0)  # stems already cached for the hand-picked sets
            else:
                vocals = separate_vocals(model, rt.load_stereo(song))
            bars = dj.vocal_level(song, song["downbeats"], vocals)
            beats = dj.vocal_level(song, song["beats"], vocals)
            collection.update_one({"_id": song["_id"]}, {"$set": {
                "vocalBars": [round(float(v), 3) for v in bars], "vocalBeats": [round(float(v), 3) for v in beats]}})
        except Exception as err:
            print(f"FAILED {song['title']}: {err}")
            continue
        done += 1
        print(f"{song['title'][:40]:<40} {song['bpm']:6.1f} BPM  singing in {np.mean(bars > 0.2):4.0%} of bars ({time.time() - t:.0f}s)")
        sys.stdout.flush()
    print(f"{done} vocal maps saved")


if __name__ == "__main__":
    main()
