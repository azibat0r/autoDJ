"""Analyze downloaded songs for DJ transitions and store the results in MongoDB.

Runs inside WSL (Ubuntu) with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/analyze.py"

Options:
    --song TEXT   only analyze songs whose title contains TEXT
    --limit N     stop after N songs
    --force       re-analyze songs that are already done
"""

import argparse
import ntpath
import os
import time

import essentia
import essentia.standard as es
import numpy as np
from beat_this.inference import Audio2Beats
from dotenv import load_dotenv
from pymongo import MongoClient

essentia.log.infoActive = False
essentia.log.warningActive = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")

load_dotenv(os.path.join(BASE_DIR, ".env"))

ANALYSIS_VERSION = 2
SAMPLE_RATE = 44100
FRAME_SIZE = 4096
HOP_SIZE = 2048
BASS_CUTOFF_HZ = 150
PHRASE_BARS = 4
NOVELTY_HALF_WIDTH = 4  # bars on each side of a candidate boundary
MIN_SECTION_BARS = 4

PITCH_CLASSES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
# Same tables as api/routes/playlist.js, indexed by pitch class
CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]


def detect_key(audio):
    name, scale, strength = es.KeyExtractor(profileType="edma", sampleRate=SAMPLE_RATE)(audio)
    key = PITCH_CLASSES[name]
    mode = 1 if scale == "major" else 0
    camelot = f"{CAMELOT_MAJOR[key]}B" if mode else f"{CAMELOT_MINOR[key]}A"
    return {
        "key": key,
        "mode": mode,
        "camelot": camelot,
        "keyName": f"{name} {scale}",
        "keyStrength": round(float(strength), 3),
    }


def frame_features(audio):
    """Per-frame RMS, bass energy and MFCCs, plus the time at the centre of each frame."""
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC(sampleRate=SAMPLE_RATE, inputSize=FRAME_SIZE // 2 + 1, numberCoefficients=13)
    bass_bins = int(BASS_CUTOFF_HZ * FRAME_SIZE / SAMPLE_RATE) + 1

    times, rms, bass, mfccs = [], [], [], []
    for i, frame in enumerate(es.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True)):
        spec = spectrum(window(frame))
        power = spec ** 2
        times.append((i * HOP_SIZE + FRAME_SIZE / 2) / SAMPLE_RATE)
        rms.append(np.sqrt(np.mean(frame ** 2)))
        bass.append(power[:bass_bins].sum() / (power.sum() + 1e-12) * np.mean(frame ** 2))
        mfccs.append(mfcc(spec)[1])

    return np.array(times), np.array(rms), np.array(bass), np.array(mfccs)


def bar_features(downbeats, duration, times, rms, bass, mfccs):
    """Average the frame features over each bar (downbeat to next downbeat)."""
    bar_length = np.median(np.diff(downbeats))
    edges = np.append(downbeats, min(downbeats[-1] + bar_length, duration))
    bar_index = np.searchsorted(edges, times, side="right") - 1

    n_bars = len(downbeats)
    bar_rms = np.zeros(n_bars)
    bar_bass = np.zeros(n_bars)
    bar_mfcc = np.zeros((n_bars, mfccs.shape[1]))
    for b in range(n_bars):
        in_bar = bar_index == b
        if in_bar.any():
            bar_rms[b] = rms[in_bar].mean()
            bar_bass[b] = bass[in_bar].mean()
            bar_mfcc[b] = mfccs[in_bar].mean(axis=0)

    return edges, bar_rms, bar_bass, bar_mfcc


def normalize(values):
    ref = np.percentile(values, 95)
    return np.clip(values / ref, 0, 1) if ref > 0 else np.zeros_like(values)


def novelty_curve(features, half):
    """Foote novelty: how much the music changes at the start of each bar."""
    feats = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    similarity = feats @ feats.T

    offsets = np.arange(-half, half) + 0.5
    taper = np.exp(-2 * (offsets / half) ** 2)
    sign = np.sign(offsets)
    kernel = np.outer(taper, taper) * np.outer(sign, sign)

    padded = np.pad(similarity, half, mode="reflect")
    n = len(features)
    return np.array([(padded[i:i + 2 * half, i:i + 2 * half] * kernel).sum() for i in range(n)])


def find_boundaries(novelty):
    """Pick section boundaries from novelty peaks, snapped to the 4-bar phrase grid."""
    n = len(novelty)
    half = NOVELTY_HALF_WIDTH

    # Phrase grid offset: which bar (0-3) the strongest changes tend to land on
    offset = int(np.argmax([novelty[o::PHRASE_BARS].sum() for o in range(PHRASE_BARS)]))

    threshold = novelty.mean() + 0.5 * novelty.std()
    peaks = [
        i for i in range(1, n)
        if novelty[i] >= threshold and novelty[i] == novelty[max(0, i - half):i + half + 1].max()
    ]

    snapped = {}
    for p in peaks:
        nearest = p + ((offset - p) % PHRASE_BARS)
        if nearest - p > PHRASE_BARS // 2:
            nearest -= PHRASE_BARS
        b = nearest if abs(nearest - p) <= 1 and 0 < nearest < n else p
        snapped[b] = max(snapped.get(b, 0), novelty[p])

    # Keep the strongest boundaries that leave every section at least MIN_SECTION_BARS long
    kept = [0, n]
    for b in sorted(snapped, key=snapped.get, reverse=True):
        if all(abs(b - k) >= MIN_SECTION_BARS for k in kept):
            kept.append(b)
    return sorted(kept), offset


def build_sections(boundaries, edges, bar_energy, bar_bass):
    median_energy = np.median(bar_energy)
    low, high = np.percentile(bar_energy, [33, 66])
    sections = []

    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        energy = float(bar_energy[start:end].mean())
        prev_energy = sections[-1]["energy"] if sections else None
        is_last = idx == len(boundaries) - 2

        if idx == 0 and energy < 0.85 * median_energy:
            label = "intro"
        elif is_last and energy < 0.85 * median_energy:
            label = "outro"
        elif prev_energy is not None and energy - prev_energy >= 0.2:
            label = "drop"
        elif prev_energy is not None and prev_energy - energy >= 0.2:
            label = "breakdown"
        else:
            label = "section"

        sections.append({
            "label": label,
            "level": "low" if energy < low else "high" if energy > high else "mid",
            "startBar": int(start),
            "endBar": int(end),
            "start": round(float(edges[start]), 3),
            "end": round(float(edges[end]), 3),
            "energy": round(energy, 3),
            "bass": round(float(bar_bass[start:end].mean()), 3),
        })

    return sections


def estimate_bpm(beats):
    # Beat times come in 20ms steps, so the median gap alone rounds the tempo coarsely.
    # Averaging every gap close to the median recovers the precise tempo.
    intervals = np.diff(beats)
    median = np.median(intervals)
    steady = intervals[np.abs(intervals - median) < 0.1 * median]
    return round(float(60 / steady.mean()), 2)


def analyze(path, beat_tracker):
    audio = es.MonoLoader(filename=path, sampleRate=SAMPLE_RATE)()
    duration = len(audio) / SAMPLE_RATE

    beats, downbeats = beat_tracker(audio, SAMPLE_RATE)
    if len(downbeats) < 2 * MIN_SECTION_BARS:
        raise ValueError(f"only {len(downbeats)} downbeats found")

    times, rms, bass, mfccs = frame_features(audio)
    edges, bar_rms, bar_bass, bar_mfcc = bar_features(downbeats, duration, times, rms, bass, mfccs)
    bar_energy = normalize(bar_rms)
    bar_bass_norm = normalize(bar_bass)

    features = np.column_stack([bar_mfcc, bar_energy * 3, bar_bass_norm * 3])
    novelty = novelty_curve(features, NOVELTY_HALF_WIDTH)
    boundaries, phrase_offset = find_boundaries(novelty)

    mean_rms = float(rms.mean())
    return {
        "bpm": estimate_bpm(beats),
        **detect_key(audio),
        "duration": round(duration, 3),
        "energy": mean_rms,
        "loudness": round(float(20 * np.log10(mean_rms + 1e-12)), 2),
        "beats": np.round(beats, 3).tolist(),
        "downbeats": np.round(downbeats, 3).tolist(),
        "phraseOffset": phrase_offset,
        "barEnergy": np.round(bar_energy, 3).tolist(),
        "barBass": np.round(bar_bass_norm, 3).tolist(),
        "sections": build_sections(boundaries, edges, bar_energy, bar_bass_norm),
        "analysisVersion": ANALYSIS_VERSION,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Timeouts so a connection dropped by laptop sleep fails and gets retried instead of hanging forever
    client = MongoClient(
        os.environ["MONGO_URL"],
        socketTimeoutMS=30000,
        connectTimeoutMS=20000,
        serverSelectionTimeoutMS=30000,
    )
    songs_collection = client.get_default_database("autodj")["songs"]

    downloaded = {name for name in os.listdir(DOWNLOAD_DIR) if name.lower().endswith(".mp3")}
    songs = [
        song
        for song in songs_collection.find({"filePath": {"$exists": True}})
        if song["filePath"] and ntpath.basename(song["filePath"]) in downloaded
    ]
    total = len(songs)
    if args.song:
        songs = [s for s in songs if args.song.lower() in s.get("title", "").lower()]
    if not args.force:
        songs = [s for s in songs if s.get("analysisVersion") != ANALYSIS_VERSION]
    if args.limit:
        songs = songs[: args.limit]

    print(f"Found {total} songs with an MP3 in {DOWNLOAD_DIR}")
    print(f"Analyzing {len(songs)}\n")

    beat_tracker = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=True)
    started = time.time()

    for i, song in enumerate(songs, start=1):
        title = song.get("title", "Unknown")
        prefix = f"[{i}/{len(songs)}] {title}"
        path = os.path.join(DOWNLOAD_DIR, ntpath.basename(song["filePath"]))

        try:
            t = time.time()
            result = analyze(path, beat_tracker)
            songs_collection.update_one({"_id": song["_id"]}, {"$set": result})

            labels = " ".join(s["label"][0].upper() for s in result["sections"])
            print(
                f"{prefix}: {result['bpm']} BPM, {result['keyName']} ({result['camelot']}), "
                f"{len(result['downbeats'])} bars, sections [{labels}] ({time.time() - t:.1f}s)"
            )
        except Exception as err:
            print(f"{prefix}: skipped (error - {err})")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
