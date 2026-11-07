"""Render DJ transitions between two analyzed songs to WAV files.

Runs inside WSL (Ubuntu) with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/render_transition.py --from 'Song A' --to 'Song B' --out-bar 112"

Song B is always beatmatched: time-stretched (pitch unchanged) so its bars line up with song A's,
and starts on one of A's downbeats. Two songs' vocals never play at the same time: B's vocals
(separated with Demucs) stay muted until A's are gone.

Techniques (--technique; "all" renders acapella, blend, filter and echo):

  blend     B's mids/highs fade in with its bass cut, basslines swap halfway, A's mids/highs fade out
  filter    B sweeps in through a closing high-pass filter, A sweeps out through an opening one
  echo      A's last beat echoes out on the beat as B slams in on its drop
  cut       hard cut on the downbeat from A straight into B's drop
  loop      A's first outro bar loops under B, rolling 4 -> 2 -> 1 -> 1/2 beats before B takes over
  acapella  stems: B's instrumental comes in under A, then A's vocals ride over B's beat
  layered   acapella, with B building up stem by stem: drums, then instruments, then bass at the swap
  layered_echo  layered, with A's last vocal line echoing out instead of fading
  layered_short for okay matches: B builds in over 8 bars, then a full switch (no vocal ride)
  vocal_preview B's vocals come in first over A's beat, then a sudden switch to all of B (for beats that clash)
  handover      stem-by-stem swap: drums, then instruments, then bass; layers are swapped, never stacked
  eq            EQ only, like djay Pro's Automix: B plays underneath with its EQ down, then the EQs swap
  eq_fast       eq for high-energy mixes: A is dropped within 2 bars of the swap so its vocals don't linger
  eq_filter     eq_fast, but A sinks away under a closing low-pass over those 2 bars instead of just dropping
  eq_layered    layered recreated with EQ only: B's highs, then mids, then bass at a sudden swap; A's mids ride on
"""

import argparse
import json
import ntpath
import os
import re

import essentia
import essentia.standard as es
import numpy as np
from dotenv import load_dotenv
from pedalboard import Delay, HighpassFilter, Limiter, LowpassFilter, time_stretch
from pedalboard.io import AudioFile
from pymongo import MongoClient
from scipy.signal import butter, sosfilt, sosfiltfilt

essentia.log.infoActive = False
essentia.log.warningActive = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
RENDER_DIR = os.path.join(BASE_DIR, "renders")
STEM_DIR = os.path.join(BASE_DIR, "stems")  # the first hand-picked songs
# The library scan (stems_scan.py) keeps its stems inside WSL instead: ~60 MB a song, too much for OneDrive to sync
LIBRARY_STEM_DIR = "/root/autodj-stems"

load_dotenv(os.path.join(BASE_DIR, ".env"))

SAMPLE_RATE = 44100  # must match analyze.py so beat times line up with the audio
BASS_HZ = 200
HIGH_HZ = 2500
MAX_LOUDNESS_MATCH_DB = 6
EQ_BACKGROUND_DB = -9  # eq technique: how far the background song's mids/highs are turned down
TEMPO_BARS = 16  # bars used to measure the tempo ratio for beatmatching
CLICK_FREE = 0.008  # seconds; the shortest fade, used for cuts so they don't click
STEMS = ("drums", "bass", "other", "vocals")
TECHNIQUES = ("blend", "filter", "echo", "cut", "loop", "acapella", "layered", "layered_echo", "layered_short", "vocal_preview", "handover", "eq", "eq_fast", "eq_filter", "eq_layered", "eq_in_filter_out", "stems_eq", "smooth")
DEFAULT_TECHNIQUES = ("layered", "acapella", "blend", "filter")  # others only when asked for


# ---------- loading ----------

def find_song(collection, title):
    song = collection.find_one(
        {"title": {"$regex": f"^{re.escape(title)}$", "$options": "i"}, "analysisVersion": 2}
    )
    if not song:
        raise SystemExit(f"Song not found or not analyzed: {title}")
    return song


def safe_name(text):
    return re.sub(r'[\\/:*?"<>|]', "", text).strip()[:120]


def load_stereo(song):
    """Decode with the same loader and resampler as analyze.py, so analysis timestamps match exactly."""
    path = os.path.join(DOWNLOAD_DIR, ntpath.basename(song["filePath"]))
    audio, sample_rate, channels = es.AudioLoader(filename=path)()[:3]
    audio = audio.T  # (channels, samples)
    if sample_rate != SAMPLE_RATE:
        resample = es.Resample(inputSampleRate=sample_rate, outputSampleRate=SAMPLE_RATE)
        audio = np.array([resample(np.ascontiguousarray(ch)) for ch in audio])
    if channels == 1:
        audio = np.vstack([audio, audio])
    return audio.astype(np.float32)


def stem_paths(song):
    """Where a song's stems are: the original folder if they were made there, otherwise the library folder."""
    name = safe_name(f"{song['artist']} - {song['title']}")
    folder = os.path.join(STEM_DIR, name)
    if not os.path.exists(os.path.join(folder, "vocals.flac")) and os.path.isdir(LIBRARY_STEM_DIR):
        folder = os.path.join(LIBRARY_STEM_DIR, name)
    return {stem: os.path.join(folder, f"{stem}.flac") for stem in STEMS}


def load_stems(song, audio=None):
    """Vocals / drums / bass / other from Demucs, cached as FLAC in python/stems/<song>/.

    `audio` is only needed (and loaded if missing) when the stems aren't cached yet.
    """
    paths = stem_paths(song)
    folder = os.path.dirname(paths["vocals"])
    if all(os.path.exists(p) for p in paths.values()):
        stems = {}
        for stem, path in paths.items():
            with AudioFile(path) as f:
                stems[stem] = f.read(f.frames)
        return stems

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    print(f"Separating stems for {song['title']} (first time only)...")
    if audio is None:
        audio = load_stereo(song)
    model = get_model("htdemucs")
    model.eval()
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cuda", split=True, overlap=0.25, progress=False)[0]
    sources = (sources * ref.std() + ref.mean()).cpu().numpy()

    os.makedirs(folder, exist_ok=True)
    stems = {}
    for name, source in zip(model.sources, sources):
        stems[name] = source.astype(np.float32)
        with AudioFile(paths[name], "w", SAMPLE_RATE, num_channels=2, bit_depth=16) as f:
            f.write(stems[name])
    return stems


# ---------- audio helpers ----------

def split_bands(audio):
    """Split into bass / mids / highs. Zero-phase filters, and the three bands sum back to the original exactly."""
    low = sosfiltfilt(butter(4, BASS_HZ, "low", fs=SAMPLE_RATE, output="sos"), audio, axis=-1)
    low_mid = sosfiltfilt(butter(4, HIGH_HZ, "low", fs=SAMPLE_RATE, output="sos"), audio, axis=-1)
    return low, low_mid - low, audio - low_mid


def sweep_filter(audio, kind, keyframes, block=256):
    """High- or low-pass filter whose cutoff moves over time. keyframes: (seconds, Hz), interpolated on a log scale."""
    key_times, key_hz = zip(*keyframes)
    out = np.empty_like(audio)
    zi = np.zeros((1, audio.shape[0], 2))
    for start in range(0, audio.shape[1], block):
        t = (start + block / 2) / SAMPLE_RATE
        cutoff = float(np.exp(np.interp(t, key_times, np.log(key_hz))))
        sos = butter(2, min(cutoff, 0.45 * SAMPLE_RATE), kind, fs=SAMPLE_RATE, output="sos")
        out[:, start:start + block], zi = sosfilt(sos, audio[:, start:start + block], axis=-1, zi=zi)
    return out


def with_edges(chunk):
    """Tiny fade at both ends so a repeated chunk doesn't click."""
    n = min(int(CLICK_FREE * SAMPLE_RATE), chunk.shape[1] // 4)
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    chunk = chunk.copy()
    chunk[:, :n] *= ramp
    chunk[:, -n:] *= ramp[::-1]
    return chunk


def fmt(seconds):
    seconds = round(float(seconds), 2)
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


class Transition:
    """Shared timeline for one A -> B transition.

    Output time 0 is A's downbeat `pre` bars before `out_bar`. B's `in_bar` downbeat lands on A's `out_bar`.
    bar(k) is the output time of bar k of the overlap, on A's beat grid.
    """

    def __init__(self, song_a, song_b, audio_a, audio_b, out_bar, in_bar, overlap, pre, post):
        self.song_a, self.song_b = song_a, song_b
        self.db_a = np.array(song_a["downbeats"])
        self.db_b = np.array(song_b["downbeats"])
        self.out_bar, self.in_bar, self.overlap = out_bar, in_bar, overlap
        self.start_bar = max(0, out_bar - pre)

        if out_bar + max(overlap, 1) >= len(self.db_a):
            raise SystemExit(f"{song_a['title']} only has {len(self.db_a)} bars; out-bar + overlap is too late")
        if in_bar + max(overlap, 1) >= len(self.db_b):
            raise SystemExit(f"{song_b['title']} only has {len(self.db_b)} bars; in-bar + overlap is too late")

        # Beatmatch: speed B so its bars take as long as A's around the transition
        tempo_bars = min(TEMPO_BARS, len(self.db_a) - 1 - out_bar, len(self.db_b) - 1 - in_bar)
        a_span = self.db_a[out_bar + tempo_bars] - self.db_a[out_bar]
        b_span = self.db_b[in_bar + tempo_bars] - self.db_b[in_bar]
        self.speed = b_span / a_span
        self.beat = a_span / tempo_bars / 4

        self.a0 = self.db_a[self.start_bar]
        self.a_end = self.db_a[min(out_bar + overlap + 1, len(self.db_a) - 1)]
        self.b_offset = self.db_a[out_bar] - self.a0
        self.last_b_bar = min(in_bar + overlap + post, len(self.db_b) - 1)
        self.b_start, self.b_end = self.db_b[in_bar], self.db_b[self.last_b_bar]

        self.gain_db = float(np.clip(song_a["loudness"] - song_b["loudness"], -MAX_LOUDNESS_MATCH_DB, MAX_LOUDNESS_MATCH_DB))
        self.audio_a, self.audio_b = audio_a, audio_b
        self.seg_a = self.prep_a(audio_a)
        self.seg_b = self.prep_b(audio_b)
        self.n = int(round(self.b_offset * SAMPLE_RATE)) + self.seg_b.shape[1]
        self.total = self.n / SAMPLE_RATE

    def bar(self, k):
        return self.db_a[self.out_bar + k] - self.a0

    def prep_a(self, audio):
        """Crop A (or one of its stems) to this transition, placed on the output timeline."""
        seg = audio[:, int(self.a0 * SAMPLE_RATE):int(self.a_end * SAMPLE_RATE)]
        out = np.zeros((2, max(seg.shape[1], 1)), dtype=np.float32)
        out[:, :seg.shape[1]] = seg
        return out

    def prep_b(self, audio):
        """Crop, beatmatch and loudness-match B (or one of its stems)."""
        seg = audio[:, int(self.b_start * SAMPLE_RATE):int(self.b_end * SAMPLE_RATE)]
        if abs(self.speed - 1) > 0.001:
            seg = time_stretch(seg, SAMPLE_RATE, stretch_factor=self.speed)
        return (seg * 10 ** (self.gain_db / 20)).astype(np.float32)

    def place_a(self, x):
        out = np.zeros((2, self.n), dtype=np.float32)
        m = min(x.shape[1], self.n)
        out[:, :m] = x[:, :m]
        return out

    def place_b(self, x):
        out = np.zeros((2, self.n), dtype=np.float32)
        at = int(round(self.b_offset * SAMPLE_RATE))
        m = min(x.shape[1], self.n - at)
        out[:, at:at + m] = x[:, :m]
        return out

    def env(self, keyframes):
        """Gain curve over the output from (seconds, position 0-1) keyframes, shaped so crossfades keep constant power."""
        times = np.arange(self.n) / SAMPLE_RATE
        key_times, key_values = zip(*keyframes)
        return np.sin(np.interp(times, key_times, key_values) * np.pi / 2).astype(np.float32)

    def cut_env(self, at, before, after):
        return self.env([(0, before), (at - CLICK_FREE, before), (at, after)])

    def stems(self):
        if not hasattr(self, "_stems"):
            self._stems = (
                {k: self.prep_a(v) for k, v in load_stems(self.song_a, self.audio_a).items()},
                {k: self.prep_b(v) for k, v in load_stems(self.song_b, self.audio_b).items()},
            )
        return self._stems

    def b_vocals_after(self, at):
        """B placed on the timeline with its vocals muted until `at`, so two songs' vocals never overlap."""
        stems_b = self.stems()[1]
        instrumental = self.place_b(stems_b["drums"] + stems_b["bass"] + stems_b["other"])
        vocals = self.place_b(stems_b["vocals"]) * self.env([(0, 0), (at, 0), (at + self.beat, 1)])
        return instrumental + vocals


# ---------- techniques ----------
# Each returns (mix before mastering, list of (seconds, description) markers).

def blend(t):
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    a_low, a_mid, a_high = split_bands(t.seg_a)
    b_low, b_mid, b_high = split_bands(t.b_vocals_after(t.bar(ov)))

    mix = (
        t.place_a(a_low) * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + t.place_a(a_mid + a_high) * t.env([(0, 1), (swap, 1), (t.bar(ov), 0)])
        + b_low * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + (b_mid + b_high) * t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), 1)])
    )
    return mix, [
        (t.bar(0), "B's instrumental starts fading in (mids/highs only, bass cut)"),
        (t.bar(quarter), "B mids/highs at full volume"),
        (swap, f"bass swap on bar {half} of the overlap"),
        (t.bar(ov), "A fully faded out; B's vocals come in"),
    ]


def filter_sweep(t):
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat

    # B: starts thin (high-pass at 1 kHz), opens up to full range, bass arrives at the halfway downbeat
    b = sweep_filter(t.b_vocals_after(t.bar(ov)), "high", [(t.bar(0), 1000), (swap - 2 * t.beat, 150), (swap, 20), (t.total, 20)])
    b *= t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), 1)])

    # A: bass drops out at the halfway downbeat, then the high-pass climbs until A is just a thin hiss
    a = sweep_filter(t.place_a(t.seg_a), "high", [(0, 20), (swap, 20), (swap_done, 250), (t.bar(ov), 2500)])
    a *= t.env([(0, 1), (t.bar(ov - max(2, ov // 4)), 1), (t.bar(ov), 0)])

    return a + b, [
        (t.bar(0), "B enters through a high-pass filter (thin, no bass)"),
        (swap, "filters cross: B opens to full range with bass, A's bass is filtered out"),
        (t.bar(ov), "A filtered away completely; B's vocals come in"),
    ]


def echo_out(t):
    cut = t.bar(0)
    a = t.place_a(t.seg_a)
    dry = a * t.cut_env(cut, 1, 0)

    # Feed A's last beat into a beat-synced delay; the echoes keep going after A is cut
    send = a * t.env([(0, 0), (cut - t.beat - CLICK_FREE, 0), (cut - t.beat, 1), (cut - CLICK_FREE, 1), (cut, 0)])
    wet = Delay(delay_seconds=float(t.beat), feedback=0.55, mix=1.0)(send, SAMPLE_RATE)
    wet = HighpassFilter(cutoff_frequency_hz=400)(wet, SAMPLE_RATE)
    wet = wet * 0.9 * t.env([(0, 1), (cut, 1), (t.bar(4), 0)])

    b = t.b_vocals_after(t.bar(4)) * t.cut_env(cut, 0, 1)
    return dry + wet + b, [
        (cut - t.beat, "echo catches A's last beat"),
        (cut, "A cut; B's beat drops in at full volume while A's echoes fade"),
        (t.bar(4), "echoes gone; B's vocals come in"),
    ]


def hard_cut(t):
    cut = t.bar(0)
    mix = t.place_a(t.seg_a) * t.cut_env(cut, 1, 0) + t.place_b(t.seg_b) * t.cut_env(cut, 0, 1)
    return mix, [(cut, "hard cut from A into B on the downbeat")]


def loop_roll(t):
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat

    # Loop source: A's bar at out_bar, so the first repeat is seamless with what A would have played
    src = t.seg_a[:, int(t.bar(0) * SAMPLE_RATE):int(t.bar(1) * SAMPLE_RATE)]
    loop = np.zeros((2, t.n), dtype=np.float32)
    for k in range(ov):
        beats = 4 if k < ov - 4 else 2 if k < ov - 2 else 1 if k == ov - 2 else 0.5
        chunk = with_edges(src[:, :int(src.shape[1] * beats / 4)])
        pos, end = int(t.bar(k) * SAMPLE_RATE), int(t.bar(k + 1) * SAMPLE_RATE)
        while pos < end:
            m = min(chunk.shape[1], end - pos)
            loop[:, pos:pos + m] += chunk[:, :m]
            pos += m

    dry = t.place_a(t.seg_a) * t.cut_env(t.bar(0), 1, 0)
    loop = sweep_filter(loop, "high", [(0, 20), (swap, 20), (swap_done, 200), (t.bar(ov), 800)])
    loop *= t.cut_env(t.bar(ov), 1, 0)

    b_low, b_mid, b_high = split_bands(t.seg_b)
    b = (
        t.place_b(b_low) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(b_mid + b_high) * t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), 1)])
    )
    return dry + loop + b, [
        (t.bar(0), "A's bar starts looping (4 beats) while B fades in"),
        (swap, "bass swap: loop loses its bass, B's bass comes in"),
        (t.bar(ov - 4), "loop roll: 2-beat loop"),
        (t.bar(ov - 2), "1-beat loop"),
        (t.bar(ov - 1), "half-beat loop"),
        (t.bar(ov), "loop released, B carries on"),
    ]


def acapella(t):
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    stems_a, stems_b = t.stems()

    a_inst = t.place_a(stems_a["drums"] + stems_a["bass"] + stems_a["other"])
    a_vox = t.place_a(stems_a["vocals"])
    mix = (
        a_inst * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + a_vox * t.env([(0, 1), (t.bar(ov - 4), 1), (t.bar(ov), 0)])
        + t.place_b(stems_b["drums"] + stems_b["other"]) * t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), 1)])
        + t.place_b(stems_b["bass"]) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(stems_b["vocals"]) * t.env([(0, 0), (t.bar(ov), 0), (t.bar(ov) + t.beat, 1)])
    )
    return mix, [
        (t.bar(0), "B's instrumental (no vocals, no bass) fades in under A"),
        (swap, "A's instrumental drops out: A's vocals now ride over B's full beat"),
        (t.bar(ov - 4), "A's vocals start fading"),
        (t.bar(ov), "B's own vocals come in"),
    ]


def acapella_layered(t, echo=False):
    """Acapella, but B builds up stem by stem: drums, then instruments, then bass at the swap."""
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    stems_a, stems_b = t.stems()

    a_inst = t.place_a(stems_a["drums"] + stems_a["bass"] + stems_a["other"])
    a_vox = t.place_a(stems_a["vocals"])
    # A's whole instrumental drops out at the swap. A gradual, layer-by-layer exit was tried and the sudden drop
    # was preferred (9/10 vs worse)
    mix = (
        a_inst * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + t.place_b(stems_b["drums"]) * t.env([(0, 0), (t.bar(0), 0), (t.bar(1), 1)])
        + t.place_b(stems_b["other"]) * t.env([(0, 0), (t.bar(quarter), 0), (t.bar(quarter + 1), 1)])
        + t.place_b(stems_b["bass"]) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(stems_b["vocals"]) * t.env([(0, 0), (t.bar(ov), 0), (t.bar(ov) + t.beat, 1)])
    )
    markers = [
        (t.bar(0), "B's drums come in under A"),
        (t.bar(quarter), "B's instruments join"),
        (swap, "swap: B's bass in, A's instrumental out, A's vocals ride over B's full beat"),
    ]

    if not echo:
        mix += a_vox * t.env([(0, 1), (t.bar(ov - 4), 1), (t.bar(ov), 0)])
        markers += [(t.bar(ov - 4), "A's vocals start fading"), (t.bar(ov), "B's own vocals come in")]
        return mix, markers

    # A's vocals play until 2 bars before the end, and their last 2 beats echo out on the beat.
    # The echoes are gone by the time B's vocals start, so the vocals never overlap.
    cut = t.bar(ov - 2)
    mix += a_vox * t.cut_env(cut, 1, 0)
    send = a_vox * t.env([(0, 0), (cut - 2 * t.beat - CLICK_FREE, 0), (cut - 2 * t.beat, 1), (cut - CLICK_FREE, 1), (cut, 0)])
    wet = Delay(delay_seconds=float(t.beat), feedback=0.5, mix=1.0)(send, SAMPLE_RATE)
    wet = HighpassFilter(cutoff_frequency_hz=300)(wet, SAMPLE_RATE)
    mix += wet * t.env([(0, 1), (cut, 1), (t.bar(ov), 0)])
    markers += [(cut, "A's last vocal line echoes out"), (t.bar(ov), "echoes gone, B's own vocals come in")]
    return mix, markers


def handover(t):
    """Stem-by-stem handover: each layer of A is swapped for B's, never stacked (from blind-test feedback).

    bar 0     B's drums in; A's drums filtered away over the same bar, so only one drum kit plays
    bars 1-h  A's melody/instruments sink under a closing low-pass while B's rise, so B takes precedence
    bar h     sudden bass swap (h = half the overlap)
    after h   only A's vocals are left over B's full beat; they end 2 bars before B's vocals start (a breather)
              - or, for an 8-bar overlap, at the end where the engine found a vocal phrase gap
    """
    ov, half = t.overlap, t.overlap // 2
    stems_a, stems_b = t.stems()
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    b_vocals_in = t.bar(ov)
    a_vocals_out = t.bar(ov - 2) if ov >= 16 else b_vocals_in

    a_drums = sweep_filter(t.place_a(stems_a["drums"]), "high", [(0, 20), (t.bar(0), 20), (t.bar(1), 600)])
    a_drums *= t.env([(0, 1), (t.bar(0), 1), (t.bar(1), 0)])
    a_other = sweep_filter(t.place_a(stems_a["other"]), "low", [(0, 20000), (t.bar(1), 20000), (swap, 400)])
    a_other *= t.env([(0, 1), (t.bar(1), 1), (swap, 0)])
    a_vocals = t.place_a(stems_a["vocals"])
    a_vocals *= (t.env([(0, 1), (t.bar(ov - 4), 1), (a_vocals_out, 0)]) if ov >= 16 else t.cut_env(a_vocals_out, 1, 0))

    mix = (
        a_drums + a_other + a_vocals
        + t.place_a(stems_a["bass"]) * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + t.place_b(stems_b["drums"]) * t.env([(0, 0), (t.bar(0), 0), (t.bar(1), 1)])
        + t.place_b(stems_b["other"]) * t.env([(0, 0), (t.bar(1), 0), (swap, 1)])
        + t.place_b(stems_b["bass"]) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(stems_b["vocals"]) * t.env([(0, 0), (b_vocals_in, 0), (b_vocals_in + t.beat, 1)])
    )
    markers = [
        (t.bar(0), "drum swap: B's drums in, A's drums filtered away"),
        (t.bar(1), "A's instruments sink under a low-pass while B's rise"),
        (swap, "bass swap; A's vocals now alone over B's full beat"),
    ]
    if ov >= 16:
        markers += [(t.bar(ov - 4), "A's vocals fade"), (a_vocals_out, "breather: beat only, no vocals")]
    markers += [(b_vocals_in, "B's vocals in")]
    return mix, markers


def eq_swap(t, fast=False, filter_out=False):
    """EQ only, no stems, modelled on djay Pro's Automix: one song is always in the background with its EQ down.

    bars 0-q  B comes in underneath: bass killed, mids/highs rise to the background level
    bar h     handover (h = half the overlap): bass swap on the downbeat, B's mids/highs up to full
              over a bar while A's go down to the background level
    bars h-ov A, now in the background, fades out
              - fast (high-energy mixes): A drops to the background within a beat and is gone 2 bars after the swap
              - filter_out: A instead sinks under a low-pass closing from 20 kHz to 300 Hz over those 2 bars
    """
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    bg = float(np.arcsin(10 ** (EQ_BACKGROUND_DB / 20)) * 2 / np.pi)  # env position that gives that gain
    fast = fast or filter_out
    a_gone = t.bar(half + 2) if fast else t.bar(ov)
    a_low, a_mid, a_high = split_bands(t.seg_a)
    b_low, b_mid, b_high = split_bands(t.seg_b)

    a_top = t.place_a(a_mid + a_high)
    if filter_out:
        a_top = sweep_filter(a_top, "low", [(0, 20000), (swap, 20000), (a_gone, 300)])
        a_top *= t.env([(0, 1), (swap, 1), (a_gone - t.beat, bg), (a_gone, 0)])
    else:
        a_top *= t.env([(0, 1), (swap, 1), (swap_done if fast else t.bar(half + 1), bg), (a_gone, 0)])

    mix = (
        t.place_a(a_low) * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + a_top
        + t.place_b(b_low) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(b_mid + b_high) * t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), bg), (swap, bg), (swap_done if fast else t.bar(half + 1), 1)])
    )
    return mix, [
        (t.bar(0), f"B comes in underneath: bass cut, mids/highs rising to {EQ_BACKGROUND_DB} dB"),
        (swap, "handover: bass swap, B's EQ back to normal, A's " + ("sinking under a low-pass" if filter_out else "turned down")),
        (a_gone, "A gone"),
    ]


def eq_in_filter_out(t):
    """EQ on the song coming in, filter on the song going out (how the user hears djay Pro's Automix).

    bars 0-q  B's highs come in first, bass killed
    bars q-h  B's mids join; highs and mids sit at the background level
    bar h     bass swap on the downbeat (A's one EQ move, so two basslines never play at once),
              B's mids/highs up to full over a bar
    bars h-ov A sinks under a low-pass closing from 20 kHz to 250 Hz, fading out over the last two beats
    """
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    bg = float(np.arcsin(10 ** (EQ_BACKGROUND_DB / 20)) * 2 / np.pi)
    a_low, a_mid, a_high = split_bands(t.seg_a)
    b_low, b_mid, b_high = split_bands(t.seg_b)

    a_top = sweep_filter(t.place_a(a_mid + a_high), "low", [(0, 20000), (swap, 20000), (t.bar(ov), 250)])
    mix = (
        t.place_a(a_low) * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + a_top * t.env([(0, 1), (t.bar(ov) - 2 * t.beat, 1), (t.bar(ov), 0)])
        + t.place_b(b_low) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(b_high) * t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), bg), (swap, bg), (t.bar(half + 1), 1)])
        + t.place_b(b_mid) * t.env([(0, 0), (t.bar(quarter), 0), (swap, bg), (t.bar(half + 1), 1)])
    )
    return mix, [
        (t.bar(0), f"B's highs come in at {EQ_BACKGROUND_DB} dB, bass cut"),
        (t.bar(quarter), "B's mids join"),
        (swap, "bass swap, B's EQ back to normal; A starts sinking under a low-pass"),
        (t.bar(ov), "A gone"),
    ]


def stems_eq(t, breath_bars=2):
    """eq_in_filter_out with the vocals on their own stems, so a song's voice never hands straight over to the other's.

    bars 0-q    B's instrumental comes in: highs first, bass killed, B's vocals muted
    bars q-h    B's mids join at the background level
    bar h-2     A's vocals fade out over a beat: `breath_bars` of instrumental only before B sings
    bar h       bass swap, B's EQ back to normal and B's vocals in; A's instrumental starts sinking under a low-pass
    bar ov      A gone
    """
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    breath = t.bar(max(1, half - breath_bars))
    bg = float(np.arcsin(10 ** (EQ_BACKGROUND_DB / 20)) * 2 / np.pi)
    stems_a, stems_b = t.stems()
    a_low, a_mid, a_high = split_bands(stems_a["drums"] + stems_a["bass"] + stems_a["other"])
    b_low, b_mid, b_high = split_bands(stems_b["drums"] + stems_b["bass"] + stems_b["other"])

    a_top = sweep_filter(t.place_a(a_mid + a_high), "low", [(0, 20000), (swap, 20000), (t.bar(ov), 250)])
    mix = (
        t.place_a(stems_a["vocals"]) * t.env([(0, 1), (breath - t.beat, 1), (breath, 0)])
        + t.place_a(a_low) * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + a_top * t.env([(0, 1), (t.bar(ov) - 2 * t.beat, 1), (t.bar(ov), 0)])
        + t.place_b(b_low) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
        + t.place_b(b_high) * t.env([(0, 0), (t.bar(0), 0), (t.bar(quarter), bg), (swap, bg), (t.bar(half + 1), 1)])
        + t.place_b(b_mid) * t.env([(0, 0), (t.bar(quarter), 0), (swap, bg), (t.bar(half + 1), 1)])
        + t.place_b(stems_b["vocals"]) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
    )
    return mix, [
        (t.bar(0), f"B's instrumental comes in: highs at {EQ_BACKGROUND_DB} dB, bass cut, vocals muted"),
        (t.bar(quarter), "B's mids join"),
        (breath, f"A's vocals out: {breath_bars} bars of instrumental only"),
        (swap, "bass swap, B's EQ back to normal and B's vocals in; A sinking under a low-pass"),
        (t.bar(ov), "A gone"),
    ]


CLASH_DB = -6  # a band at least this loud (vs the song's loud bars) in both songs clashes: one must be nearly off
CLASH_BACKGROUND_DB = -24
QUIET_VOCALS = 0.15  # a beat counts as a gap between words below 15% of the song's loud vocal level


def band_levels(stems, start, end):
    """Bass / mids / highs of the instrumental between two output times, in dB vs the song's loud bars."""
    bands = split_bands((stems["drums"] + stems["bass"] + stems["other"]).mean(0, keepdims=True))
    window = slice(int(start * SAMPLE_RATE), int(end * SAMPLE_RATE))
    out = []
    for band in bands:
        x = band[0]
        bars = np.array([np.sqrt(np.mean(c ** 2)) + 1e-9 for c in np.array_split(x, max(1, len(x) // (2 * SAMPLE_RATE)))])
        out.append(float(20 * np.log10(np.sqrt(np.mean(x[window] ** 2)) / np.percentile(bars, 95))))
    return out


def beat_vocals(vocals, first_beat, beat, count):
    """Vocal level per beat on the output timeline, 0-1 against the clip's loud vocal beats."""
    mono = vocals.mean(0)
    level = np.array([np.sqrt(np.mean(mono[int((first_beat + i * beat) * SAMPLE_RATE):int((first_beat + (i + 1) * beat) * SAMPLE_RATE)] ** 2))
                      for i in range(count)])
    ref = np.percentile(level, 90)
    return level / ref if ref > 0 else level


def smooth(t):
    """Stems + EQ, every move over bars rather than beats, for a handover between two cooled-down parts.

    Each band is checked in both songs over the overlap: where both are strong (they clash), the song in the
    background keeps that band nearly off (CLASH_BACKGROUND_DB) instead of at the usual background level.
    bars 0-h   B's highs, then mids, rise slowly to their background level; B's bass stays out
    bar h      bass crossfade over a full bar
    bars h-h+2 B's mids/highs rise to full while A's sink to the background level
    bars h-ov  A closes under a low-pass and fades out over the last two bars
    Vocals: A's fade only in a gap between words, B's come in only at a gap before a phrase, never both at once.
    """
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap = t.bar(half)
    stems_a, stems_b = t.stems()
    a_low, a_mid, a_high = split_bands(stems_a["drums"] + stems_a["bass"] + stems_a["other"])
    b_low, b_mid, b_high = split_bands(stems_b["drums"] + stems_b["bass"] + stems_b["other"])

    level_a = band_levels({k: t.place_a(v) for k, v in stems_a.items()}, t.bar(0), t.bar(ov))
    level_b = band_levels({k: t.place_b(v) for k, v in stems_b.items()}, t.bar(0), t.bar(ov))
    pos = lambda db: float(np.arcsin(min(1.0, 10 ** (db / 20))) * 2 / np.pi)  # env position for a gain in dB
    bg = [pos(CLASH_BACKGROUND_DB if la >= CLASH_DB and lb >= CLASH_DB else EQ_BACKGROUND_DB)
          for la, lb in zip(level_a, level_b)]
    names = ["bass", "mids", "highs"]
    clashes = [n for n, la, lb in zip(names, level_a, level_b) if la >= CLASH_DB and lb >= CLASH_DB]

    # Vocals: the first gap of 2+ quiet beats in A from the start of the overlap, and B's first quiet beat
    # at or after the swap that is followed by singing (so B's first word arrives whole)
    beats = ov * 4
    va = beat_vocals(t.place_a(stems_a["vocals"]), t.bar(0), t.beat, beats)
    vb = beat_vocals(t.place_b(stems_b["vocals"]), t.bar(0), t.beat, beats)
    a_gap = next((i for i in range(beats - 1) if va[i] < QUIET_VOCALS and va[i + 1] < QUIET_VOCALS), 0)
    a_vocals_out = t.bar(0) + a_gap * t.beat
    b_gap = next((i for i in range(half * 4, beats) if vb[i] < QUIET_VOCALS and (i >= a_gap + 2)), half * 4)
    b_vocals_in = t.bar(0) + b_gap * t.beat

    a_top = sweep_filter(t.place_a(a_mid + a_high), "low", [(0, 20000), (swap, 20000), (t.bar(ov), 250)])
    a_mid_f, a_high_f = split_bands(a_top)[1:]
    mix = (
        t.place_a(stems_a["vocals"]) * t.env([(0, 1), (a_vocals_out, 1), (a_vocals_out + 2 * t.beat, 0)])
        + t.place_a(a_low) * t.env([(0, 1), (swap, 1), (t.bar(half + 1), 0)])
        + a_mid_f * t.env([(0, 1), (swap, 1), (t.bar(half + 2), bg[1]), (t.bar(ov - 2), bg[1]), (t.bar(ov), 0)])
        + a_high_f * t.env([(0, 1), (swap, 1), (t.bar(half + 2), bg[2]), (t.bar(ov - 2), bg[2]), (t.bar(ov), 0)])
        + t.place_b(b_low) * t.env([(0, 0), (swap, 0), (t.bar(half + 1), 1)])
        + t.place_b(b_high) * t.env([(0, 0), (t.bar(0), 0), (swap, bg[2]), (t.bar(half + 2), 1)])
        + t.place_b(b_mid) * t.env([(0, 0), (t.bar(quarter), 0), (swap, bg[1]), (t.bar(half + 2), 1)])
        + t.place_b(stems_b["vocals"]) * t.env([(0, 0), (b_vocals_in, 0), (b_vocals_in + t.beat, 1)])
    )
    print(f"  band levels over the overlap (bass/mids/highs, dB): A {[round(x) for x in level_a]}  B {[round(x) for x in level_b]}"
          f"  -> clashing: {', '.join(clashes) or 'none'}")
    held = [n for n in clashes if n != "bass"]  # B's bass is always out until the swap
    return mix, sorted([
        (t.bar(0), "B's highs start rising" + (f" (clash in {', '.join(held)}: kept at {CLASH_BACKGROUND_DB} dB)" if held else "")),
        (t.bar(quarter), "B's mids start rising"),
        (a_vocals_out, "A's vocals fade out, in a gap between words"),
        (swap, "bass crossfade over one bar; A starts closing under a low-pass"),
        (b_vocals_in, "B's vocals in, at the start of a phrase"),
        (t.bar(half + 2), "B's EQ fully open, A at the background level"),
        (t.bar(ov), "A gone"),
    ], key=lambda m: m[0])


def eq_layered(t):
    """The layered structure (best rated so far, Amen -> REQUIRE) with EQ bands standing in for stems.

    bar 0      B's highs in (hi-hats, cymbals: the nearest band to its drums)
    bar q      B's mids join at the background level
    bar h      sudden swap: A's bass and highs drop, B's bass in, so A's mids (its vocals) ride over B's beat
    bars ov-4+ the mids swap: A's fade out while B's rise to full, as B's vocals come in
    """
    ov, half, quarter = t.overlap, t.overlap // 2, max(1, t.overlap // 4)
    swap, swap_done = t.bar(half), t.bar(half) + t.beat
    bg = float(np.arcsin(10 ** (EQ_BACKGROUND_DB / 20)) * 2 / np.pi)
    a_low, a_mid, a_high = split_bands(t.seg_a)
    b_low, b_mid, b_high = split_bands(t.seg_b)

    mix = (
        t.place_a(a_low + a_high) * t.env([(0, 1), (swap, 1), (swap_done, 0)])
        + t.place_a(a_mid) * t.env([(0, 1), (t.bar(ov - 4), 1), (t.bar(ov), 0)])
        + t.place_b(b_high) * t.env([(0, 0), (t.bar(0), 0), (t.bar(1), 1)])
        + t.place_b(b_mid) * t.env([(0, 0), (t.bar(quarter), 0), (t.bar(quarter + 1), bg), (t.bar(ov - 4), bg), (t.bar(ov), 1)])
        + t.place_b(b_low) * t.env([(0, 0), (swap, 0), (swap_done, 1)])
    )
    return mix, [
        (t.bar(0), "B's highs come in under A"),
        (t.bar(quarter), "B's mids join in the background"),
        (swap, "swap: A's bass and highs out, B's bass in; A's mids ride over B's beat"),
        (t.bar(ov - 4), "mids swap: A's fade out, B's rise"),
        (t.bar(ov), "A gone"),
    ]


def layered_short(t):
    """For okay matches: B builds in (drums, then instruments) and at the end of the overlap there's a full switch.

    No vocal ride: A plays whole (vocals included) until the switch, then B's bass and vocals arrive with it.
    Use an 8-bar overlap, switching where A's vocal phrase has ended.
    """
    ov, half = t.overlap, t.overlap // 2
    stems_a, stems_b = t.stems()
    switch = t.bar(ov)
    mix = (
        t.place_a(t.seg_a) * t.cut_env(switch, 1, 0)
        + t.place_b(stems_b["drums"]) * t.env([(0, 0), (t.bar(0), 0), (t.bar(1), 1)])
        + t.place_b(stems_b["other"]) * t.env([(0, 0), (t.bar(half), 0), (t.bar(half + 1), 1)])
        + t.place_b(stems_b["bass"] + stems_b["vocals"]) * t.cut_env(switch, 0, 1)
    )
    return mix, [
        (t.bar(0), "B's drums come in under A"),
        (t.bar(half), "B's instruments join"),
        (switch, "full switch: A out completely, B's bass and vocals in"),
    ]


def vocal_preview(t):
    """B's vocals arrive first over A's beat, then a sudden switch to all of B. Only one beat ever plays.

    For pairs whose beats clash when layered. Use a short overlap (8 bars) and an in-bar where B has vocals.
    """
    ov = t.overlap
    stems_a, stems_b = t.stems()
    mix = (
        t.place_a(stems_a["drums"] + stems_a["bass"] + stems_a["other"]) * t.cut_env(t.bar(ov), 1, 0)
        + t.place_a(stems_a["vocals"]) * t.cut_env(t.bar(0), 1, 0)
        + t.place_b(stems_b["vocals"]) * t.cut_env(t.bar(0), 0, 1)
        + t.place_b(stems_b["drums"] + stems_b["bass"] + stems_b["other"]) * t.cut_env(t.bar(ov), 0, 1)
    )
    return mix, [
        (t.bar(0), "A's vocals out, B's vocals in over A's beat"),
        (t.bar(ov), "sudden switch: A's beat out, all of B in"),
    ]


RENDERERS = {
    "eq": eq_swap,
    "eq_fast": lambda t: eq_swap(t, fast=True),
    "eq_filter": lambda t: eq_swap(t, fast=True, filter_out=True),
    "eq_layered": eq_layered,
    "eq_in_filter_out": eq_in_filter_out,
    "stems_eq": stems_eq,
    "smooth": smooth,
    "handover": handover,
    "layered_short": layered_short,
    "vocal_preview": vocal_preview,
    "layered": acapella_layered,
    "layered_echo": lambda t: acapella_layered(t, echo=True),
    "blend": blend,
    "filter": filter_sweep,
    "echo": echo_out,
    "cut": hard_cut,
    "loop": loop_roll,
    "acapella": acapella,
}


def master(t, mix):
    edge = t.env([(0, 0), (0.3, 1), (t.total - 8 * t.beat, 1), (t.total, 0)])
    mix = Limiter(threshold_db=-1.0)(mix * edge, SAMPLE_RATE)
    # The limiter isn't a hard ceiling, so cap the peak at -1 dBFS to avoid clipping
    ceiling = 10 ** (-1 / 20)
    peak = np.abs(mix).max()
    return mix * (ceiling / peak) if peak > ceiling else mix


def first_drop(song, max_bar=32):
    for section in song["sections"]:
        if section["label"] == "drop" and 0 < section["startBar"] <= max_bar:
            return section["startBar"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="song_a", required=True)
    parser.add_argument("--to", dest="song_b", required=True)
    parser.add_argument("--out-bar", type=int, required=True, help="bar of song A where song B comes in")
    parser.add_argument("--in-bar", type=int, default=0, help="bar of song B that starts on --out-bar")
    parser.add_argument("--overlap", type=int, default=16, help="bars both songs play together")
    parser.add_argument("--pre", type=int, default=16, help="bars of song A to play before the transition")
    parser.add_argument("--post", type=int, default=16, help="bars of song B to play after the transition")
    parser.add_argument("--technique", choices=TECHNIQUES + ("all",), default="all")
    args = parser.parse_args()

    collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    song_a = find_song(collection, args.song_a)
    song_b = find_song(collection, args.song_b)
    audio_a, audio_b = load_stereo(song_a), load_stereo(song_b)

    folder = os.path.join(RENDER_DIR, safe_name(f"{song_a['title']} to {song_b['title']}"))
    os.makedirs(folder, exist_ok=True)
    techniques = DEFAULT_TECHNIQUES if args.technique == "all" else (args.technique,)

    for technique in techniques:
        # Echo and cut have no overlap, so B comes in on its first drop for maximum impact
        in_bar = args.in_bar
        if technique in ("echo", "cut") and args.in_bar == 0:
            in_bar = first_drop(song_b) or 0

        t = Transition(song_a, song_b, audio_a, audio_b, args.out_bar, in_bar, args.overlap, args.pre, args.post)
        mix, markers = RENDERERS[technique](t)
        mix = master(t, mix)

        wav_path = os.path.join(folder, f"{technique}.wav")
        with AudioFile(wav_path, "w", SAMPLE_RATE, num_channels=2, bit_depth=16) as f:
            f.write(mix)
        markers = [(0, f"{song_a['title']} from {fmt(t.a0)} (bar {t.start_bar})")] + markers
        with open(os.path.join(folder, f"{technique}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "from": song_a["title"], "to": song_b["title"], "technique": technique,
                "outBar": args.out_bar, "inBar": in_bar, "overlapBars": args.overlap,
                "speed": round(float(t.speed), 5), "bGainDb": round(t.gain_db, 2),
                "markers": [{"time": round(float(s), 3), "event": e} for s, e in markers],
            }, f, indent=2, ensure_ascii=False)

        print(f"\n{technique.upper()}  ->  {wav_path}")
        print(f"  {song_b['title']} starts at its bar {in_bar} ({fmt(t.b_start)}), speed x{t.speed:.4f}, gain {t.gain_db:+.1f} dB")
        for s, event in markers:
            print(f"  {fmt(s)}  {event}")


if __name__ == "__main__":
    main()
