"""Build one continuous mix: pick a chain of compatible songs and every entry/exit point together, render it with EQ.

Runs inside WSL with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/make_mix.py"

  1. Candidates: for every pair of songs with close tempos, every way to go from any phrase of A to any phrase of B
     is scored (vibe + how well the points suit the eq swap). Hard rules: keys at most one Camelot step apart, and
     never both songs singing in the same bar of the overlap (EQ can't mute vocals, so the timing has to avoid it).
  2. Optimisation: a beam search over whole mixes picks the song order AND every entry/exit point together, for the
     best total. Each song plays at least MIN_SOLO_BARS on its own between entering and handing over, so a slightly
     weaker A -> B can win if it leaves B in a better place for B -> C.
  3. Render: the part of each song that plays is time-stretched once to the mix tempo (no tempo jumps), set to one
     loudness, and the transitions are the `eq` swap. Songs render in parallel; one limiter pass over the finished mix.

A theme narrows the ~400 ready songs to a pool the search can handle, e.g. a calm amapiano mix:
    python make_mix.py --genre amapiano --energy low --out renders/mix/amapiano.wav

--stems uses only songs split by stems_scan.py and mixes them with stems + EQ instead (the `smooth` handover):
transitions go where A has cooled down and B is building (from the per-bar stemBands, not section labels), A's vocals
fade in a pause between phrases and B's come in at the next pause after the swap, and a band strong in both songs
is held nearly off in the background song.
    python make_mix.py --stems --genre r&b --energy low --songs 3 --out renders/mix/rnb_calm.wav
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations

import numpy as np
from pymongo import MongoClient

import auto_dj as dj
import lastfm_tags as lfm
import render_transition as rt

MIN_SOLO_BARS = 32
MAX_MIX_TEMPO_GAP = 0.03  # every song within 3% of the mix tempo, so the stretch stays inaudible
VOCALS_ON = 0.2  # a bar counts as singing above 20% of the song's loud vocal level (ignores ad-libs and stem bleed)
BEAM = 3000  # partial mixes kept at each step of the search
OPTIONS_PER_PAIR = 4  # different entry bars kept per next song, so later transitions have room to choose
POOL_MAX = 24  # songs the search considers; pairs grow with the square of this
ENERGY = {"low": (0, 0.4), "mid": (0.3, 0.7), "high": (0.6, 1)}  # felt-energy rank bands (they overlap a little)
WORKERS = 4  # songs rendered at once (Rubber Band uses one core each)
MARGIN = 1.0  # seconds kept either side of the used part of a song while stretching, trimmed off after
SR = rt.SAMPLE_RATE
STEM_OVERLAPS = (16, 8)
LONG_VIBE = 0.7  # stems: only pairs this close in vibe get the 16-bar overlap ("greater matches can handle longer")
COOLED_DB = 6  # stems: A this far under its loud bars over the overlap counts as fully cooled down
MIN_BAR_DB = -12  # stems: no bar of A before the swap or of B after it under this, or the mix falls into a hole


def four_beat_bars(song):
    """The downbeats are 4 beats apart at the song's tempo. When the tracker misses (Dide O came out with 3-beat
    bars), the render would stretch the song to make its bars fit, 0.76x for Dide O."""
    return abs(np.median(np.diff(song["downbeats"])) * song["bpm"] / 240 - 1) < 0.02


def theme_pool(lib, titles, genres=(), energy=None, bpm=None, rank_among=None):
    """Songs that fit the theme and share one tempo (within MAX_MIX_TEMPO_GAP), at most POOL_MAX of them.

    Without --bpm, the tempo is the one most of the matching songs share. Felt energy is ranked among `rank_among`
    (default: `titles`), so narrowing the songs first doesn't shift what counts as calm."""
    energies = dj.felt_energy({t: lib.profiles[t] for t in rank_among or titles})
    cache = lfm.load_cache()
    fits = [t for t in titles
            if (not genres or set(genres) & set(lfm.genres_for(cache, t, lib.songs[t]["artist"])))
            and (not energy or ENERGY[energy][0] <= energies[t] <= ENERGY[energy][1])]
    tempo = {t: lib.songs[t]["bpm"] for t in fits}
    if not bpm:
        # Each candidate centre's pool spans +-half the allowed gap, so every pair in it is within the gap
        counts = Counter({c: sum(abs(b - c) / c <= MAX_MIX_TEMPO_GAP / 2 for b in tempo.values()) for c in tempo.values()})
        bpm = counts.most_common(1)[0][0] if counts else 0
    near = [t for t in fits if abs(tempo[t] - bpm) / bpm <= MAX_MIX_TEMPO_GAP / 2] if bpm else []
    return sorted(near, key=lambda t: abs(tempo[t] - bpm))[:POOL_MAX], bpm


def candidates(lib, a, b):
    """Every valid eq transition from a to b as (score, out_bar, in_bar, overlap, key steps, parts), best first."""
    sa, sb = lib.songs[a], lib.songs[b]
    va, vb = lib.bar_vocals(a), lib.bar_vocals(b)
    n_a, n_b = len(sa["downbeats"]), len(sb["downbeats"])
    level = {bar: s["level"] for s in sa["sections"] for bar in range(s["startBar"], s["endBar"])}
    energy_a, energy_b = np.array(sa["barEnergy"]), np.array(sb["barEnergy"])
    b_starts = {s["startBar"] for s in sb["sections"]}
    vibe, _ = lib.vibe(a, b)
    found = []

    for min_score, _, overlap in dj.TIERS:  # better matches may hold the longer overlap
        half, quarter = overlap // 2, max(1, overlap // 4)
        outs = [x for x in range(4, n_a - overlap - 1) if (x - sa["phraseOffset"]) % 4 == 0]
        ins = [y for y in range(0, n_b - overlap - 4) if (y - sb["phraseOffset"]) % 4 == 0]
        keys_b = {y: dj.keys_between(lib.profiles[b], y, y + overlap) for y in ins}
        sing_b = {y: vb[y:y + overlap] > VOCALS_ON for y in ins}

        for x in outs:
            keys_a = dj.keys_between(lib.profiles[a], x, x + overlap)
            sing_a = va[x:x + overlap] > VOCALS_ON
            calm = 1 - np.mean([level.get(z) == "high" for z in range(x + quarter, x + overlap)])  # A calms as B builds
            quiet_after_swap = 1 - min(1, va[x + half:x + overlap].mean() / 0.6)  # A not mid-sentence once it's faded
            before_swap = energy_a[x + half - 2:x + half].mean()
            for y in ins:
                if np.any(sing_a & sing_b[y]):
                    continue  # the vocal rule: never two songs singing at once
                steps = max(dj.cam_dist(ka, kb) for ka in keys_a for kb in keys_b[y])
                if steps > dj.MAX_KEY_STEPS:
                    continue
                parts = {
                    "calm": float(calm),
                    "quietAfterSwap": float(quiet_after_swap),
                    "vocalsArriveOnTime": float(min(1, vb[y + overlap:y + overlap + 4].mean() / 0.6)),
                    "key": 1.0 if steps == 0 else 0.8,
                    # B takes over on a section start: best right at the bass swap, so its drop arrives as A's bass
                    # leaves. Can't Be Stopped -> FInally swapped onto FInally's thin intro, its drop only after
                    "onNewSection": 1.0 if y + half in b_starts else 0.5 if y + overlap in b_starts else 0.0,
                    # The mix doesn't sag at the handover: B's bars after the swap are about as full as A's before
                    "energyHolds": float(np.clip(1 - max(0, before_swap - energy_b[y + half:y + half + 2].mean()) / 0.3, 0, 1)),
                }
                score = 100 * (0.5 * vibe + 0.5 * np.mean(list(parts.values())))
                if score >= min_score:
                    found.append((score, x, y, overlap, steps, {k: round(v, 2) for k, v in parts.items()}))
    return sorted(found, key=lambda c: -c[0])


def bar_level(song):
    """Per bar instrumental level in dB (the three bands' power together), 0 = as loud as the song's loud bars."""
    return 10 * np.log10(np.mean(10 ** (np.array(song["stemBands"]) / 10), axis=1))


def bar_beats(song):
    """Index into song["beats"] of every bar's first beat."""
    return np.searchsorted(song["beats"], np.array(song["downbeats"]) - 0.05)


def vocal_handover(sa, sb, x, y, overlap):
    """(beat A's vocals fade, beat B's vocals come in), counted in beats from the start of the overlap, or None.

    A's vocals fade at the start of its longest pause (2+ quiet beats) in the first half, so the cut lands at the end
    of a phrase rather than inside a word. B's come in at its first quiet beat at or after the swap and at least a bar
    after A's, so its first word arrives whole after a breath of no vocals."""
    span = overlap * 4 + 16
    va = np.array(sa["vocalBeats"])[bar_beats(sa)[x]:][:span]
    vb = np.array(sb["vocalBeats"])[bar_beats(sb)[y]:][:span]
    quiet_a, quiet_b = va < rt.QUIET_VOCALS, vb < rt.QUIET_VOCALS
    runs = []
    for i in range(overlap * 2 + 1):
        if quiet_a[i] and (i == 0 or not quiet_a[i - 1]):
            n = np.argmin(np.append(quiet_a[i:], False))
            if n >= 2:
                runs.append((n, -i))
    if not runs:
        return None
    a_out = -max(runs)[1]
    b_in = next((i for i in range(max(overlap * 2, a_out + 4), min(overlap * 4 + 8, len(vb))) if quiet_b[i]), None)
    return None if b_in is None else (a_out, b_in)


def stem_candidates(lib, a, b):
    """Every valid stem transition from a to b as (score, out_bar, in_bar, overlap, key steps, parts), best first."""
    sa, sb = lib.songs[a], lib.songs[b]
    level_a, level_b = bar_level(sa), bar_level(sb)
    bands_a, bands_b = np.array(sa["stemBands"]), np.array(sb["stemBands"])
    va, vb = np.array(sa["vocalBeats"]), np.array(sb["vocalBeats"])
    beats_a, beats_b = bar_beats(sa), bar_beats(sb)
    n_a, n_b = len(sa["downbeats"]), len(sb["downbeats"])
    vibe, _ = lib.vibe(a, b)
    found = []

    for overlap in STEM_OVERLAPS:
        if overlap > min(STEM_OVERLAPS) and vibe < LONG_VIBE:
            continue
        half, quarter = overlap // 2, max(1, overlap // 4)
        outs = [x for x in range(8, n_a - overlap - 1) if (x - sa["phraseOffset"]) % 4 == 0]
        ins = [y for y in range(0, n_b - overlap - 12) if (y - sb["phraseOffset"]) % 4 == 0]
        keys_b = {y: dj.keys_between(lib.profiles[b], y, y + overlap) for y in ins}
        # B builds: its first half under the bars after the overlap, i.e. it grows into the handover
        builds = {y: float(np.clip((level_b[y + overlap:y + overlap + 8].mean() - level_b[y:y + half].mean()) / 4, 0, 1)) for y in ins}
        loud_b = {y: bands_b[y:y + overlap, 1:].mean(0) >= rt.CLASH_DB for y in ins}

        for x in outs:
            keys_a = dj.keys_between(lib.profiles[a], x, x + overlap)
            during = level_a[x + quarter:x + overlap].mean()
            # A cooled down: well under its loud bars, and quieter than just before the overlap
            cooled = 0.5 * np.clip(-during / COOLED_DB, 0, 1) + 0.5 * np.clip((level_a[x - 8:x].mean() - during) / 3, 0, 1)
            loud_a = bands_a[x:x + overlap, 1:].mean(0) >= rt.CLASH_DB
            if level_a[x:x + half + 1].min() < MIN_BAR_DB:
                continue
            for y in ins:
                if level_b[y + half:y + overlap].min() < MIN_BAR_DB:
                    continue
                steps = max(dj.cam_dist(ka, kb) for ka in keys_a for kb in keys_b[y])
                if steps > dj.MAX_KEY_STEPS:
                    continue
                vocals = vocal_handover(sa, sb, x, y, overlap)
                if vocals is None:
                    continue  # A never pauses, or B never does after the swap: no clean vocal handover
                a_out, b_in = vocals
                muted = va[beats_a[x] + a_out:beats_a[x] + overlap * 4].mean() if a_out < overlap * 4 else 0.0
                parts = {
                    "aCooled": float(cooled),
                    "bBuilds": builds[y],
                    # Mids / highs strong in both songs clash; the render holds them nearly off, but fewer is better
                    "noClash": 1 - float(np.mean(loud_a & loud_b[y])),
                    "aPhraseEnds": float(1 - np.clip(muted / 0.4, 0, 1)),  # little of A's singing muted after its fade
                    "bVocalsArrive": float(min(1, vb[beats_b[y] + b_in:beats_b[y] + b_in + 16].mean() / 0.5)),
                    "key": 1.0 if steps == 0 else 0.8,
                }
                score = 100 * (0.4 * vibe + 0.6 * np.mean(list(parts.values())))
                found.append((score, x, y, overlap, steps, {k: round(v, 2) for k, v in parts.items()}))
    return sorted(found, key=lambda c: -c[0])


def plan_mix(lib, titles, length, fixed=None, avoid=(), max_avoided=0, find_candidates=candidates):
    """Song order and every entry/exit bar for the best total score, by beam search over whole mixes.

    At most `max_avoided` songs from `avoid` (e.g. the last mix) may be used."""
    bpm = {t: lib.songs[t]["bpm"] for t in titles}
    bars = {t: len(lib.songs[t]["downbeats"]) for t in titles}
    cands = {}
    for a, b in permutations(titles, 2):
        if fixed and (a, b) not in zip(fixed, fixed[1:]):
            continue
        if abs(bpm[a] - bpm[b]) / bpm[b] <= MAX_MIX_TEMPO_GAP:
            found = find_candidates(lib, a, b)
            if found:
                cands[a, b] = found
    print(f"{len(cands)} song pairs have at least one valid transition")

    # A partial mix: (total score, songs so far, last song's entry bar, overlap it entered with, transitions)
    beam = [(0.0, (t,), 0, 0, ()) for t in (fixed[:1] if fixed else titles) if t not in avoid or max_avoided > 0]
    for step in range(1, length):
        last = step == length - 1
        grown = []
        for total, chain, entry, overlap_in, plan in beam:
            a = chain[-1]
            for b in ([fixed[step]] if fixed else [t for t in titles if t not in chain]):
                if b in avoid and sum(t in avoid for t in chain) >= max_avoided:
                    continue
                tempos = [bpm[t] for t in chain + (b,)]
                if (a, b) not in cands or max(tempos) / min(tempos) - 1 > MAX_MIX_TEMPO_GAP:
                    continue
                entries = set()
                for score, x, y, overlap, steps, parts in cands[a, b]:
                    if x < entry + overlap_in + MIN_SOLO_BARS:
                        continue  # A hasn't had its time on its own yet
                    # B needs its own solo time, plus room to hand over to the next song unless it's the last
                    if y + overlap + MIN_SOLO_BARS + (0 if last else dj.TIERS[-1][2] + 1) > bars[b] or y in entries:
                        continue
                    entries.add(y)
                    grown.append((total + score, chain + (b,), y, overlap, plan + ({
                        "from": a, "to": b, "outBar": x, "inBar": y, "overlap": overlap, "keySteps": steps,
                        "score": round(score, 1), "parts": parts},)))
                    if len(entries) == OPTIONS_PER_PAIR:
                        break
        beam = sorted(grown, key=lambda s: -s[0])[:BEAM]
        if not beam:
            return None, None
    return list(beam[0][1]), list(beam[0][4])


def env(times, keyframes):
    """Constant-power gain curve from (seconds, position 0-1) keyframes, same shaping as render_transition."""
    key_times, key_values = zip(*keyframes)
    return np.sin(np.interp(times, key_times, key_values) * np.pi / 2).astype(np.float32)


def crop(audio, speed, start, end):
    """The part of a song that plays, stretched to the mix tempo: times are on the stretched song's clock (end None =
    to the end). Only that part is stretched (plus MARGIN either side, so Rubber Band has settled at the edges),
    which is most of the render time saved."""
    total = audio.shape[1] / SR / speed
    end = total if end is None else min(end, total)
    lo, hi = max(0.0, start - MARGIN), min(total, end + MARGIN)
    seg = audio[:, int(lo * speed * SR):int(hi * speed * SR)]
    if abs(speed - 1) > 0.001:
        seg = rt.time_stretch(seg, SR, stretch_factor=speed)
    return seg[:, int(round((start - lo) * SR)):][:, :int(round((end - start) * SR))]


def render_song(song, speed, start, end, offset, gain_db, envelopes):
    """One song's part of the mix with EQ envelopes. Returns (sample where it lands in the mix, audio)."""
    seg = crop(rt.load_stereo(song), speed, start, end) * 10 ** (gain_db / 20)
    times = offset + start + np.arange(seg.shape[1]) / SR

    low, mid, high = rt.split_bands(seg)
    g_low = np.ones(seg.shape[1], dtype=np.float32)
    g_mh = np.ones(seg.shape[1], dtype=np.float32)
    for low_keys, mh_keys in envelopes:
        g_low *= env(times, low_keys)
        g_mh *= env(times, mh_keys)
    print(f"    {song['title']} (x{speed:.4f}) done")
    return int(round((offset + start) * SR)), (low * g_low + (mid + high) * g_mh).astype(np.float32)


def render_song_stems(song, speed, start, end, offset, gain_db, envelopes, lowpass):
    """One song's part of the mix from its stems: `envelopes` maps bass / mids / highs / vocals to keyframe lists,
    `lowpass` is (mix seconds, Hz) keyframes closing the mids and highs as the song leaves, or None."""
    stems = rt.load_stems(song)
    gain = 10 ** (gain_db / 20)
    instrumental = crop(stems["drums"] + stems["bass"] + stems["other"], speed, start, end) * gain
    vocals = crop(stems["vocals"], speed, start, end) * gain
    times = offset + start + np.arange(instrumental.shape[1]) / SR
    curve = lambda part: np.prod([env(times, k) for k in envelopes[part]], axis=0) if envelopes[part] else 1.0

    low, mid, high = rt.split_bands(instrumental)
    top = mid * curve("mids") + high * curve("highs")
    if lowpass:
        at = max(0, int(round((lowpass[0][0] - offset - start) * SR)))
        top[:, at:] = rt.sweep_filter(top[:, at:], "low", [(t - offset - start - at / SR, hz) for t, hz in lowpass])
    print(f"    {song['title']} (x{speed:.4f}) done")
    out = low * curve("bass") + top + vocals[:, :low.shape[1]] * curve("vocals")
    return int(round((offset + start) * SR)), out.astype(np.float32)


def timeline(songs, plan):
    """Mix bar length, each song's stretch, its downbeats on its stretched clock, and where it lands in the mix:
    song i's time t plays at offsets[i] + t."""
    bar_lens = [np.median(np.diff(s["downbeats"])) for s in songs]
    mix_bar = float(np.median(bar_lens))
    speeds = [bl / mix_bar for bl in bar_lens]
    downbeats = [np.array(s["downbeats"]) / sp for s, sp in zip(songs, speeds)]
    offsets = [0.0]
    for i, p in enumerate(plan):
        k = np.arange(p["overlap"] + 1)
        # Least-squares alignment over the whole overlap, so any small tempo drift is split evenly
        offsets.append(float(np.mean(offsets[i] + downbeats[i][p["outBar"] + k] - downbeats[i + 1][p["inBar"] + k])))
    return mix_bar, speeds, downbeats, offsets


def song_spans(songs, plan, downbeats):
    """(start, end) of each song on its stretched clock: from where it enters to the end of its outgoing overlap."""
    return [(0.0 if i == 0 else downbeats[i][plan[i - 1]["inBar"]],
             None if i == len(songs) - 1 else downbeats[i][plan[i]["outBar"] + plan[i]["overlap"]] + 0.05)
            for i in range(len(songs))]


def loudness_gains(songs):
    target = float(np.median([s["loudness"] for s in songs]))
    return [float(np.clip(target - s["loudness"], -rt.MAX_LOUDNESS_MATCH_DB, rt.MAX_LOUDNESS_MATCH_DB)) for s in songs]


def finish(pieces, out_path):
    """Sum the songs, one limiter pass, cap the peak at -1 dBFS and write the WAV. Returns the mix."""
    n = max(at + x.shape[1] for at, x in pieces)
    mix = np.zeros((2, n), dtype=np.float32)
    for at, x in pieces:
        mix[:, at:at + x.shape[1]] += x
    mix = rt.Limiter(threshold_db=-1.0)(mix, SR)
    ceiling = 10 ** (-1 / 20)
    peak = np.abs(mix).max()
    if peak > ceiling:
        mix *= ceiling / peak
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rt.AudioFile(out_path, "w", SR, num_channels=2, bit_depth=16) as f:
        f.write(mix)
    return mix


def tracklist_of(songs, trans):
    return [{"time": round(0.0 if i == 0 else trans[i - 1]["start"], 2), "title": song["title"], "artist": song["artist"],
             "bpm": song["bpm"], "camelot": song["camelot"]} for i, song in enumerate(songs)]


def render(lib, chain, plan, out_path):
    songs = [lib.songs[t] for t in chain]
    mix_bar, speeds, downbeats, offsets = timeline(songs, plan)
    bg = float(np.arcsin(10 ** (rt.EQ_BACKGROUND_DB / 20)) * 2 / np.pi)

    # Transition keyframes on the mix timeline, measured on the outgoing song's bars
    trans = []
    for i, p in enumerate(plan):
        bar = lambda j, i=i, p=p: offsets[i] + downbeats[i][p["outBar"] + j]
        ov, half, quarter = p["overlap"], p["overlap"] // 2, max(1, p["overlap"] // 4)
        beat = mix_bar / 4
        trans.append({
            "a_low": [(bar(half), 1), (bar(half) + beat, 0)],
            "a_mh": [(bar(half), 1), (bar(half + 1), bg), (bar(ov), 0)],
            "b_low": [(bar(half), 0), (bar(half) + beat, 1)],
            "b_mh": [(bar(0), 0), (bar(quarter), bg), (bar(half), bg), (bar(half + 1), 1)],
            "start": bar(0), "swap": bar(half), "end": bar(ov),
        })

    jobs = []
    for i, (song, (start, end), gain) in enumerate(zip(songs, song_spans(songs, plan, downbeats), loudness_gains(songs))):
        envelopes = ([(trans[i - 1]["b_low"], trans[i - 1]["b_mh"])] if i > 0 else []) + \
                    ([(trans[i]["a_low"], trans[i]["a_mh"])] if i < len(plan) else [])
        jobs.append((song, speeds[i], start, end, offsets[i], gain, envelopes))
    print(f"  rendering {len(songs)} songs, {WORKERS} at a time ...")
    with ProcessPoolExecutor(WORKERS) as pool:
        mix = finish(list(pool.map(render_song, *zip(*jobs))), out_path)
    return mix.shape[1] / SR, 60 * 4 / mix_bar, tracklist_of(songs, trans), trans


def render_stems(lib, chain, plan, out_path):
    """The mix with stems + EQ: the `smooth` handover from render_transition, on the whole mix's timeline.

    bars 0-h    B's highs, then its mids, rise to the background level (held nearly off where both songs are
                strong in that band); B's bass and vocals stay out
    vocal fade  A's vocals fade over 2 beats in its longest pause of the first half
    bar h       bass crossfade over a bar; A's mids/highs start closing under a low-pass
    bars h-h+2  B fully open, A down to the background level, gone by the end of the overlap
    vocals in   B's vocals at its first pause after the swap, a bar or more after A's went"""
    songs = [lib.songs[t] for t in chain]
    mix_bar, speeds, downbeats, offsets = timeline(songs, plan)
    pos = lambda db: float(np.arcsin(10 ** (db / 20)) * 2 / np.pi)  # env position for a gain in dB
    beat = mix_bar / 4
    beat_time = lambda k, bar, j: offsets[k] + songs[k]["beats"][bar_beats(songs[k])[bar] + j] / speeds[k]

    envelopes = [{"bass": [], "mids": [], "highs": [], "vocals": []} for _ in songs]
    lowpass = [None] * len(songs)
    trans = []
    for i, p in enumerate(plan):
        sa, sb = songs[i], songs[i + 1]
        x, y, ov = p["outBar"], p["inBar"], p["overlap"]
        half, quarter = ov // 2, max(1, ov // 4)
        bar = lambda j, i=i, x=x: offsets[i] + downbeats[i][x + j]
        swap = bar(half)
        a_out, b_in = vocal_handover(sa, sb, x, y, ov)
        a_vocals, b_vocals = beat_time(i, x, a_out), beat_time(i + 1, y, b_in)

        loud_a = np.array(sa["stemBands"])[x:x + ov].mean(0) >= rt.CLASH_DB
        loud_b = np.array(sb["stemBands"])[y:y + ov].mean(0) >= rt.CLASH_DB
        bg = [pos(rt.CLASH_BACKGROUND_DB if la and lb else rt.EQ_BACKGROUND_DB) for la, lb in zip(loud_a, loud_b)]

        envelopes[i]["vocals"].append([(a_vocals, 1), (a_vocals + 2 * beat, 0)])
        envelopes[i]["bass"].append([(swap, 1), (bar(half + 1), 0)])
        for band, level in (("mids", bg[1]), ("highs", bg[2])):
            envelopes[i][band].append([(swap, 1), (bar(half + 2), level), (bar(ov - 2), level), (bar(ov), 0)])
        lowpass[i] = [(swap, 20000), (bar(ov), 250)]
        envelopes[i + 1]["bass"].append([(swap, 0), (bar(half + 1), 1)])
        envelopes[i + 1]["highs"].append([(bar(0), 0), (swap, bg[2]), (bar(half + 2), 1)])
        envelopes[i + 1]["mids"].append([(bar(quarter), 0), (swap, bg[1]), (bar(half + 2), 1)])
        envelopes[i + 1]["vocals"].append([(b_vocals, 0), (b_vocals + beat, 1)])
        trans.append({"start": bar(0), "swap": swap, "end": bar(ov), "aVocalsOut": a_vocals, "bVocalsIn": b_vocals,
                      "heldDown": [n for n, la, lb in zip(("mids", "highs"), loud_a[1:], loud_b[1:]) if la and lb]})

    jobs = [(song, speeds[i], start, end, offsets[i], gain, envelopes[i], lowpass[i]) for i, (song, (start, end), gain)
            in enumerate(zip(songs, song_spans(songs, plan, downbeats), loudness_gains(songs)))]
    print(f"  rendering {len(songs)} songs from stems, {WORKERS} at a time ...")
    with ProcessPoolExecutor(WORKERS) as pool:
        mix = finish(list(pool.map(render_song_stems, *zip(*jobs))), out_path)
    return mix.shape[1] / SR, 60 * 4 / mix_bar, tracklist_of(songs, trans), trans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", type=int, default=4)
    parser.add_argument("--genre", nargs="+", default=[], choices=list(lfm.FAMILIES), help="theme: any of these genres")
    parser.add_argument("--energy", choices=list(ENERGY), help="theme: felt energy")
    parser.add_argument("--bpm", type=float, help="theme: mix tempo (default: the one most matching songs share)")
    parser.add_argument("--chain", nargs="+", help="use these songs in this order (the points are still optimised)")
    parser.add_argument("--exclude", nargs="+", default=[], help="leave these songs out")
    parser.add_argument("--avoid", nargs="+", default=[], help="use at most --max-avoided of these songs (e.g. the last mix)")
    parser.add_argument("--max-avoided", type=int, default=2)
    parser.add_argument("--stems", action="store_true", help="only songs with stems, mixed with stems + EQ")
    parser.add_argument("--pool", nargs="+", help="pick from these songs only (the order and points are still optimised)")
    parser.add_argument("--out", default=os.path.join(rt.RENDER_DIR, "mix", "test_mix.wav"))
    args = parser.parse_args()

    collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    lib = dj.Library(collection, dj.load_profiles())
    titles = [t for t in lib.profiles if lib.ready(t) and four_beat_bars(lib.songs[t])]
    ready = list(titles)
    if args.stems:
        titles = [t for t in titles if "stemBands" in lib.songs[t] and "vocalBeats" in lib.songs[t]]
    find = lambda name: next(t for t in ready if t.lower().startswith(name.lower()))
    excluded = {find(e) for e in args.exclude}
    titles = [t for t in titles if t not in excluded]
    if (args.genre or args.energy or args.bpm) and not args.chain:
        titles, bpm = theme_pool(lib, titles, args.genre, args.energy, args.bpm, rank_among=ready)
        print(f"theme pool: {len(titles)} songs around {bpm:.1f} BPM: {', '.join(titles)}")

    if args.pool:
        titles = [find(p) for p in args.pool]
    fixed = [find(c) for c in args.chain] if args.chain else None
    avoid = {find(a) for a in args.avoid}
    chain, plan = plan_mix(lib, titles, len(fixed) if fixed else args.songs, fixed, avoid, args.max_avoided,
                           stem_candidates if args.stems else candidates)
    if not chain:
        raise SystemExit("no mix found where every transition follows the rules")
    print(f"Chain (total score {sum(p['score'] for p in plan):.0f}): " + "  ->  ".join(chain))
    for p in plan:
        sa, sb = lib.songs[p["from"]], lib.songs[p["to"]]
        print(f"  {p['from']} {dj.fmt_bar(sa, p['outBar'])}  ->  {p['to']} {dj.fmt_bar(sb, p['inBar'])}, "
              f"{p['overlap']} bars, score {p['score']}, key steps {p['keySteps']}  {p['parts']}")

    length, bpm, tracklist, trans = (render_stems if args.stems else render)(lib, chain, plan, args.out)
    with open(args.out[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump({"bpm": round(bpm, 2), "length": round(length, 1), "tracklist": tracklist, "transitions": plan},
                  f, indent=2, ensure_ascii=False)

    print(f"\nRendered {args.out}  ({rt.fmt(length)} at {bpm:.1f} BPM)")
    for i, t in enumerate(tracklist):
        print(f"  {rt.fmt(t['time'])}  {t['title']} - {t['artist']}  ({t['bpm']} BPM, {t['camelot']})")
        if i < len(trans):
            t = trans[i]
            print(f"  {rt.fmt(t['start'])}    transition starts; swap at {rt.fmt(t['swap'])}, done {rt.fmt(t['end'])}")
            if "aVocalsOut" in t:
                print(f"           vocals out {rt.fmt(t['aVocalsOut'])}, in {rt.fmt(t['bVocalsIn'])}"
                      + (f"; held down (strong in both): {', '.join(t['heldDown'])}" if t["heldDown"] else ""))


if __name__ == "__main__":
    main()
