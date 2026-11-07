"""Pick the next song and the transition point automatically, then render the transition.

Vibe data (sound, mood, per-section keys) comes from the song sets in data/ (vibe_test_set.py), so only
songs in them can be used. Runs inside WSL with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/auto_dj.py --from 'Amen'"

Transition length follows match quality (better matches can hold longer transitions):
  great match  "layered"        16 bars: B builds in, swap at bar 8, A's vocals ride over B's beat until bar 16
  okay match   "layered_short"   8 bars: B builds in, then a full switch to B at bar 8 (no vocal ride)
  poor match   no transition: warn and suggest the best alternatives instead

Rules, from listening tests:
  - keys at the mixed sections must be the same or one Camelot step apart (a 2-step gap clashed even with a matching vibe)
  - among those, prefer the closest vibe; movement (steady groove vs stop-start) matters more than sound similarity
  - never switch while A is mid-sentence: A's vocals must be quiet right after a full switch
  - A should calm down while B builds: no high-energy sections of A while B's instruments stack up
  - two songs' vocals never play together; B's vocals are muted until A's are gone
"""

import argparse
import json
import os

import numpy as np
from pymongo import MongoClient

import render_transition as rt

DATA_DIR = os.path.join(rt.BASE_DIR, "data")
DATA = os.path.join(DATA_DIR, "test_set.json")  # also holds the background sample for sound similarity


def load_profiles():
    """Merge every song set in data/ (test_set.json, fresh_set.json, ...)."""
    merged = json.load(open(DATA, encoding="utf-8"))
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith("_set.json") and name != os.path.basename(DATA):
            merged["test"].update(json.load(open(os.path.join(DATA_DIR, name), encoding="utf-8"))["test"])
    return merged
MAX_KEY_STEPS = 1
MAX_TEMPO_GAP = 0.06
MOODS = ["happy", "sad", "aggressive", "relaxed", "party", "danceability"]
MOVEMENT = ["energySwing", "bassSwing", "avgBass"]

# Provisional tiers, set from the few engine picks rated so far: Amen -> REQUIRE (76) 9/10 long;
# Shining -> Happy Birthday (72) 8/10 long; Happy Birthday -> Amen and Shining -> Leave it Behind (70)
# were "okay"/disliked at 16 bars, and the latter was judged too long.
TIERS = [  # (minimum score, technique, overlap bars)
    (72, "layered", 16),
    (60, "layered_short", 8),
]


def cam_dist(a, b):
    na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    d = min(abs(na - nb), 12 - abs(na - nb))
    return d if la == lb else d + 1


def keys_between(profile, start_bar, end_bar):
    """Keys of every section that overlaps bars [start_bar, end_bar)."""
    return {s["key"] for s in profile["sections"] if s["startBar"] < end_bar and s["endBar"] > start_bar}


def tempo_gap(bpm_a, bpm_b):
    """Relative tempo difference, treating half and double time as matching."""
    return min(abs(bpm_a * f - bpm_b) / bpm_b for f in (0.5, 1, 2))


def vocal_level(song, times, vocals=None):
    """Vocal loudness between consecutive times, from the Demucs vocal stem, scaled so 1 = the song's loud vocal parts."""
    if vocals is None:
        vocals = rt.load_stems(song)["vocals"].mean(axis=0)
    edges = (np.append(times, song["duration"]) * rt.SAMPLE_RATE).astype(int)
    level = np.array([np.sqrt(np.mean(vocals[a:b] ** 2)) if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
    ref = np.percentile(level, 90)
    return np.clip(level / ref, 0, 1) if ref > 0 else level


def vocal_per_bar(song):
    return vocal_level(song, song["downbeats"])


def felt_energy(profiles):
    """Felt energy from the mood models, as a rank 0 (calmest) to 1 (most energetic) among `profiles`.

    The stored "energy" is loudness: relaxed Down Flat came out near the top."""
    titles = list(profiles)
    raw = np.array([profiles[t]["party"] + profiles[t]["aggressive"] - profiles[t]["relaxed"] for t in titles])
    return dict(zip(titles, raw.argsort().argsort() / max(1, len(titles) - 1)))


class Library:
    def __init__(self, collection, profiles):
        self.profiles = profiles["test"]
        self.center = np.mean(profiles["background"], axis=0)
        self.songs = {t: None for t in self.profiles}
        for song in collection.find({"title": {"$in": list(self.profiles)}, "analysisVersion": 2}):  # one query, not one per song
            self.songs[song["title"]] = song
        self._bar_vocals, self._beat_vocals = {}, {}

        spread = {k: [] for k in MOVEMENT}
        for d in collection.find({"analysisVersion": 2}, {"barEnergy": 1, "barBass": 1}):
            e, b = np.array(d["barEnergy"]), np.array(d["barBass"])
            body = slice(len(e) // 10, -(len(e) // 10) or None)
            spread["energySwing"].append(e[body].std())
            spread["bassSwing"].append(b[body].std())
            spread["avgBass"].append(b[body].mean())
        self.spread = {k: np.std(v) for k, v in spread.items()}

    def ready(self, title):
        """Has everything mixing needs: vibe data, and a vocal map (stored by vocal_maps.py, or cached stems)."""
        song = self.songs.get(title)
        return bool(song) and ("vocalBars" in song or all(os.path.exists(p) for p in rt.stem_paths(song).values()))

    def bar_vocals(self, title):
        if title not in self._bar_vocals:
            song = self.songs[title]
            self._bar_vocals[title] = np.array(song["vocalBars"]) if "vocalBars" in song else vocal_per_bar(song)
        return self._bar_vocals[title]

    def beat_vocals(self, title):
        if title not in self._beat_vocals:
            song = self.songs[title]
            self._beat_vocals[title] = np.array(song["vocalBeats"]) if "vocalBeats" in song else vocal_level(song, song["beats"])
        return self._beat_vocals[title]

    def vibe(self, a, b):
        pa, pb = self.profiles[a], self.profiles[b]
        x, y = np.array(pa["embedding"]) - self.center, np.array(pb["embedding"]) - self.center
        sound = float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
        movement = float(np.mean([abs(pa[k] - pb[k]) / self.spread[k] for k in MOVEMENT]))
        mood = float(np.linalg.norm([pa[m] - pb[m] for m in MOODS]))
        # Movement weighs most: Happy Birthday -> Amen had the best sound match but different movement and didn't work
        score = 0.25 * (sound + 1) / 2 + 0.5 * np.exp(-movement) + 0.25 * np.exp(-2 * mood)
        return score, {"sound": round(sound, 2), "movementDiff": round(movement, 2), "moodDiff": round(mood, 2)}

    def vocals_cut_at(self, title, bar):
        """How much A is singing in the 2 beats right after `bar`'s downbeat, i.e. what a switch there would chop."""
        song = self.songs[title]
        first_beat = int(np.searchsorted(song["beats"], song["downbeats"][bar] - 0.05))
        return float(self.beat_vocals(title)[first_beat:first_beat + 2].mean())

    def best_points(self, a, b, technique, overlap, out_bars=None, in_bars=None):
        """Best (score, out_bar, in_bar, key steps, parts) for this technique, or None if the keys never fit.

        out_bars / in_bars limit the search (default: A's second half, B's first third)."""
        sa, sb = self.songs[a], self.songs[b]
        va, vb = self.bar_vocals(a), self.bar_vocals(b)
        n_a, n_b = len(sa["downbeats"]), len(sb["downbeats"])
        level = {bar: s["level"] for s in sa["sections"] for bar in range(s["startBar"], s["endBar"])}
        half = overlap // 2
        best = None
        out_bars = range(n_a // 2, n_a - overlap - 1) if out_bars is None else out_bars
        in_bars = range(0, max(1, n_b // 3)) if in_bars is None else in_bars

        for out_bar in out_bars:
            if (out_bar - sa["phraseOffset"]) % 4:
                continue
            position = np.clip((out_bar / n_a - 0.4) / 0.3, 0, 1)
            if technique == "layered":
                # A calms down while B stacks up (bars 4-12); A's vocals ride over B's beat after the swap (8-16)
                calm = 1 - np.mean([level[x] == "high" for x in range(out_bar + 4, out_bar + 12)])
                ride = min(1, va[out_bar + half:out_bar + overlap].mean() / 0.6)
                not_chopped = 1.0  # A's vocals fade out over bars 12-16, nothing is cut
            else:
                # A calms down while B's instruments join (bars 4-8); switch only at the end of A's vocal phrase
                calm = 1 - np.mean([level[x] == "high" for x in range(out_bar + half, out_bar + overlap)])
                ride = 1.0
                not_chopped = 1 - min(1, self.vocals_cut_at(a, out_bar + overlap) / 0.4)

            for in_bar in in_bars:
                if (in_bar - sb["phraseOffset"]) % 4 or in_bar + overlap + 4 >= n_b:
                    continue
                steps = max(cam_dist(ka, kb)
                            for ka in keys_between(self.profiles[a], out_bar, out_bar + overlap)
                            for kb in keys_between(self.profiles[b], in_bar, in_bar + overlap))
                if steps > MAX_KEY_STEPS:
                    continue
                lost = vb[in_bar:in_bar + overlap].mean()                      # B's vocals we have to mute
                arrival = vb[in_bar + overlap:in_bar + overlap + 4].mean()      # B's vocals when they come in

                parts = {
                    "rideVocals": ride,
                    "notChopped": not_chopped,
                    "fewMutedVocals": 1 - min(1, lost / 0.6),
                    "vocalsArriveOnTime": min(1, arrival / 0.6),
                    "lateInSong": float(position),
                    "key": 1.0 if steps == 0 else 0.8,
                    "calm": float(calm),
                }
                score = (0.15 * parts["rideVocals"] + 0.15 * parts["notChopped"] + 0.15 * parts["fewMutedVocals"]
                         + 0.15 * parts["vocalsArriveOnTime"] + 0.1 * parts["lateInSong"] + 0.1 * parts["key"]
                         + 0.2 * parts["calm"])
                if best is None or score > best[0]:
                    best = (score, out_bar, in_bar, steps, {k: round(float(v), 2) for k, v in parts.items()})
        return best


def fmt_bar(song, bar):
    return f"bar {bar} ({rt.fmt(song['downbeats'][bar])})"


def plan_transition(lib, a, b):
    """Score b as the next song after a, trying the techniques from longest to shortest.

    Returns (score, technique, overlap, points, vibe parts), with technique None if no tier fits.
    """
    vibe, vibe_parts = lib.vibe(a, b)
    shortest = None
    for min_score, technique, overlap in TIERS:
        points = lib.best_points(a, b, technique, overlap)
        if not points:
            continue
        total = 100 * (0.5 * vibe + 0.5 * points[0])
        if total >= min_score:
            return total, technique, overlap, points, vibe_parts
        shortest = (total, None, overlap, points, vibe_parts)  # poor match: report the shortest technique's score
    return shortest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="song_a", required=True)
    parser.add_argument("--to", dest="song_b", help="ask for a specific next song instead of the engine's pick")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--force", action="store_true", help="render a poor match anyway, with the shortest technique")
    parser.add_argument("--out", help="WAV path to write instead of renders/auto/<A> to <B>.wav")
    parser.add_argument("--technique", choices=rt.TECHNIQUES, help="render with this technique at the engine's chosen points")
    args = parser.parse_args()

    collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    lib = Library(collection, load_profiles())
    find = lambda name: next((t for t in lib.profiles if t.lower().startswith(name.lower())), None)
    a = find(args.song_a)
    if not a:
        raise SystemExit(f"{args.song_a} is not in the analysed song sets: {', '.join(lib.profiles)}")
    song_a = lib.songs[a]

    print(f"Now playing: {a} ({song_a['bpm']} BPM, {song_a['camelot']})\n")
    ranked = []
    for b in lib.profiles:
        if b == a:
            continue
        if tempo_gap(song_a["bpm"], lib.songs[b]["bpm"]) > MAX_TEMPO_GAP:
            print(f"  x {b}: tempo too far ({lib.songs[b]['bpm']} BPM)")
            continue
        plan = plan_transition(lib, a, b)
        if not plan:
            print(f"  x {b}: no transition point where the keys are within {MAX_KEY_STEPS} step")
            continue
        ranked.append((b, *plan))

    ranked.sort(key=lambda r: (r[2] is None, -r[1]))  # good matches first, best score first
    print()
    for b, total, technique, overlap, (ps, out_bar, in_bar, steps, pp), vp in ranked:
        verdict = f"{technique} ({overlap} bars)" if technique else "POOR MATCH"
        print(f"  {total:3.0f}  {b}  ->  {verdict}")
        print(f"       vibe {vp}")
        print(f"       leave at {fmt_bar(song_a, out_bar)}, enter at {fmt_bar(lib.songs[b], in_bar)}, key steps {steps}  {pp}")

    good = [r for r in ranked if r[2]]
    if args.song_b:
        b = find(args.song_b)
        choice = next((r for r in ranked if r[0] == b), None)
        if not choice or not choice[2]:
            print(f"\nWARNING: {args.song_b} is not a good match after {a}.")
            if good:
                print("Better next songs: " + ", ".join(f"{r[0]} ({r[1]:.0f})" for r in good[:3]))
            if not (choice and args.force):
                raise SystemExit(1)
            b_, total, _, _, points, vp = choice
            choice = (b_, total, TIERS[-1][1], TIERS[-1][2], points, vp)
            print(f"--force: rendering it anyway with {TIERS[-1][1]}")
    elif not good:
        print(f"\nWARNING: no good next song after {a} in the analysed song sets.")
        raise SystemExit(1)
    else:
        choice = good[0]

    if args.no_render:
        return
    b, _, technique, overlap, (_, out_bar, in_bar, _, _), _ = choice
    technique = args.technique or technique
    song_b = lib.songs[b]
    t = rt.Transition(song_a, song_b, rt.load_stereo(song_a), rt.load_stereo(song_b), out_bar, in_bar, overlap, 16, 16)
    mix, markers = rt.RENDERERS[technique](t)
    mix = rt.master(t, mix)

    wav_path = args.out or os.path.join(rt.RENDER_DIR, "auto", rt.safe_name(f"{a} to {b}") + ".wav")
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    with rt.AudioFile(wav_path, "w", rt.SAMPLE_RATE, num_channels=2, bit_depth=16) as f:
        f.write(mix)
    if not args.out:  # blind-test renders get no plan file, so the WAV name gives nothing away
        with open(wav_path[:-4] + ".json", "w", encoding="utf-8") as f:
            json.dump({"from": a, "to": b, "outBar": out_bar, "inBar": in_bar, "technique": technique, "overlapBars": overlap,
                       "markers": [{"time": round(float(s), 3), "event": e} for s, e in markers]}, f, indent=2, ensure_ascii=False)

    print(f"\nRendered {wav_path}  ({technique}, {overlap} bars)")
    for s, event in [(0, f"{a} from {rt.fmt(t.a0)} (bar {t.start_bar})")] + markers:
        print(f"  {rt.fmt(s)}  {event}")


if __name__ == "__main__":
    main()
