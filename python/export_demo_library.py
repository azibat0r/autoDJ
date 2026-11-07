"""Export the analysed songs (tempo, key, energy, moods, genre) to demo/library.json for the AutoDJ demo page.

Runs inside WSL with the ~/autodj-env environment:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-env/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/export_demo_library.py"
"""

import json
import os

from pymongo import MongoClient

import auto_dj as dj
import lastfm_tags as lfm

DEMO_DIR = os.path.join(os.path.dirname(dj.rt.BASE_DIR), "demo")
OUT = os.path.join(DEMO_DIR, "library.json")
PAGE = os.path.join(DEMO_DIR, "player.html")  # the page embeds the library so it works opened straight from disk
EMBED_START, EMBED_END = '<script id="library" type="application/json">', "</script>"


def main():
    collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    profiles = dj.load_profiles()["test"]
    found = {s["title"]: s for s in collection.find({"title": {"$in": list(profiles)}, "analysisVersion": 2},
                                                     {"title": 1, "artist": 1, "bpm": 1, "camelot": 1})}
    songs = [(t, found[t]) for t in profiles if t in found]
    energies = dj.felt_energy({t: profiles[t] for t, _ in songs})
    cache = lfm.load_cache()

    library = []
    for title, song in songs:
        energy = energies[title]
        track_tags, artist_tags = lfm.cached_tags(cache, title, song["artist"])
        genres = lfm.genres_for(cache, title, song["artist"])
        library.append({
            "title": title,
            "artist": song["artist"],
            "bpm": round(song["bpm"], 1),
            "camelot": song["camelot"],
            "energy": round(float(energy), 2),
            "moods": {m: round(float(profiles[title][m]), 2) for m in dj.MOODS},
            "genres": genres,
            "lastfmTags": [t for t, _ in artist_tags[:5]],
        })
        print(f"{title[:28]:28} {song['bpm']:6.1f} {song['camelot']:4} energy {energy:.2f}  {', '.join(genres) or '?':22} {[t for t, _ in artist_tags[:4]]}")

    json.dump(cache, open(lfm.CACHE, "w", encoding="utf-8"), ensure_ascii=False)  # any songs tagged just now
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=1, ensure_ascii=False)
    print(f"\n{len(library)} songs -> {OUT}")

    if os.path.exists(PAGE):
        page = open(PAGE, encoding="utf-8").read()
        start = page.index(EMBED_START) + len(EMBED_START)
        end = page.index(EMBED_END, start)
        page = page[:start] + json.dumps(library, ensure_ascii=False).replace("</", "<\\/") + page[end:]
        with open(PAGE, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"embedded in {PAGE}")


if __name__ == "__main__":
    main()
