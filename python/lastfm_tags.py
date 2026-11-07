"""Genre tags from Last.fm (listener tags like "afrobeats", "amapiano", "r&b"), since Spotify no longer returns genres.

Needs LASTFM_API_KEY in python/.env (free key from https://www.last.fm/api/account/create).
    python lastfm_tags.py "Happy Birthday" "The Boy"
    python lastfm_tags.py --all     (fetch and cache genres for every analysed song)
"""

import json
import os
import sys
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CACHE = os.path.join(BASE_DIR, "data", "lastfm_tags.json")
API = "https://ws.audioscrobbler.com/2.0/"
MIN_COUNT = 10  # Last.fm tag weight (0-100); lower tags are mostly noise
NOT_GENRES = {"seen live", "favorites", "favourite", "favorite songs", "spotify", "beautiful", "love", "my playlist",
              "dj", "deejay", "producer", "remix", "mix", "female vocalists", "male vocalists", "nigeria", "nigerian",
              "south africa", "south african", "ghana", "ghanaian", "uk", "usa", "canadian"}


def call(method, **params):
    query = urllib.parse.urlencode({"method": method, "api_key": os.environ["LASTFM_API_KEY"], "format": "json",
                                    "autocorrect": 1, **params})
    with urllib.request.urlopen(f"{API}?{query}", timeout=15) as r:
        return json.load(r)


def clean(tags):
    return [(t["name"].lower(), int(t["count"])) for t in tags
            if int(t["count"]) >= MIN_COUNT and t["name"].lower() not in NOT_GENRES]


def tags_for(artist, title):
    """(track tags, artist tags combined over every credited artist).

    Last.fm merges different artists that share a name, so one credited artist can come back with the wrong
    genre; summing the tags of all of them lets the others outvote it."""
    parts = [p.strip() for p in artist.split(",") if p.strip()]  # "Beekay & Kaypee" stays one duo
    track, votes = [], {}
    for name in parts:
        track = track or clean(call("track.gettoptags", artist=name, track=title).get("toptags", {}).get("tag", []))
        for tag, count in clean(call("artist.gettoptags", artist=name).get("toptags", {}).get("tag", [])):
            votes[tag] = votes.get(tag, 0) + count
    return track, sorted(votes.items(), key=lambda v: -v[1])


# Last.fm tags -> the genre families mixes (and the demo's AI) can ask for
FAMILIES = {
    "amapiano": ["amapiano"],
    "afrobeats": ["afrobeats", "afrobeat", "afropop", "afro pop", "afro-pop", "afrofusion", "afro fusion", "highlife", "afro"],
    "r&b": ["rnb", "r&b", "alternative rnb", "soul", "neo-soul", "contemporary r&b"],
    "hip-hop": ["hip-hop", "hip hop", "rap", "trap"],
    "dancehall": ["dancehall", "reggae", "roots reggae"],
    "house": ["house", "deep house", "afro house", "electronic", "dance"],
}

# Checked by hand: Last.fm matched a different artist with the same name for these
CORRECTIONS = {
    "Mokotla": ["amapiano"],
    "FInally": ["afrobeats"],
    "Wata": ["afrobeats"],
}


def families_for(tags):
    found = []
    for tag, _ in tags:
        for family, names in FAMILIES.items():
            if tag in names and family not in found:
                found.append(family)
    return found[:2]


def load_cache():
    return json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def cached_tags(cache, title, artist):
    """tags_for, fetched once per song and kept in data/lastfm_tags.json (as lists, so they round-trip)."""
    if title not in cache:
        track, artist_tags = tags_for(artist, title)
        cache[title] = {"track": track, "artist": artist_tags}
    return [tuple(t) for t in cache[title]["track"]], [tuple(t) for t in cache[title]["artist"]]


def genres_for(cache, title, artist):
    """Up to two genre families, from cached tags (track tags first) or the hand corrections."""
    track, artist_tags = cached_tags(cache, title, artist)
    return CORRECTIONS.get(title) or families_for(track + artist_tags)


def fetch_all(songs):
    cache = load_cache()
    todo = [s for s in songs.find({"analysisVersion": 2}, {"title": 1, "artist": 1}) if s["title"] not in cache]
    print(f"{len(todo)} songs to tag")
    for i, s in enumerate(todo):
        try:
            print(f"{s['title'][:35]:35} {', '.join(genres_for(cache, s['title'], s['artist'])) or '?'}")
        except Exception as err:
            print(f"FAILED {s['title']}: {err}")
        if i % 20 == 19 or i == len(todo) - 1:
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        sys.stdout.flush()


def main():
    songs = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    if sys.argv[1:] == ["--all"]:
        return fetch_all(songs)
    for name in sys.argv[1:]:
        song = songs.find_one({"title": {"$regex": f"^{name}", "$options": "i"}}, {"title": 1, "artist": 1})
        if not song:
            print(f"{name}: not in the library")
            continue
        track, artist = tags_for(song["artist"], song["title"])
        print(f"{song['title']} - {song['artist']}")
        print(f"  song tags:   {', '.join(f'{t} ({c})' for t, c in track[:8]) or 'none'}")
        print(f"  artist tags: {', '.join(f'{t} ({c})' for t, c in artist[:8]) or 'none'}")


if __name__ == "__main__":
    main()
