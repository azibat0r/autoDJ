"""Replace the MP3 for specific songs whose YouTube download was wrong, then mark them for re-analysis.

Usage:
    python python/redownload.py "Highlife Led Me Here" "Pele" "Magik"

Each argument is matched against song titles (case-insensitive, exact). The new MP3 is the first
YouTube result whose length matches the Spotify track. Run analyze.py afterwards in WSL.
"""

import ntpath
import os
import re
import sys

import spotipy
from pymongo import MongoClient
from spotipy.oauth2 import SpotifyClientCredentials

from ingest import DOWNLOAD_DIR, download_mp3, find_youtube_url


def spotify_duration(sp, title, artist):
    first_artist = artist.split(",")[0].strip()
    results = sp.search(q=f"track:{title} artist:{first_artist}", type="track", limit=5)
    for track in results["tracks"]["items"]:
        if track["name"].lower() == title.lower():
            return track["duration_ms"] / 1000
    items = results["tracks"]["items"]
    return items[0]["duration_ms"] / 1000 if items else None


def main():
    titles = sys.argv[1:]
    if not titles:
        print(__doc__)
        return

    songs_collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        )
    )

    for title in titles:
        song = songs_collection.find_one({"title": {"$regex": f"^{re.escape(title)}$", "$options": "i"}})
        if not song:
            print(f"{title}: not found in database")
            continue

        prefix = f"{song['title']} - {song['artist']}"
        duration = spotify_duration(sp, song["title"], song["artist"])
        if not duration:
            print(f"{prefix}: not found on Spotify, skipping")
            continue

        print(f"{prefix}: Spotify length {int(duration // 60)}:{int(duration % 60):02d}, searching YouTube...")
        url = find_youtube_url(song["title"], song["artist"], duration)
        if not url:
            print(f"{prefix}: no YouTube result with a matching length, skipping")
            continue

        old_path = os.path.join(DOWNLOAD_DIR, ntpath.basename(song["filePath"]))
        name = os.path.splitext(os.path.basename(old_path))[0]
        if os.path.exists(old_path):
            os.remove(old_path)

        print(f"{prefix}: downloading {url}")
        file_path = download_mp3(url, name)

        songs_collection.update_one(
            {"_id": song["_id"]},
            {"$set": {"filePath": file_path}, "$unset": {"analysisVersion": ""}},
        )
        print(f"{prefix}: replaced, marked for re-analysis")


if __name__ == "__main__":
    main()
