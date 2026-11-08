"""Download liked Spotify songs that aren't in the library yet; analyze.py and stems_scan.py do the analysis.

    python ingest.py --minutes 30
Most recently liked first. Resumable: songs already in the database are skipped, as are other versions of them.
"""

import argparse
import os
import re
import time

import spotipy
import yt_dlp
from dotenv import load_dotenv
from pymongo import MongoClient
from spotipy.oauth2 import SpotifyOAuth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")

load_dotenv(os.path.join(BASE_DIR, ".env"))

SEARCH_RESULTS = 5
DURATION_TOLERANCE = 15  # seconds a YouTube result may differ from the Spotify track length
MIN_SECONDS, MAX_SECONDS = 90, 480  # skips interludes and hour-long sets


def get_spotify():
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
            scope="user-library-read",
            cache_path=os.path.join(BASE_DIR, ".spotify_cache"),
        )
    )


def fetch_liked_songs(sp):
    songs = []
    page = sp.current_user_saved_tracks(limit=50)
    while page:
        for item in page["items"]:
            track = item["track"]
            if not track:
                continue
            songs.append(
                {
                    "title": track["name"],
                    "artist": ", ".join(artist["name"] for artist in track["artists"]),
                    "duration": track["duration_ms"] / 1000,
                }
            )
        page = sp.next(page) if page["next"] else None
    return songs


def safe_filename(text):
    return re.sub(r'[\\/:*?"<>|]', "", text).strip()[:150]


def find_youtube_url(title, artist, duration=None):
    """First top result whose length is within DURATION_TOLERANCE of the Spotify track.

    Tries "artist title" first, which finds official uploads most reliably, then
    "title artist audio". Without a duration, returns the first result.
    """
    options = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    for query in (f"{artist} {title}", f"{title} {artist} audio"):
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"ytsearch{SEARCH_RESULTS}:{query}", download=False)
        entries = [e for e in info.get("entries") or [] if e]

        if duration is None:
            if entries:
                return entries[0]["url"]
            continue

        for entry in entries:
            if entry.get("duration") and abs(entry["duration"] - duration) <= DURATION_TOLERANCE:
                return entry["url"]
    return None


def download_mp3(url, name):
    options = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, f"{name}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    return os.path.join(DOWNLOAD_DIR, f"{name}.mp3")


def plain(text):
    """Text without version notes or punctuation, so "Song (Remix)" and "Song - Live" count as "Song"."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)|\[.*?\]| - .*$", "", text.lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, help="stop starting new downloads after this long")
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    songs_collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]
    have = list(songs_collection.find({}, {"title": 1, "artist": 1}))
    titles = {s["title"] for s in have}  # the mix code looks songs up by title, so titles must stay unique
    versions = {(plain(s["title"]), plain(s["artist"].split(",")[0])) for s in have}

    print("Fetching liked songs from Spotify...")
    songs = fetch_liked_songs(get_spotify())
    print(f"Found {len(songs)} liked songs")

    started, saved, failed = time.time(), 0, 0
    for i, song in enumerate(songs, start=1):
        title, artist = song["title"], song["artist"]
        prefix = f"[{i}/{len(songs)}] {title} - {artist}"
        version = (plain(title), plain(artist.split(",")[0]))

        if title in titles or version in versions or not MIN_SECONDS <= song["duration"] <= MAX_SECONDS:
            continue
        if args.minutes and time.time() - started > args.minutes * 60:
            print(f"time budget of {args.minutes:.0f} min used")
            break

        try:
            print(f"{prefix}: searching YouTube...")
            url = find_youtube_url(title, artist, song["duration"])
            if not url:
                print(f"{prefix}: no YouTube result with a matching length, skipping")
                failed += 1
                continue

            file_path = download_mp3(url, safe_filename(f"{artist} - {title}"))
            songs_collection.insert_one({"title": title, "artist": artist, "filePath": file_path})
            titles.add(title)
            versions.add(version)
            saved += 1
            print(f"{prefix}: saved ({time.time() - started:.0f}s in)")
        except Exception as err:
            failed += 1
            print(f"{prefix}: failed - {err}")

    print(f"Done. {saved} songs saved, {failed} not found or failed.")


if __name__ == "__main__":
    main()
