import os
import re

import librosa
import numpy as np
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


def analyze(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Pitch class with the strongest average chroma energy (0 = C, 11 = B)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = int(np.argmax(chroma.mean(axis=1)))

    energy = float(librosa.feature.rms(y=y).mean())

    return {"bpm": round(bpm, 2), "key": key, "energy": energy}


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    songs_collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]

    print("Fetching liked songs from Spotify...")
    songs = fetch_liked_songs(get_spotify())
    print(f"Found {len(songs)} liked songs\n")

    for i, song in enumerate(songs, start=1):
        title, artist = song["title"], song["artist"]
        prefix = f"[{i}/{len(songs)}] {title} - {artist}"

        if songs_collection.find_one({"title": title, "artist": artist}):
            print(f"{prefix}: already in database, skipping")
            continue

        try:
            print(f"{prefix}: searching YouTube...")
            url = find_youtube_url(title, artist, song["duration"])
            if not url:
                print(f"{prefix}: no YouTube result with a matching length, skipping")
                continue

            print(f"{prefix}: downloading {url}")
            file_path = download_mp3(url, safe_filename(f"{artist} - {title}"))

            print(f"{prefix}: analyzing...")
            features = analyze(file_path)

            songs_collection.insert_one(
                {"title": title, "artist": artist, **features, "filePath": file_path}
            )
            print(
                f"{prefix}: saved (bpm={features['bpm']}, key={features['key']}, "
                f"energy={features['energy']:.4f})"
            )
        except Exception as err:
            print(f"{prefix}: failed - {err}")

    print("\nDone.")


if __name__ == "__main__":
    main()
