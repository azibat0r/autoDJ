import os

import librosa
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")

load_dotenv(os.path.join(BASE_DIR, ".env"))

# Krumhansl-Schmuckler key profiles, starting from the tonic
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Below this correlation gap the chroma result is ambiguous, so tonnetz decides
MODE_MARGIN = 0.05

# A song with all of these fields has already been re-analyzed
NEW_FIELDS = ("mode", "beats", "energy_sections")


def detect_mode(y, sr, key=None):
    y_harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr).mean(axis=1)

    if key is None:
        key = int(np.argmax(chroma))

    major_corr = np.corrcoef(chroma, np.roll(MAJOR_PROFILE, key))[0, 1]
    minor_corr = np.corrcoef(chroma, np.roll(MINOR_PROFILE, key))[0, 1]

    if abs(major_corr - minor_corr) >= MODE_MARGIN:
        return 1 if major_corr > minor_corr else 0

    # Tonnetz rows 2-3 are the minor-third axis, rows 4-5 the major-third axis
    tonnetz = librosa.feature.tonnetz(chroma=librosa.feature.chroma_cqt(y=y_harmonic, sr=sr))
    minor_strength = np.abs(tonnetz[2:4]).mean()
    major_strength = np.abs(tonnetz[4:6]).mean()
    return 1 if major_strength >= minor_strength else 0


def get_beats(y, sr):
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return [round(float(t), 3) for t in beats]


def get_energy_sections(y, sections=8):
    rms = librosa.feature.rms(y=y)[0]
    return [float(chunk.mean()) if len(chunk) else 0.0 for chunk in np.array_split(rms, sections)]


def main():
    songs_collection = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]

    downloaded = {
        os.path.normcase(os.path.join(DOWNLOAD_DIR, name))
        for name in os.listdir(DOWNLOAD_DIR)
        if name.lower().endswith(".mp3")
    }

    songs = [
        song
        for song in songs_collection.find({"filePath": {"$exists": True}})
        if song["filePath"] and os.path.normcase(os.path.abspath(song["filePath"])) in downloaded
    ]
    total = len(songs)
    songs = [song for song in songs if not all(field in song for field in NEW_FIELDS)]
    print(f"Found {total} songs with an MP3 in {DOWNLOAD_DIR}")
    print(f"{total - len(songs)} already analyzed, {len(songs)} left to do\n")

    for i, song in enumerate(songs, start=1):
        title = song.get("title", "Unknown")
        file_path = song["filePath"]
        prefix = f"[{i}/{len(songs)}] {title}"

        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True)

            updates = {
                "mode": detect_mode(y, sr, song.get("key")),
                "beats": get_beats(y, sr),
                "energy_sections": get_energy_sections(y),
            }

            songs_collection.update_one({"_id": song["_id"]}, {"$set": updates})

            mode_name = "major" if updates["mode"] == 1 else "minor"
            print(f"{prefix}: updated ({mode_name}, {len(updates['beats'])} beats)")
        except Exception as err:
            print(f"{prefix}: skipped (error - {err})")

    print("\nDone.")


if __name__ == "__main__":
    main()
