"""Vibe, movement and per-section key profiles for a song set. Writes python/data/<set>_set.json.

Runs inside WSL with the TensorFlow build of Essentia (~/autodj-tf) and models in /root/essentia-models:
    wsl -d Ubuntu-24.04 -u root -- bash -c "source ~/autodj-tf/bin/activate && python /mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python/vibe_test_set.py fresh"
"""
import json
import ntpath
import os
import random
import sys
import time

import essentia
import essentia.standard as es
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

essentia.log.infoActive = False
essentia.log.warningActive = False

PY = "/mnt/c/Users/aziba/OneDrive/Desktop/AutoDj/python"
DL = PY + "/downloads/"
MODELS = "/root/essentia-models/"
load_dotenv(PY + "/.env")
col = MongoClient(os.environ["MONGO_URL"]).get_default_database("autodj")["songs"]

SETS = {
    "test": ["Shining (feat. Beyoncé & JAŸ-Z)", "E No Easy", "Amen", "Leave it Behind",
             "Pure Love Mi Give Gal", "Happy Birthday", "REQUIRE (with Olamide)", "Loyalty"],
    # Fresh songs the engine's rules were not tuned on, for blind tests
    "fresh": ["Feels", "Ear Candy", "Wata", "Mokotla", "Kakalika", "Tonga", "PRESIDENTIAL", "Royal Jama",
              "The Boy", "Down Flat", "worst behaviour", "FInally", "Can't Be Stopped", "In A Loop"],
}
SET = sys.argv[1] if len(sys.argv) > 1 else "test"  # python vibe_test_set.py [test|fresh|library [minutes]]
OUT = PY + f"/data/{SET}_set.json"
if SET == "library":
    # The rest of the library, closest to 112 BPM first (where the hand-picked sets sit, so new songs can join their
    # mixes), until the time budget runs out. Saved after every song and resumable.
    MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    picked = [t for s in SETS.values() for t in s]
    TEST = [d["title"] for d in sorted(col.find({"analysisVersion": 2, "title": {"$nin": picked}}, {"title": 1, "bpm": 1}),
                                       key=lambda d: abs(d["bpm"] - 112))]
else:
    MINUTES = None
    TEST = SETS[SET]
PITCH = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
         "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
CAM_MAJ = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
CAM_MIN = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]


def camelot(key, scale):
    return f"{CAM_MAJ[PITCH[key]]}B" if scale == "major" else f"{CAM_MIN[PITCH[key]]}A"


effnet = es.TensorflowPredictEffnetDiscogs(graphFilename=MODELS + "discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
genre_model = es.TensorflowPredict2D(graphFilename=MODELS + "genre_discogs400-discogs-effnet-1.pb",
                                     input="serving_default_model_Placeholder", output="PartitionedCall:0")
genre_classes = json.load(open(MODELS + "genre_discogs400-discogs-effnet-1.json"))["classes"]
heads = {}
for name in ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party", "danceability"]:
    meta = json.load(open(MODELS + f"{name}-discogs-effnet-1.json"))
    positive = meta["classes"].index(name.replace("mood_", "") if name != "danceability" else "danceable")
    heads[name.replace("mood_", "")] = (es.TensorflowPredict2D(graphFilename=MODELS + f"{name}-discogs-effnet-1.pb", output="model/Softmax"), positive)


def vibe(path):
    audio16 = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
    emb = effnet(audio16)
    genres = genre_model(emb).mean(axis=0)
    top = np.argsort(genres)[::-1][:3]
    return {
        "embedding": emb.mean(axis=0).tolist(),
        "genres": [[genre_classes[i], round(float(genres[i]), 3)] for i in top],
        **{k: round(float(m(emb)[:, pos].mean()), 3) for k, (m, pos) in heads.items()},
    }


def movement(d):
    e, b = np.array(d["barEnergy"]), np.array(d["barBass"])
    body = slice(len(e) // 10, -(len(e) // 10) or None)
    return {"energySwing": round(float(e[body].std()), 3), "bassSwing": round(float(b[body].std()), 3),
            "avgBass": round(float(b[body].mean()), 3), "avgEnergy": round(float(e[body].mean()), 3)}


def section_keys(d, path):
    audio = es.MonoLoader(filename=path, sampleRate=44100)()
    out = []
    for s in d["sections"]:
        seg = audio[int(s["start"] * 44100):int(s["end"] * 44100)]
        k1 = camelot(*es.KeyExtractor(profileType="edma")(seg)[:2])
        k2 = camelot(*es.KeyExtractor(profileType="bgate")(seg)[:2])
        agreed = k1 == k2
        out.append({"start": s["start"], "startBar": s["startBar"], "endBar": s["endBar"], "label": s["label"],
                    "key": k1 if agreed else d["camelot"], "confident": agreed})
    return out


result = {"test": {}, "background": []}
if SET == "library" and os.path.exists(OUT):
    result = json.load(open(OUT))  # resume
started = time.time()
for title in TEST:
    if title in result["test"]:
        continue
    if MINUTES and time.time() - started > MINUTES * 60:
        print(f"time budget of {MINUTES:.0f} min used; {len(result['test'])} songs done, run again to continue")
        break
    t = time.time()
    try:
        d = col.find_one({"title": title})
        path = DL + ntpath.basename(d["filePath"])
        result["test"][title] = {"artist": d["artist"], "bpm": d["bpm"], "camelot": d["camelot"],
                                 **vibe(path), **movement(d), "sections": section_keys(d, path)}
    except Exception as err:
        print(f"FAILED {title}: {err}")
        continue
    v = result["test"][title]
    print(f"{title[:30]:<30} {v['genres'][0][0]:<32} dance {v['danceability']:.2f} happy {v['happy']:.2f} "
          f"party {v['party']:.2f} aggr {v['aggressive']:.2f} relax {v['relaxed']:.2f} sad {v['sad']:.2f} ({time.time() - t:.0f}s)")
    sys.stdout.flush()
    if SET == "library":
        json.dump(result, open(OUT, "w"), ensure_ascii=False)

if SET == "test":  # the background sample is only needed once
    random.seed(7)
    others = [d for d in col.find({"analysisVersion": 2, "title": {"$nin": TEST}}, {"filePath": 1, "title": 1})]
    for d in random.sample(others, 40):
        try:
            result["background"].append(vibe(DL + ntpath.basename(d["filePath"]))["embedding"])
        except Exception as err:
            print("background skip", d["title"], err)
    print(f"background embeddings: {len(result['background'])}")

json.dump(result, open(OUT, "w"), ensure_ascii=False)
print("saved", OUT)
