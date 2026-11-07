const express = require('express');
const axios = require('axios');
const jwt = require('jsonwebtoken');

const router = express.Router();

const CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
const CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10];

function getCamelot(key, mode) {
  if (key == null || key < 0) return null;
  return mode === 1 ? CAMELOT_MAJOR[key] : CAMELOT_MINOR[key];
}

function isCamelotCompatible(a, b) {
  if (a == null || b == null) return false;
  if (a === b) return true;
  const diff = Math.abs(a - b);
  return diff === 1 || diff === 11;
}

function orderTracks(tracks) {
  const remaining = [...tracks];
  const ordered = [remaining.shift()];

  while (remaining.length) {
    const current = ordered[ordered.length - 1];
    let minDiff = Infinity;
    let candidates = [];

    for (const track of remaining) {
      const diff = Math.abs(track.tempo - current.tempo);
      if (diff < minDiff) {
        minDiff = diff;
        candidates = [track];
      } else if (diff === minDiff) {
        candidates.push(track);
      }
    }

    const compatible = candidates.find((track) =>
      isCamelotCompatible(current.camelot, track.camelot)
    );
    const next = compatible || candidates[0];

    ordered.push(next);
    remaining.splice(remaining.indexOf(next), 1);
  }

  return ordered;
}

router.post('/generate', async (req, res) => {
  const { trackIds } = req.body;
  const { token } = req.cookies;

  if (!token) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  if (!Array.isArray(trackIds) || trackIds.length < 2) {
    return res.status(400).json({ error: 'At least 2 track IDs are required' });
  }

  let accessToken;
  try {
    ({ access_token: accessToken } = jwt.verify(token, process.env.JWT_SECRET));
  } catch (err) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }

  const spotify = axios.create({
    baseURL: 'https://api.spotify.com/v1',
    headers: { Authorization: `Bearer ${accessToken}` },
  });

  try {
    const featureResponses = await Promise.all(
      trackIds.map((id) => spotify.get(`/audio-features/${id}`))
    );

    const tracks = trackIds.map((id, i) => {
      const { tempo, key, mode } = featureResponses[i].data;
      return { id, tempo, camelot: getCamelot(key, mode) };
    });

    const ordered = orderTracks(tracks);

    const { data: me } = await spotify.get('/me');

    const { data: playlist } = await spotify.post(`/users/${me.id}/playlists`, {
      name: 'AutoDJ Mixtape',
      public: false,
    });

    await spotify.post(`/playlists/${playlist.id}/tracks`, {
      uris: ordered.map((track) => `spotify:track:${track.id}`),
    });

    res.json({ url: playlist.external_urls.spotify });
  } catch (err) {
    console.error('Playlist generation failed:', err.response?.data || err.message);
    res.status(500).json({ error: 'Failed to generate playlist' });
  }
});

module.exports = router;
