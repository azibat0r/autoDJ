import { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import { UserContext } from '../UserContext.jsx';

export default function SearchPage() {
  const { user } = useContext(UserContext);
  const accessToken = user?.access_token;

  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState([]);
  const [generating, setGenerating] = useState(false);
  const [playlistUrl, setPlaylistUrl] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!accessToken || !query.trim()) {
      setResults([]);
      return;
    }

    const timeoutId = setTimeout(() => {
      axios
        .get('https://api.spotify.com/v1/search', {
          params: { q: query, type: 'track', limit: 10 },
          headers: { Authorization: `Bearer ${accessToken}` },
        })
        .then(({ data }) => setResults(data.tracks.items))
        .catch(() => setResults([]));
    }, 400);

    return () => clearTimeout(timeoutId);
  }, [query, accessToken]);

  function addSong(track) {
    setSelected((prev) =>
      prev.some((song) => song.id === track.id) ? prev : [...prev, track]
    );
  }

  function removeSong(id) {
    setSelected((prev) => prev.filter((song) => song.id !== id));
  }

  async function handleGenerateMixtape() {
    setGenerating(true);
    setError(null);
    setPlaylistUrl(null);

    try {
      const { data } = await axios.post('/api/playlist/generate', {
        trackIds: selected.map((track) => track.id),
      });
      setPlaylistUrl(data.url);
    } catch (err) {
      setError('Failed to generate mixtape. Please try again.');
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto', padding: '32px 16px' }}>
      <h1 style={{ fontSize: '32px', margin: '0 0 24px' }}>Search Songs</h1>

      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search for a song..."
        style={{
          width: '100%',
          padding: '12px 16px',
          borderRadius: '8px',
          border: '1px solid var(--border)',
          fontSize: '16px',
          boxSizing: 'border-box',
        }}
      />

      <ul style={{ listStyle: 'none', margin: '24px 0', padding: 0 }}>
        {results.map((track) => {
          const isSelected = selected.some((song) => song.id === track.id);
          const art = track.album.images[track.album.images.length - 1];

          return (
            <li
              key={track.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '8px 0',
                borderBottom: '1px solid var(--border)',
                textAlign: 'left',
              }}
            >
              {art && (
                <img
                  src={art.url}
                  alt=""
                  width="48"
                  height="48"
                  style={{ borderRadius: '4px' }}
                />
              )}
              <div style={{ flex: 1 }}>
                <div style={{ color: 'var(--text-h)' }}>{track.name}</div>
                <div style={{ fontSize: '14px' }}>
                  {track.artists.map((artist) => artist.name).join(', ')}
                </div>
              </div>
              <button
                type="button"
                onClick={() => addSong(track)}
                disabled={isSelected}
                style={{
                  padding: '6px 14px',
                  borderRadius: '999px',
                  border: 'none',
                  background: isSelected ? 'var(--border)' : 'var(--accent)',
                  color: '#fff',
                  cursor: isSelected ? 'default' : 'pointer',
                }}
              >
                {isSelected ? 'Added' : 'Add'}
              </button>
            </li>
          );
        })}
      </ul>

      <h2 style={{ fontSize: '20px', margin: '0 0 12px' }}>
        Selected Songs ({selected.length})
      </h2>

      <ul style={{ listStyle: 'none', margin: '0 0 24px', padding: 0 }}>
        {selected.map((track) => (
          <li
            key={track.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '8px 0',
              borderBottom: '1px solid var(--border)',
              textAlign: 'left',
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ color: 'var(--text-h)' }}>{track.name}</div>
              <div style={{ fontSize: '14px' }}>
                {track.artists.map((artist) => artist.name).join(', ')}
              </div>
            </div>
            <button
              type="button"
              onClick={() => removeSong(track.id)}
              style={{
                padding: '6px 14px',
                borderRadius: '999px',
                border: '1px solid var(--border)',
                background: 'transparent',
                color: 'var(--text-h)',
                cursor: 'pointer',
              }}
            >
              Remove
            </button>
          </li>
        ))}
      </ul>

      <button
        type="button"
        onClick={handleGenerateMixtape}
        disabled={selected.length < 2 || generating}
        style={{
          width: '100%',
          padding: '14px',
          borderRadius: '999px',
          border: 'none',
          background:
            selected.length < 2 || generating ? 'var(--border)' : 'var(--accent)',
          color: '#fff',
          fontSize: '16px',
          cursor: selected.length < 2 || generating ? 'not-allowed' : 'pointer',
        }}
      >
        {generating ? 'Generating...' : 'Generate Mixtape'}
      </button>

      {error && (
        <p style={{ color: 'crimson', marginTop: '16px' }}>{error}</p>
      )}

      {playlistUrl && (
        <a
          href={playlistUrl}
          target="_blank"
          rel="noreferrer"
          style={{
            display: 'block',
            textAlign: 'center',
            marginTop: '16px',
            color: 'var(--accent)',
          }}
        >
          Open in Spotify
        </a>
      )}
    </div>
  );
}
