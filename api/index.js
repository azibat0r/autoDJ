const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const cookieParser = require('cookie-parser');
const axios = require('axios');
const jwt = require('jsonwebtoken');
const querystring = require('querystring');
require('dotenv').config();

const app = express();

app.use(express.json());
app.use(cookieParser());
app.use(cors({
  credentials: true,
  origin: ['http://127.0.0.1:5173', 'http://localhost:5173'],
}));

mongoose.connect(process.env.MONGO_URL);

app.get('/api/test', (req, res) => {
  res.json('server is running');
});

app.get('/api/auth/login', (req, res) => {
  const scope = [
    'user-read-private',
    'user-read-email',
    'playlist-modify-public',
    'playlist-modify-private',
  ].join(' ');

  const params = querystring.stringify({
    response_type: 'code',
    client_id: process.env.SPOTIFY_CLIENT_ID,
    scope,
    redirect_uri: process.env.REDIRECT_URI,
  });

  res.redirect(`https://accounts.spotify.com/authorize?${params}`);
});

app.get('/api/auth/callback', async (req, res) => {
  const { code } = req.query;

  try {
    const tokenResponse = await axios.post(
      'https://accounts.spotify.com/api/token',
      querystring.stringify({
        grant_type: 'authorization_code',
        code,
        redirect_uri: process.env.REDIRECT_URI,
      }),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Authorization:
            'Basic ' +
            Buffer.from(
              `${process.env.SPOTIFY_CLIENT_ID}:${process.env.SPOTIFY_CLIENT_SECRET}`
            ).toString('base64'),
        },
      }
    );

    const { access_token } = tokenResponse.data;

    const token = jwt.sign({ access_token }, process.env.JWT_SECRET, {
      expiresIn: '1h',
    });

    res.cookie('token', token, {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
    });

    res.redirect('http://127.0.0.1:5173/search');
  } catch (err) {
    console.error('Spotify token exchange failed:', err.response?.data || err.message);
    res.status(500).json({ error: 'Authentication failed' });
  }
});

app.get('/api/auth/me', (req, res) => {
  const { token } = req.cookies;

  if (!token) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  try {
    const { access_token } = jwt.verify(token, process.env.JWT_SECRET);
    res.json({ access_token });
  } catch (err) {
    res.status(401).json({ error: 'Invalid or expired token' });
  }
});

app.use('/api/playlist', require('./routes/playlist'));

app.listen(4000, () => {
  console.log('Server running on port 4000');
});