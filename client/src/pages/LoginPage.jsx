import { useContext } from 'react';
import { Navigate } from 'react-router-dom';
import { UserContext } from '../UserContext.jsx';

export default function LoginPage() {
  const { user, ready } = useContext(UserContext);

  if (ready && user) {
    return <Navigate to="/search" replace />;
  }

  function handleLogin() {
    window.location.href = 'http://127.0.0.1:4000/api/auth/login';
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        gap: '8px',
        textAlign: 'center',
      }}
    >
      <h1 style={{ margin: 0 }}>AutoDJ</h1>
      <p>Automate your DJ experience</p>
      <button
        type="button"
        onClick={handleLogin}
        style={{
          marginTop: '24px',
          padding: '12px 24px',
          borderRadius: '999px',
          border: 'none',
          background: 'var(--accent)',
          color: '#fff',
          fontSize: '16px',
          cursor: 'pointer',
        }}
      >
        Login with Spotify
      </button>
    </div>
  );
}
