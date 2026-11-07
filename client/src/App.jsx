import { Routes, Route } from 'react-router-dom'
import LoginPage from './pages/LoginPage.jsx'
import SearchPage from './pages/SearchPage.jsx'
import './App.css'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LoginPage />} />
      <Route path="/search" element={<SearchPage />} />
    </Routes>
  )
}

export default App
