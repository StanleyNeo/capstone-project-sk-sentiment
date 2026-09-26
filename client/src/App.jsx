import { useState } from 'react';
import axios from 'axios';
import './App.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:5001';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Connect to FastAPI on Render
      const response = await axios.post(`${API}/sentiment`, {
        text: text,
      });
      setResult(response.data.data);
    } catch (err) {
      console.error(err);
      setError('Failed to get sentiment. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>IMDB Sentiment Analyzer</h1>
      <p>Enter a movie review to see if it's positive or negative.</p>

      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type your review here..."
          rows="5"
          cols="50"
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-card">
          <h2>Verdict: <span className={result.label}>{result.label.toUpperCase()}</span></h2>
          <p>Confidence: {(result.score * 100).toFixed(2)}%</p>
          
          {/* Confidence Bar */}
          <div className="confidence-bar-container">
            <div
              className={`confidence-bar ${result.label}`}
              style={{ width: `${result.score * 100}%` }}
            ></div>
          </div>
          <p className="model-info">Model: {result.model} (v{result.version})</p>
        </div>
      )}
    </div>
  );
}

export default App;