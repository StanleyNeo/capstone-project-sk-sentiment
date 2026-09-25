// Day 11: Express Gateway for the IMDB Sentiment API.
// Proxies /api/sentiment to the FastAPI service on port 5001.
const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:5001';

app.use(cors());
app.use(express.json());

// Health check for the gateway itself
app.get('/health', (req, res) => {
    res.json({ success: true, service: 'express-gateway', status: 'up' });
});

// Proxy endpoint
app.post('/api/sentiment', async (req, res) => {
    try {
        const response = await axios.post(`${FASTAPI_URL}/sentiment`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Proxy error:', error.message);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to reach sentiment API' 
        });
    }
});

app.listen(PORT, () => {
    console.log(`====================================================`);
    console.log(` EXPRESS GATEWAY  v1.0`);
    console.log(`====================================================`);
    console.log(` Port: ${PORT}     URL: http://localhost:${PORT}`);
    console.log(` Proxying /api/sentiment -> ${FASTAPI_URL}/sentiment`);
    console.log(`====================================================`);
});