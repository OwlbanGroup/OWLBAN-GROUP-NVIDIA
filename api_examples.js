// JPMorgan Financial APIs - JavaScript Examples
// This file demonstrates how to consume the APIs using fetch and axios

// ==========================================
// USING FETCH API
// ==========================================

// 1. Health Check (No authentication required)
function checkHealth() {
    fetch('http://localhost:5000/health')
        .then(response => response.json())
        .then(data => {
            console.log('Server Health:', data);
            // Output: { status: "healthy", timestamp: "2025-...", version: "1.0.0" }
        })
        .catch(error => console.error('Health check failed:', error));
}

// 2. User Login (Get authentication token)
function loginUser(username, password) {
    fetch('http://localhost:5000/user/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: username, // e.g., 'testuser'
            password: password  // e.g., 'testpass'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.token) {
            console.log('Login successful! Token:', data.token);
            localStorage.setItem('authToken', data.token); // Store token
            return data.token;
        } else {
            console.error('Login failed:', data.error);
        }
    })
    .catch(error => console.error('Login error:', error));
}

// 3. Get JPMorgan Data (Requires authentication)
function getJPMorganData() {
    const token = localStorage.getItem('authToken');

    if (!token) {
        console.error('No authentication token found. Please login first.');
        return;
    }

    fetch('http://localhost:5000/api/jpmorgan-data', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (response.status === 401) {
            console.error('Authentication failed. Token may be invalid or expired.');
            return response.json();
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            console.error('API Error:', data.error);
        } else {
            console.log('JPMorgan Financial Data:', data);
            // Use the data in your dashboard
            setDashboardData(data);
        }
    })
    .catch(error => console.error('API request failed:', error));
}

// Example usage in a dashboard component
function setDashboardData(data) {
    // Update financial metrics
    document.getElementById('revenue').textContent = `$${data.financial_metrics.revenue.toLocaleString()}`;
    document.getElementById('net-income').textContent = `$${data.financial_metrics.net_income.toLocaleString()}`;

    // Update stock ticker
    document.getElementById('stock-price').textContent = `$${data.stock_ticker.current_price}`;
    document.getElementById('stock-change').textContent = `${data.stock_ticker.change > 0 ? '+' : ''}${data.stock_ticker.change}`;

    // Update assets list
    const assetsList = document.getElementById('assets-list');
    assetsList.innerHTML = '';
    data.assets.forEach(asset => {
        const li = document.createElement('li');
        li.textContent = `${asset.name}: $${asset.value.toLocaleString()}`;
        assetsList.appendChild(li);
    });
}

// ==========================================
// USING AXIOS (if available)
// ==========================================

// Uncomment and install axios first: npm install axios

/*
const axios = require('axios');

// Health Check with axios
function checkHealthAxios() {
    axios.get('http://localhost:5000/health')
        .then(response => {
            console.log('Server Health:', response.data);
        })
        .catch(error => {
            console.error('Health check failed:', error.response?.data || error.message);
        });
}

// Login with axios
function loginUserAxios(username, password) {
    axios.post('http://localhost:5000/user/login', {
        username: username,
        password: password
    })
    .then(response => {
        const token = response.data.token;
        console.log('Login successful! Token:', token);
        localStorage.setItem('authToken', token);
        return token;
    })
    .catch(error => {
        console.error('Login failed:', error.response?.data?.error || error.message);
    });
}

// Get JPMorgan Data with axios
function getJPMorganDataAxios() {
    const token = localStorage.getItem('authToken');

    if (!token) {
        console.error('No authentication token found. Please login first.');
        return;
    }

    axios.get('http://localhost:5000/api/jpmorgan-data', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        console.log('JPMorgan Financial Data:', response.data);
        setDashboardData(response.data);
    })
    .catch(error => {
        if (error.response?.status === 401) {
            console.error('Authentication failed. Token may be invalid or expired.');
        } else {
            console.error('API Error:', error.response?.data?.error || error.message);
        }
    });
}
*/

// ==========================================
// COMPLETE WORKFLOW EXAMPLE
// ==========================================

function initializeDashboard() {
    // Step 1: Check server health
    checkHealth();

    // Step 2: Login to get token (in real app, this would be user-initiated)
    // loginUser('testuser', 'testpass').then(() => {
    //     // Step 3: Fetch protected data
    //     getJPMorganData();
    // });

    // For demo purposes, assume token is already stored
    getJPMorganData();
}

// Error handling utilities
function handleApiError(error, context) {
    console.error(`API Error in ${context}:`, error);

    if (error.response) {
        // Server responded with error status
        switch (error.response.status) {
            case 401:
                console.error('Authentication required. Please login.');
                // Redirect to login page
                break;
            case 403:
                console.error('Access forbidden.');
                break;
            case 500:
                console.error('Server error. Please try again later.');
                break;
            default:
                console.error('Unexpected error occurred.');
        }
    } else if (error.request) {
        // Network error
        console.error('Network error. Please check your connection.');
    } else {
        // Other error
        console.error('An unexpected error occurred.');
    }
}

// Export functions for use in other modules
// module.exports = { checkHealth, loginUser, getJPMorganData, initializeDashboard };
