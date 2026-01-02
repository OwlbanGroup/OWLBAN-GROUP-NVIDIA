# Frontend Dashboard Integration Guide

Complete guide for integrating the financial API endpoints with a vanilla JavaScript dashboard.

---

## 📋 Overview

This guide provides ready-to-use code for connecting your dashboard to the NestJS financial endpoints we've implemented:

- **Backend:** NestJS REST API (already implemented)
- **Frontend:** Vanilla JavaScript dashboard
- **Data Flow:** Dashboard → NestJS API → Database/JPMorgan API

---

## 🎯 Quick Start

### Prerequisites

1. ✅ NestJS backend running on `http://localhost:3000`
2. ✅ Financial endpoints implemented (already done)
3. ✅ Dashboard HTML file ready

### File Structure

```
jpmorgan_financial_apis/
├── nestjs-backend/          # Backend (already implemented)
│   └── src/financial/       # Financial endpoints
└── dashboard/               # Frontend (to be created)
    ├── index.html
    ├── dashboard.js
    └── styles.css
```

---

## 🔧 Backend Configuration (Already Done)

Your backend already has these endpoints:

```
GET /api/financial/summary      - Financial summary
GET /api/financial/assets       - Assets breakdown
GET /api/financial/performance  - Performance metrics
GET /api/financial/stocks       - Stock holdings
GET /api/system/status          - System status
```

### Enable CORS (if not already enabled)

Update `nestjs-backend/src/main.ts`:

```typescript
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Enable CORS for dashboard
  app.enableCors({
    origin: ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8080'],
    credentials: true,
  });
  
  await app.listen(process.env.PORT || 3000);
  console.log(`Application is running on: ${await app.getApplicationUrl()}`);
}
bootstrap();
```

---

## 🎨 Frontend Implementation

### 1. Dashboard HTML

Create `dashboard/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JPMorgan Financial Dashboard</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header">
            <h1>JPMorgan Financial Dashboard</h1>
            <div class="status-bar">
                <span id="statusIndicator" class="badge">Loading...</span>
                <span id="lastUpdate" class="text-muted">--</span>
            </div>
        </header>

        <!-- Summary Cards -->
        <section class="summary-section">
            <div class="card">
                <h3>Total Balance</h3>
                <p id="totalBalance" class="metric-value">$0</p>
                <span class="metric-label">USD</span>
            </div>
            <div class="card">
                <h3>Total Assets</h3>
                <p id="totalAssets" class="metric-value">$0</p>
                <span class="metric-label">USD</span>
            </div>
            <div class="card">
                <h3>Accounts</h3>
                <p id="accountsCount" class="metric-value">0</p>
                <span class="metric-label">Active</span>
            </div>
            <div class="card">
                <h3>Transactions</h3>
                <p id="transactionsCount" class="metric-value">0</p>
                <span class="metric-label">Last 30 days</span>
            </div>
        </section>

        <!-- Performance Chart -->
        <section class="chart-section">
            <div class="card">
                <h3>Performance Trends</h3>
                <canvas id="performanceChart"></canvas>
            </div>
        </section>

        <!-- Assets Table -->
        <section class="table-section">
            <div class="card">
                <h3>Assets Breakdown</h3>
                <table id="assetsTable">
                    <thead>
                        <tr>
                            <th>Account</th>
                            <th>Type</th>
                            <th>Balance</th>
                            <th>Currency</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="4" class="text-center">Loading...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Stocks Table -->
        <section class="table-section">
            <div class="card">
                <h3>Stock Holdings</h3>
                <table id="stocksTable">
                    <thead>
                        <tr>
                            <th>Account</th>
                            <th>Name</th>
                            <th>Value</th>
                            <th>Currency</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="4" class="text-center">Loading...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Recent Transactions -->
        <section class="table-section">
            <div class="card">
                <h3>Recent Transactions</h3>
                <table id="transactionsTable">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Description</th>
                            <th>Amount</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="4" class="text-center">Loading...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>
    </div>

    <script src="dashboard.js"></script>
</body>
</html>
```

---

### 2. Dashboard JavaScript

Create `dashboard/dashboard.js`:

```javascript
// Configuration
const API_BASE = 'http://localhost:3000/api';

// Utility Functions
async function fetchJson(path) {
  try {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    return await res.json();
  } catch (error) {
    console.error(`Error fetching ${path}:`, error);
    throw error;
  }
}

function formatCurrency(value, currency = 'USD') {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(value || 0);
}

function formatNumber(value) {
  return new Intl.NumberFormat('en-US').format(value || 0);
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// Load Summary Data
async function loadSummary() {
  try {
    const data = await fetchJson('/financial/summary');

    document.getElementById('totalBalance').textContent =
      formatCurrency(data.totalBalance, data.currency);
    document.getElementById('totalAssets').textContent =
      formatCurrency(data.totalBalance, data.currency);
    document.getElementById('accountsCount').textContent =
      formatNumber(data.accountsCount);
    document.getElementById('transactionsCount').textContent =
      formatNumber(data.recentTransactionsCount);

    // Load transactions table
    loadTransactions(data.recentTransactions);
  } catch (error) {
    console.error('Error loading summary:', error);
    showError('summary');
  }
}

// Load Assets Data
async function loadAssets() {
  try {
    const data = await fetchJson('/financial/assets');
    const tbody = document.getElementById('assetsTable').querySelector('tbody');

    tbody.innerHTML = '';

    if (!data.assetsByAccount || data.assetsByAccount.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center">No assets found</td></tr>';
      return;
    }

    data.assetsByAccount.forEach((asset) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${asset.accountName}</td>
        <td><span class="badge badge-info">${asset.accountType}</span></td>
        <td class="text-right">${formatCurrency(asset.balance, asset.currency)}</td>
        <td>${asset.currency}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (error) {
    console.error('Error loading assets:', error);
    showError('assets');
  }
}

// Load Stocks Data
async function loadStocks() {
  try {
    const data = await fetchJson('/financial/stocks');
    const tbody = document.getElementById('stocksTable').querySelector('tbody');

    tbody.innerHTML = '';

    if (!data.stocks || data.stocks.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center">No stock holdings found</td></tr>';
      return;
    }

    data.stocks.forEach((stock) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${stock.accountId}</td>
        <td>${stock.accountName}</td>
        <td class="text-right">${formatCurrency(stock.totalValue, stock.currency)}</td>
        <td>${stock.currency}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (error) {
    console.error('Error loading stocks:', error);
    showError('stocks');
  }
}

// Load Transactions
function loadTransactions(transactions) {
  const tbody = document.getElementById('transactionsTable').querySelector('tbody');
  tbody.innerHTML = '';

  if (!transactions || transactions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center">No recent transactions</td></tr>';
    return;
  }

  transactions.slice(0, 10).forEach((tx) => {
    const tr = document.createElement('tr');
    const amountClass = tx.type === 'CREDIT' ? 'text-green' : 'text-red';
    const amountPrefix = tx.type === 'CREDIT' ? '+' : '-';

    tr.innerHTML = `
      <td>${formatDate(tx.date)}</td>
      <td>${tx.description}</td>
      <td class="${amountClass} text-right">${amountPrefix}${formatCurrency(tx.amount, tx.currency)}</td>
      <td><span class="badge ${tx.type === 'CREDIT' ? 'badge-success' : 'badge-danger'}">${tx.type}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// Load Performance Chart
let performanceChart;

async function loadPerformance() {
  try {
    const data = await fetchJson('/financial/performance');
    const ctx = document.getElementById('performanceChart').getContext('2d');

    if (performanceChart) {
      performanceChart.destroy();
    }

    // Extract trend data
    const labels = data.trends.map(t => t.period);
    const values = data.trends.map(t => t.balance);

    performanceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Balance Trend',
            data: values,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.1)',
            tension: 0.4,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return `Balance: ${formatCurrency(context.parsed.y)}`;
              }
            }
          },
        },
        scales: {
          y: {
            beginAtZero: false,
            ticks: {
              callback: function(value) {
                return formatCurrency(value);
              }
            }
          },
        },
      },
    });
  } catch (error) {
    console.error('Error loading performance:', error);
    showError('performance');
  }
}

// Load System Status
async function loadStatus() {
  try {
    const status = await fetchJson('/system/status');

    const indicator = document.getElementById('statusIndicator');
    const lastUpdate = document.getElementById('lastUpdate');

    indicator.textContent = status.status === 'operational' ? 'Live' : 'Offline';
    indicator.className = status.status === 'operational' ? 'badge badge-success' : 'badge badge-danger';

    lastUpdate.textContent = `Last updated: ${formatDate(status.timestamp)}`;
  } catch (error) {
    console.error('Error loading status:', error);
    const indicator = document.getElementById('statusIndicator');
    indicator.textContent = 'Error';
    indicator.className = 'badge badge-danger';
  }
}

// Error Handling
function showError(section) {
  console.error(`Error in ${section} section`);
  // You can add visual error indicators here
}

// Initialize Dashboard
async function initializeDashboard() {
  console.log('Initializing dashboard...');

  try {
    // Load all data in parallel
    await Promise.all([
      loadSummary(),
      loadAssets(),
      loadStocks(),
      loadPerformance(),
      loadStatus(),
    ]);

    console.log('Dashboard initialized successfully');

    // Set up auto-refresh every 30 seconds
    setInterval(() => {
      console.log('Refreshing dashboard data...');
      loadSummary();
      loadAssets();
      loadStocks();
      loadPerformance();
      loadStatus();
    }, 30000);
  } catch (error) {
    console.error('Error initializing dashboard:', error);
    alert('Failed to load dashboard. Please check if the backend is running.');
  }
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initializeDashboard();
});
```

---

### 3. Dashboard Styles

Create `dashboard/styles.css`:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background: #f5f7fa;
  color: #333;
  line-height: 1.6;
}

.dashboard-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.dashboard-header {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dashboard-header h1 {
  font-size: 24px;
  color: #2563eb;
}

.status-bar {
  display: flex;
  gap: 15px;
  align-items: center;
}

.badge {
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
}

.badge-success {
  background: #10b981;
  color: white;
}

.badge-danger {
  background: #ef4444;
  color: white;
}

.badge-info {
  background: #3b82f6;
  color: white;
}

.text-muted {
  color: #6b7280;
  font-size: 14px;
}

.summary-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.card {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.card h3 {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 10px;
  text-transform: uppercase;
  font-weight: 600;
}

.metric-value {
  font-size: 32px;
  font-weight: 700;
  color: #1f2937;
  margin: 10px 0;
}

.metric-label {
  font-size: 12px;
  color: #9ca3af;
}

.chart-section {
  margin-bottom: 20px;
}

.chart-section .card {
  height: 400px;
}

#performanceChart {
  height: 320px !important;
}

.table-section {
  margin-bottom: 20px;
}

table {
  width: 100%;
  border-collapse: collapse;
}

thead {
  background: #f9fafb;
}

th {
  padding: 12px;
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  border-bottom: 2px solid #e5e7eb;
}

td {
  padding: 12px;
  border-bottom: 1px solid #e5e7eb;
  font-size: 14px;
}

tr:hover {
  background: #f9fafb;
}

.text-center {
  text-align: center;
}

.text-right {
  text-align: right;
}

.text-green {
  color: #10b981;
  font-weight: 600;
}

.text-red {
  color: #ef4444;
  font-weight: 600;
}

@media (max-width: 768px) {
  .dashboard-header {
    flex-direction: column;
    gap: 15px;
  }

  .summary-section {
    grid-template-columns: 1fr;
  }

  table {
    font-size: 12px;
  }

  th, td {
    padding: 8px;
  }
}
```

---

## 🚀 Running the Dashboard

### Option 1: Using Python HTTP Server

```bash
cd jpmorgan_financial_apis/dashboard
python -m http.server 8080
```

Open: `http://localhost:8080`

### Option 2: Using Node.js serve

```bash
cd jpmorgan_financial_apis/dashboard
npx serve .
```

### Option 3: Using VS Code Live Server

1. Install "Live Server" extension
2. Right-click `index.html`
3. Select "Open with Live Server"

---

## 🔄 Data Flow

```
Dashboard (Browser)
    ↓ HTTP GET
NestJS API (/api/financial/*)
    ↓ Query
Database (PostgreSQL)
    ↓ Return Data
NestJS API (Transform)
    ↓ JSON Response
Dashboard (Update UI)
```

---

## 🧪 Testing

### 1. Test Backend Endpoints

```bash
# Test summary
curl http://localhost:3000/api/financial/summary

# Test assets
curl http://localhost:3000/api/financial/assets

# Test performance
curl http://localhost:3000/api/financial/performance

# Test stocks
curl http://localhost:3000/api/financial/stocks

# Test status
curl http://localhost:3000/api/system/status
```

### 2. Test Dashboard

1. Open browser console (F12)
2. Check for errors
3. Verify API calls in Network tab
4. Confirm data loads correctly

---

## 🐛 Troubleshooting

### CORS Errors

**Error:** `Access to fetch at 'http://localhost:3000' from origin 'http://localhost:8080' has been blocked by CORS policy`

**Solution:** Enable CORS in `main.ts` (see Backend Configuration section)

### Connection Refused

**Error:** `Failed to fetch`

**Solution:** 
1. Verify backend is running: `npm run start:dev`
2. Check port is correct (default: 3000)
3. Test endpoint directly: `curl http://localhost:3000/api/financial/summary`

### Empty Data

**Error:** Tables show "No data found"

**Solution:**
1. Check database has data
2. Verify database connection in backend
3. Check backend logs for errors

---

## 📊 Customization

### Change API Base URL

```javascript
// In dashboard.js
const API_BASE = 'https://your-production-api.com/api';
```

### Change Refresh Interval

```javascript
// In dashboard.js, change from 30000 (30 seconds) to desired value
setInterval(() => {
  // refresh code
}, 60000); // 60 seconds
```

### Add New Metrics

1. Add HTML element in `index.html`
2. Fetch data in `dashboard.js`
3. Update DOM with new data

---

## 🎨 Advanced Features

### Add Loading Spinners

```javascript
function showLoading(elementId) {
  document.getElementById(elementId).innerHTML = '<div class="spinner"></div>';
}

function hideLoading(elementId) {
  // Remove spinner
}
```

### Add Error Messages

```javascript
function showError(section, message) {
  const element = document.getElementById(`${section}Error`);
  element.textContent = message;
  element.style.display = 'block';
}
```

### Add Filters

```javascript
async function loadAssets(filter = 'all') {
  const data = await fetchJson(`/financial/assets?type=${filter}`);
  // Update UI
}
```

---

## 📝 Complete Example

See the files created in `jpmorgan_financial_apis/dashboard/`:
- `index.html` - Dashboard structure
- `dashboard.js` - API integration logic
- `styles.css` - Styling

---

## ✅ Checklist

- [ ] Backend running on port 3000
- [ ] CORS enabled in backend
- [ ] Dashboard files created
- [ ] Dashboard served via HTTP server
- [ ] Browser console shows no errors
- [ ] Data loads successfully
- [ ] Charts render correctly
- [ ] Tables populate with data
- [ ] Auto-refresh works

---

**Last Updated:** January 2, 2025  
**Status:** Ready for Integration
