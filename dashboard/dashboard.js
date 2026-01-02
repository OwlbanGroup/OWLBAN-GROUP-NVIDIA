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
    alert('Failed to load dashboard. Please check if the backend is running on http://localhost:3000');
  }
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initializeDashboard();
});
