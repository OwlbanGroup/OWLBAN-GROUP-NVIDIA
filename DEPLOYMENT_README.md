# JPMorgan Financial APIs - Railway Deployment Guide

## 🚀 Quick Railway Deployment

### Prerequisites
- Railway account (https://railway.com)
- GitHub account (for connecting repository)

### Step 1: Prepare Your Repository
1. Ensure all files are committed to your Git repository
2. The `railway.json` configuration file is already created in your project root

### Step 2: Deploy to Railway
1. Go to [Railway.com](https://railway.com) and sign in
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select the `jpmorgan_financial_apis` repository
4. Railway will automatically detect the Python application and start deployment

### Step 3: Configure Environment Variables
In your Railway project dashboard, go to "Variables" and add these required variables:

#### Required Variables
```
SECRET_KEY=your-32-character-secret-key
JWT_SECRET_KEY=your-32-character-jwt-secret-key
DATABASE_URL=postgresql://user:password@host:5432/database
ALLOWED_ORIGINS=https://your-app-name.railway.app
```

#### Optional Variables
```
REDIS_URL=redis://user:password@host:port/db
TOKEN_CLIENT_ID=your-jpmorgan-client-id
TOKEN_CLIENT_SECRET=your-jpmorgan-client-secret
APOLLO_API_KEY=your-apollo-api-key
AUDIT_LOG_ENABLED=true
LOG_LEVEL=INFO
```

### Step 4: Database Setup
1. Add a PostgreSQL database to your Railway project
2. Railway will provide a `DATABASE_URL` automatically
3. The application will create tables automatically on first run

### Step 5: Verify Deployment
1. Once deployed, Railway will provide a URL (e.g., `https://your-app-name.railway.app`)
2. Test the health endpoint: `https://your-app-name.railway.app/health`
3. Check the API documentation: `https://your-app-name.railway.app/`

## 📋 API Endpoints Available

### Core Endpoints
- `GET /health` - Health check
- `GET /` - API information and available endpoints
- `POST /user/register` - User registration
- `POST /user/login` - User authentication
- `GET /user/profile` - User profile (requires JWT)

### Financial Data Endpoints
- `GET /api/jpmorgan-data` - JPMorgan financial metrics and stock data
- `GET /private-bank/accounts` - Private banking accounts
- `GET /private-bank/wealth` - Wealth management portfolio
- `GET /private-bank/investments` - Investment portfolio

### Business Management
- `GET /businesses` - List businesses
- `POST /businesses` - Create business
- `GET /assets` - List assets
- `POST /assets` - Create asset

### Audit & Monitoring
- `GET /audit/logs` - Query audit logs
- `GET /audit/summary` - Audit statistics
- `GET /metrics` - Prometheus metrics

### Data Enrichment (Apollo.io)
- `POST /enrichment/contact` - Enrich contact information
- `POST /enrichment/company` - Enrich company information
- `GET /enrichment/search/contacts` - Search contacts
- `GET /enrichment/search/companies` - Search companies

## 🔧 Troubleshooting

### Common Issues

**Application won't start:**
- Check that all required environment variables are set
- Verify DATABASE_URL is correct
- Check Railway build logs for Python dependency errors

**Database connection errors:**
- Ensure PostgreSQL database is added to your Railway project
- Verify DATABASE_URL environment variable
- Check database credentials

**Port binding errors:**
- Railway automatically assigns PORT environment variable
- The app is configured to use this automatically

### Logs and Monitoring
- View application logs in Railway dashboard under "Logs" tab
- Check build logs for deployment issues
- Monitor resource usage in Railway metrics

## 🔒 Security Notes

- The application uses JWT authentication for protected endpoints
- Audit logging is enabled by default
- CORS is configured for Railway domain
- All secrets should be set as Railway environment variables (not in code)

## 📞 Support

For Railway-specific issues:
- Railway Documentation: https://docs.railway.com/
- Railway Community: https://discord.gg/railway

For application-specific issues:
- Check the API documentation at `/` endpoint
- Review application logs in Railway dashboard

---

**🎉 Your JPMorgan Financial APIs are now deployed on Railway!**
