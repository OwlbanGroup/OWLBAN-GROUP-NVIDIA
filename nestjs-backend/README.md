# JPMorgan Financial APIs - NestJS Backend

A production-ready NestJS application for financial services with JPMorgan integration, featuring comprehensive security, monitoring, and best practices.

## 🚀 Features

### Core Features
- ✅ **TypeScript** - Full type safety
- ✅ **NestJS Framework** - Modular architecture
- ✅ **PostgreSQL** - Robust database with TypeORM
- ✅ **JWT Authentication** - Secure authentication
- ✅ **API Versioning** - URI-based versioning
- ✅ **Swagger Documentation** - Auto-generated API docs

### Security Features
- 🔒 **Helmet** - Security headers
- 🔒 **CORS** - Configurable cross-origin requests
- 🔒 **Rate Limiting** - Throttling with @nestjs/throttler
- 🔒 **Input Validation** - Class-validator for DTOs
- 🔒 **Environment Validation** - Type-safe configuration

### Monitoring & Observability
- 📊 **Health Checks** - Database, memory, and disk monitoring
- 📊 **Logging** - Structured logging with interceptors
- 📊 **Error Handling** - Global exception filters

### Performance
- ⚡ **Connection Pooling** - Optimized database connections
- ⚡ **Compression** - Response compression
- ⚡ **Caching Ready** - Redis integration prepared

## 📋 Prerequisites

- Node.js >= 18.x
- PostgreSQL >= 14.x
- npm or yarn
- Docker & Docker Compose (optional)

## 🛠️ Installation

### 1. Clone and Install Dependencies

```bash
cd nestjs-backend
npm install
```

### 2. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
NODE_ENV=development
PORT=3000
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=jpmorgan_financial_db
JWT_SECRET=your_super_secret_key
```

### 3. Database Setup

Create the database:

```bash
createdb jpmorgan_financial_db
```

Run migrations (when available):

```bash
npm run migration:run
```

## 🚀 Running the Application

### Development Mode

```bash
npm run start:dev
```

The application will be available at:
- API: http://localhost:3000/api
- Swagger Docs: http://localhost:3000/api/docs
- Health Check: http://localhost:3000/health

### Production Mode

```bash
npm run build
npm run start:prod
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## 📁 Project Structure

```
nestjs-backend/
├── src/
│   ├── config/                 # Configuration modules
│   │   ├── config.module.ts    # Global config module
│   │   ├── database.config.ts  # Database configuration
│   │   └── env.validation.ts   # Environment validation
│   ├── database/               # Database module
│   │   └── database.module.ts
│   ├── health/                 # Health check endpoints
│   │   ├── health.controller.ts
│   │   └── health.module.ts
│   ├── common/                 # Shared utilities
│   │   ├── filters/            # Exception filters
│   │   └── interceptors/       # Request/response interceptors
│   ├── auth/                   # Authentication module
│   ├── users/                  # User management
│   ├── organizations/          # Organization management
│   ├── bank-connections/       # Bank connections
│   ├── accounts/               # Account management
│   ├── balances/               # Balance information
│   ├── transactions/           # Transaction management
│   ├── payments/               # Payment processing
│   ├── payroll/                # Payroll management
│   ├── petty-cash/             # Petty cash management
│   ├── corporate/              # Corporate services
│   ├── connectors/             # External integrations
│   │   └── jpmorgan/           # JPMorgan connector
│   ├── app.module.ts           # Root module
│   └── main.ts                 # Application entry point
├── test/                       # Test files
├── .env.example                # Environment template
├── docker-compose.yml          # Docker composition
├── Dockerfile                  # Docker image definition
├── nest-cli.json               # NestJS CLI config
├── package.json                # Dependencies
├── tsconfig.json               # TypeScript config
└── README.md                   # This file
```

## 🔧 Key Improvements Over Original Code

### 1. Configuration Management
- ✅ Centralized configuration with validation
- ✅ Type-safe environment variables
- ✅ Separate database configuration module

### 2. Security Enhancements
- ✅ Helmet for security headers
- ✅ Rate limiting to prevent abuse
- ✅ CORS configuration
- ✅ Input validation with class-validator
- ✅ Global exception handling

### 3. Database Optimization
- ✅ Connection pooling configuration
- ✅ Retry logic for failed connections
- ✅ SSL support for production
- ✅ Proper migration setup

### 4. Monitoring & Observability
- ✅ Health check endpoints (liveness, readiness)
- ✅ Structured logging with interceptors
- ✅ Request/response logging
- ✅ Error tracking

### 5. Developer Experience
- ✅ Swagger API documentation
- ✅ Docker support
- ✅ Hot reload in development
- ✅ TypeScript strict mode
- ✅ Comprehensive error messages

## 🏥 Health Checks

The application provides three health check endpoints:

### General Health Check
```bash
GET /health
```
Checks database, memory, and disk health.

### Liveness Probe
```bash
GET /health/liveness
```
Verifies the application is running.

### Readiness Probe
```bash
GET /health/readiness
```
Checks if the application is ready to accept traffic.

## 📚 API Documentation

Access the interactive Swagger documentation at:
```
http://localhost:3000/api/docs
```

Features:
- Interactive API testing
- Request/response schemas
- Authentication testing
- Example payloads

## 🧪 Testing

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Test coverage
npm run test:cov
```

## 🔐 Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **JWT Secret**: Use strong, random secrets in production
3. **Database Credentials**: Rotate regularly
4. **Rate Limiting**: Adjust based on your needs
5. **CORS**: Configure specific origins in production
6. **SSL/TLS**: Enable for database connections in production

## 📊 Performance Optimization

1. **Connection Pooling**: Configured for optimal database performance
2. **Compression**: Enabled for response compression
3. **Caching**: Redis integration ready
4. **Query Optimization**: Use indexes and proper relations

## 🚢 Deployment

### Environment Variables for Production

```env
NODE_ENV=production
PORT=3000
DB_HOST=your-db-host
DB_PORT=5432
DB_USER=your-db-user
DB_PASSWORD=strong-password
DB_NAME=jpmorgan_financial_db
JWT_SECRET=very-strong-secret-key
CORS_ORIGIN=https://yourdomain.com
THROTTLE_TTL=60
THROTTLE_LIMIT=100
```

### Docker Production Deployment

```bash
# Build production image
docker build -t jpmorgan-api:latest .

# Run with docker-compose
docker-compose -f docker-compose.yml up -d
```

### Kubernetes Deployment

Health check configuration for K8s:

```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 3000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 3000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check existing documentation
- Review Swagger API docs

## 🔄 Migration from Original Code

If migrating from the original AppModule:

1. Install new dependencies: `npm install`
2. Copy `.env.example` to `.env` and configure
3. Update module imports to use new structure
4. Run database migrations
5. Test all endpoints
6. Update deployment configurations

## 📈 Roadmap

- [ ] Add Redis caching layer
- [ ] Implement GraphQL API
- [ ] Add WebSocket support
- [ ] Implement audit logging
- [ ] Add Prometheus metrics
- [ ] Implement distributed tracing
- [ ] Add API rate limiting per user
- [ ] Implement data encryption at rest

---

**Built with ❤️ using NestJS**
