# Load Testing Suite for JPMorgan Financial APIs

This directory contains load testing scripts for the JPMorgan Financial APIs using Locust.

## Prerequisites

Install the required dependencies:

```bash
pip install -r ../requirements.txt
```

## Files

- `locustfile.py` - Main Locust test script with user scenarios
- `package.json` - Artillery configuration (legacy, kept for reference)
- `artillery-config.yml` - Artillery configuration (legacy, kept for reference)
- `processors/telemetry-processor.js` - Artillery data processor (legacy, kept for reference)

## Running Load Tests

### Using Locust (Recommended)

1. **Web Interface Mode** (Interactive):
   ```bash
   locust -f locustfile.py --host http://localhost:5000
   ```
   Then open http://localhost:8089 in your browser to control the test.

2. **Headless Mode** (Automated):
   ```bash
   # Warm-up test (60 seconds, 5 users/sec)
   locust -f locustfile.py --host http://localhost:5000 --no-web --run-time 60 --spawn-rate 5 --users 50

   # Load test (300 seconds, 20 users/sec)
   locust -f locustfile.py --host http://localhost:5000 --no-web --run-time 300 --spawn-rate 20 --users 1000

   # Stress test (120 seconds, 50 users/sec)
   locust -f locustfile.py --host http://localhost:5000 --no-web --run-time 120 --spawn-rate 50 --users 2000

   # Spike test (60 seconds, 100 users/sec)
   locust -f locustfile.py --host http://localhost:5000 --no-web --run-time 60 --spawn-rate 100 --users 5000
   ```

3. **Distributed Testing**:
   ```bash
   # Master node
   locust -f locustfile.py --master --host http://your-api-host

   # Worker nodes (run on different machines)
   locust -f locustfile.py --worker --master-host=master-ip
   ```

### Using Artillery (Legacy)

```bash
npm install
npm run test:smoke    # Smoke test
npm run test:load     # Load test
npm run test:stress   # Stress test
npm run test:spike    # Spike test
```

## Test Scenarios

The load tests simulate real-world usage patterns with the following scenarios:

1. **Submit Telemetry Data** (70% weight)
   - POST `/api/telemetry` with realistic telemetry data
   - Tests the main data ingestion endpoint

2. **Health Check** (20% weight)
   - GET `/health`
   - Tests system health monitoring

3. **Get Telemetry Metrics** (10% weight)
   - GET `/api/telemetry/metrics`
   - Tests metrics retrieval functionality

4. **Submit GPU Telemetry** (10% weight)
   - POST `/api/telemetry/gpu` with GPU-specific data
   - Tests GPU telemetry ingestion

## Configuration

### Environment Variables

Set these environment variables to configure the tests:

```bash
export API_HOST=http://localhost:5000  # Target API host
export LOCUST_USERS=1000              # Number of users
export LOCUST_SPAWN_RATE=20           # Users spawned per second
export LOCUST_RUN_TIME=300            # Test duration in seconds
```

### Customizing Test Data

The `generate_telemetry_data()` method in `locustfile.py` uses Faker to generate realistic test data. Modify this method to customize the data generation.

## Monitoring and Results

### Locust Web Interface

When running in web mode, Locust provides:
- Real-time statistics (RPS, response times, failures)
- Charts and graphs
- Request/response details
- Downloadable CSV reports

### Key Metrics to Monitor

- **Response Time**: P95 should be < 500ms for telemetry endpoints
- **Error Rate**: Should be < 1%
- **Throughput**: Target 1000+ RPS for load tests
- **Resource Usage**: Monitor CPU, memory, and database connections

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
- name: Run Load Tests
  run: |
    pip install locust faker
    locust -f load-testing/locustfile.py --host ${{ secrets.API_HOST }} --no-web --run-time 60 --spawn-rate 5 --users 100 --csv results
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the API is running and accessible
2. **High Error Rates**: Check API logs for issues, consider reducing load
3. **Memory Issues**: Reduce user count or add more worker nodes
4. **Slow Response Times**: Check database performance and API optimization

### Performance Tuning

- Increase `wait_time` to reduce load
- Adjust task weights based on real usage patterns
- Use distributed testing for high loads
- Monitor system resources during tests

## Best Practices

1. **Test Environment**: Always test against staging/production-like environments
2. **Gradual Load Increase**: Start with low loads and gradually increase
3. **Monitor Resources**: Watch CPU, memory, disk I/O, and network
4. **Realistic Data**: Use production-like data volumes and patterns
5. **Regular Testing**: Include load tests in CI/CD pipeline
6. **Baseline Metrics**: Establish performance baselines for regression testing

## Contributing

When adding new test scenarios:

1. Follow the existing pattern in `locustfile.py`
2. Add appropriate task weights
3. Include proper error handling
4. Update this README with new scenarios
5. Test locally before committing
