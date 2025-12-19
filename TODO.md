# TODO: Add Query Parameters for Filtering JPMorgan Financial Data

## Tasks
- [ ] Modify `/api/jpmorgan-data` endpoint in `app.py` to accept query parameters: `env`, `region`, `payment_type`, and `status`
- [ ] Update mock data generation to include multiple data entries with different values for env, region, payment_type, and status
- [ ] Implement filtering logic to return data matching the provided query parameters
- [ ] Test the endpoint with various query parameter combinations
- [ ] Verify behavior when no filters are provided (should return all data)
