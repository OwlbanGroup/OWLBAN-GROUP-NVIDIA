# TODO: Add Query Parameters for Filtering JPMorgan Financial Data

## Tasks
- [x] Modify `/api/jpmorgan-data` endpoint in `app_final.py` to accept query parameters: `env`, `region`, `payment_type`, and `status`
- [x] Update mock data generation to include multiple data entries with different values for env, region, payment_type, and status
- [x] Implement filtering logic to return data matching the provided query parameters
- [x] Test the endpoint with various query parameter combinations
- [x] Verify behavior when no filters are provided (should return all data)
