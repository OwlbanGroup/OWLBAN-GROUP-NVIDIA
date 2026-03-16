# TODO - Production Runnable Fixes

- [x] Remove fragile custom PostgreSQL startup command from `docker-compose.production.yml`
- [x] Disable Windows-incompatible `node-exporter` service mounts from `docker-compose.production.yml`
- [x] Add/ensure `restart: unless-stopped` for `certbot`
- [x] Validate final compose with `docker compose -f jpmorgan_financial_apis/docker-compose.production.yml config`
- [ ] Switch `jpmorgan-api` from pull-only image to local build fallback in `docker-compose.production.yml`
- [ ] Rebuild and start production stack with `docker compose -f jpmorgan_financial_apis/docker-compose.production.yml up -d --build`
