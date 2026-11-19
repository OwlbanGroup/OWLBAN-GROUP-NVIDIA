# Public TLS and GoDaddy DNS

This runbook configures `owlbangroup.com` and `www.owlbangroup.com`.

## 1. Create GoDaddy DNS API credentials

Create production API credentials in the GoDaddy developer portal. Never add the key or secret to source files.

Set them only in the deployment shell:

```powershell
$env:GODADDY_API_KEY="<GoDaddy production API key>"
$env:GODADDY_API_SECRET="<GoDaddy production API secret>"
```

## 2. Configure public A records

Run this once the public server IPv4 address is known:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/configure-godaddy-dns.ps1 `
  -PublicIp "<public IPv4 address>" `
  -Domain "owlbangroup.com"
```

This creates or replaces:

- `owlbangroup.com` -> public IPv4
- `www.owlbangroup.com` -> public IPv4

## 3. Wait for DNS

Verify DNS resolution before requesting certificates:

```powershell
Resolve-DnsName owlbangroup.com
Resolve-DnsName www.owlbangroup.com
```

Both names should resolve to the public server address.

## 4. Obtain a trusted certificate

For a public server with reachable ports 80 and 443, use an ACME client such as Certbot or Caddy.

The certificate must include:

- `owlbangroup.com`
- `www.owlbangroup.com`

Install the chain and private key at:

- `deploy/tls/fullchain.pem`
- `deploy/tls/privkey.pem`

Replace the local self-signed certificate. The private key must remain excluded from Git.

## 5. Set production browser origins

Update `.env.production` so CORS lists only:

```text
CORS_ORIGINS=https://owlbangroup.com,https://www.owlbangroup.com
```

Generate or refresh the production environment with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-production.ps1 -Force `
  -Domain owlbangroup.com `
  -CorsOrigin "https://owlbangroup.com,https://www.owlbangroup.com"
```

## 6. Deploy and verify

```powershell
docker compose -f docker-compose.production.yml config
docker compose -f docker-compose.production.yml up --build -d
curl.exe https://owlbangroup.com/health
curl.exe https://www.owlbangroup.com/health
```

Verify the certificate chain, certificate expiry, redirect behavior, and authenticated endpoints. Configure certificate renewal before the certificate expires.
