# TLS certificate files

Place the active TLS certificate chain at `fullchain.pem` and its matching private key at `privkey.pem`.

These are mounted read-only by the production Compose stack. Restrict key access and never commit certificate private keys.
