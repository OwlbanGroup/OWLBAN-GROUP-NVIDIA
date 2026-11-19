"""Generate a development-only TLS certificate and private key."""

from __future__ import annotations

import argparse
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate_certificate(domain: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domain)])
    san_names: list[x509.GeneralName] = [x509.DNSName(domain)]
    if domain.startswith("www."):
        san_names.append(x509.DNSName(domain.removeprefix("www.")))
    elif "." in domain:
        san_names.append(x509.DNSName(f"www.{domain}"))

    try:
        san_names.append(x509.IPAddress(ipaddress.ip_address(domain)))
    except ValueError:
        pass

    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_names), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    fullchain_path = output_dir / "fullchain.pem"
    privkey_path = output_dir / "privkey.pem"
    fullchain_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    privkey_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="localhost")
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/tls"))
    args = parser.parse_args()
    generate_certificate(args.domain, args.output_dir)
