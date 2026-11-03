"""
Apply (or revert) the OSCAR BROOME revenue-service update.

OSCAR-BROOME-REVENUE is recorded as a gitlink in this repository, so edits made
inside it are tracked by neither repo. This script is the record of the update;
it is idempotent and refuses to guess when a file has drifted.

    python .tools/apply_update.py             # apply (default root = this repo)
    python .tools/apply_update.py --check     # report only, write nothing
    python .tools/apply_update.py --revert    # undo

What the update does
--------------------
server_with_auth.js   no hardcoded owner password: OSCAR_OWNER_PASSWORD is
                      required to pin it, otherwise a random one is generated
                      and printed once on first seed.
e2e-test.js           owner credentials come from OSCAR_OWNER_EMAIL /
                      OSCAR_OWNER_PASSWORD instead of literals.
verify-oscar-login.mjs tolerates a clone without a data/ directory.
.env.example          OSCAR_OWNER_PASSWORD empty by default + the upstream
                      OVERRIDE_*_PASS variables the clone requires.
docker-compose.yml    OSCAR_JWT_SECRET / OSCAR_OWNER_* wired through.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMODULE = ROOT / "OSCAR-BROOME-REVENUE"

SERVER = "server_with_auth.js"
E2E = "e2e-test.js"
VERIFY = "verify-oscar-login.mjs"

INDENT = "    "


def ind(text: str, level: int = 1) -> str:
    """Indent every non-blank line of text by `level` four-space units."""
    pad = INDENT * level
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


# --------------------------------------------------------------------------- #
# Submodule edits: (filename, original text, updated text)
# --------------------------------------------------------------------------- #

SERVER_OLD_PASSWORD = """const OWNER_USERNAME = process.env.OSCAR_OWNER_USERNAME || 'Oscar Broome';
const OWNER_PASSWORD = process.env.OSCAR_OWNER_PASSWORD || 'OscarBroome!2026';
"""
SERVER_NEW_PASSWORD = """const OWNER_USERNAME = process.env.OSCAR_OWNER_USERNAME || 'Oscar Broome';
// No password is ever baked into the repo: without OSCAR_OWNER_PASSWORD the
// seed below generates a random one and prints it to the console exactly once.
const OWNER_PASSWORD = process.env.OSCAR_OWNER_PASSWORD;
"""

SERVER_OLD_SEED = ind("""users[OWNER_EMAIL] = {
    email: OWNER_EMAIL,
    username: OWNER_USERNAME,
    passwordHash: await bcrypt.hash(OWNER_PASSWORD, 10),
""")
SERVER_NEW_SEED = ind("""// Generate-and-print keeps zero-config startups working without shipping a
// known password. Only printed on first seed — later boots find the record.
const generated = !OWNER_PASSWORD;
const password = OWNER_PASSWORD || crypto.randomBytes(12).toString('base64url');
users[OWNER_EMAIL] = {
    email: OWNER_EMAIL,
    username: OWNER_USERNAME,
    passwordHash: await bcrypt.hash(password, 10),
""")

SERVER_OLD_SEED_LOG = ind("""console.log(`Seeded owner account for Oscar Broome (${OWNER_EMAIL})`);
}
""")
SERVER_NEW_SEED_LOG = (
    "    console.log(`Seeded owner account for Oscar Broome (${OWNER_EMAIL})`);\n"
    "    if (generated) {\n"
    "        console.log(`Generated owner password (set OSCAR_OWNER_PASSWORD to pin it): ${password}`);\n"
    "    }\n"
    "}\n"
)

E2E_OLD_SPAWN = """    const port = await freePort();
    const baseUrl = `http://127.0.0.1:${port}`;
    const child = spawn(process.execPath, [path.join(APP_DIR, 'server_with_auth.js')], {
"""
E2E_NEW_SPAWN = """    const port = await freePort();
    const baseUrl = `http://127.0.0.1:${port}`;
    // Credentials are supplied through the environment and never committed.
    const ownerEmail = process.env.OSCAR_OWNER_EMAIL || 'oscar.broome@jpmorgan.com';
    const ownerPassword = process.env.OSCAR_OWNER_PASSWORD;
    if (!ownerPassword) {
        console.error('OSCAR_OWNER_PASSWORD must be set so the owner account can be seeded and signed in.');
        process.exit(2);
    }
    const child = spawn(process.execPath, [path.join(APP_DIR, 'server_with_auth.js')], {
"""

E2E_OLD_LOGIN = """        await page.fill('#le', 'oscar.broome@jpmorgan.com');
        await page.fill('#lp', 'OscarBroome!2026');
"""
E2E_NEW_LOGIN = """        await page.fill('#le', ownerEmail);
        await page.fill('#lp', ownerPassword);
"""

# The waitForFunction callback runs inside the page, so the e-mail has to be
# passed in as an argument rather than closed over.
E2E_OLD_WAIT = """        await page.waitForFunction(() =>
            document.getElementById('content')?.style.display === 'block' &&
            document.querySelectorAll('#reportTable tr').length === 3 &&
            document.getElementById('profileInfo')?.innerText.includes('oscar.broome@jpmorgan.com'),
        null, { timeout: 20000 });
"""
E2E_NEW_WAIT = """        await page.waitForFunction((email) =>
            document.getElementById('content')?.style.display === 'block' &&
            document.querySelectorAll('#reportTable tr').length === 3 &&
            document.getElementById('profileInfo')?.innerText.includes(email),
        ownerEmail, { timeout: 20000 });
"""

E2E_OLD_PROFILE = """            /oscar\\.broome@jpmorgan\\.com/.test(data.profile || '') &&
"""
E2E_NEW_PROFILE = """            (data.profile || '').includes(ownerEmail) &&
"""

E2E_OLD_BAD = """        await bad.fill('#le', 'oscar.broome@jpmorgan.com');
"""
E2E_NEW_BAD = """        await bad.fill('#le', ownerEmail);
"""

VERIFY_OLD_TMP = """const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oscar-auth-'));
fs.cpSync(path.join(ROOT, 'data'), path.join(tmp, 'data'), { recursive: true });
"""
VERIFY_NEW_TMP = """const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oscar-auth-'));
const dataDir = path.join(ROOT, 'data');
if (fs.existsSync(dataDir)) {
    fs.cpSync(dataDir, path.join(tmp, 'data'), { recursive: true });
}
"""

EDITS = [
    (SERVER, SERVER_OLD_PASSWORD, SERVER_NEW_PASSWORD),
    (SERVER, SERVER_OLD_SEED, SERVER_NEW_SEED),
    (SERVER, SERVER_OLD_SEED_LOG, SERVER_NEW_SEED_LOG),
    (E2E, E2E_OLD_SPAWN, E2E_NEW_SPAWN),
    (E2E, E2E_OLD_LOGIN, E2E_NEW_LOGIN),
    (E2E, E2E_OLD_WAIT, E2E_NEW_WAIT),
    (E2E, E2E_OLD_PROFILE, E2E_NEW_PROFILE),
    (E2E, E2E_OLD_BAD, E2E_NEW_BAD),
    (VERIFY, VERIFY_OLD_TMP, VERIFY_NEW_TMP),
]

# --------------------------------------------------------------------------- #
# Root-repo edits
# --------------------------------------------------------------------------- #

ENV_OLD = "OSCAR_OWNER_PASSWORD=OscarBroome!2026\n"
ENV_NEW = (
    "# No default: unset seeds the owner with a random password (printed once to the\n"
    "# console). Set a strong value here to pin reproducible credentials.\n"
    "OSCAR_OWNER_PASSWORD=\n"
)
ENV_OVERRIDE_ANCHOR = "OSCAR_JWT_SECRET=\n"
ENV_OVERRIDE_BLOCK = """
# --- Upstream OSCAR-BROOME-REVENUE clone (OwlbanGroup/OSCAR-BROOME-REVENUE) ---
# scripts/verify-oscar-login.mjs boots that ESM server; middleware/authOverride.js
# refuses to start unless all three override passwords are set.
OVERRIDE_ADMIN_PASS=
OVERRIDE_MANAGER_PASS=
OVERRIDE_SUPER_ADMIN_PASS=
"""

COMPOSE_JWT_OLD = "- JWT_SECRET=${JWT_SECRET:-oscar_broome_jwt_secret_change_me}\n"
COMPOSE_JWT_NEW = "- JWT_SECRET=${OSCAR_JWT_SECRET:-oscar_broome_jwt_secret_change_me}\n"
COMPOSE_ANCHOR = "API_SERVER_URL=http://api-server:8000\n"
COMPOSE_OWNERS = (
    "      - OSCAR_OWNER_EMAIL=${OSCAR_OWNER_EMAIL:-oscar.broome@jpmorgan.com}\n"
    "      - OSCAR_OWNER_USERNAME=${OSCAR_OWNER_USERNAME:-Oscar Broome}\n"
    "      - OSCAR_OWNER_PASSWORD=${OSCAR_OWNER_PASSWORD:-OscarBroome!2026}\n"
)
COMPOSE_OWNERS_RE = re.compile(
    r"[ \t]*- OSCAR_OWNER_EMAIL=\$\{OSCAR_OWNER_EMAIL:-[^\n]*\n"
    r"[ \t]*- OSCAR_OWNER_USERNAME=\$\{OSCAR_OWNER_USERNAME:-[^\n]*\n"
    r"[ \t]*- OSCAR_OWNER_PASSWORD=[^\n]*\n"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class Drift(Exception):
    """A file no longer matches the update; stop instead of guessing."""


def read(path: Path) -> tuple[str, bool]:
    raw = path.read_text(encoding="utf-8")
    return raw, "\r\n" in raw


def write(path: Path, text: str, crlf: bool) -> None:
    path.write_text(text.replace("\n", "\r\n") if crlf else text, encoding="utf-8")


def replace_once(path: Path, old: str, new: str, check: bool, revert: bool) -> str:
    text, crlf = read(path)
    flat = text.replace("\r\n", "\n")
    source, target = (new, old) if revert else (old, new)
    if target in flat:
        return "ALREADY"
    count = flat.count(source)
    if count != 1:
        raise Drift(f"{path.name}: anchor matched {count} times (expected 1)")
    if not check:
        write(path, flat.replace(source, target), crlf)
    return "PATCH"


def apply_all(root: Path, repo: Path, check: bool, revert: bool) -> list[str]:
    results: list[str] = []

    for name, old, new in EDITS:
        try:
            status = replace_once(repo / name, old, new, check, revert)
        except Drift as exc:
            raise Drift(str(exc)) from exc
        results.append(f"{status} {name}")

    env_path = root / ".env.example"
    if not env_path.exists():
        results.append("MISSING .env.example")
    else:
        text, crlf = read(env_path)
        flat = text.replace("\r\n", "\n")
        changed = False
        source, target = (ENV_NEW, ENV_OLD) if revert else (ENV_OLD, ENV_NEW)
        if target not in flat:
            if flat.count(source) != 1:
                raise Drift(".env.example: OSCAR_OWNER_PASSWORD line not found once")
            flat = flat.replace(source, target)
            changed = True
        want_overrides = not revert
        has_overrides = "OVERRIDE_ADMIN_PASS" in flat
        if want_overrides and not has_overrides:
            if flat.count(ENV_OVERRIDE_ANCHOR) != 1:
                raise Drift(".env.example: OSCAR_JWT_SECRET= anchor not found once")
            flat = flat.replace(ENV_OVERRIDE_ANCHOR, ENV_OVERRIDE_ANCHOR + ENV_OVERRIDE_BLOCK, 1)
            changed = True
        elif revert and has_overrides:
            flat = flat.replace(ENV_OVERRIDE_BLOCK, "", 1)
            changed = True
        if changed and not check:
            write(env_path, flat, crlf)
        results.append(("PATCH" if changed else "ALREADY") + " .env.example")

    compose_path = root / "docker-compose.yml"
    if not compose_path.exists():
        results.append("MISSING docker-compose.yml")
    else:
        text, crlf = read(compose_path)
        flat = text.replace("\r\n", "\n")
        changed = False
        # The JWT secret moved to OSCAR_JWT_SECRET (see .env.example).
        if revert:
            if COMPOSE_JWT_NEW in flat:
                flat = flat.replace(COMPOSE_JWT_NEW, COMPOSE_JWT_OLD, 1)
                changed = True
        elif COMPOSE_JWT_OLD in flat:
            flat = flat.replace(COMPOSE_JWT_OLD, COMPOSE_JWT_NEW, 1)
            changed = True
        # Owner env block: removed on revert, appended to the OSCAR service otherwise.
        if revert:
            if COMPOSE_OWNERS_RE.search(flat):
                flat = COMPOSE_OWNERS_RE.sub("", flat, count=1)
                changed = True
        elif "OSCAR_OWNER_USERNAME=" not in flat:
            # Anchor on the 3001 service, not just the URL: api-server URLs occur
            # several times in the file.
            service = re.search(r"\n  oscar-broome:\n.*?" + re.escape(COMPOSE_ANCHOR),
                                flat, re.S)
            if not service:
                raise Drift("docker-compose.yml: oscar-broome service / API_SERVER_URL not found")
            end = service.end()
            flat = flat[:end] + COMPOSE_OWNERS + flat[end:]
            changed = True
        if changed and not check:
            write(compose_path, flat, crlf)
        results.append(("PATCH" if changed else "ALREADY") + " docker-compose.yml")

    return results


def main() -> int:
    """Parse arguments, then apply or revert the update."""
    parser = argparse.ArgumentParser(
        description="Apply or revert the OSCAR BROOME revenue-service update.")
    parser.add_argument("--submodule", type=Path, default=None,
                        help=f"OSCAR-BROOME-REVENUE checkout (default {DEFAULT_SUBMODULE})")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repo holding .env.example / docker-compose.yml (default: this repo)")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--revert", action="store_true", help="undo the update")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    root = args.root.resolve()
    repo = args.submodule or Path(os.environ.get("OSCAR_SUBMODULE", DEFAULT_SUBMODULE))
    repo = repo.resolve()
    if not repo.is_dir():
        print(f"submodule not found: {repo}", file=sys.stderr)
        return 2

    try:
        results = apply_all(root, repo, args.check, args.revert)
    except Drift as exc:
        print(f"drift: {exc} - apply this part by hand", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"submodule": str(repo), "root": str(root), "results": results}, indent=2))
        return 0

    verb = "Would apply" if args.check else ("Reverted" if args.revert else "Applied")
    print(f"{verb} OSCAR update (submodule {repo}, root {root})")
    for line in results:
        print("  " + line)
    if not args.check and not args.revert:
        print("\nNext: cd OSCAR-BROOME-REVENUE; $env:OSCAR_OWNER_PASSWORD='...'; npm test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
