"""
Access the live TimescaleDB instances that run in Docker.

The postgres servers listen on a NON-default port (socket
/run/postgresql/.s.PGSQL.<port>), so we exec into the container and let it find
its own port + credentials from the standard POSTGRES_* env. No host port needed.
"""
from __future__ import annotations
import shlex
import subprocess

# container -> default db
CONTAINERS = {"trading": "trading-timescaledb", "crypto": "crypto-timescaledb"}

_INNER_PREFIX = (
    'PORT=$(ls /run/postgresql/.s.PGSQL.* 2>/dev/null | grep -oE "[0-9]+" | sort -u | head -1); '
    'PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p "$PORT" -U "$POSTGRES_USER" '
    '-d {db} -v ON_ERROR_STOP=1 '
)


def query(container: str, db: str, sql: str) -> str:
    """Run a SQL query, return stdout (tab-separated, no align)."""
    inner = _INNER_PREFIX.format(db=db) + "-P pager=off -At -F$'\\t' -c " + shlex.quote(sql)
    out = subprocess.run(["docker", "exec", container, "bash", "-c", inner],
                         capture_output=True, text=True, check=True)
    return out.stdout


def copy_to_file(container: str, db: str, sql: str, out_path: str) -> int:
    """Stream a `COPY (...) TO STDOUT WITH CSV HEADER` into a host file. Returns bytes."""
    copy_sql = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"
    inner = _INNER_PREFIX.format(db=db) + "-c " + shlex.quote(copy_sql)
    with open(out_path, "wb") as f:
        subprocess.run(["docker", "exec", container, "bash", "-c", inner], stdout=f, check=True)
    import os
    return os.path.getsize(out_path)
