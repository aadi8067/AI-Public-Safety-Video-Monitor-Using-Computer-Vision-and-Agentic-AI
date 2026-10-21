"""
Executes schema.sql against public_safety_db.
Uses psycopg3 (already installed). Edit DB_PASSWORD below if needed.
"""
import psycopg
import sys
import getpass
from pathlib import Path

# ── connection settings ────────────────────────────────────────────────────
DB_HOST  = "localhost"
DB_PORT  = 5432
DB_NAME  = "public_safety_db"
DB_USER  = "postgres"
# getpass prompts securely — password is not shown or stored anywhere
DB_PASSWORD = getpass.getpass("Enter PostgreSQL password for 'postgres': ")
# ──────────────────────────────────────────────────────────────────────────

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

def main():
    print(f"\nConnecting to {DB_NAME} on {DB_HOST}:{DB_PORT} as {DB_USER}...")
    try:
        conn = psycopg.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            autocommit=False
        )
    except Exception as e:
        print(f"[ERROR] Could not connect: {e}")
        sys.exit(1)

    print("Connected successfully.")

    sql = SCHEMA_FILE.read_text(encoding="utf-8")

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("Schema executed successfully — no errors.")
    except Exception as e:
        print(f"[ERROR] Schema execution failed: {e}")
        conn.close()
        sys.exit(1)

    # ── verification: list all tables created ─────────────────────────────
    expected = [
        "sessions", "persons", "events", "incidents",
        "incident_events", "response_actions", "reports", "evidence_frames"
    ]

    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        found = [row[0] for row in cur.fetchall()]

    conn.close()

    print("\n── Tables in public_safety_db (public schema) ──")
    all_ok = True
    for t in expected:
        status = "OK" if t in found else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  {status:7s}  {t}")

    extra = [t for t in found if t not in expected]
    if extra:
        print(f"\n  Extra tables (not from this schema): {extra}")

    print()
    if all_ok:
        print("All 8 tables created successfully.")
    else:
        print("[WARNING] Some tables are missing — review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
