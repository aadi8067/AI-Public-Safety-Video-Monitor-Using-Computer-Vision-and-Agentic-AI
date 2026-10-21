"""
One-time script: creates a restricted PostgreSQL application user
'safety_app' with access only to public_safety_db.
Safe to re-run — uses IF NOT EXISTS pattern.
"""
import getpass
import sys
import psycopg

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "public_safety_db"

print("Enter credentials for the postgres SUPERUSER (to create the new user):")
admin_password = getpass.getpass("  postgres password: ")

print("\nChoose a password for the new app user 'safety_app':")
app_password = getpass.getpass("  safety_app password: ")
app_password2 = getpass.getpass("  confirm password   : ")

if app_password != app_password2:
    print("[ERROR] Passwords do not match.")
    sys.exit(1)

if len(app_password) < 8:
    print("[ERROR] Password must be at least 8 characters.")
    sys.exit(1)

# Connect as superuser with autocommit — CREATE ROLE cannot run in a transaction
conn = psycopg.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
    user="postgres", password=admin_password,
    autocommit=True
)
cur = conn.cursor()

# Create role only if it does not already exist
cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'safety_app'")
if cur.fetchone():
    print("\n'safety_app' role already exists — updating password.")
    cur.execute(f"ALTER ROLE safety_app WITH PASSWORD '{app_password}'")
else:
    cur.execute(
        f"CREATE ROLE safety_app WITH LOGIN PASSWORD '{app_password}' "
        f"NOSUPERUSER NOCREATEDB NOCREATEROLE"
    )
    print("\nRole 'safety_app' created.")

# Grant connect + usage + full DML on all current and future tables
cur.execute("GRANT CONNECT ON DATABASE public_safety_db TO safety_app")
cur.execute("GRANT USAGE ON SCHEMA public TO safety_app")
cur.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO safety_app")
cur.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO safety_app")

# Future tables/sequences created by postgres are also accessible
cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO safety_app")
cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO safety_app")

conn.close()
print("Permissions granted.")
print("\nNext: add to your .env file:")
print(f"  DB_PASSWORD={app_password}")
print("  (user = safety_app)")
