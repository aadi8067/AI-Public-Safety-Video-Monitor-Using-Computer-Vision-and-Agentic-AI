"""One-time script: verify live DB columns match schema.sql expectations."""
import getpass, sys, psycopg

DB_HOST, DB_PORT, DB_NAME, DB_USER = "localhost", 5432, "public_safety_db", "postgres"
DB_PASSWORD = getpass.getpass("Postgres password: ")

EXPECTED = {
    "sessions":         {"id","filename","file_path","fps","total_frames","duration_sec","status","output_dir","created_at","updated_at"},
    "persons":          {"id","session_id","stable_id","first_frame","last_frame","is_threat"},
    "events":           {"id","session_id","event_uid","event_type","person_id","frame","timestamp_sec","confidence","severity","details","created_at"},
    "incidents":        {"id","session_id","incident_uid","start_time","end_time","duration_sec","event_types","person_ids","event_count","risk_score","risk_severity","risk_explanation","created_at"},
    "incident_events":  {"incident_id","event_id"},
    "response_actions": {"id","incident_id","action_code","action_label","triggered_at"},
    "reports":          {"id","incident_id","summary","what_happened","risk_reason","persons","generated_at"},
    "evidence_frames":  {"id","incident_id","event_id","filename","file_path","image_url","frame","timestamp_sec","exists_on_disk","created_at"},
}

conn = psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
cur  = conn.cursor()

all_ok = True
for table, expected_cols in EXPECTED.items():
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
    """, (table,))
    live_cols = {row[0] for row in cur.fetchall()}
    missing = expected_cols - live_cols
    extra   = live_cols - expected_cols
    if missing or extra:
        all_ok = False
        print(f"  MISMATCH  {table}")
        if missing: print(f"            missing : {missing}")
        if extra:   print(f"            extra   : {extra}")
    else:
        print(f"  OK        {table}  ({len(live_cols)} columns)")

conn.close()
print("\nAll tables match schema.sql." if all_ok else "\nSchema mismatch — review above.")
sys.exit(0 if all_ok else 1)
