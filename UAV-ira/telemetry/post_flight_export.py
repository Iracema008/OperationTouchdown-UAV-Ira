""" Reads the SQLite telem db and exports to CSV for analysis in Excel/Python """
# python post_flight_export.py flight_log.db output.csv


import sys
import sqlite3
import csv


def export_to_csv(db_path: str, csv_path: str):
    """Export telemetry database to CSV."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM telemetry ORDER BY t_sec")
    rows = cursor.fetchall()
    
    if not rows:
        print("No data in database")
        return
    
    # Get column names
    col_names = [desc[0] for desc in cursor.description]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)
    
    conn.close()
    print(f"Exported {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python post_flight_export.py <db_path> <csv_path>")
        sys.exit(1)
    
    export_to_csv(sys.argv[1], sys.argv[2])
