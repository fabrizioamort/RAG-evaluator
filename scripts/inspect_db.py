import sqlite3
from pathlib import Path


def inspect_db():
    root_path = Path(__file__).parent.parent
    db_path = root_path / "platform" / "backend" / "storage" / "dev.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Listing latest 5 evaluations:")
    cursor.execute(
        "SELECT id, summary_metrics, tags, notes FROM evaluations ORDER BY created_at DESC LIMIT 5"
    )
    rows = cursor.fetchall()

    for row in rows:
        print(f"ID: {row[0]}")
        print(f"Notes: {row[3]}")
        print(f"Tags: {row[2]}")
        print(f"Metrics: {row[1]}")
        print("-" * 20)

    conn.close()


if __name__ == "__main__":
    inspect_db()
