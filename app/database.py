import sqlite3

DB_NAME = "crowd_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total INTEGER,
            zoneA INTEGER,
            zoneB INTEGER,
            zoneC INTEGER,
            zoneD INTEGER
        )
    """)

    conn.commit()
    conn.close()


def insert_log(timestamp, total, zoneA, zoneB, zoneC, zoneD):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO crowd_logs (timestamp, total, zoneA, zoneB, zoneC, zoneD)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, total, zoneA, zoneB, zoneC, zoneD))

    conn.commit()
    conn.close()


def get_last_logs(limit=30):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, total, zoneA, zoneB, zoneC, zoneD
        FROM crowd_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    rows.reverse()

    return [
        {
            "timestamp": r[0],
            "total": r[1],
            "zoneA": r[2],
            "zoneB": r[3],
            "zoneC": r[4],
            "zoneD": r[5],
        }
        for r in rows
    ]