import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "time_tracker.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS clients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        address TEXT,
        hourly_rate REAL DEFAULT 0.0
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id INTEGER,
        project TEXT NOT NULL,
        description TEXT,
        start_ts TEXT NOT NULL,
        end_ts TEXT,
        duration_min INTEGER,
        FOREIGN KEY (client_id) REFERENCES clients (id)
    )
    ''')
    # Add client_id column if it doesn't exist
    try:
        cur.execute('ALTER TABLE entries ADD COLUMN client_id INTEGER REFERENCES clients(id)')
    except sqlite3.OperationalError:
        pass  # Column already exists
    cur.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        name TEXT PRIMARY KEY,
        target_hours REAL DEFAULT 0.0
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS invoices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_number TEXT UNIQUE NOT NULL,
        client_id INTEGER NOT NULL,
        invoice_date TEXT NOT NULL,
        due_date TEXT NOT NULL,
        total_hours REAL NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT DEFAULT 'unpaid',
        FOREIGN KEY (client_id) REFERENCES clients (id)
    )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print(f"Initialized database at {DB_PATH}")
