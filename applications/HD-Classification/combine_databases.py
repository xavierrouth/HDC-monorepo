#!/usr/bin/env python3
"""
Combine sweep_results1.db and sweep_results2.db into a new combined database.
"""

import sqlite3
import os

RESULTS_DIR = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Classification/results"
DB1_PATH = os.path.join(RESULTS_DIR, "sweep_results1.db")
DB2_PATH = os.path.join(RESULTS_DIR, "sweep_results2.db")
COMBINED_PATH = os.path.join(RESULTS_DIR, "sweep_results_combined.db")


def combine_databases():
    """Combine two sweep results databases into one."""
    
    # Check if input databases exist
    if not os.path.exists(DB1_PATH):
        print(f"Error: {DB1_PATH} not found")
        return
    
    if not os.path.exists(DB2_PATH):
        print(f"Error: {DB2_PATH} not found")
        return
    
    print(f"Combining databases:")
    print(f"  DB1: {DB1_PATH}")
    print(f"  DB2: {DB2_PATH}")
    print(f"  Output: {COMBINED_PATH}\n")
    
    # Remove existing combined database if it exists
    if os.path.exists(COMBINED_PATH):
        print(f"Removing existing combined database...")
        os.remove(COMBINED_PATH)
    
    # Create new combined database
    conn_combined = sqlite3.connect(COMBINED_PATH)
    cursor_combined = conn_combined.cursor()
    
    # Create table structure (copy from DB1)
    conn1 = sqlite3.connect(DB1_PATH)
    cursor1 = conn1.cursor()
    
    # Get table schema
    cursor1.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sweep_results'")
    create_table_sql = cursor1.fetchone()[0]
    
    # Create table in combined database
    cursor_combined.execute(create_table_sql)
    print(f"Created table structure in combined database\n")
    
    # Copy data from DB1
    cursor1.execute("SELECT * FROM sweep_results")
    db1_rows = cursor1.fetchall()
    
    # Get column names (excluding id which is auto-increment)
    cursor1.execute("PRAGMA table_info(sweep_results)")
    columns = [col[1] for col in cursor1.fetchall()]
    id_index = columns.index('id')
    columns_no_id = [col for i, col in enumerate(columns) if i != id_index]
    
    placeholders = ','.join(['?' for _ in columns_no_id])
    insert_sql = f"INSERT INTO sweep_results ({','.join(columns_no_id)}) VALUES ({placeholders})"
    
    for row in db1_rows:
        # Exclude the id column
        row_no_id = [val for i, val in enumerate(row) if i != id_index]
        cursor_combined.execute(insert_sql, row_no_id)
    
    conn1.close()
    
    print(f"Copied {len(db1_rows)} rows from {os.path.basename(DB1_PATH)}")
    
    # Copy data from DB2
    conn2 = sqlite3.connect(DB2_PATH)
    cursor2 = conn2.cursor()
    
    cursor2.execute("SELECT * FROM sweep_results")
    db2_rows = cursor2.fetchall()
    
    for row in db2_rows:
        # Exclude the id column
        row_no_id = [val for i, val in enumerate(row) if i != id_index]
        cursor_combined.execute(insert_sql, row_no_id)
    
    conn2.close()
    
    print(f"Copied {len(db2_rows)} rows from {os.path.basename(DB2_PATH)}")
    
    # Commit and close
    conn_combined.commit()
    
    # Get final count
    cursor_combined.execute("SELECT COUNT(*) FROM sweep_results")
    total_count = cursor_combined.fetchone()[0]
    
    # Get success count
    cursor_combined.execute("SELECT COUNT(*) FROM sweep_results WHERE success = 1")
    success_count = cursor_combined.fetchone()[0]
    
    conn_combined.close()
    
    print(f"\n{'='*80}")
    print(f"COMBINATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total rows in combined database: {total_count}")
    print(f"Successful experiments: {success_count}")
    print(f"Failed experiments: {total_count - success_count}")
    print(f"\nCombined database saved to: {COMBINED_PATH}")
    
    # Show some statistics
    conn_combined = sqlite3.connect(COMBINED_PATH)
    cursor_combined = conn_combined.cursor()
    
    cursor_combined.execute("""
        SELECT encoding_dim, COUNT(*) as count, 
               AVG(CASE WHEN success=1 THEN accuracy END) as avg_accuracy
        FROM sweep_results
        GROUP BY encoding_dim
        ORDER BY encoding_dim
    """)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY BY ENCODING DIMENSION")
    print(f"{'='*80}")
    for row in cursor_combined.fetchall():
        enc_dim, count, avg_acc = row
        if avg_acc is not None:
            print(f"  Encoding Dim {enc_dim}: {count} experiments, avg accuracy: {avg_acc:.4f}")
        else:
            print(f"  Encoding Dim {enc_dim}: {count} experiments (all failed)")
    
    conn_combined.close()


if __name__ == '__main__':
    combine_databases()
