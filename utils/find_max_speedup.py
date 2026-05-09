import argparse
import sqlite3
import pandas as pd
import numpy as np
import sys
import pickle
import pprint
import zlib

def has_column(conn, table, column):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    return column in cols

def _load_config(db_path, config_id):
    """Load and return unpickled configuration for given Configuration.id, or None."""
    if config_id is None:
        return None
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT data FROM Configuration WHERE id = ?", (int(config_id),))
        row = cur.fetchone()
    finally:
        if conn is not None:
            conn.close()
    if not row or row[0] is None:
        return None
    data = row[0]
    # Try to decompress; if it's not compressed, proceed with original bytes
    try:
        data = zlib.decompress(data)
    except Exception:
        pass
    try:
        return pickle.loads(data)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Find max speedup subject to accuracy cutoff using first run as baseline.")
    parser.add_argument("tuning_run_id", type=int, help="Tuning run ID to analyze")
    parser.add_argument("accuracy_cutoff", type=float, help="Accuracy cutoff")
    parser.add_argument("--db", default="/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db",
                        help="Path to SQLite DB")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        include_energy = has_column(conn, "result", "energy")
        include_cid = has_column(conn, "result", "configuration_id")
        cols = ["collection_date", "accuracy", "time"] \
             + (["energy"] if include_energy else []) \
             + (["configuration_id"] if include_cid else [])
        col_str = ", ".join(cols)
        query = f"""
            SELECT {col_str}
            FROM result
            WHERE tuning_run_id = ?
            ORDER BY collection_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(args.tuning_run_id,))
    finally:
        conn.close()

    if df.empty:
        print("No data for the specified tuning_run_id.")
        sys.exit(1)

    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    df = df.dropna(subset=["collection_date", "time", "accuracy"])
    if df.empty:
        print("No valid rows after cleaning data.")
        sys.exit(1)

    baseline_row = df.iloc[0]
    baseline_time = float(baseline_row["time"])
    baseline_accuracy = float(baseline_row["accuracy"])
    baseline_energy = float(baseline_row["energy"]) if include_energy and pd.notna(baseline_row.get("energy")) else None
    baseline_cid = int(baseline_row["configuration_id"]) if include_cid and pd.notna(baseline_row.get("configuration_id")) else None

    if baseline_time <= 0:
        print("Baseline time is non-positive; cannot compute speedup.")
        sys.exit(1)

    df = df.copy()
    df = df[df["time"] > 0]
    if df.empty:
        print("No rows with positive time.")
        sys.exit(1)

    df["speedup"] = baseline_time / df["time"]
    df = df[df["accuracy"] >= args.accuracy_cutoff]
    if df.empty:
        print("No points meet the accuracy cutoff.")
        sys.exit(1)

    idx = df["speedup"].idxmax()
    best = df.loc[idx]
    best_cid = int(best["configuration_id"]) if include_cid and pd.notna(best.get("configuration_id")) else None

    print("Baseline (first run):")
    print(f"  collection_date: {baseline_row['collection_date']}")
    print(f"  time: {baseline_time:.6f}")
    print(f"  accuracy: {baseline_accuracy:.6f}")
    if include_energy:
        print(f"  energy: {baseline_energy:.6f}" if baseline_energy is not None else "  energy: N/A")
    if include_cid:
        print(f"  configuration_id: {baseline_cid}")
        bcfg = _load_config(args.db, baseline_cid)
        print("  configuration:")
        pprint.pprint(bcfg if bcfg is not None else "N/A")

    print("\nBest point under accuracy cutoff:")
    print(f"  collection_date: {best['collection_date']}")
    print(f"  time: {float(best['time']):.6f}")
    print(f"  accuracy: {float(best['accuracy']):.6f}")
    print(f"  speedup: {float(best['speedup']):.6f}x")
    if include_energy:
        energy_val = float(best["energy"]) if pd.notna(best.get("energy")) else None
        print(f"  energy: {energy_val:.6f}" if energy_val is not None else "  energy: N/A")
    if include_cid:
        print(f"  configuration_id: {best_cid}")
        mcfg = _load_config(args.db, best_cid)
        print("  configuration:")
        pprint.pprint(mcfg if mcfg is not None else "N/A")

if __name__ == "__main__":
    main()
