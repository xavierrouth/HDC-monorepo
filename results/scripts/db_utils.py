"""
Shared OpenTuner DB helpers for HDC eval scripts.

Covers:
  - Sidecar readers (.device, .searchspace)
  - Config deserialization (zlib+pickle blob)
  - DB lookup by prefix/suffix/device or prefix/dataset/suffix/device
  - Seed time, valid-time array, top-k config extraction
"""

import ast
import glob
import json
import os
import pickle
import sqlite3
import zlib

import numpy as np

_DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dbs")


# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------

def db_device(db_path: str) -> str:
    """Read .device sidecar; returns 'gpu' if absent."""
    sidecar = db_path + ".device"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return f.read().strip()
    return "gpu"


def search_space_size(db_path: str) -> str | None:
    """Read .searchspace sidecar; returns None if absent."""
    sidecar = db_path + ".searchspace"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return f.read().strip()
    return None


# ---------------------------------------------------------------------------
# Config deserialization
# ---------------------------------------------------------------------------

def decode_config(data) -> dict | None:
    """Deserialize an OpenTuner configuration.data blob (zlib+pickle or plain pickle)."""
    if data is None:
        return None
    try:
        if isinstance(data, (bytes, bytearray)):
            try:
                return pickle.loads(zlib.decompress(data))
            except Exception:
                return pickle.loads(data)
        if isinstance(data, str):
            try:
                return json.loads(data)
            except Exception:
                return ast.literal_eval(data)
        if isinstance(data, dict):
            return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DB lookup — eval_redperf style (prefix + suffix, no dataset tag)
# ---------------------------------------------------------------------------

def best_db(db_prefix: str, db_suffix: str, device: str = "gpu",
            db_dir: str = _DEFAULT_DB_DIR) -> str | None:
    """Return the most recent DB for prefix+suffix+device with >0 rows, or None."""
    candidates = sorted(
        glob.glob(os.path.join(db_dir, f"{db_prefix}_{db_suffix}_[0-9]*.db")),
        reverse=True,
    )
    for db_path in candidates:
        if db_device(db_path) != device:
            continue
        try:
            conn = sqlite3.connect(db_path)
            n = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            conn.close()
            if n > 0:
                return db_path
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# DB lookup — eval_transfer style (prefix + dataset tag + suffix)
# ---------------------------------------------------------------------------

def db_glob(db_prefix: str, dataset: str, db_suffix: str,
            strict: bool = False,
            db_dir: str = _DEFAULT_DB_DIR) -> list[str]:
    """Return DB paths matching prefix/dataset/suffix, sorted newest-first.

    strict=False (default): also include legacy DBs without a dataset tag
    (for ISOLET runs predating the dataset-in-filename convention).
    strict=True: only return DBs whose filename contains the dataset name.
    """
    pat = os.path.join(db_dir, f"{db_prefix}_{dataset.lower()}_*{db_suffix}*.db")
    found = sorted(glob.glob(pat), reverse=True)
    if not strict:
        pat2 = os.path.join(db_dir, f"{db_prefix}_{db_suffix}*.db")
        found += sorted(glob.glob(pat2), reverse=True)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def find_best_transfer_db(db_prefix: str, dataset: str,
                          db_suffix: str = "redperf_endonly",
                          device: str | None = None,
                          db_dir: str = _DEFAULT_DB_DIR) -> str | None:
    """Return the most recent DB for (prefix, dataset, suffix) with >0 rows.

    Tries device-matching DBs first; falls back to any device if none found.
    """
    candidates = db_glob(db_prefix, dataset, db_suffix, db_dir=db_dir)

    def _device_ok(p: str) -> bool:
        if device is None:
            return True
        sidecar = p + ".device"
        if not os.path.exists(sidecar):
            return True
        return open(sidecar).read().strip().lower() == device.lower()

    def _has_rows(p: str) -> bool:
        try:
            conn = sqlite3.connect(p)
            n = conn.execute("SELECT COUNT(*) FROM result WHERE time < 1e30").fetchone()[0]
            conn.close()
            return n > 0
        except Exception:
            return False

    # Prefer device-matching
    if device:
        for p in candidates:
            if _device_ok(p) and _has_rows(p):
                return p
    # Fallback: any device
    for p in candidates:
        if _has_rows(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Result queries
# ---------------------------------------------------------------------------

# Every reported number is taken from the first TRUNCATE_HOURS of a run, so that
# series recorded with a longer budget (HD-Clustering at 5h, HD-HashTable GPU at
# 5h, HD-Classification GPU's ON series at 4.9h) are comparable with the 3h ones
# and consistent with the "3-hour tuning budget" the figures claim. Set to None
# to use each database in full.
TRUNCATE_HOURS: float | None = 3.0


def _cutoff(conn) -> str | None:
    """Timestamp TRUNCATE_HOURS after a run's first result, in the DB's own format."""
    if TRUNCATE_HOURS is None:
        return None
    row = conn.execute(
        "SELECT MIN(collection_date) FROM result WHERE collection_date IS NOT NULL"
    ).fetchone()
    if not row or not row[0]:
        return None
    import datetime
    start = datetime.datetime.fromisoformat(row[0])
    end = start + datetime.timedelta(hours=TRUNCATE_HOURS)
    # Match the stored format exactly: a space separator, not the 'T' that
    # isoformat() emits -- string comparison against 'T' silently matches
    # everything, because ' ' sorts below 'T'.
    return end.strftime("%Y-%m-%d %H:%M:%S.%f")


def seed_time(db_path: str) -> float | None:
    """Return the execution time of the first evaluated (seed) config."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT time FROM result WHERE time < 1e30 ORDER BY collection_date LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else None


def load_times(db_path: str, accuracy_cutoff: float) -> np.ndarray:
    """Return array of times for valid configs (accuracy >= cutoff, time < 1e30)."""
    conn = sqlite3.connect(db_path)
    cut = _cutoff(conn)
    if cut is None:
        rows = conn.execute(
            "SELECT time FROM result WHERE accuracy >= ? AND time < 1e30",
            (accuracy_cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT time FROM result WHERE accuracy >= ? AND time < 1e30 "
            "AND (collection_date IS NULL OR collection_date <= ?)",
            (accuracy_cutoff, cut),
        ).fetchall()
    conn.close()
    return np.array([r[0] for r in rows], dtype=float)


def extract_top_configs(db_path: str, accuracy_target: float,
                        k: int = 1) -> list[dict]:
    """Return the top-k fastest valid configs from the DB, decoded, fastest-first."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT c.data
            FROM result r
            JOIN desired_result dr ON r.id = dr.result_id
            JOIN configuration c  ON dr.configuration_id = c.id
            WHERE r.accuracy >= ? AND r.time < 1e30
              AND (r.collection_date IS NULL OR r.collection_date <= ?)
            ORDER BY r.time ASC
            LIMIT ?
            """,
            (accuracy_target, _cutoff(conn) or "9999-12-31 23:59:59.999999", k),
        ).fetchall()
    finally:
        conn.close()
    return [cfg for row in rows if (cfg := decode_config(row[0])) is not None]
