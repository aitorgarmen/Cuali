# pgdb.py
"""Simple PostgreSQL accessor for Batch operations.

Connection details may be provided via a full DSN string parameter or the
QUALIF_PG_DSN environment variable. Alternatively, individual settings can be
specified using QUALIF_PG_HOST, QUALIF_PG_PORT, QUALIF_PG_NAME, QUALIF_PG_USER,
and QUALIF_PG_PASSWORD. When no DSN is available (or psycopg2 is missing) a
mock in-memory backend is used. This mirrors the behaviour of DatabaseAccessor
used in other tabs.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
try:
    from sshtunnel import SSHTunnelForwarder  # type: ignore
except ImportError:
    SSHTunnelForwarder = None

import hashlib
import struct

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - psycopg2 might not be available
    psycopg2 = None  # type: ignore
    
def md5_password(password: str, user: str) -> str:
    """Generate MD5-prefixed PostgreSQL password hash for user."""
    m = hashlib.md5()
    m.update(password.encode() + user.encode())
    return "md5" + m.hexdigest()


class BatchDB:
    def __init__(self,
                 dsn: Optional[str] = None,
                 host: Optional[str] = None,
                 port: Optional[str] = None,
                 dbname: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 ssh_host: Optional[str] = None,
                 ssh_port: Optional[int] = None,
                 ssh_user: Optional[str] = None,
                 ssh_password: Optional[str] = None,
                 ssh_pkey: Optional[str] = None) -> None:
        # Determine DSN from parameter or environment
        if dsn is None:
            dsn = os.getenv("QUALIF_PG_DSN")
        # Normalize DSN: treat literal 'None' or empty as no DSN
        if isinstance(dsn, str) and dsn.strip().lower() in ("none", ""):
            dsn = None
        # Check required driver for real connection
        if (dsn is not None) and psycopg2 is None:
            raise RuntimeError("psycopg2 no está instalado. Ejecuta: pip install psycopg2-binary")
        # If no DSN, attempt using provided or env DB/SSH params only if DB host/dbname set
        if dsn is None:
            host = host or os.getenv("QUALIF_PG_HOST")
            port = port or os.getenv("QUALIF_PG_PORT")
            dbname = dbname or os.getenv("QUALIF_PG_NAME")
            # Treat literal 'None' strings as missing
            if isinstance(host, str) and host.strip().lower() == 'none':
                host = None
            if isinstance(dbname, str) and dbname.strip().lower() == 'none':
                dbname = None
            user = user or os.getenv("QUALIF_PG_USER")
            password = password or os.getenv("QUALIF_PG_PASSWORD")
            ssh_host = ssh_host or os.getenv("QUALIF_PG_SSH_HOST")
            ssh_port = ssh_port or (os.getenv("QUALIF_PG_SSH_PORT") and int(os.getenv("QUALIF_PG_SSH_PORT")))
            ssh_user = ssh_user or os.getenv("QUALIF_PG_SSH_USER")
            ssh_password = ssh_password or os.getenv("QUALIF_PG_SSH_PASSWORD")
            ssh_pkey = ssh_pkey or os.getenv("QUALIF_PG_SSH_PKEY")
            # Only build tunnel and DSN if we have valid DB host and dbname
            if host and dbname and str(host).strip().lower() != "none" and str(dbname).strip().lower() != "none":
                # Ensure SSHTunnelForwarder is available for SSH tunnel
                if ssh_host and SSHTunnelForwarder is None:
                    raise RuntimeError("sshtunnel no está instalado. Ejecuta: pip install sshtunnel")
                # Establish SSH tunnel if configured
                if ssh_host:
                    try:
                        tunnel = SSHTunnelForwarder(
                            (ssh_host, ssh_port or 22),
                            ssh_username=ssh_user,
                            ssh_password=ssh_password,
                            ssh_pkey=ssh_pkey,
                            remote_bind_address=(host, int(port)),
                            local_bind_address=("127.0.0.1",)
                        )
                        tunnel.start()
                        # SSHTunnelForwarder by default binds to 0.0.0.0; use localhost for DB connection
                        bind_port = tunnel.local_bind_port
                        host = '127.0.0.1'
                        port = str(bind_port)
                        self._tunnel = tunnel
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"Error al crear túnel SSH: {e}")
                # Build DSN
                parts = [f"host={host}", f"dbname={dbname}"]
                if port:
                    parts.append(f"port={port}")
                if user:
                    parts.append(f"user={user}")
                if password:
                    parts.append(f"password={password}")
                dsn = " ".join(parts)
        # sanitize DSN: if it contains invalid None values, treat as no DSN
        if isinstance(dsn, str) and ("host=None" in dsn or "dbname=None" in dsn):
            dsn = None
        # determine if using mock or real backend
        self._mock = dsn is None or psycopg2 is None
        # track actual connection status
        self._connected = False
        if self._mock:
            self._batches = {}
            self._customers: set[str] = set()
        else:
            try:
                self.conn = psycopg2.connect(dsn)
                self.conn.autocommit = True
                self.cur = self.conn.cursor()
                self._connected = True
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Error al conectar PostgreSQL: {e}")
                # fallback to mock on connection error
                self._mock = True
                self._batches = {}
                self._connected = False

    # ------------------------------------------------------------------
    def list_batches(self) -> List[str]:
        if self._mock:
            return sorted(self._batches.keys())
        self.cur.execute("SELECT batch_id FROM batch ORDER BY batch_id")
        return [row[0] for row in self.cur.fetchall()]
    # Alias for backward compatibility with DatabaseAccessor
    def batch_numbers(self) -> List[str]:
        """Alias to list_batches to provide batch_numbers interface."""
        return self.list_batches()

    # ------------------------------------------------------------------
    def list_customers(self) -> List[str]:
        """Return the list of existing customers."""
        if self._mock:
            return sorted(self._customers)
        self.cur.execute("SELECT name FROM customer ORDER BY name")
        return [row[0] for row in self.cur.fetchall()]
    # ------------------------------------------------------------------
    def add_pre_measurement(self, batch_id: str, bolt_id: str, frame: Dict[str, Any]) -> None:
        """Insert a single pre_measurement row into the database."""
        if self._mock:
            # Skip persistence in mock
            return
        # Prepare fields
        sql = (
            "INSERT INTO pre_measurement "
            "(batch_id, bolt_id, freq, gain, pulse, pico1, pct_diff, tof, temp, force, maxcorrx, maxcorry, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        # Convert dat2 and dat3 to bytes
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            frame.get('freq'),
            frame.get('gain'),
            frame.get('pulse'),
            frame.get('pico1'),
            frame.get('porcentaje_diferencia'),
            frame.get('tof'),
            frame.get('temp'),
            frame.get('force'),
            frame.get('maxcorrx'),
            frame.get('maxcorry'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def add_one4_measurement(self, batch_id: str, bolt_id: str, frame: Dict[str, Any]) -> None:
        """Insert a single one4_measurement row into the database."""
        if self._mock:
            # Skip persistence in mock
            return
        sql = (
            "INSERT INTO one4_measurement "
            "(batch_id, bolt_id, freq, gain, pulse, pico1, pct_diff, tof, temp, force, "
            "maxcorrx, maxcorry, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            frame.get('freq'),
            frame.get('gain'),
            frame.get('pulse'),
            frame.get('pico1'),
            frame.get('porcentaje_diferencia'),
            frame.get('tof'),
            frame.get('temp'),
            frame.get('force'),
            frame.get('maxcorrx'),
            frame.get('maxcorry'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def add_one4_loading(self, batch_id: str, bolt_id: str, seq: int, frame: Dict[str, Any]) -> None:
        """Insert a single one4_loading row into the database."""
        if self._mock:
            # Skip persistence in mock
            return
        sql = (
            "INSERT INTO one4_loading "
            "(batch_id, bolt_id, seq, tof, force, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            seq,
            frame.get('tof'),
            frame.get('force'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def add_one10_loading(self, batch_id: str, bolt_id: str, seq: int, frame: Dict[str, Any]) -> None:
        """Insert a single one10_loading row into the database."""
        if self._mock:
            # Skip persistence in mock
            return
        sql = (
            "INSERT INTO one10_loading "
            "(batch_id, bolt_id, seq, tof, force, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            seq,
            frame.get('tof'),
            frame.get('force'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def add_one10_measurement(self, batch_id: str, bolt_id: str, stage: str, frame: Dict[str, Any]) -> None:
        """Insert a single one10_measurement row into the database."""
        if self._mock:
            # Skip persistence in mock
            return
        sql = (
            "INSERT INTO one10_measurement "
            "(batch_id, bolt_id, stage, freq, gain, pulse, pico1, pct_diff, tof, temp, force_load_cell, "
            "maxcorrx, maxcorry, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            stage,
            frame.get('freq'),
            frame.get('gain'),
            frame.get('pulse'),
            frame.get('pico1'),
            frame.get('porcentaje_diferencia'),
            frame.get('tof'),
            frame.get('temp'),
            frame.get('force'),
            frame.get('maxcorrx'),
            frame.get('maxcorry'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def add_bending_measurement(
        self, batch_id: str, bolt_id: str, position: int, stage: str, frame: Dict[str, Any]
    ) -> None:
        """Insert a bending_measurement row into the database."""
        if self._mock:
            return
        sql = (
            "INSERT INTO bending_measurement "
            "(batch_id, bolt_id, position, tof, force, force_load_cell, stage, freq, gain, pulse, "
            "pico1, pct_diff, temp, maxcorrx, maxcorry, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            position,
            frame.get('tof'),
            frame.get('force'),
            frame.get('force_load_cell'),
            stage,
            frame.get('freq'),
            frame.get('gain'),
            frame.get('pulse'),
            frame.get('pico1'),
            frame.get('porcentaje_diferencia'),
            frame.get('temp'),
            frame.get('maxcorrx'),
            frame.get('maxcorry'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def list_batches_with_one10_process(self) -> List[str]:
        """Return batches having at least one bolt with both initial and final
        stages present in ``one10_measurement``.

        In mock mode returns an empty list.
        """
        if self._mock:
            return []
        sql = (
            "SELECT DISTINCT batch_id FROM ("
            "  SELECT batch_id, bolt_id, COUNT(DISTINCT stage) AS nstages"
            "  FROM one10_measurement"
            "  GROUP BY batch_id, bolt_id"
            ") t WHERE nstages >= 2 ORDER BY batch_id"
        )
        try:
            self.cur.execute(sql)
            return [row[0] for row in self.cur.fetchall()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def fetch_one10_initial_final(self, batch_id: str, bolt_ids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Fetch latest initial/final one10 measurements per bolt.

        Returns mapping: {bolt_id: {"initial": row|None, "final": row|None}}
        where row contains keys: stage, tof, force_load_cell, measured_at.
        """
        if self._mock:
            return {}
        params: List[Any] = [str(batch_id)]
        cond = "batch_id=%s"
        if bolt_ids:
            ids = list({str(b) for b in bolt_ids})
            placeholders = ",".join(["%s"] * len(ids))
            cond += f" AND bolt_id IN ({placeholders})"
            params.extend(ids)
        sql = (
            f"SELECT bolt_id, stage, tof, force_load_cell, measured_at "
            f"FROM one10_measurement WHERE {cond} ORDER BY measured_at DESC"
        )
        self.cur.execute(sql, tuple(params))
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for bolt_id, stage, tof, flc, ts in self.cur.fetchall() or []:
            d = out.setdefault(str(bolt_id), {"initial": None, "final": None})
            if stage in ("initial", "final") and d.get(stage) is None:
                d[stage] = {
                    "stage": stage,
                    "tof": tof,
                    "force_load_cell": flc,
                    "measured_at": ts,
                }
        return out

    # ------------------------------------------------------------------
    def list_completed_batches(self) -> List[str]:
        """Return batch_ids marked as completed (is_completed=true)."""
        if self._mock:
            out = []
            for bid, data in (self._batches or {}).items():
                if data.get("attrs", {}).get("is_completed"):
                    out.append(bid)
            return sorted(out)
        # ensure column exists
        try:
            self.cur.execute(
                "ALTER TABLE batch ADD COLUMN IF NOT EXISTS is_completed BOOLEAN NOT NULL DEFAULT FALSE"
            )
        except Exception:
            pass
        try:
            self.cur.execute(
                "SELECT batch_id FROM batch WHERE is_completed ORDER BY batch_id"
            )
            return [r[0] for r in self.cur.fetchall() or []]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def fetch_temp_loading_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        """Fetch temp_loading rows for a batch ordered by measured_at desc.

        Returns list of dicts with keys: bolt_id, seq, tof, temp, dat2, dat3, measured_at.
        """
        if self._mock:
            return []
        bid = str(batch_id)
        # Prefer measured_at order, fallback to seq when timestamp missing
        try:
            self.cur.execute(
                "SELECT bolt_id, seq, tof, temp, dat2, dat3, measured_at "
                "FROM temp_loading WHERE batch_id=%s ORDER BY measured_at DESC",
                (bid,),
            )
        except Exception:
            self.cur.execute(
                "SELECT bolt_id, seq, tof, temp, dat2, dat3, NULL as measured_at "
                "FROM temp_loading WHERE batch_id=%s ORDER BY seq DESC",
                (bid,),
            )
        rows = self.cur.fetchall() or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "bolt_id": r[0],
                    "seq": r[1],
                    "tof": r[2],
                    "temp": r[3],
                    "dat2": r[4],
                    "dat3": r[5],
                    "measured_at": r[6],
                }
            )
        return out

    # ------------------------------------------------------------------
    def update_xi(self, batch_id: str, xi: float) -> None:
        """Update only the xi field for a batch."""
        if self._mock:
            data = self._batches.setdefault(batch_id, {"attrs": {}, "bolts": []})
            attrs = data.setdefault("attrs", {})
            attrs["xi"] = str(xi)
            return
        self.cur.execute("UPDATE batch SET xi=%s WHERE batch_id=%s", (float(xi), str(batch_id)))

    # ------------------------------------------------------------------
    def add_bending_loading(self, batch_id: str, bolt_id: str, position: int, seq: int, frame: Dict[str, Any]) -> None:
        """Insert a single bending_loading row into the database."""
        if self._mock:
            return
        sql = (
            "INSERT INTO bending_loading "
            "(batch_id, bolt_id, position, seq, tof, force, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            int(position),
            int(seq),
            frame.get('tof'),
            frame.get('force'),
            psycopg2.Binary(dat2_bytes) if dat2_bytes is not None else None,
            psycopg2.Binary(dat3_bytes) if dat3_bytes is not None else None,
        )
        self.cur.execute(sql, params)

    def add_temp_loading(self, batch_id: str, bolt_id: str, seq: int, frame: Dict[str, Any]) -> None:
        """Insert a temp_loading row into the database."""
        if self._mock:
            return
        sql = (
            "INSERT INTO temp_loading "
            "(batch_id, bolt_id, seq, tof, temp, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        params = (
            batch_id,
            bolt_id,
            seq,
            frame.get('tof'),
            frame.get('temp'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    def add_temp_measurement(self, batch_id: str, bolt_id: str, frame: Dict[str, Any]) -> None:
        """Insert a temp_measurement row into the database."""
        if self._mock:
            return
        sql = (
            "INSERT INTO temp_measurement "
            "(batch_id, bolt_id, freq, gain, pulse, pico1, pct_diff, tof, temp, maxcorrx, maxcorry, dat2, dat3) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        dat2 = frame.get('dat2')
        dat3 = frame.get('dat3')
        dat2_bytes = memoryview(dat2).tobytes() if hasattr(dat2, 'tobytes') else dat2
        dat3_bytes = memoryview(dat3).tobytes() if hasattr(dat3, 'tobytes') else dat3
        pct_diff = frame.get('pct_diff')
        if pct_diff is None:
            pct_diff = frame.get('porcentaje_diferencia')
        params = (
            batch_id,
            bolt_id,
            frame.get('freq'),
            frame.get('gain'),
            frame.get('pulse'),
            frame.get('pico1'),
            pct_diff,
            frame.get('tof'),
            frame.get('temp'),
            frame.get('maxcorrx'),
            frame.get('maxcorry'),
            psycopg2.Binary(dat2_bytes),
            psycopg2.Binary(dat3_bytes),
        )
        self.cur.execute(sql, params)

    def add_temp_tof_loading(self, batch_id: str, bolt_id: str, seq: int, *,
                              setpoint: Optional[float], oven_temp: Optional[float],
                              tof: Optional[float], freq: Optional[float],
                              gain: Optional[float], pulse: Optional[float]) -> None:
        """Insert a temp_tof_loading row (stabilization time series).

        Stores minimal fields needed to reproduce the stabilization graph.
        Does not store dat2/dat3 by design.
        """
        if self._mock:
            return
        sql = (
            "INSERT INTO temp_tof_loading "
            "(batch_id, bolt_id, seq, setpoint, oven_temp, tof, freq, gain, pulse) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        params = (
            batch_id,
            bolt_id,
            int(seq) if seq is not None else None,
            setpoint,
            oven_temp,
            tof,
            int(freq) if freq is not None else None,
            int(gain) if gain is not None else None,
            int(pulse) if pulse is not None else None,
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def fetch_temp_measurements(
        self,
        batch_id: str,
        bolt_id: str,
        freq: Optional[int] = None,
        gain: Optional[int] = None,
        pulse: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch temp_measurement rows for a bolt, optionally filtered by combo.

        Returns latest rows first. Each row is a dict with keys:
        freq, gain, pulse, tof, temp, measured_at.
        """
        if self._mock:
            return []
        conds = ["batch_id=%s", "bolt_id=%s"]
        params: List[Any] = [str(batch_id), str(bolt_id)]
        if freq is not None:
            conds.append("freq=%s")
            params.append(int(freq))
        if gain is not None:
            conds.append("gain=%s")
            params.append(int(gain))
        if pulse is not None:
            conds.append("pulse=%s")
            params.append(int(pulse))
        where = " AND ".join(conds)
        sql = (
            f"SELECT freq, gain, pulse, tof, temp, measured_at "
            f"FROM temp_measurement WHERE {where} ORDER BY measured_at DESC"
        )
        self.cur.execute(sql, tuple(params))
        rows = self.cur.fetchall() or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "freq": r[0],
                    "gain": r[1],
                    "pulse": r[2],
                    "tof": r[3],
                    "temp": r[4],
                    "measured_at": r[5],
                }
            )
        return out
    # ------------------------------------------------------------------
    def add_baseline(
        self,
        batch_id: str,
        bolt_id: str,
        pallet_num: int,
        frame: Dict[str, Any],
        is_valid: Optional[bool] = None,
        measured_at: Optional[Any] = None,
    ) -> None:
        """Insert a single baseline row into the database."""
        if self._mock:
            return
        # ensure bolt exists to satisfy FK constraint
        self.cur.execute(
            "INSERT INTO bolt (batch_id, bolt_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (batch_id, bolt_id),
        )
        sql = (
            "INSERT INTO baseline "
            "(batch_id, bolt_id, pallet_num, is_valid, freq, gain, pulse, pico1, pct_diff, "
            "tof, temp, force, maxcorrx, maxcorry, dat2, dat3, measured_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        # Convert possible numpy arrays or lists into raw bytes
        def _to_bytes(data: Any, fmt: str) -> bytes | None:
            if data is None:
                return None
            if isinstance(data, (bytes, bytearray, memoryview)):
                return bytes(data)
            if hasattr(data, "tobytes"):
                try:
                    return data.tobytes()
                except Exception:
                    pass
            try:
                return struct.pack("<" + str(len(data)) + fmt, *data)
            except Exception:
                return None

        dat2_bytes = _to_bytes(frame.get("dat2"), "h")
        dat3_bytes = _to_bytes(frame.get("dat3"), "i")
        params = (
            batch_id,
            bolt_id,
            pallet_num,
            is_valid,
            frame.get('freq'),
            frame.get('gain'),
            frame.get('pulse'),
            frame.get('pico1'),
            frame.get('porcentaje_diferencia'),
            frame.get('tof'),
            frame.get('temp'),
            frame.get('force'),
            frame.get('maxcorrx'),
            frame.get('maxcorry'),
            psycopg2.Binary(dat2_bytes) if dat2_bytes is not None else None,
            psycopg2.Binary(dat3_bytes) if dat3_bytes is not None else None,
            measured_at or frame.get("measured_at"),
        )
        self.cur.execute(sql, params)

    # ------------------------------------------------------------------
    def latest_valid_baselines(self, batch_id: str) -> list[Dict[str, Any]]:
        """Return latest valid baseline row per bolt for the given batch."""
        if self._mock:
            return []
        self.cur.execute(
            """
            SELECT b.measured_at, b.bolt_id, b.tof, b.temp, b.dat2, b.dat3
            FROM baseline b
            JOIN (
                SELECT bolt_id, MAX(measured_at) AS m
                FROM baseline
                WHERE batch_id=%s AND is_valid
                GROUP BY bolt_id
            ) latest
            ON b.bolt_id = latest.bolt_id AND b.measured_at = latest.m AND b.batch_id = %s
            WHERE b.is_valid
            ORDER BY b.bolt_id
            """,
            (batch_id, batch_id),
        )
        rows = self.cur.fetchall() or []
        return [
            {
                "measured_at": r[0],
                "bolt_id": r[1],
                "tof": r[2],
                "temp": r[3],
                "dat2": r[4],
                "dat3": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    def baseline_export_rows(self, batch_id: str) -> list[Dict[str, Any]]:
        """Return per-bolt baseline info for export.

        Each item contains the bolt id, timestamp of the first valid
        measurement, its ToF and temperature, and the timestamp of the last
        valid measurement.
        """
        if self._mock:
            return []
        self.cur.execute(
            """
            SELECT bolt_id,
                   MIN(measured_at) AS first_ts,
                   (ARRAY_AGG(tof ORDER BY measured_at))[1]  AS first_tof,
                   (ARRAY_AGG(temp ORDER BY measured_at))[1] AS first_temp,
                   MAX(measured_at) AS last_ts
            FROM baseline
            WHERE batch_id=%s AND is_valid
            GROUP BY bolt_id
            ORDER BY bolt_id
            """,
            (batch_id,),
        )
        rows = self.cur.fetchall() or []
        return [
            {
                "bolt_id": r[0],
                "first_ts": r[1],
                "tof": r[2],
                "temp": r[3],
                "last_ts": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    def get_pre_tof(
        self, batch_id: str, bolt_id: str, freq: int, gain: int, pulse: int
    ) -> Optional[float]:
        """Fetch most recent ToF from ``pre_measurement`` for given combo.

        Returns ``None`` when not found or when using the mock backend."""
        if self._mock:
            return None
        self.cur.execute(
            (
                "SELECT tof FROM pre_measurement "
                "WHERE batch_id=%s AND bolt_id=%s AND freq=%s AND gain=%s AND pulse=%s "
                "ORDER BY measured_at DESC LIMIT 1"
            ),
            (batch_id, bolt_id, freq, gain, pulse),
        )
        row = self.cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None

    # ------------------------------------------------------------------
    def get_baseline_params(self, batch_id: str) -> tuple[int, int, int]:
        """Return (frequency, gain, cycles_coarse) for a batch.
        Raises ValueError if any parameter is missing."""
        if self._mock:
            attrs = self._batches.get(batch_id, {}).get("attrs", {})
            freq = attrs.get("frequency")
            gain = attrs.get("gain")
            pulse = attrs.get("cycles_coarse")
            if None in (freq, gain, pulse):
                raise ValueError("Parámetros no definidos para el batch")
            return int(freq), int(gain), int(pulse)
        self.cur.execute(
            "SELECT frequency, gain, cycles_coarse FROM batch WHERE batch_id=%s",
            (batch_id,),
        )
        row = self.cur.fetchone()
        if not row or None in row:
            raise ValueError("Parámetros no definidos para el batch")
        return int(row[0]), int(row[1]), int(row[2])

    def get_batch_params(self, batch_id: str) -> Dict[str, Any]:
        """Return frequency, gain, pulse and length/ToF for a batch."""
        if self._mock:
            attrs = self._batches.get(batch_id, {}).get("attrs", {})
            freq = attrs.get("frequency", 0.0)
            gain = attrs.get("gain", 0.0)
            pulse = attrs.get("cycles_coarse", 0)
            ul = attrs.get("ultrasonic_length", 0.0)
            ref_tof = attrs.get("reference_tof")
            result = {
                "freq": float(freq),
                "gain": float(gain),
                "pulse": float(pulse),
                "ul": float(ul),
            }
            if ref_tof is not None:
                result["tof"] = float(ref_tof)
            return result
        self.cur.execute(
            "SELECT frequency, gain, cycles_coarse, ultrasonic_length, reference_tof "
            "FROM batch WHERE batch_id=%s",
            (batch_id,),
        )
        row = self.cur.fetchone()
        if not row:
            raise KeyError(f"Batch {batch_id} no encontrado")
        freq, gain, pulse, ul, ref_tof = row
        result = {
            "freq": float(freq) if freq is not None else 0.0,
            "gain": float(gain) if gain is not None else 0.0,
            "pulse": float(pulse) if pulse is not None else 0.0,
            "ul": float(ul) if ul is not None else 0.0,
        }
        if ref_tof is not None:
            result["tof"] = float(ref_tof)
        return result

    # ------------------------------------------------------------------
    def create_batch(self, batch_id: str) -> None:
        if self._mock:
            if batch_id in self._batches:
                raise ValueError("Batch already exists")
            self._batches[batch_id] = {"attrs": {}, "bolts": []}
        else:
            self.cur.execute("INSERT INTO batch (batch_id) VALUES (%s)", (batch_id,))

    # ------------------------------------------------------------------
    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        if self._mock:
            return self._batches.get(batch_id, {"attrs": {}, "bolts": []})
        self.cur.execute(
            "SELECT metric, length, ultrasonic_length, grade, manufacturer, customer_part_number, "
            "additional_comment, application_description, nut_or_tapped_hole, joint_length, max_load, "
            "target_load, min_load, min_temp, max_temp, frequency, gain, cycles_coarse, cycles_fine, "
            "temperature, reference_tof, temp_gradient, short_temporal_window, short_signal_power_first_window, "
            "long_temporal_window, long_signal_power_first_window, short_correlation_window, long_correlation_window, "
            "temporal_signal_power, correlation_signal_power, xi, alpha1, alpha2, alpha3, customer "
            "FROM batch WHERE batch_id=%s",
            (batch_id,),
        )
        row = self.cur.fetchone()
        if not row:
            return {"attrs": {}, "bolts": []}
        keys = [
            "metric",
            "length",
            "ultrasonic_length",
            "grade",
            "manufacturer",
            "customer_part_number",
            "additional_comment",
            "application_description",
            "nut_or_tapped_hole",
            "joint_length",
            "max_load",
            "target_load",
            "min_load",
            "min_temp",
            "max_temp",
            "frequency",
            "gain",
            "cycles_coarse",
            "cycles_fine",
            "temperature",
            "reference_tof",
            "temp_gradient",
            "short_temporal_window",
            "short_signal_power_first_window",
            "long_temporal_window",
            "long_signal_power_first_window",
            "short_correlation_window",
            "long_correlation_window",
            "temporal_signal_power",
            "correlation_signal_power",
            "xi",
            "alpha1",
            "alpha2",
            "alpha3",
            "customer",
        ]
        attrs = {k: "" if v is None else str(v) for k, v in zip(keys, row)}
        self.cur.execute(
            "SELECT bolt_id FROM bolt WHERE batch_id=%s ORDER BY bolt_id", (batch_id,)
        )
        bolts = [r[0] for r in self.cur.fetchall()]
        return {"attrs": attrs, "bolts": bolts}

    # ------------------------------------------------------------------
    def get_device_params(self, batch_id: str) -> tuple[float, int, float, float]:
        """Return (temp_gradient, long_correlation_window, xi, alpha1) for batch.
        Falls back to (-103.0, 10, 0.0, 0.0) when not found.

        Note: alpha1 is returned as the second alpha parameter to be sent to the
        device at address 0x19. Historically only the first two values were
        returned; callers updated to consume all four will now also get xi/alpha.
        """
        default = (-103.0, 10, 0.0, 0.0)
        try:
            if self._mock:
                attrs = self._batches.get(batch_id, {}).get("attrs", {})
                return (
                    float(attrs.get("temp_gradient", default[0])),
                    int(attrs.get("long_correlation_window", default[1])),
                    float(attrs.get("xi", default[2])),
                    float(attrs.get("alpha1", default[3])),
                )
            self.cur.execute(
                "SELECT temp_gradient, long_correlation_window, xi, alpha1 FROM batch WHERE batch_id=%s",
                (batch_id,),
            )
            row = self.cur.fetchone()
            if row:
                diftemp = float(row[0]) if row[0] is not None else default[0]
                long_corr = int(row[1]) if row[1] is not None else default[1]
                xi = float(row[2]) if row[2] is not None else default[2]
                alpha1 = float(row[3]) if row[3] is not None else default[3]
                return diftemp, long_corr, xi, alpha1
        except Exception:
            pass
        return default

    # ------------------------------------------------------------------
    def set_batch_attrs(self, batch_id: str, attrs: Dict[str, str]) -> None:
        if self._mock:
            if batch_id not in self._batches:
                self._batches[batch_id] = {"attrs": {}, "bolts": []}
            self._batches[batch_id]["attrs"] = attrs.copy()
            cust = attrs.get("customer")
            if cust:
                self._customers.add(cust)
            return
        # field definitions: column -> converter
        fields: List[tuple[str, Any]] = [
            ("metric", str),
            ("length", float),
            ("ultrasonic_length", float),
            ("grade", str),
            ("manufacturer", str),
            ("customer_part_number", str),
            ("additional_comment", str),
            ("application_description", str),
            ("nut_or_tapped_hole", str),
            ("joint_length", float),
            ("max_load", float),
            ("target_load", float),
            ("min_load", float),
            ("min_temp", float),
            ("max_temp", float),
            ("frequency", float),
            ("gain", float),
            ("cycles_coarse", int),
            ("cycles_fine", int),
            ("temperature", float),
            ("reference_tof", float),
            ("temp_gradient", float),
            ("short_temporal_window", int),
            ("short_signal_power_first_window", int),
            ("long_temporal_window", int),
            ("long_signal_power_first_window", int),
            ("short_correlation_window", int),
            ("long_correlation_window", int),
            ("temporal_signal_power", int),
            ("correlation_signal_power", int),
            ("xi", float),
            ("alpha1", float),
            ("alpha2", float),
            ("alpha3", float),
            ("customer", str),
        ]
        values = []
        for col, conv in fields:
            val = attrs.get(col)
            if conv is float:
                val = _to_float(val)
            elif conv is int:
                val = _to_int(val)
            else:
                val = val
            values.append(val)
        set_clause = ", ".join(f"{c}=%s" for c, _ in fields)
        sql = f"UPDATE batch SET {set_clause} WHERE batch_id=%s"
        self.cur.execute(sql, (*values, batch_id))

        customer = attrs.get("customer")
        if customer:
            self.cur.execute(
                "SELECT customer_id FROM customer WHERE name=%s", (customer,)
            )
            row = self.cur.fetchone()
            if row:
                cid = row[0]
            else:
                self.cur.execute(
                    "INSERT INTO customer (name) VALUES (%s) RETURNING customer_id",
                    (customer,),
                )
                cid = self.cur.fetchone()[0]
            self.cur.execute(
                "DELETE FROM customer_batch WHERE batch_id=%s", (batch_id,)
            )
            self.cur.execute(
                "INSERT INTO customer_batch (customer_id, batch_id) VALUES (%s, %s)",
                (cid, batch_id),
            )
        else:
            self.cur.execute(
                "DELETE FROM customer_batch WHERE batch_id=%s", (batch_id,)
            )

    # ------------------------------------------------------------------
    # Batch completion state
    # ------------------------------------------------------------------
    def _ensure_batch_completed_column(self) -> None:
        """Ensure boolean column is_completed exists in batch table."""
        if self._mock:
            return
        try:
            self.cur.execute(
                "ALTER TABLE batch ADD COLUMN IF NOT EXISTS is_completed BOOLEAN NOT NULL DEFAULT FALSE"
            )
        except Exception:
            pass

    def set_batch_completed(self, batch_id: str, completed: bool) -> None:
        """Set completion flag for a batch."""
        if self._mock:
            data = self._batches.setdefault(batch_id, {"attrs": {}, "bolts": []})
            data.setdefault("attrs", {})["is_completed"] = bool(completed)
            return
        self._ensure_batch_completed_column()
        self.cur.execute(
            "UPDATE batch SET is_completed=%s WHERE batch_id=%s",
            (bool(completed), str(batch_id)),
        )

    def get_batch_completed(self, batch_id: str) -> bool:
        """Return completion flag for a batch (defaults to False)."""
        if self._mock:
            return bool(self._batches.get(batch_id, {}).get("attrs", {}).get("is_completed", False))
        self._ensure_batch_completed_column()
        self.cur.execute("SELECT is_completed FROM batch WHERE batch_id=%s", (str(batch_id),))
        row = self.cur.fetchone()
        return bool(row[0]) if row and row[0] is not None else False

    # ------------------------------------------------------------------
    def update_alpha1(self, batch_id: str, alpha1: float) -> None:
        """Update only the alpha1 field for a batch."""
        if self._mock:
            data = self._batches.setdefault(batch_id, {"attrs": {}, "bolts": []})
            attrs = data.setdefault("attrs", {})
            attrs["alpha1"] = str(alpha1)
            return
        self.cur.execute("UPDATE batch SET alpha1=%s WHERE batch_id=%s", (float(alpha1), str(batch_id)))

    # ------------------------------------------------------------------
    def list_bolts(self, batch_id: str) -> List[str]:
        if self._mock:
            return list(self._batches.get(batch_id, {}).get("bolts", []))
        self.cur.execute(
            "SELECT bolt_id FROM bolt WHERE batch_id=%s ORDER BY bolt_id", (batch_id,)
        )
        return [r[0] for r in self.cur.fetchall()]

    def list_batches_with_temp_process(self, min_temps: int = 5) -> List[str]:
        """Return batches that have at least one bolt with measurements at
        ``min_temps`` different temperatures recorded in ``temp_measurement``.

        Uses integer-rounded temperatures to group nearby readings, which is
        robust against small sensor noise (e.g., 19.9 and 20.1 both count as 20).
        In mock mode, returns an empty list.
        """
        if self._mock:
            return []
        # Group by batch and bolt, count distinct rounded temperatures, then
        # keep those with at least ``min_temps`` distinct temps.
        sql = (
            "SELECT DISTINCT batch_id FROM ("
            "  SELECT batch_id, bolt_id, COUNT(DISTINCT ROUND(temp::numeric, 0)) AS ntemps"
            "  FROM temp_measurement"
            "  GROUP BY batch_id, bolt_id"
            ") t WHERE ntemps >= %s ORDER BY batch_id"
        )
        try:
            self.cur.execute(sql, (int(min_temps),))
            return [row[0] for row in self.cur.fetchall()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def find_batch_by_bolt(self, bolt_id: str) -> Optional[str]:
        """Find and return the batch_id that contains the given ``bolt_id``.

        Looks up the ``bolt`` table by exact ``bolt_id``. If not found, falls
        back to checking ``bolt_alias`` in case some deployments only
        materialize aliases. Returns ``None`` if no association is found.
        In mock mode, searches the in-memory structure.
        """
        if not bolt_id:
            return None
        if self._mock:
            for bid, data in self._batches.items():
                if bolt_id in set(data.get("bolts", [])):
                    return bid
            return None
        # 1) Direct bolt table lookup
        try:
            self.cur.execute(
                "SELECT batch_id FROM bolt WHERE bolt_id=%s LIMIT 1",
                (str(bolt_id),),
            )
            row = self.cur.fetchone()
            if row:
                return str(row[0])
        except Exception:
            # Defensive: ignore transient DB errors here; caller will handle None
            pass
        # 2) Fallback via alias table (some datasets may reference only aliases)
        try:
            self.cur.execute(
                "SELECT batch_id FROM bolt_alias WHERE bolt_id=%s LIMIT 1",
                (str(bolt_id),),
            )
            row = self.cur.fetchone()
            if row:
                return str(row[0])
        except Exception:
            pass
        return None
    # ------------------------------------------------------------------
    def bolt_exists(self, batch_id: str, bolt: str | int) -> bool:
        """Check if a bolt (alias number or ID) exists for the given batch."""
        if self._mock:
            bolts = self._batches.get(batch_id, {}).get("bolts", [])
            return str(bolt) in bolts
        # Try alias lookup first
        try:
            num = int(bolt)
            bid = str(batch_id)
            self.cur.execute(
                "SELECT 1 FROM bolt_alias WHERE batch_id=%s AND bolt_num=%s LIMIT 1",
                (bid, num)
            )
            if self.cur.fetchone():
                return True
        except (ValueError, TypeError):
            pass
        # Fallback to bolt_id lookup
        bid = str(batch_id)
        self.cur.execute(
            "SELECT 1 FROM bolt WHERE batch_id=%s AND bolt_id=%s LIMIT 1",
            (bid, str(bolt))
        )
        return self.cur.fetchone() is not None
    
    def get_bolt_num(self, batch_id: str, bolt_id: str) -> int | None:
        """Return the bolt_num (alias) for a given batch and bolt ID."""
        if self._mock:
            try:
                return int(bolt_id)
            except (ValueError, TypeError):
                return None
        bid = str(batch_id)
        self.cur.execute(
            "SELECT bolt_num FROM bolt_alias WHERE batch_id=%s AND bolt_id=%s LIMIT 1",
            (bid, str(bolt_id))
        )
        row = self.cur.fetchone()
        return int(row[0]) if row else None
    # ------------------------------------------------------------------
    def params_for(self, batch_id: str, bolt: str | int = None) -> Dict[str, Any]:
        """Return parameters (frequency, gain, pulse) stored in ``batch``.

        Also returns ``ul`` (bolt length) and optional reference ``tof`` when
        available.  ``bolt`` is accepted for API compatibility but ignored."""
        # Reuse get_batch_params to honour both real and mock backends
        return self.get_batch_params(str(batch_id))
    # ------------------------------------------------------------------
    def valid_combinations(self, batch_id: str) -> List[tuple[float, float, float]]:
        """Return all pre-valid combos for the batch, ignoring bolt."""
        if self._mock:
            return []
        bid = str(batch_id)
        self.cur.execute(
            "SELECT freq, gain, pulse FROM pre_valid_combo WHERE batch_id=%s",
            (bid,)
        )
        return [(float(f), float(g), float(p)) for f, g, p in self.cur.fetchall()]


    def one4_valid_combinations(self, batch_id: str) -> List[tuple[float, float, float]]:
        """Return valid combos from ``one4_valid_combo`` for temperature scans."""
        if self._mock:
            return []
        bid = str(batch_id)
        self.cur.execute(
            "SELECT freq, gain, pulse FROM one4_valid_combo WHERE batch_id=%s",
            (bid,),
        )
        return [(float(f), float(g), float(p)) for f, g, p in self.cur.fetchall()]

    # ------------------------------------------------------------------
    def add_bolts(self, batch_id: str, bolts: Iterable[str]) -> None:
        if self._mock:
            self._batches.setdefault(batch_id, {"attrs": {}, "bolts": []})[
                "bolts"
            ].extend(bolts)
            return
        for bolt in bolts:
            self.cur.execute(
                "INSERT INTO bolt (batch_id, bolt_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (batch_id, bolt),
            )

    # ------------------------------------------------------------------
    def delete_one10_data(self, batch_id: str, bolt_id: str) -> None:
        """Delete all one10 data for the given bolt (measurements + loading)."""
        if self._mock:
            # mock backend keeps nothing persistent
            return
        bid = str(batch_id)
        self.cur.execute("DELETE FROM one10_loading WHERE batch_id=%s AND bolt_id=%s", (bid, str(bolt_id)))
        self.cur.execute("DELETE FROM one10_measurement WHERE batch_id=%s AND bolt_id=%s", (bid, str(bolt_id)))

    def delete_bending_data(self, batch_id: str, bolt_id: str, position: Optional[int] = None) -> None:
        """Delete bending data for a bolt and optional position.

        If position is None, deletes all positions for that bolt.
        """
        if self._mock:
            return
        bid = str(batch_id)
        if position is None:
            self.cur.execute("DELETE FROM bending_loading WHERE batch_id=%s AND bolt_id=%s", (bid, str(bolt_id)))
            self.cur.execute("DELETE FROM bending_measurement WHERE batch_id=%s AND bolt_id=%s", (bid, str(bolt_id)))
        else:
            self.cur.execute(
                "DELETE FROM bending_loading WHERE batch_id=%s AND bolt_id=%s AND position=%s",
                (bid, str(bolt_id), int(position)),
            )
            self.cur.execute(
                "DELETE FROM bending_measurement WHERE batch_id=%s AND bolt_id=%s AND position=%s",
                (bid, str(bolt_id), int(position)),
            )

    # ------------------------------------------------------------------
    def remove_bolts(self, batch_id: str, bolts: Iterable[str]) -> None:
        if self._mock:
            data = self._batches.get(batch_id)
            if not data:
                return
            data["bolts"] = [b for b in data["bolts"] if b not in set(bolts)]
            return
        for bolt in bolts:
            self.cur.execute(
                "DELETE FROM bolt WHERE batch_id=%s AND bolt_id=%s", (batch_id, bolt)
            )
    
    def add_bolt_alias(self, batch_id: str, bolt_id: str, bolt_num: int) -> None:
        """Associate a bolt alias number to a bolt in a batch."""
        if self._mock:
            # mock backend: skip alias persistence
            return
        sql = (
            "INSERT INTO bolt_alias (batch_id, bolt_id, bolt_num) VALUES (%s, %s, %s) "
            "ON CONFLICT (batch_id, bolt_num) DO UPDATE SET bolt_id = EXCLUDED.bolt_id"
        )
        self.cur.execute(sql, (batch_id, bolt_id, bolt_num))

    def list_bolt_aliases(self, batch_id: str) -> List[tuple[str, Optional[int]]]:
        """List all bolts in batch with their alias numbers, if any."""
        if self._mock:
            return [(b, None) for b in self.list_bolts(batch_id)]
        self.cur.execute(
            "SELECT b.bolt_id, ba.bolt_num"
            " FROM bolt b"
            " LEFT JOIN bolt_alias ba"
            " ON b.batch_id=ba.batch_id AND b.bolt_id=ba.bolt_id"
            " WHERE b.batch_id=%s"
            " ORDER BY ba.bolt_num NULLS LAST, b.bolt_id",
            (batch_id,)
        )
        return [(row[0], row[1]) for row in self.cur.fetchall()]
    
    def get_bolt_alias(self, batch_id: str, bolt_id: str) -> Optional[int]:
        """Retrieve the alias number for a given bolt_id in a batch."""
        if self._mock:
            return None
        self.cur.execute(
            "SELECT bolt_num FROM bolt_alias WHERE batch_id=%s AND bolt_id=%s LIMIT 1",
            (batch_id, bolt_id)
        )
        row = self.cur.fetchone()
        return row[0] if row else None

    def is_connected(self) -> bool:
        """Return True if connected to a real PostgreSQL database."""
        return self._connected


def _to_int(val: Optional[str]) -> Optional[int]:
    try:
        return int(val) if val not in (None, "") else None
    except ValueError:
        return None


def _to_float(val: Optional[str]) -> Optional[float]:
    try:
        return float(val) if val not in (None, "") else None
    except ValueError:
        return None


# (batch,0,xi,aplhatemp,0,0,'Customer','Description',part number,nominalJL,targetLoad,maxLoad,minLoad,'Application description',frequency,gain,cyclesInCourse,cyclesInFine,Tempgradient,correlationShort,correlationLong,temporalShort,temporalLong,0,0,506,0,196);
