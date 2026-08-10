#!/usr/bin/env python3
"""
MLOps.dev — Local API Server
Raghunathareddy GR <hello@mlops.dev>

This is the real API server that the SDK talks to.
It runs locally during development and on your infrastructure in production.

Usage:
    pip install flask flask-cors
    python server/api.py

    # Then use the SDK pointing at local:
    export MLOPS_API_KEY=demo
    export MLOPS_API_URL=http://localhost:8000/v1
    mlops status
"""

import os
import json

def safe_json(s, default=None):
    if default is None:
        default = {}
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default
import time
import uuid
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, request, jsonify, g
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = Path(__file__).parent / "mlops.db"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            name TEXT,
            status TEXT DEFAULT 'online',
            hw_class TEXT DEFAULT 'jetson_orin',
            arch TEXT DEFAULT 'arm64',
            model_name TEXT DEFAULT '',
            model_tag TEXT DEFAULT '',
            model_format TEXT DEFAULT 'onnx',
            drift_score REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            last_seen TEXT,
            agent_version TEXT DEFAULT '0.6.0',
            os TEXT DEFAULT 'linux',
            ram_mb INTEGER DEFAULT 14,
            cpu_pct REAL DEFAULT 0.0,
            temp_c REAL DEFAULT 45.0,
            uptime_s INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            tag TEXT NOT NULL,
            format TEXT DEFAULT 'onnx',
            variant TEXT DEFAULT 'all',
            size_bytes INTEGER DEFAULT 0,
            sha256 TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            active_devices INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            UNIQUE(name, tag, variant)
        );
        CREATE TABLE IF NOT EXISTS deployments (
            id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_tag TEXT NOT NULL,
            status TEXT DEFAULT 'completed',
            stage INTEGER DEFAULT 1,
            total_stages INTEGER DEFAULT 1,
            target TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            health_gate TEXT DEFAULT '{}',
            stages TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS drift_alerts (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            device_name TEXT DEFAULT '',
            kl_score REAL NOT NULL,
            severity TEXT NOT NULL,
            monitor TEXT DEFAULT 'input_distribution',
            model_name TEXT DEFAULT '',
            model_tag TEXT DEFAULT '',
            detected_at TEXT DEFAULT (datetime('now')),
            resolved_at TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            event_type TEXT NOT NULL,
            model_name TEXT,
            model_tag TEXT,
            status TEXT,
            msg TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Insert demo API key (hash of "demo")
        INSERT OR IGNORE INTO api_keys (id, key_hash, name)
        VALUES ('key_demo', 'demo', 'Demo key');

        -- Insert demo devices
        INSERT OR IGNORE INTO devices
            (id, name, hw_class, model_name, model_tag, model_format, drift_score, latency_ms, last_seen, temp_c, uptime_s, status)
        VALUES
            ('jetson-prod-01', 'Factory Floor A — Camera 1', 'jetson_orin', 'defect-detector', 'v1.0', 'tensorrt', 0.12, 3.8, datetime('now'), 52.4, 864000, 'online'),
            ('jetson-prod-02', 'Factory Floor A — Camera 2', 'jetson_orin', 'defect-detector', 'v1.0', 'tensorrt', 0.08, 4.1, datetime('now'), 48.9, 864000, 'online'),
            ('jetson-nano-01', 'Inspection Line B — Camera 1', 'jetson_nano', 'defect-detector', 'v1.0', 'onnx', 0.45, 19.2, datetime('now'), 61.2, 432000, 'warning'),
            ('jetson-nano-02', 'Inspection Line B — Camera 2', 'jetson_nano', 'defect-detector', 'v1.0', 'tflite', 0.72, 21.4, datetime('now'), 67.8, 432000, 'drift'),
            ('rpi5-edge-01',   'Packaging Station — Camera 1', 'rpi5', 'defect-detector', 'v0.9', 'tflite', 0.03, 48.2, datetime('now'), 41.5, 172800, 'online'),
            ('rpi5-edge-02',   'Packaging Station — Camera 2', 'rpi5', '', '', '', 0.0, 0.0, datetime('now'), 38.2, 7200, 'offline');

        -- Insert demo models
        INSERT OR IGNORE INTO models (id, name, tag, format, variant, size_bytes, sha256, active_devices)
        VALUES
            ('mv_001', 'defect-detector', 'v1.0', 'onnx',      'all',         7400000, 'abc123def456abc123def456abc123def456abc123def456abc123def456abc1', 5),
            ('mv_002', 'defect-detector', 'v1.0', 'tensorrt',  'jetson_orin', 12800000,'def456abc123def456abc123def456abc123def456abc123def456abc123def4', 2),
            ('mv_003', 'defect-detector', 'v1.0', 'tflite',    'rpi5',        4200000, 'bcd789efg012bcd789efg012bcd789efg012bcd789efg012bcd789efg012bcd7', 2),
            ('mv_004', 'defect-detector', 'v0.9', 'onnx',      'all',         7100000, 'efg012hij345efg012hij345efg012hij345efg012hij345efg012hij345efg0', 1);

        -- Insert demo drift alert
        INSERT OR IGNORE INTO drift_alerts (id, device_id, device_name, kl_score, severity, model_name, model_tag)
        VALUES
            ('alert_001', 'jetson-nano-02', 'Inspection Line B — Camera 2', 0.72, 'alert', 'defect-detector', 'v1.0'),
            ('alert_002', 'jetson-nano-01', 'Inspection Line B — Camera 1', 0.45, 'warning', 'defect-detector', 'v1.0');
    """)
    db.commit()
    db.close()

# ── Auth middleware ───────────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing Authorization header"}), 401
        key = auth.split(" ", 1)[1].strip()
        db = get_db()
        row = db.execute(
            "SELECT id FROM api_keys WHERE key_hash = ?", (key,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Invalid API key. Get yours at mlops.dev/dashboard"}), 401
        return f(*args, **kwargs)
    return decorated

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def row_to_dict(row):
    return dict(row) if row else None

# ── Health ────────────────────────────────────────────────────────
@app.route("/v1/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0", "uptime_s": int(time.time() % 864000)})

# ── Status ────────────────────────────────────────────────────────
@app.route("/v1/status")
@require_auth
def status():
    db = get_db()
    total    = db.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
    online   = db.execute("SELECT COUNT(*) FROM devices WHERE status='online'").fetchone()[0]
    offline  = db.execute("SELECT COUNT(*) FROM devices WHERE status='offline'").fetchone()[0]
    drifting = db.execute("SELECT COUNT(*) FROM devices WHERE status IN ('drift','warning')").fetchone()[0]
    active_d = db.execute("SELECT COUNT(*) FROM deployments WHERE status='running'").fetchone()[0]
    return jsonify({
        "total_devices": total, "online": online,
        "offline": offline, "drifting": drifting,
        "active_deployments": active_d, "api_version": "1.0.0",
    })

# ── Devices ───────────────────────────────────────────────────────
@app.route("/v1/devices")
@require_auth
def devices_list():
    db = get_db()
    q = "SELECT * FROM devices WHERE 1=1"
    params = []
    if request.args.get("status"):
        q += " AND status=?"
        params.append(request.args["status"])
    if request.args.get("hw_class"):
        q += " AND hw_class=?"
        params.append(request.args["hw_class"])
    if request.args.get("model"):
        q += " AND model_name=?"
        params.append(request.args["model"])
    limit  = min(int(request.args.get("limit",  100)), 500)
    offset = int(request.args.get("offset", 0))
    q += f" ORDER BY id LIMIT {limit} OFFSET {offset}"
    rows = [row_to_dict(r) for r in db.execute(q, params).fetchall()]
    for r in rows:
        r["metadata"] = safe_json(r.get("metadata"))
    total = db.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
    return jsonify({"data": rows, "total": total, "limit": limit, "offset": offset})

@app.route("/v1/devices/<device_id>")
@require_auth
def devices_get(device_id):
    db = get_db()
    row = row_to_dict(db.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone())
    if not row:
        return jsonify({"error": f"Device not found: {device_id}"}), 404
    row["metadata"] = safe_json(row.get("metadata"))
    return jsonify({"data": row})

@app.route("/v1/devices/<device_id>", methods=["DELETE"])
@require_auth
def devices_delete(device_id):
    db = get_db()
    db.execute("DELETE FROM devices WHERE id=?", (device_id,))
    db.commit()
    return jsonify({"deleted": device_id})

@app.route("/v1/devices/<device_id>/logs")
@require_auth
def devices_logs(device_id):
    limit = min(int(request.args.get("limit", 50)), 1000)
    db    = get_db()
    rows  = db.execute(
        "SELECT * FROM audit_log WHERE device_id=? ORDER BY created_at DESC LIMIT ?",
        (device_id, limit)
    ).fetchall()
    return jsonify({"data": [dict(r) for r in rows]})

@app.route("/v1/devices/<device_id>/config", methods=["PATCH"])
@require_auth
def devices_config(device_id):
    data = request.get_json(silent=True) or {}
    db   = get_db()
    row  = db.execute("SELECT id FROM devices WHERE id=?", (device_id,)).fetchone()
    if not row:
        return jsonify({"error": "Device not found"}), 404
    # Apply supported config fields
    allowed = ["drift_warn", "drift_alert"]
    if "drift_alert" in data:
        db.execute("UPDATE devices SET status='online' WHERE id=? AND drift_score < ?",
                   (device_id, data["drift_alert"]))
    db.commit()
    return jsonify({"device_id": device_id, "config": data, "applied": True})

# ── Models ────────────────────────────────────────────────────────
@app.route("/v1/models")
@require_auth
def models_list():
    db   = get_db()
    rows = db.execute("SELECT * FROM models ORDER BY name, created_at DESC").fetchall()
    # Group by name
    groups = {}
    for row in rows:
        d = dict(row)
        d["metadata"] = safe_json(d.get("metadata"))
        name = d["name"]
        if name not in groups:
            groups[name] = {"id": d["id"], "name": name, "versions": []}
        groups[name]["versions"].append(d)
    return jsonify({"data": list(groups.values())})

@app.route("/v1/models/<name>")
@require_auth
def models_get(name):
    db   = get_db()
    rows = db.execute("SELECT * FROM models WHERE name=? ORDER BY created_at DESC", (name,)).fetchall()
    if not rows:
        return jsonify({"error": f"Model not found: {name}"}), 404
    versions = []
    for row in rows:
        d = dict(row)
        d["metadata"] = safe_json(d.get("metadata"))
        versions.append(d)
    return jsonify({"data": {"id": versions[0]["id"], "name": name, "versions": versions}})

@app.route("/v1/models", methods=["POST"])
@require_auth
def models_push():
    # Accept multipart form upload
    if "model" not in request.files:
        return jsonify({"error": "No model file in request"}), 400

    file     = request.files["model"]
    name     = request.form.get("name", "")
    tag      = request.form.get("tag",  "latest")
    fmt      = request.form.get("format", "onnx")
    variant  = request.form.get("variant", "all")
    sha256   = request.form.get("sha256", "")
    metadata_raw = request.form.get("metadata", "{}")
    try:
        metadata = json.dumps(json.loads(metadata_raw))
    except Exception:
        # metadata might be a Python repr dict string like "{'key': 'val'}"
        # convert safely
        try:
            import ast
            metadata = json.dumps(ast.literal_eval(metadata_raw))
        except Exception:
            metadata = "{}"

    if not name:
        return jsonify({"error": "name is required"}), 400

    # Save file
    model_dir = MODELS_DIR / name / tag / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / file.filename
    file.save(str(save_path))
    size_bytes = save_path.stat().st_size

    # Compute SHA-256 if not provided
    if not sha256:
        h = hashlib.sha256()
        with open(save_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    # Upsert model version
    mv_id = f"mv_{uuid.uuid4().hex[:8]}"
    db    = get_db()
    try:
        db.execute("""
            INSERT INTO models (id, name, tag, format, variant, size_bytes, sha256, metadata)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(name, tag, variant) DO UPDATE SET
                size_bytes=excluded.size_bytes,
                sha256=excluded.sha256,
                metadata=excluded.metadata,
                created_at=datetime('now')
        """, (mv_id, name, tag, fmt, variant, size_bytes, sha256, metadata))
        db.commit()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    row = row_to_dict(db.execute(
        "SELECT * FROM models WHERE name=? AND tag=? AND variant=?",
        (name, tag, variant)
    ).fetchone())
    row["metadata"] = safe_json(row.get("metadata"))

    # Log it
    db.execute(
        "INSERT INTO audit_log (id, event_type, model_name, model_tag, status, msg) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), "model_push", name, tag, "success",
         f"Pushed {name}:{tag} ({variant}, {size_bytes//1024}KB)")
    )
    db.commit()

    return jsonify({"data": row}), 201

@app.route("/v1/models/<name>/<tag>", methods=["DELETE"])
@require_auth
def models_delete(name, tag):
    db = get_db()
    # Check not active
    active = db.execute(
        "SELECT COUNT(*) FROM devices WHERE model_name=? AND model_tag=?",
        (name, tag)
    ).fetchone()[0]
    if active > 0:
        return jsonify({
            "error": f"Cannot delete {name}:{tag} — it is active on {active} device(s). "
                     "Deploy a different version first."
        }), 409
    db.execute("DELETE FROM models WHERE name=? AND tag=?", (name, tag))
    db.commit()
    return jsonify({"deleted": f"{name}:{tag}"})

# ── Deployments ───────────────────────────────────────────────────
@app.route("/v1/deployments", methods=["POST"])
@require_auth
def deployments_create():
    data       = request.get_json(silent=True) or {}
    model_name = data.get("model_name", "")
    model_tag  = data.get("model_tag",  "latest")
    target     = data.get("target", "")
    stages     = data.get("stages", [])
    health_gate= data.get("health_gate", {})

    if not model_name or not target:
        return jsonify({"error": "model_name and target are required"}), 400

    db    = get_db()
    dep_id = f"dep_{uuid.uuid4().hex[:8]}"
    total_stages = max(len(stages), 1)

    # Simulate instant completion for demo
    dep_status = "completed"

    # Apply model update to matching devices
    if target == "all":
        db.execute(
            "UPDATE devices SET model_name=?, model_tag=?, last_seen=datetime('now') WHERE status != 'offline'",
            (model_name, model_tag)
        )
    elif target in ("jetson_orin","jetson_nano","rpi5","rpi4","coral","x86_64","arm_custom"):
        db.execute(
            "UPDATE devices SET model_name=?, model_tag=?, last_seen=datetime('now') WHERE hw_class=? AND status != 'offline'",
            (model_name, model_tag, target)
        )
    else:
        # Specific device ID
        row = db.execute("SELECT id FROM devices WHERE id=?", (target,)).fetchone()
        if not row:
            return jsonify({"error": f"Device not found: {target}"}), 404
        db.execute(
            "UPDATE devices SET model_name=?, model_tag=?, last_seen=datetime('now') WHERE id=?",
            (model_name, model_tag, target)
        )

    db.execute("""
        INSERT INTO deployments (id, model_name, model_tag, status, stage, total_stages, target, health_gate, stages)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (dep_id, model_name, model_tag, dep_status, total_stages, total_stages,
          target, json.dumps(health_gate), json.dumps(stages)))

    db.execute(
        "INSERT INTO audit_log (id, event_type, device_id, model_name, model_tag, status, msg) VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), "deployment", target, model_name, model_tag, dep_status,
         f"Deployed {model_name}:{model_tag} to {target}")
    )
    db.commit()

    row = row_to_dict(db.execute("SELECT * FROM deployments WHERE id=?", (dep_id,)).fetchone())
    row["stages"]      = json.loads(row.get("stages") or "[]")
    row["health_gate"] = json.loads(row.get("health_gate") or "{}")
    return jsonify({"data": row}), 201

@app.route("/v1/deployments")
@require_auth
def deployments_list():
    db     = get_db()
    limit  = min(int(request.args.get("limit", 20)), 100)
    status = request.args.get("status")
    q      = "SELECT * FROM deployments"
    params = []
    if status:
        q += " WHERE status=?"
        params.append(status)
    q += f" ORDER BY created_at DESC LIMIT {limit}"
    rows = []
    for row in db.execute(q, params).fetchall():
        d = dict(row)
        d["stages"]      = json.loads(d.get("stages") or "[]")
        d["health_gate"] = json.loads(d.get("health_gate") or "{}")
        rows.append(d)
    return jsonify({"data": rows})

@app.route("/v1/deployments/<dep_id>")
@require_auth
def deployments_get(dep_id):
    db  = get_db()
    row = row_to_dict(db.execute("SELECT * FROM deployments WHERE id=?", (dep_id,)).fetchone())
    if not row:
        return jsonify({"error": f"Deployment not found: {dep_id}"}), 404
    row["stages"]      = json.loads(row.get("stages") or "[]")
    row["health_gate"] = json.loads(row.get("health_gate") or "{}")
    return jsonify({"data": row})

@app.route("/v1/deployments/rollback", methods=["POST"])
@require_auth
def deployments_rollback():
    data      = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    model_name= data.get("model_name")
    model_tag = data.get("model_tag")
    db        = get_db()

    if device_id:
        if model_name and model_tag:
            db.execute(
                "UPDATE devices SET model_name=?, model_tag=?, last_seen=datetime('now') WHERE id=?",
                (model_name, model_tag, device_id)
            )
        affected = 1
    else:
        if model_name and model_tag:
            db.execute(
                "UPDATE devices SET model_name=?, model_tag=?, last_seen=datetime('now') WHERE status != 'offline'",
                (model_name, model_tag)
            )
        affected = db.execute("SELECT COUNT(*) FROM devices WHERE status != 'offline'").fetchone()[0]

    db.execute(
        "INSERT INTO audit_log (id, event_type, device_id, model_name, model_tag, status, msg) VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), "rollback", device_id or "fleet",
         model_name or "previous", model_tag or "previous",
         "queued", f"Rollback queued for {device_id or 'fleet'}")
    )
    db.commit()
    return jsonify({"status": "queued", "affected_devices": affected})

@app.route("/v1/deployments/<dep_id>/rollback", methods=["POST"])
@require_auth
def deployment_rollback(dep_id):
    db = get_db()
    db.execute("UPDATE deployments SET status='rolled_back' WHERE id=?", (dep_id,))
    db.commit()
    return jsonify({"status": "rolled_back", "deployment_id": dep_id})

# ── Drift ─────────────────────────────────────────────────────────
@app.route("/v1/drift")
@require_auth
def drift_report():
    db       = get_db()
    total    = db.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
    healthy  = db.execute("SELECT COUNT(*) FROM devices WHERE drift_score < 0.4 AND status='online'").fetchone()[0]
    warning  = db.execute("SELECT COUNT(*) FROM devices WHERE drift_score >= 0.4 AND drift_score < 0.7").fetchone()[0]
    drifting = db.execute("SELECT COUNT(*) FROM devices WHERE drift_score >= 0.7").fetchone()[0]
    offline  = db.execute("SELECT COUNT(*) FROM devices WHERE status='offline'").fetchone()[0]

    avg_kl_row = db.execute(
        "SELECT AVG(drift_score) FROM devices WHERE status != 'offline'"
    ).fetchone()
    avg_kl = round(float(avg_kl_row[0] or 0.0), 3)

    worst = db.execute(
        "SELECT id, drift_score FROM devices ORDER BY drift_score DESC LIMIT 1"
    ).fetchone()
    worst_id = worst["id"]   if worst else ""
    worst_kl = worst["drift_score"] if worst else 0.0

    alerts = db.execute(
        "SELECT * FROM drift_alerts WHERE resolved_at IS NULL ORDER BY kl_score DESC"
    ).fetchall()

    return jsonify({"data": {
        "total_devices":   total,
        "healthy":         healthy,
        "warning":         warning,
        "drifting":        drifting,
        "offline":         offline,
        "fleet_avg_kl":    avg_kl,
        "worst_device_id": worst_id,
        "worst_kl":        round(float(worst_kl), 3),
        "alerts":          [dict(a) for a in alerts],
    }})

@app.route("/v1/drift/alerts")
@require_auth
def drift_alerts():
    db       = get_db()
    resolved = request.args.get("resolved", "false").lower() == "true"
    if resolved:
        rows = db.execute(
            "SELECT * FROM drift_alerts WHERE resolved_at IS NOT NULL ORDER BY resolved_at DESC"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM drift_alerts WHERE resolved_at IS NULL ORDER BY kl_score DESC"
        ).fetchall()
    return jsonify({"data": [dict(r) for r in rows]})

@app.route("/v1/drift/<device_id>/history")
@require_auth
def drift_history(device_id):
    hours    = int(request.args.get("hours", 24))
    # Return synthetic history for demo
    import random, math
    now   = time.time()
    base  = 0.12
    points = []
    for i in range(hours * 12):  # 5-min intervals
        ts       = now - (hours * 3600 - i * 300)
        kl       = max(0, base + 0.05 * math.sin(i * 0.3) + random.uniform(-0.02, 0.02))
        points.append({
            "ts":       datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kl_score": round(kl, 4),
            "monitor":  "input_distribution",
        })
    return jsonify({"device_id": device_id, "data": points[-50:]})  # last 50 points

@app.route("/v1/drift/<device_id>/baseline/reset", methods=["POST"])
@require_auth
def drift_reset(device_id):
    db = get_db()
    db.execute(
        "UPDATE devices SET drift_score=0.0, status=CASE WHEN status='drift' THEN 'online' WHEN status='warning' THEN 'online' ELSE status END WHERE id=?",
        (device_id,)
    )
    db.execute(
        "UPDATE drift_alerts SET resolved_at=datetime('now') WHERE device_id=? AND resolved_at IS NULL",
        (device_id,)
    )
    db.execute(
        "INSERT INTO audit_log (id, event_type, device_id, status, msg) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), "drift_reset", device_id, "success",
         f"Drift baseline reset for {device_id}")
    )
    db.commit()
    return jsonify({"device_id": device_id, "reset": True, "msg": "Recalibrating over next 200 inferences"})

@app.route("/v1/drift/baseline/reset-fleet", methods=["POST"])
@require_auth
def drift_reset_fleet():
    data     = request.get_json(silent=True) or {}
    hw_class = data.get("hw_class")
    model    = data.get("model")
    db       = get_db()

    q = "UPDATE devices SET drift_score=0.0, status=CASE WHEN status IN ('drift','warning') THEN 'online' ELSE status END WHERE 1=1"
    params = []
    if hw_class:
        q += " AND hw_class=?"
        params.append(hw_class)
    if model:
        q += " AND model_name=?"
        params.append(model)
    db.execute(q, params)
    count = db.execute("SELECT changes()").fetchone()[0]
    db.commit()
    return jsonify({"reset": True, "count": count})

# ── Audit ─────────────────────────────────────────────────────────
@app.route("/v1/audit")
@require_auth
def audit():
    db     = get_db()
    limit  = min(int(request.args.get("limit", 100)), 10000)
    q      = "SELECT * FROM audit_log WHERE 1=1"
    params = []
    if request.args.get("device_id"):
        q += " AND device_id=?"; params.append(request.args["device_id"])
    if request.args.get("event_type"):
        q += " AND event_type=?"; params.append(request.args["event_type"])
    if request.args.get("since"):
        q += " AND created_at >= ?"; params.append(request.args["since"])
    if request.args.get("until"):
        q += " AND created_at <= ?"; params.append(request.args["until"])
    q += f" ORDER BY created_at DESC LIMIT {limit}"
    rows = [dict(r) for r in db.execute(q, params).fetchall()]
    fmt = request.args.get("format","json")
    if fmt == "csv":
        import csv, io
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        return jsonify({"csv": buf.getvalue()})
    return jsonify({"data": rows, "total": len(rows)})

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MLOps.dev API Server")
    print("  Raghunathareddy GR — CEO & Founder")
    print("=" * 55)
    print(f"  URL:      http://localhost:8000")
    print(f"  API:      http://localhost:8000/v1")
    print(f"  Demo key: demo")
    print()
    print("  SDK usage:")
    print("    export MLOPS_API_KEY=demo")
    print("    export MLOPS_API_URL=http://localhost:8000/v1")
    print("    mlops status")
    print("    mlops devices list")
    print("=" * 55)
    init_db()
    app.run(host="0.0.0.0", port=8000, debug=False)
