#!/usr/bin/env python3
"""
KALI Web Frontend — Flask backend
Supports local folder paths, chunked uploads, log throttling,
background jobs, JSON persistence, and production-ready structure.
"""
import json, os, queue, subprocess, sys, threading, uuid, urllib.request, urllib.error, sqlite3
from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request, send_file

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
JOBS_FILE  = BASE_DIR / "jobs.json"
# Scripts live in src/kali_apps/ (same source used by the pip-installable CLI
# package). This lets the web UI and the `kali-*` CLI commands share one copy
# of each script instead of drifting apart.
SCRIPTS_DIR = BASE_DIR / "src" / "kali_apps"
SCRIPTS    = {
    "hash":       SCRIPTS_DIR / "pykali_hash.py",
    "KALI_non_hash":     SCRIPTS_DIR / "KALI_non_hash.py",
    "tree":       SCRIPTS_DIR / "kali_tree.py",
    "outbreak":   SCRIPTS_DIR / "kali_outbreak.py",       # not included in this bundle
    "tensor":     SCRIPTS_DIR / "kali_tensor.py",
    "downloader":   SCRIPTS_DIR / "kali_downloader.py",
    "KALI_hash":           SCRIPTS_DIR / "KALI_hash.py",
    "mash_compare":      SCRIPTS_DIR / "kali_mash_comparison.py",  # not included in this bundle
    "plot_mantel":       SCRIPTS_DIR / "plot_mantel.py",           # not included in this bundle
    "prepare_metadata":  SCRIPTS_DIR / "prepare_metadata.py",
}
GENOME_DOWNLOADS_DIR = BASE_DIR / "downloads"
GENOME_DOWNLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]      = 500 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"]   = True
app.jinja_env.auto_reload             = True

@app.after_request
def no_cache(response):
    if "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
    return response

# ── Job store — persisted to jobs.json ──────────────────────────
_job_queues: dict = {}   # job_id → queue.Queue (in-memory only)

def _load_jobs() -> dict:
    if JOBS_FILE.exists():
        try: return json.loads(JOBS_FILE.read_text())
        except: pass
    return {}

def _save_jobs(jobs: dict) -> None:
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))

def _update_job(job_id: str, **kwargs) -> None:
    jobs = _load_jobs()
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        _save_jobs(jobs)


# ── Background runner ────────────────────────────────────────────
LOG_THROTTLE = 30   # send 1 in every N lines to the browser

def _run_job(job_id: str, cmd: list, output_base: str) -> None:
    q = _job_queues[job_id]
    _update_job(job_id, status="running")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            preexec_fn=os.setsid  # detach from parent process group
        )
        line_count = 0
        for line in proc.stdout:
            line = line.rstrip()
            line_count += 1
            # Always store in log; throttle what goes to browser
            jobs = _load_jobs()
            if job_id in jobs:
                jobs[job_id]["log"].append(line)
                _save_jobs(jobs)
            # Send every important line + throttled regular lines
            important = any(kw in line.lower() for kw in
                            ("saved", "error", "done", "complete", "k=", "total time", "✓"))
            if important or line_count % LOG_THROTTLE == 0:
                q.put(("log", line))

        proc.wait()
        if proc.returncode == 0:
            files = sorted(f.name for f in OUTPUT_DIR.glob(
                f"{Path(output_base).name}*"))
            _update_job(job_id, status="done", files=files)
            q.put(("done", json.dumps({"files": files})))
        else:
            _update_job(job_id, status="error")
            q.put(("error", f"Script exited with code {proc.returncode}"))
    except Exception as e:
        _update_job(job_id, status="error")
        q.put(("error", str(e)))


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Chunked file upload — saves each file to disk immediately."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400
    saved = []
    for f in files:
        if f.filename:
            dest = UPLOAD_DIR / Path(f.filename).name
            f.save(dest)
            saved.append(Path(f.filename).name)
    return jsonify({"files": saved})


@app.route("/run", methods=["POST"])
def run():
    data     = request.json or {}
    script   = data.get("script", "hash")
    k_list   = data.get("k", [3])
    bins     = data.get("bins", 50)
    metric   = data.get("metric", "cosine")
    mode     = data.get("mode", "genome")
    reduce   = data.get("reduce", "mean")
    combine  = data.get("combine", False)
    plot     = data.get("plot", False)
    files    = data.get("files", [])        # uploaded filenames
    folder   = data.get("folder", "").strip()  # local folder path

    if not files and not folder:
        return jsonify({"error": "No genome files or folder specified"}), 400

    # Auto-clear old uploads when starting a new run with fresh uploads
    if files and data.get("clear_before_run", False):
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.name not in files:
                f.unlink()

    job_id      = str(uuid.uuid4())[:8]
    output_base = str(OUTPUT_DIR / f"kali_{job_id}")

    # Build genome input — prefer local folder, fall back to uploaded files
    if folder:
        genome_args = [folder]
    else:
        genome_args = [str(UPLOAD_DIR / f) for f in files]

    if script == "hash":
        cmd = [
            sys.executable, str(SCRIPTS["hash"]),
            "-g", *genome_args,
            "-k", *[str(k) for k in k_list],
            "-b", str(bins),
            "-d", metric, "-m", mode, "-r", reduce,
            "-o", output_base, "-v",
        ]
    else:
        g_input = folder if folder else str(UPLOAD_DIR)
        cmd = [
            sys.executable, str(SCRIPTS["KALI_non_hash"]),
            "-g", g_input,
            "-k", *[str(k) for k in k_list],
            "-b", str(bins), "-d", metric,
            "-r", reduce,
            "-o", output_base, "-v",
        ]

    if combine:
        cmd.append("--combine")
    if plot and script != "hash":
        cmd.append("--plot")

    # Persist job to disk
    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [], "cmd": " ".join(cmd)}
    _save_jobs(jobs)

    _job_queues[job_id] = queue.Queue()
    threading.Thread(
        target=_run_job, args=(job_id, cmd, output_base), daemon=True
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    if job_id not in _job_queues:
        jobs = _load_jobs()
        if job_id not in jobs:
            return Response("data: {\"error\": \"unknown job\"}\n\n",
                            mimetype="text/event-stream")
        j = jobs[job_id]

        # If job was running when server restarted, re-launch it
        if j["status"] in ("running", "queued") and j.get("cmd"):
            import shlex
            cmd = shlex.split(j["cmd"])
            _job_queues[job_id] = __import__("queue").Queue()
            # find output base from cmd
            try:
                out_idx = cmd.index("-o") + 1
                out_base = cmd[out_idx]
            except (ValueError, IndexError):
                out_base = str(OUTPUT_DIR / ("kali_" + job_id))
            __import__("threading").Thread(
                target=_run_job, args=(job_id, cmd, out_base), daemon=True
            ).start()
            # fall through to generate() below

        else:
            def replay():
                for line in j["log"][-200:]:
                    yield "event: log\ndata: " + line + "\n\n"
                if j["status"] == "done":
                    yield "event: done\ndata: " + json.dumps({"files": j["files"]}) + "\n\n"
                elif j["status"] == "error":
                    yield "event: error\ndata: Job failed\n\n"
            return Response(replay(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache",
                                     "X-Accel-Buffering": "no"})

    if job_id not in _job_queues:
        return Response("event: error\ndata: Queue lost\n\n",
                        mimetype="text/event-stream")

    def generate():
        q = _job_queues[job_id]
        while True:
            try:
                event_type, data = q.get(timeout=30)
                yield f"event: {event_type}\ndata: {data}\n\n"
                if event_type in ("done", "error"):
                    break
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/jobs")
def list_jobs():
    return jsonify(_load_jobs())


@app.route("/status/<job_id>")
def status(job_id):
    jobs = _load_jobs()
    if job_id not in jobs:
        return jsonify({"error": "unknown"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<filename>")
def download(filename):
    path = OUTPUT_DIR / filename
    if not path.exists(): return "Not found", 404
    return send_file(path, as_attachment=True)


@app.route("/preview/<filename>")
def preview(filename):
    path = OUTPUT_DIR / filename
    if not path.exists(): return "Not found", 404
    return send_file(path)


@app.route("/uploads")
def list_uploads():
    return jsonify({"files": sorted(f.name for f in UPLOAD_DIR.iterdir() if f.is_file())})


@app.route("/clear-uploads", methods=["POST"])
def clear_uploads():
    """Delete all files in the uploads folder."""
    removed = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
            removed.append(f.name)
    return jsonify({"removed": removed, "count": len(removed)})


@app.route("/validate-folder", methods=["POST"])
def validate_folder():
    """Check a local path — accepts both a folder and a single file."""
    folder = (request.json or {}).get("folder", "").strip()
    p = Path(folder)
    if not p.exists():
        return jsonify({"valid": False, "error": "Path not found"})
    exts = {".fasta", ".fa", ".fna", ".fas", ".ffn", ".fastq", ".fq"}
    gz   = {e + ".gz" for e in exts}
    if p.is_file():
        if p.suffix.lower() in exts or any(p.name.lower().endswith(e) for e in gz):
            return jsonify({"valid": True, "count": 1,
                            "path": str(p), "type": "file"})
        return jsonify({"valid": False, "error": "Not a recognised FASTA file"})
    count = sum(1 for f in p.iterdir()
                if f.suffix.lower() in exts
                or f.name.lower().endswith(tuple(gz)))
    return jsonify({"valid": True, "count": count,
                    "path": str(p), "type": "folder"})


@app.route("/build-tree", methods=["POST"])
def build_tree():
    """Run kali_tree.py on a finished distance matrix CSV."""
    data          = request.json or {}
    csv_file      = data.get("csv_file", "")
    method        = data.get("method", "upgma")
    group_col     = data.get("group_col", "")
    metadata_file = data.get("metadata_file", "").strip()

    csv_path = OUTPUT_DIR / csv_file
    if not csv_path.exists():
        return jsonify({"error": f"CSV not found: {csv_file}"}), 404

    if not SCRIPTS["tree"].exists():
        return jsonify({"error": "kali_tree.py not found in app folder"}), 500

    # Resolve metadata file — check outputs, downloads, and absolute path
    meta_path = None
    if metadata_file:
        for search_dir in [OUTPUT_DIR, GENOME_DOWNLOADS_DIR, Path(".")]:
            candidate = search_dir / metadata_file if not Path(metadata_file).is_absolute() else Path(metadata_file)
            if candidate.exists():
                meta_path = candidate
                break
            # Also search recursively in downloads subfolders
            matches = list(search_dir.rglob(metadata_file)) if search_dir.exists() else []
            if matches:
                meta_path = matches[0]
                break
        if not meta_path and Path(metadata_file).exists():
            meta_path = Path(metadata_file)

    # Output base — same name as CSV but with _tree suffix
    out_base = str(csv_path.with_suffix("")) + "_tree"
    html_out = Path(out_base + "_tree.html")

    cmd = [
        sys.executable, str(SCRIPTS["tree"]),
        "-i", str(csv_path),
        "-o", out_base,
        "-m", method,
    ]
    if meta_path:
        cmd += ["--metadata", str(meta_path)]
    if group_col:
        cmd += ["--group-col", group_col]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            err_msg = (result.stderr or result.stdout or "Unknown error")[-600:]
            return jsonify({"error": f"kali_tree.py failed (code {result.returncode}): {err_msg}"}), 500
        # kali_tree saves: out_base + "_tree.html"
        # Try both possible output names
        possible = [
            Path(out_base + "_tree.html"),
            Path(out_base + ".html"),
            html_out,
        ]
        found_html = next((p for p in possible if p.exists()), None)
        if not found_html:
            return jsonify({"error": f"Tree HTML not generated. cmd={' '.join(cmd)}"}), 500
        return jsonify({"html_file": found_html.name,
                        "nwk_file":  out_base + ".nwk",
                        "metadata_used": str(meta_path) if meta_path else None})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Tree build timed out (>120s)"}), 500
    except Exception as e:
        return jsonify({"error": f"Exception: {str(e)} | cmd={' '.join(cmd)}"}), 500


@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    """
    Accept one or more distance matrix CSV files uploaded directly from the
    user's hard drive and save them into OUTPUT_DIR so that /build-tree can
    use them immediately — no prior pykali/hash run required.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    saved = []
    errors = []
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if not name.lower().endswith(".csv"):
            errors.append(f"{name}: only .csv files are accepted")
            continue
        dest = OUTPUT_DIR / name
        f.save(dest)
        saved.append(name)

    if not saved:
        return jsonify({"error": "; ".join(errors) or "No valid CSV files"}), 400

    return jsonify({"saved": saved, "errors": errors})


@app.route("/models")
def list_models():
    """List all .pkl model files in the models/ folder."""
    models = sorted(f.name for f in MODELS_DIR.glob("*.pkl"))
    return jsonify({"models": models})



@app.route("/run-outbreak", methods=["POST"])
def run_outbreak():
    data       = request.json or {}
    inbox      = data.get("inbox", "").strip()
    model      = data.get("model", "")
    k_list     = data.get("k", [3, 4])
    bin_size = data.get("bin_size", 500)
    z_thresh   = data.get("z_threshold", 2.0)
    single     = data.get("single_file", "").strip()

    if not inbox and not single:
        return jsonify({"error": "No inbox folder or file specified"}), 400
    if not SCRIPTS["outbreak"].exists():
        return jsonify({"error": "kali_outbreak.py not found"}), 500

    job_id  = str(__import__("uuid").uuid4())[:8]
    out_dir = str(OUTPUT_DIR / f"outbreak_{job_id}")

    model_path = str(MODELS_DIR / model) if model else ""

    if single:
        cmd = [
            sys.executable, str(SCRIPTS["outbreak"]),
            "add",
            "--genome", single,
            "--outdir", out_dir,
            "--model",  model_path,
            "-k",      *[str(k) for k in k_list],
            "--bin-size", str(bin_size),
            "-z",      str(z_thresh),
        ]
    else:
        cmd = [
            sys.executable, str(SCRIPTS["outbreak"]),
            "watch",
            "--inbox",    inbox,
            "--outdir",   out_dir,
            "--model",    model_path,
            "-k",        *[str(k) for k in k_list],
            "--bin-size",   str(bin_size),
            "-z",        str(z_thresh),
            "--interval","10",
        ]

    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [], "cmd": " ".join(cmd)}
    _save_jobs(jobs)

    _job_queues[job_id] = __import__("queue").Queue()
    __import__("threading").Thread(
        target=_run_job, args=(job_id, cmd, out_dir), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/outbreak-state", methods=["GET"])
def outbreak_state():
    """Return current outbreak monitor state."""
    run_id = request.args.get("run_id", "")
    state_file = OUTPUT_DIR / f"outbreak_{run_id}" / "outbreak_state.json"
    if not state_file.exists():
        return jsonify({"error": "No outbreak state found"}), 404
    with open(state_file) as f:
        return jsonify(json.load(f))


@app.route("/run-tensor", methods=["POST"])
def run_tensor():
    data       = request.json or {}
    mode       = data.get("mode", "fasta")      # "fasta" or "matrices"
    files      = data.get("files", [])
    folder     = data.get("folder", "").strip()
    matrices   = data.get("matrices", [])       # list of CSV paths for mode 2
    labels     = data.get("labels", [])         # layer labels for mode 2
    k_list     = data.get("k", [2, 3, 4, 5])
    bin_size = data.get("bin_size", 500)
    fusion     = data.get("fusion", "variance")
    meta       = data.get("meta", {})           # dict of KEY:VALUE for --meta
    name_map   = data.get("name_map", {})        # dict of ID:DisplayName for --name-map
    labels_col = data.get("labels_col", "")      # CSV path for silhouette scoring

    if not SCRIPTS["tensor"].exists():
        return jsonify({"error": "kali_tensor.py not found"}), 500

    job_id      = str(__import__("uuid").uuid4())[:8]
    output_base = str(OUTPUT_DIR / f"tensor_{job_id}")

    if mode == "matrices":
        # Mode 2 — pre-computed matrices
        # Resolve paths relative to outputs folder
        resolved = []
        for m in matrices:
            p = Path(m)
            if not p.is_absolute():
                p = OUTPUT_DIR / m
            if p.exists():
                resolved.append(str(p))
            else:
                return jsonify({"error": f"Matrix not found: {m}"}), 400

        if len(resolved) < 2:
            return jsonify({"error": "Need at least 2 matrix CSV files"}), 400

        cmd = [
            sys.executable, str(SCRIPTS["tensor"]),
            "--matrices", *resolved,
            "--fusion", fusion,
            "--compare-single",
            "-o", output_base, "-v",
        ]
        # Always use file stems as labels
        cmd += ["--labels"] + [Path(p).stem for p in resolved]

        # Silhouette scoring — pass labels CSV if provided
        if labels_col and Path(labels_col).exists():
            cmd += ["--labels-col", labels_col]

    else:
        # Mode 1 — raw FASTA
        if not files and not folder:
            return jsonify({"error": "No files or folder specified"}), 400

        if folder:
            p = Path(folder)
            exts = {".fasta", ".fa", ".fna", ".fas", ".ffn"}
            input_args = sorted(str(f) for f in p.iterdir()
                                if f.is_file() and f.suffix.lower() in exts) if p.is_dir() else [str(p)]
        else:
            input_args = [str(UPLOAD_DIR / f) for f in files]

        cmd = [
            sys.executable, str(SCRIPTS["tensor"]),
            "-g", *input_args,
            "-k", *[str(k) for k in k_list],
            "-b", str(bin_size),
            "--fusion", fusion,
            "--compare-single",
            "-o", output_base, "-v",
        ]

        # Silhouette scoring — pass labels CSV if provided
        if labels_col and Path(labels_col).exists():
            cmd += ["--labels-col", labels_col]

    # Append any custom --meta KEY=VALUE pairs
    if meta and isinstance(meta, dict):
        meta_args = [f"{k}={v}" for k, v in meta.items() if k and v]
        if meta_args:
            cmd += ["--meta"] + meta_args

    # Append genome name-map entries
    if name_map and isinstance(name_map, dict):
        map_args = [f"{k}={v}" for k, v in name_map.items() if k and v]
        if map_args:
            cmd += ["--name-map"] + map_args

    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [], "cmd": " ".join(cmd)}
    _save_jobs(jobs)
    _job_queues[job_id] = __import__("queue").Queue()
    __import__("threading").Thread(
        target=_run_job, args=(job_id, cmd, output_base), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/download-genome", methods=["POST"])
def download_genome():
    """Download genomes from NCBI using kali_downloader.py."""
    data = request.json or {}

    # Accept both single and list forms
    taxon      = data.get("taxon",  "").strip()
    taxid      = data.get("taxid",  "").strip()
    taxons     = data.get("taxons", [])   # list of taxon names
    taxids_l   = data.get("taxids", [])   # list of taxids
    accessions = data.get("accessions", [])
    limit      = min(int(data.get("limit", 100)), 500)
    complete   = data.get("complete", True)
    prefix     = data.get("prefix", "genome").strip() or "genome"

    # Merge single + list
    all_taxons = list(taxons) + ([taxon] if taxon and taxon not in taxons else [])
    all_taxids = list(taxids_l) + ([taxid] if taxid and taxid not in taxids_l else [])

    if not all_taxons and not all_taxids and not accessions:
        return jsonify({"error": "Select at least one taxon or provide accessions"}), 400
    if not SCRIPTS["downloader"].exists():
        return jsonify({"error": "kali_downloader.py not found"}), 500

    job_id  = str(__import__("uuid").uuid4())[:8]
    out_dir = str(GENOME_DOWNLOADS_DIR / job_id)

    cmd = [sys.executable, str(SCRIPTS["downloader"]),
           "-o", out_dir, "--limit", str(limit), "--prefix", prefix]

    if all_taxons:
        cmd += ["--taxon"] + all_taxons
    elif all_taxids:
        cmd += ["--taxid"] + all_taxids
    elif accessions:
        cmd += ["--accessions"] + accessions

    if not complete:
        cmd.append("--no-complete")

    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [],
                    "cmd": " ".join(cmd)}
    _save_jobs(jobs)
    _job_queues[job_id] = __import__("queue").Queue()
    __import__("threading").Thread(
        target=_run_job,
        args=(job_id, cmd, out_dir),
        daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/taxonomy-browse", methods=["POST"])
def taxonomy_browse():
    """Return child taxa for a given taxid using NCBI eutils."""
    import urllib.request, urllib.parse
    data  = request.json or {}
    txid  = data.get("txid", "").strip()
    level = data.get("level", "family")

    if not txid:
        return jsonify({"error": "No taxid provided"}), 400

    try:
        # Get children of this taxid
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=taxonomy&term=txid{txid}[Subtree]&retmax=200&retmode=json"
        )
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "KALI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            result = json.loads(r.read())

        child_ids = result.get("esearchresult", {}).get("idlist", [])
        count     = result.get("esearchresult", {}).get("count", "0")

        return jsonify({
            "txid":      txid,
            "count":     count,
            "child_ids": child_ids[:50],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ncbi-count", methods=["POST"])
def ncbi_count():
    """Count sequences available for a taxon."""
    import urllib.request, urllib.parse
    data    = request.json or {}
    taxon   = data.get("taxon","").strip()
    taxid   = data.get("taxid","").strip()
    complete= data.get("complete", True)

    if not taxon and not taxid:
        return jsonify({"error": "No query"}), 400

    try:
        if taxid:
            term = f"txid{taxid}[Organism:exp]"
        else:
            term = f"{taxon}[Organism]"
        if complete:
            term += " AND complete genome[Title]"

        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=nucleotide&term={urllib.parse.quote(term)}&retmax=0&retmode=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent":"KALI/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            result = json.loads(r.read())
        count = result.get("esearchresult",{}).get("count","?")
        return jsonify({"count": count, "term": term})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-mash", methods=["POST"])
def run_mash():
    """Run KALI vs MASH comparison using kali_mash_comparison.py."""
    import shutil
    data       = request.json or {}
    folder     = data.get("folder", "").strip()
    k_list     = data.get("k", [3, 4, 5])
    bin_size = data.get("bin_size", 500)
    sketch_size= data.get("sketch_size", 10000)
    mash_k     = data.get("mash_k", 21)
    metadata   = data.get("metadata", "").strip()

    if not folder:
        return jsonify({"error": "No genome folder specified"}), 400
    if not Path(folder).exists():
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    # Check MASH is installed
    if not shutil.which("mash"):
        return jsonify({"error": "MASH not installed. Run: brew install mash"}), 500

    if not SCRIPTS["mash_compare"].exists():
        return jsonify({"error": "kali_mash_comparison.py not found"}), 500

    job_id   = str(__import__("uuid").uuid4())[:8]
    out_base = str(OUTPUT_DIR / f"mash_comparison_{job_id}")
    mash_tsv = out_base + "_mash_distances.tsv"
    mash_msh = out_base + "_sketch"

    def run_mash_job(job_id, folder, k_list, bin_size,
                     sketch_size, mash_k, metadata, out_base,
                     mash_tsv, mash_msh):
        jobs = _load_jobs()
        jobs[job_id]["status"] = "running"
        _save_jobs(jobs)
        q = _job_queues[job_id]

        def emit(msg):
            q.put(("log", msg))
            jobs = _load_jobs()
            jobs[job_id]["log"].append(msg)
            _save_jobs(jobs)

        try:
            import glob as globmod
            exts = (".fasta",".fa",".fna",".fas",".ffn")
            genome_files = sorted(
                f for f in Path(folder).iterdir()
                if f.suffix.lower() in exts
            )
            if not genome_files:
                emit("No FASTA files found in folder")
                jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
                q.put(("error", "no fasta files")); return

            emit(f"Found {len(genome_files)} genome files")

            # Step 1: MASH sketch
            emit("Step 1: Sketching with MASH...")
            sketch_cmd = ["mash", "sketch", "-o", mash_msh,
                          "-k", str(mash_k), "-s", str(sketch_size)] +                          [str(f) for f in genome_files]
            r = subprocess.run(sketch_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                emit(f"MASH sketch failed: {r.stderr[:200]}")
                jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
                q.put(("error", "mash sketch failed")); return
            emit("  Sketch complete")

            # Step 2: MASH dist
            emit("Step 2: Computing MASH distances...")
            dist_cmd = ["mash", "dist",
                        mash_msh + ".msh", mash_msh + ".msh"]
            with open(mash_tsv, "w") as f:
                r = subprocess.run(dist_cmd, stdout=f, stderr=subprocess.PIPE)
            if r.returncode != 0:
                emit(f"MASH dist failed: {r.stderr.decode()[:200]}")
                jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
                q.put(("error", "mash dist failed")); return
            emit(f"  MASH distances saved")

            # Step 3: KALI distances using pykali_hash.py (same as main pipeline)
            emit("Step 3: Computing KALI distances with pykali_hash.py...")
            hash_out = out_base + "_hash"
            comp_cmd = [
                sys.executable, str(SCRIPTS["hash"]),
                "-g", folder,
                "-k"] + [str(k) for k in k_list] + [
                "-b", str(bin_size),
                "-d", "cosine",
                "-r", "mean",
                "--combine",
                "-o", hash_out,
            ]
            r = subprocess.run(comp_cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                emit(f"pykali_hash.py failed: {r.stderr[:300]}")
                jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
                q.put(("error", "pykali_hash.py failed")); return

            for line in r.stdout.splitlines():
                if line.strip(): emit(line)

            # Find combined KALI matrix
            k_range = f"k{k_list[0]}-k{k_list[-1]}"
            kali_csv = Path(f"{hash_out}_{k_range}_b{bin_size}_combined.csv")
            if not kali_csv.exists():
                # fallback to first k matrix
                kali_csv = Path(f"{hash_out}_k{k_list[0]}_b{bin_size}.csv")

            if not kali_csv.exists():
                emit("Could not find KALI output matrix")
                jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
                q.put(("error", "kali matrix not found")); return

            # Step 4: Run comparison script to correlate KALI vs MASH
            emit("Step 4: Correlating KALI vs MASH...")
            comp_cmd2 = [
                sys.executable, str(SCRIPTS["mash_compare"]),
                "--mash",    mash_tsv,
                "--kali",    str(kali_csv),
                "-o", out_base,
            ]
            if metadata and Path(metadata).exists():
                comp_cmd2 += ["--metadata", metadata]

            r2 = subprocess.run(comp_cmd2, capture_output=True, text=True, timeout=600)
            for line in r2.stdout.splitlines():
                if line.strip(): emit(line)

            # Collect output files
            saved = []
            for ext in ("_comparison.csv","_scatter.png","_summary.json"):
                p = Path(out_base + ext)
                if p.exists(): saved.append(p.name)

            emit(f"Done — {len(saved)} output files")
            jobs = _load_jobs()
            jobs[job_id]["status"] = "done"
            jobs[job_id]["files"]  = saved
            _save_jobs(jobs)
            q.put(("done", json.dumps({"files": saved})))

        except Exception as e:
            emit(f"Error: {e}")
            jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
            q.put(("error", str(e)))

    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [],
                    "cmd": f"mash comparison {folder}"}
    _save_jobs(jobs)
    _job_queues[job_id] = __import__("queue").Queue()
    __import__("threading").Thread(
        target=run_mash_job,
        args=(job_id, folder, k_list, bin_size,
              sketch_size, mash_k, metadata, out_base,
              mash_tsv, mash_msh),
        daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/run-mantel-plot", methods=["POST"])
def run_mantel_plot():
    """Run Mantel test and generate figure from existing KALI and MASH matrices."""
    data      = request.json or {}
    kali_file = data.get("kali_file", "").strip()
    mash_file = data.get("mash_file", "").strip()

    if not kali_file or not mash_file:
        return jsonify({"error": "kali_file and mash_file are required"}), 400

    if not SCRIPTS["plot_mantel"].exists():
        return jsonify({"error": "plot_mantel.py not found in app folder"}), 500

    # Resolve full paths from OUTPUT_DIR
    kali_path = OUTPUT_DIR / kali_file
    mash_path = OUTPUT_DIR / mash_file
    ani_path  = OUTPUT_DIR / "fastani_results_txt.matrix"

    if not kali_path.exists():
        return jsonify({"error": f"KALI matrix not found: {kali_file}"}), 400
    if not mash_path.exists():
        return jsonify({"error": f"MASH matrix not found: {mash_file}"}), 400
    if not ani_path.exists():
        return jsonify({"error": "fastani_results_txt.matrix not found in outputs — "
                                 "run fastANI first and place the matrix file in the outputs folder"}), 400

    job_id  = str(__import__("uuid").uuid4())[:8]
    out_png = str(OUTPUT_DIR / f"mantel_plot_{job_id}.png")

    def run_mantel_job(job_id, kali_path, mash_path, ani_path, out_png):
        jobs = _load_jobs()
        jobs[job_id]["status"] = "running"
        _save_jobs(jobs)
        q = _job_queues[job_id]

        def emit(msg):
            q.put(("log", msg))
            jobs = _load_jobs()
            jobs[job_id]["log"].append(msg)
            _save_jobs(jobs)

        try:
            emit("Running Mantel test (9,999 permutations)...")
            cmd = [
                sys.executable, str(SCRIPTS["plot_mantel"]),
                "--kali", str(kali_path),
                "--mash", str(mash_path),
                "--ani",  str(ani_path),
                "--out",  out_png,
                "--n_perm", "9999",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            for line in r.stdout.splitlines():
                if line.strip(): emit(line)

            if r.returncode != 0:
                emit(f"Error: {r.stderr[:300]}")
                jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
                q.put(("error", "plot_mantel failed")); return

            saved = []
            if Path(out_png).exists():
                saved.append(Path(out_png).name)

            emit(f"Done — Mantel figure saved")
            jobs = _load_jobs()
            jobs[job_id]["status"] = "done"
            jobs[job_id]["files"]  = saved
            _save_jobs(jobs)
            q.put(("done", json.dumps({"files": saved})))

        except Exception as e:
            emit(f"Error: {e}")
            jobs = _load_jobs(); jobs[job_id]["status"] = "error"; _save_jobs(jobs)
            q.put(("error", str(e)))

    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [],
                    "cmd": f"mantel plot"}
    _save_jobs(jobs)
    _job_queues[job_id] = __import__("queue").Queue()
    __import__("threading").Thread(
        target=run_mantel_job,
        args=(job_id, kali_path, mash_path, ani_path, out_png),
        daemon=True
    ).start()
    return jsonify({"job_id": job_id})
def prepare_metadata_route():
    """Fix NCBI metadata files for classifier training."""
    data       = request.json or {}
    inputs     = data.get("inputs", [])    # list of file paths
    output     = data.get("output", "")
    label_col  = data.get("label_col", "Genus")
    min_length = data.get("min_length", 0)

    if not inputs:
        return jsonify({"error": "No input files specified"}), 400
    if not output:
        output = str(OUTPUT_DIR / "prepared_labels.csv")

    if not SCRIPTS["prepare_metadata"].exists():
        return jsonify({"error": "prepare_metadata.py not found"}), 500

    job_id = str(__import__("uuid").uuid4())[:8]
    cmd = [
        sys.executable, str(SCRIPTS["prepare_metadata"]),
        "--input",      *inputs,
        "--output",     output,
        "--label-col",  label_col,
        "--min-length", str(min_length),
        "-v",
    ]
    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [], "cmd": " ".join(cmd)}
    _save_jobs(jobs)
    _job_queues[job_id] = __import__("queue").Queue()
    __import__("threading").Thread(
        target=_run_job, args=(job_id, cmd, output), daemon=True
    ).start()
    return jsonify({"job_id": job_id, "output": output})


# ── AI conversation database ─────────────────────────────────────────────────
AI_DB = Path(__file__).parent / "kali_ai_conversations.db"

def init_ai_db():
    """Create AI conversation tables if they do not exist."""
    conn = sqlite3.connect(AI_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id        TEXT PRIMARY KEY,
            title     TEXT,
            job_type  TEXT,
            created   TEXT,
            updated   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role            TEXT,
            content         TEXT,
            timestamp       TEXT,
            model           TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()

init_ai_db()

# ── AI interpretation: Groq / OpenRouter / Ollama + KALI file context ─────────
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_API_URL     = "http://localhost:11434/api/chat"

KALI_SYSTEM = (
    "You are KALI-AI, the built-in AI assistant for Paul Alemoh's KALI genomic analysis platform. "
    "Identity rules: your name is KALI-AI; the KALI platform was created by Paul Alemoh; never say KALI is hypothetical; "
    "never replace your product identity with the underlying model provider identity such as Mistral, Meta, OpenAI, Groq, or OpenRouter. "
    "If asked who built or created this platform, say Paul Alemoh created KALI and you assist inside it. "
    "If asked what powers you, you may say an external LLM backend powers the chat, but your role remains KALI-AI. "
    "Personality: sound warm, confident, practical, and research-focused. Speak like a bioinformatics co-pilot, not a generic chatbot. "
    "Always use the provided job logs, output files, selected files, and user message before answering. "
    "KALI analyzes genomes using alignment-free k-mer and motif-based comparison, hashed features, cosine distance, "
    "phylogenetic tree construction, HGT/anomaly detection, outbreak monitoring, tensor fusion, Mash comparison, and Random Forest classification. "
    "Explain results in clear biological language, stay grounded in supplied files, and suggest the next useful analysis step. "
    "When evidence is missing, say exactly what file/result is needed. Be concise, structured, and practical."
)

TEXT_EXTENSIONS = {
    ".txt", ".csv", ".tsv", ".json", ".log", ".nwk", ".tree", ".html", ".md",
    ".fasta", ".fa", ".fna", ".fas", ".ffn", ".fastq", ".fq"
}

MAX_FILE_CHARS = 12000
MAX_CONTEXT_CHARS = 50000

OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
]


def _safe_read_text(path: Path, max_chars: int = MAX_FILE_CHARS) -> str:
    """Read text-like KALI output safely with a character cap."""
    try:
        if not path.exists() or not path.is_file():
            return f"[Missing file: {path.name}]"
        if path.suffix.lower() not in TEXT_EXTENSIONS and not path.name.lower().endswith(".gz"):
            return f"[Skipped non-text file: {path.name}]"
        if path.name.lower().endswith(".gz"):
            return f"[Skipped compressed file: {path.name}]"
        text = path.read_text(errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[Truncated: showing first {max_chars} characters of {path.name}]"
        return text
    except Exception as e:
        return f"[Could not read {path.name}: {e}]"


def _resolve_output_file(filename: str) -> Path | None:
    """Resolve output/download/upload filenames without allowing unsafe paths."""
    if not filename:
        return None
    p = Path(filename)
    search_dirs = [OUTPUT_DIR, UPLOAD_DIR, GENOME_DOWNLOADS_DIR, MODELS_DIR]

    if p.is_absolute() and p.exists():
        return p

    safe_name = Path(filename).name
    for folder in search_dirs:
        candidate = folder / safe_name
        if candidate.exists():
            return candidate
        if folder.exists():
            matches = list(folder.rglob(safe_name))
            if matches:
                return matches[0]
    return None


def _build_ai_context(data: dict) -> str:
    """Collect user-provided results, job logs, and selected files for AI grounding."""
    parts = []

    results = (data.get("results") or "").strip()
    if results:
        parts.append("## User-provided results\n" + results[:MAX_FILE_CHARS])

    job_id = (data.get("job_id") or "").strip()
    if job_id:
        jobs = _load_jobs()
        job = jobs.get(job_id)
        if job:
            parts.append(
                "## KALI job context\n"
                f"job_id: {job_id}\n"
                f"status: {job.get('status')}\n"
                f"command: {job.get('cmd', '')}\n"
                "recent_log:\n" + "\n".join(job.get("log", [])[-200:])
            )
            for f in job.get("files", [])[:8]:
                path = _resolve_output_file(f)
                if path:
                    parts.append(f"## Output file: {path.name}\n" + _safe_read_text(path))
        else:
            parts.append(f"## KALI job context\nNo job found for job_id={job_id}")

    # User may pass files, output_files, selected_files, or attachments from the frontend.
    requested_files = []
    for key in ("files", "output_files", "selected_files"):
        val = data.get(key, [])
        if isinstance(val, str):
            requested_files.append(val)
        elif isinstance(val, list):
            requested_files.extend([str(x) for x in val])

    # If the frontend forgot to send a file/job_id, still give KALI-AI useful context
    # by loading the newest small result files from outputs/. This prevents generic
    # "I cannot access files" answers when a user asks about the latest result.
    if not requested_files and not job_id and not results:
        preferred_exts = {".json", ".csv", ".txt", ".tsv", ".html", ".nwk"}
        try:
            recent = sorted(
                [f for f in OUTPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in preferred_exts],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )[:5]
            requested_files.extend([f.name for f in recent])
            if recent:
                parts.append("## Auto-loaded recent KALI output files\n" + ", ".join(f.name for f in recent))
        except Exception as e:
            parts.append(f"## Auto-load note\nCould not list recent outputs: {e}")

    seen = set()
    for filename in requested_files[:10]:
        if filename in seen:
            continue
        seen.add(filename)
        path = _resolve_output_file(filename)
        if path:
            parts.append(f"## Selected file: {path.name}\n" + _safe_read_text(path))
        else:
            parts.append(f"## Selected file missing\nCould not find: {filename}")

    context = "\n\n".join(parts).strip()
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + f"\n\n[Context truncated to {MAX_CONTEXT_CHARS} characters]"
    return context


def _call_chat_backend(backend: str, api_key: str, model: str, messages: list, data: dict) -> str:
    """Call the selected LLM backend and return assistant text."""
    backend = (backend or "groq").lower().strip()

    if backend == "ollama":
        ollama_model = data.get("ollama_model") or model or "llama3"
        payload = json.dumps({
            "model": ollama_model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read())
        return out.get("message", {}).get("content", "")

    if not api_key:
        raise ValueError(f"{backend.capitalize()} API key required")

    if backend == "openrouter":
        api_url = OPENROUTER_API_URL
        fallback_models = data.get("openrouter_fallback_models") or OPENROUTER_FREE_MODELS
        if isinstance(fallback_models, str):
            fallback_models = [fallback_models]

        models_to_try = []
        if model:
            models_to_try.append(model)
        for m in fallback_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": data.get("referer", "http://localhost:8080"),
            "X-Title": data.get("app_title", "KALI AI"),
        }

        last_error = None
        for selected_model in models_to_try:
            payload = json.dumps({
                "model": selected_model,
                "messages": messages,
                "temperature": float(data.get("temperature", 0.3)),
                "max_tokens": int(data.get("max_tokens", 1200)),
            }).encode("utf-8")
            req = urllib.request.Request(api_url, data=payload, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    out = json.loads(resp.read())
                text = out["choices"][0]["message"]["content"]
                return text
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:800]
                last_error = urllib.error.HTTPError(e.url, e.code, body, e.headers, None)
                if e.code == 401:
                    raise last_error
                if e.code in (403, 404, 429, 502, 503):
                    continue
                raise last_error

        if last_error:
            raise last_error
        raise ValueError("No OpenRouter model available to try.")

    elif backend == "groq":
        api_url = GROQ_API_URL
        selected_model = model or "llama-3.3-70b-versatile"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        raise ValueError("Unsupported backend. Use 'openrouter', 'groq', or 'ollama'.")

    payload = json.dumps({
        "model": selected_model,
        "messages": messages,
        "temperature": float(data.get("temperature", 0.3)),
        "max_tokens": int(data.get("max_tokens", 1200)),
    }).encode("utf-8")

    req = urllib.request.Request(api_url, data=payload, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=90) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"]


@app.route("/ai-model-options", methods=["GET"])
def ai_model_options():
    """Return model options for the frontend dropdown."""
    return jsonify({
        "openrouter": OPENROUTER_FREE_MODELS,
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "ollama": ["llama3", "mistral", "phi3"],
        "default_backend": "openrouter",
        "default_openrouter_model": OPENROUTER_FREE_MODELS[0],
    })


def _enforce_kali_personality(text: str) -> str:
    """Correct common identity drift from free/open models."""
    if not text:
        return text
    lower = text.lower()
    bad_identity = (
        "kali is a hypothetical" in lower
        or "created for the purpose of this conversation" in lower
        or "i am an ai language model developed by mistral" in lower
        or "i was created by a team of engineers" in lower
    )
    if bad_identity:
        return (
            "I am KALI-AI, the AI assistant inside Paul Alemoh's KALI genomic platform. "
            "KALI is designed to support alignment-free genome analysis, k-mer/motif comparison, "
            "cosine-distance matrices, phylogenetic trees, HGT/anomaly detection, outbreak monitoring, "
            "tensor fusion, Mash comparison, and Random Forest classification.\n\n"
            "Ask me about a KALI job, output file, tree, classifier result, HGT region, or distance matrix, "
            "and I will interpret it in biological terms."
        )
    return text


@app.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    """AI interpretation endpoint with memory, OpenRouter/Groq/Ollama support, and KALI file context."""
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    data = request.json or {}
    job_type = data.get("job_type", "analysis")
    backend = (data.get("backend") or "groq").lower().strip()
    api_key = (data.get("api_key") or "").strip()
    model = (data.get("model") or "").strip()
    conversation_id = (data.get("conversation_id") or "").strip()
    user_message = (data.get("user_message") or "").strip()

    if backend not in {"groq", "openrouter", "ollama"}:
        return jsonify({"error": "Unsupported backend. Use 'openrouter', 'groq', or 'ollama'."}), 400
    if backend != "ollama" and not api_key:
        return jsonify({"error": f"{backend.capitalize()} API key required"}), 400

    now = __import__("datetime").datetime.utcnow().isoformat()
    context = _build_ai_context(data)

    conn = sqlite3.connect(AI_DB)
    try:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())[:8]
            title = f"{job_type.capitalize()} — {now[:10]}"
            conn.execute(
                "INSERT INTO conversations VALUES (?,?,?,?,?)",
                (conversation_id, title, job_type, now, now),
            )
            if context:
                conn.execute(
                    "INSERT INTO messages (conversation_id,role,content,timestamp,model) VALUES (?,?,?,?,?)",
                    (conversation_id, "user", "KALI analysis context:\n\n" + context, now, model or backend),
                )
            conn.commit()

        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        history = [{"role": r, "content": c} for r, c in rows]

        if user_message:
            prompt = user_message
            # Always attach current file/job context to the current question.
            # This is the key fix: KALI-AI should answer from the selected result file,
            # even inside an older conversation.
            if context:
                prompt = f"KALI analysis context:\n\n{context}\n\nUser question:\n{user_message}"
            history.append({"role": "user", "content": prompt})
            conn.execute(
                "INSERT INTO messages (conversation_id,role,content,timestamp,model) VALUES (?,?,?,?,?)",
                (conversation_id, "user", prompt, now, model or backend),
            )
            conn.execute("UPDATE conversations SET updated=? WHERE id=?", (now, conversation_id))
            conn.commit()

        if not history:
            return jsonify({"error": "No message content to send. Provide user_message, results, job_id, or files."}), 400

        messages = [{"role": "system", "content": KALI_SYSTEM}] + history[-20:]

        try:
            text = _call_chat_backend(backend, api_key, model, messages, data)
            text = _enforce_kali_personality(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:800]
            if backend == "openrouter":
                if e.code == 400:
                    msg = f"OpenRouter bad request (400). Model name is likely wrong. Body: {body[:300]}"
                elif e.code == 401:
                    msg = "OpenRouter rejected the API key. Check that the key starts with sk-or and is active."
                elif e.code == 403:
                    msg = "OpenRouter blocked the request. Check HTTP-Referer/X-Title headers, model access, or account credits."
                elif e.code == 404:
                    msg = "OpenRouter model not found. Check the model name."
                elif e.code == 429:
                    msg = "OpenRouter rate limit reached. Try again later or choose another model."
                else:
                    msg = f"OpenRouter API error {e.code}: {body}"
            elif backend == "groq":
                if e.code == 401:
                    msg = "Groq rejected the API key. Check console.groq.com."
                elif e.code == 403:
                    msg = "Groq blocked the request. Try another model or backend."
                elif e.code == 429:
                    msg = "Groq rate limit reached. Try again later."
                else:
                    msg = f"Groq API error {e.code}: {body}"
            else:
                msg = f"Ollama/LLM error {e.code}: {body}"
            return jsonify({"error": msg, "details": body}), e.code
        except urllib.error.URLError as e:
            if backend == "ollama":
                return jsonify({"error": "Ollama is not running. Start it, then run: ollama pull llama3"}), 503
            return jsonify({"error": f"Network/API connection error: {e}"}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        conn.execute(
            "INSERT INTO messages (conversation_id,role,content,timestamp,model) VALUES (?,?,?,?,?)",
            (conversation_id, "assistant", text, now, model or backend),
        )
        conn.execute("UPDATE conversations SET updated=? WHERE id=?", (now, conversation_id))
        conn.commit()

        return jsonify({
            "interpretation": text,
            "model": model,
            "conversation_id": conversation_id,
            "backend": backend,
            "context_used": bool(context),
        })
    finally:
        conn.close()


@app.route("/ai-conversations", methods=["GET"])
def ai_conversations():
    """List all past AI conversations."""
    conn  = sqlite3.connect(AI_DB)
    rows  = conn.execute(
        "SELECT id, title, job_type, created, updated FROM conversations ORDER BY updated DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([
        {"id": r[0], "title": r[1], "job_type": r[2], "created": r[3], "updated": r[4]}
        for r in rows
    ])


@app.route("/ai-conversation/<conv_id>", methods=["GET"])
def ai_conversation(conv_id):
    """Load full message history for a conversation."""
    conn = sqlite3.connect(AI_DB)
    msgs = conn.execute(
        "SELECT role, content, timestamp, model FROM messages WHERE conversation_id=? ORDER BY id",
        (conv_id,)
    ).fetchall()
    meta = conn.execute(
        "SELECT id, title, job_type, created, updated FROM conversations WHERE id=?",
        (conv_id,)
    ).fetchone()
    conn.close()
    if not meta:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({
        "id": meta[0], "title": meta[1], "job_type": meta[2],
        "created": meta[3], "updated": meta[4],
        "messages": [{"role": r[0], "content": r[1], "timestamp": r[2], "model": r[3]} for r in msgs]
    })


@app.route("/ai-conversation/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    """Delete a conversation and all its messages."""
    conn = sqlite3.connect(AI_DB)
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": conv_id})


@app.route("/save-setting", methods=["POST"])
def save_setting():
    """Save a user setting to the local database."""
    data  = request.json or {}
    key   = data.get("key",   "").strip()
    value = data.get("value", "").strip()
    if not key:
        return jsonify({"error": "No key provided"}), 400
    conn = sqlite3.connect(AI_DB)
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()
    return jsonify({"saved": key})


@app.route("/load-setting/<key>", methods=["GET"])
def load_setting(key):
    """Load a user setting from the local database."""
    conn = sqlite3.connect(AI_DB)
    row  = conn.execute(
        "SELECT value FROM settings WHERE key=?", (key,)
    ).fetchone()
    conn.close()
    if row:
        return jsonify({"key": key, "value": row[0]})
    return jsonify({"key": key, "value": None})



@app.route("/run-KALI-hash", methods=["POST"])
def run_KALI_hash():
    """
    Run KALI_hash.py — k-mer spacing-histogram distance.

    Captures HOW k-mers are distributed across the genome (gap distribution),
    not just how often they appear. Matches pykali at r > 0.999.

    Request body (JSON):
        files          list of uploaded filenames (or use folder)
        folder         local folder path
        k              k-mer size(s) — single int or list e.g. [3,4,5]
        bins           histogram bins per k-mer (default: 200)
        metric         cosine | euclidean | jaccard (default: cosine)
        combine        bool — produce combined matrix when multiple k (default: false)
        combine_method average | concat (default: average)
        verbose        bool (default: true)
    """
    data           = request.json or {}
    files          = data.get("files",          [])
    folder         = data.get("folder",         "").strip()
    k_raw          = data.get("k",              4)
    bins           = data.get("bins",           200)
    metric         = data.get("metric",         "cosine")
    combine        = data.get("combine",        False)
    combine_method = data.get("combine_method", "average")
    verbose        = data.get("verbose",        True)

    if not files and not folder:
        return jsonify({"error": "No genome files or folder specified"}), 400

    if "KALI_hash" not in SCRIPTS or not SCRIPTS["KALI_hash"].exists():
        return jsonify({"error": "KALI_hash.py not found in app folder"}), 500

    # k can be a single int or a list
    if isinstance(k_raw, list):
        k_list = [int(k) for k in k_raw]
    else:
        k_list = [int(k_raw)]

    job_id      = str(uuid.uuid4())[:8]
    output_base = str(OUTPUT_DIR / f"spacing_{job_id}")
    genome_arg  = folder if folder else str(UPLOAD_DIR)

    cmd = [
        sys.executable, str(SCRIPTS["KALI_hash"]),
        "-g",     genome_arg,
        "-k",     *[str(k) for k in k_list],
        "--bins", str(bins),
        "-d",     metric,
        "-o",     output_base,
    ]
    if combine and len(k_list) > 1:
        cmd += ["--combine", "--combine-method", combine_method]
    if verbose:
        cmd.append("-v")

    jobs = _load_jobs()
    jobs[job_id] = {"status": "queued", "log": [], "files": [], "cmd": " ".join(cmd)}
    _save_jobs(jobs)

    _job_queues[job_id] = queue.Queue()
    threading.Thread(
        target=_run_job, args=(job_id, cmd, output_base), daemon=True
    ).start()

    return jsonify({"job_id": job_id})

@app.errorhandler(405)
def method_not_allowed(e):
    """Return JSON for 405 instead of HTML so the browser shows a useful message."""
    return jsonify({
        "error": "Method not allowed",
        "hint": "This endpoint only accepts POST requests from the KALI interface"
    }), 405


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  KALI Web UI  →  http://localhost:{port}\n")
    print("  For production:  gunicorn -w 4 -b 0.0.0.0:8080 app:app\n")
    app.run(debug=True, threaded=True, port=port)
