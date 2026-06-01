# server.py — seq2seq chatbot backend
from flask import Flask, jsonify, request, send_file, redirect
from flask_cors import CORS
import torch, json, random, os, threading, subprocess, time, sys, zipfile, re
import requests as http_requests

from model import Encoder, Decoder, Seq2Seq
from utils import (
    sentence_to_indices, indices_to_sentence,
    build_vocab, PAD_IDX, SOS_IDX, EOS_IDX,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


# ── Discord Webhook ───────────────────────────────────────────────────────────
# Set this environment variable to your Discord webhook URL
DISCORD_WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1511103886675542167/4eMVxXWQ6j3jIYzyczADxkPPsU-9kqhgRY_DYaROQQA8HssKFI_gml9jYo-voh9QRrho",
)

def send_discord_backup(label: str, data_json_path: str = "data.json"):
    """Send the current data.json to Discord as a file attachment."""
    if not DISCORD_WEBHOOK_URL:
        log_activity("Discord webhook not configured — skipping backup.")
        return
    try:
        with open(data_json_path, "rb") as f:
            file_bytes = f.read()
        http_requests.post(
            DISCORD_WEBHOOK_URL,
            data={"content": f"📦 **data.json backup** — {label}"},
            files={"file": ("data.json", file_bytes, "application/json")},
            timeout=15,
        )
        log_activity(f"Discord backup sent: {label}")
    except Exception as e:
        log_activity(f"Discord backup failed: {e}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load():
    ck = torch.load("model.pth", weights_only=False)

    encoder = Encoder(
        vocab_size   = ck["vocab_size"],
        embed_dim    = ck["embed_dim"],
        hidden_size  = ck["hidden_size"],
        num_layers   = ck["num_layers"],
        dropout      = ck["dropout"],
    )
    decoder = Decoder(
        vocab_size   = ck["vocab_size"],
        embed_dim    = ck["embed_dim"],
        hidden_size  = ck["hidden_size"],
        num_layers   = ck["num_layers"],
        dropout      = ck["dropout"],
    )
    model = Seq2Seq(
        encoder, decoder,
        sos_idx = ck["sos_idx"],
        eos_idx = ck["eos_idx"],
        pad_idx = ck["pad_idx"],
    )
    encoder.load_state_dict(ck["encoder_state"])
    decoder.load_state_dict(ck["decoder_state"])
    model.eval()
    return model, ck


model, ck = load()
retrain_status   = "idle"
last_retrain_log = ""
activity_logs    = []


# ── Logging ───────────────────────────────────────────────────────────────────

def log_activity(message, ip=None):
    global activity_logs
    from datetime import datetime
    ts    = datetime.now().strftime("%H:%M:%S")
    msg   = f"[{ip}] {message}" if ip else message
    entry = {"time": ts, "msg": msg}
    activity_logs.append(entry)
    if len(activity_logs) > 100:
        activity_logs.pop(0)
    print(f"[LOG] {msg}")


def get_client_ip():
    """Return the real client IP, respecting X-Forwarded-For if behind a proxy."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    return request.remote_addr


# ── Auth ──────────────────────────────────────────────────────────────────────

ACCESS_PASSWORD   = "ithadbetterbetonight"
authenticated_ips = set()

@app.route("/")
def index():
    client_ip   = request.remote_addr
    provided_pw = request.args.get("pw")
    if provided_pw == ACCESS_PASSWORD:
        authenticated_ips.add(client_ip)
        return redirect("/")
    if client_ip in authenticated_ips:
        return send_file("index.html")
    return '''
        <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
        <body style="background:#000;color:#7c6bff;display:flex;flex-direction:column;
                     align-items:center;justify-content:center;min-height:100vh;
                     font-family:sans-serif;margin:0;padding:20px;box-sizing:border-box;">
        <h1 style="margin-bottom:20px;letter-spacing:-1px;font-size:clamp(24px,7vw,32px);">
            Wyntr<span style="color:#fff;">AI</span> Access</h1>
        <form method="GET" style="display:flex;flex-direction:column;gap:14px;
                                   align-items:center;width:100%;max-width:320px;">
            <p style="line-height:1.5;color:#6b6b8a;margin:0 0 10px 0;
                      text-align:center;font-size:14px;">
                Authorization is required to prevent malicious requests.</p>
            <input type="password" name="pw" placeholder="Enter Password" autofocus
                   style="padding:14px;border-radius:10px;border:1px solid #2f3336;
                          background:#080808;color:#fff;outline:none;width:100%;
                          text-align:center;font-size:16px;box-sizing:border-box;">
            <button type="submit"
                    style="padding:14px 24px;background:#7c6bff;color:#fff;border:none;
                           border-radius:10px;cursor:pointer;font-weight:bold;
                           width:100%;font-size:14px;box-sizing:border-box;">
                Unlock Interface</button>
        </form></body>
    ''', 401

@app.route("/index.css")
def serve_css():
    return send_file("index.css")


# ── Core predict endpoint ─────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    global model, ck

    sentence    = request.json["sentence"]
    vocab       = ck["vocab"]
    temperature = float(request.json.get("temperature", 0.8))
    beam_width  = int(request.json.get("beam_width",    1))
    max_len     = int(request.json.get("max_len", 40))

    src_indices = sentence_to_indices(sentence, vocab)
    if not src_indices:
        return jsonify({"response": "I didn't catch that.", "tag": "unknown", "confidence": 0.0})

    src_tensor = torch.tensor([src_indices], dtype=torch.long)

    with torch.no_grad():
        output_indices = model.generate(
            src_tensor,
            max_len     = max_len,
            temperature = temperature,
            beam_width  = beam_width,
        )

    response   = indices_to_sentence(output_indices, vocab)
    confidence = 1.0
    tag        = "generated"

    if not response.strip():
        response = "hmm..."

    # Intentionally no log here — only data merges and retrains are logged

    return jsonify({
        "tag":        tag,
        "confidence": confidence,
        "response":   response,
        "probs":      [],
        "activations": {},
        "all_words":  vocab,
        "tags":       [],
    })


# ── Merge Dataset ─────────────────────────────────────────────────────────────

@app.route("/merge-dataset", methods=["POST"])
def merge_dataset():
    client_ip = get_client_ip()
    payload   = request.json or {}
    new_convs = payload.get("conversations")

    if not isinstance(new_convs, list) or len(new_convs) == 0:
        return jsonify({"error": "conversations list is required and must not be empty"}), 400

    # Validate each entry
    for i, c in enumerate(new_convs):
        if not isinstance(c.get("input"), str):
            return jsonify({"error": f"Entry {i}: 'input' must be a string"}), 400
        if not isinstance(c.get("replies"), list) or len(c["replies"]) == 0:
            return jsonify({"error": f"Entry {i}: 'replies' must be a non-empty array"}), 400

    try:
        # 1. Send existing data.json to Discord before touching anything
        send_discord_backup(
            label=f"pre-merge backup — requested by {client_ip}",
        )

        # 2. Load, merge, write
        with open("data.json") as f:
            data = json.load(f)

        if "conversations" not in data:
            data["conversations"] = []

        existing_inputs = {c["input"].lower().strip() for c in data["conversations"]}
        added = 0

        for conv in new_convs:
            key = conv["input"].lower().strip()
            if key in existing_inputs:
                # Append new replies to existing entry
                for existing in data["conversations"]:
                    if existing["input"].lower().strip() == key:
                        if "replies" not in existing:
                            existing["replies"] = [existing.pop("reply")] if "reply" in existing else []
                        for reply in conv["replies"]:
                            if reply not in existing["replies"]:
                                existing["replies"].append(reply)
                        break
            else:
                data["conversations"].append({
                    "input":   conv["input"],
                    "replies": conv["replies"],
                })
                existing_inputs.add(key)
                added += 1

        with open("data.json", "w") as f:
            json.dump(data, f, indent=2)

        log_activity(
            f"Dataset merged — {added} new entries added ({len(new_convs)} submitted)",
            ip=client_ip,
        )
        return jsonify({"status": "ok", "added": added})

    except Exception as e:
        log_activity(f"Merge error: {e}", ip=client_ip)
        return jsonify({"error": str(e)}), 500


# ── Learn / Teach / Retrain ───────────────────────────────────────────────────

@app.route("/learn", methods=["POST", "OPTIONS"])
def learn():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    payload = request.json or {}
    pattern = payload.get("pattern")
    tag     = payload.get("tag")
    if not pattern or not tag:
        return jsonify({"error": "pattern and tag required"}), 400
    try:
        with open("data.json") as f:
            data = json.load(f)
        tag_found = False
        for intent in data["intents"]:
            if intent["tag"] != tag:
                continue
            tag_found = True
            if "pairs" in intent:
                existing = [p["pattern"] for p in intent["pairs"]]
                if pattern not in existing:
                    intent["pairs"].append({"pattern": pattern, "responses": []})
            else:
                if pattern not in intent.get("patterns", []):
                    intent.setdefault("patterns", []).append(pattern)
            break
        if not tag_found:
            data["intents"].append({"tag": tag, "patterns": [pattern], "responses": []})
        with open("data.json", "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/teach", methods=["POST"])
def teach():
    payload       = request.json or {}
    pattern       = payload.get("pattern")
    response_text = payload.get("response")
    
    if not pattern or not response_text:
        return jsonify({"error": "pattern and response required"}), 400
        
    with open("data.json") as f:
        data = json.load(f)
        
    if "conversations" not in data:
        data["conversations"] = []
        
    matched_conv = None
    for conv in data["conversations"]:
        if conv.get("input", "").lower().strip() == pattern.lower().strip():
            matched_conv = conv
            break
            
    if matched_conv:
        if "replies" not in matched_conv:
            old_reply = matched_conv.pop("reply", None)
            matched_conv["replies"] = [old_reply] if old_reply else []
        if response_text not in matched_conv["replies"]:
            matched_conv["replies"].append(response_text)
    else:
        data["conversations"].append({
            "input": pattern,
            "replies": [response_text]
        })
    
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
        
    return jsonify({"status": "ok"})


def _retrain_and_reload():
    global retrain_status, last_retrain_log, model, ck
    retrain_status   = "running"
    last_retrain_log = ""

    ip = _retrain_and_reload._trigger_ip if hasattr(_retrain_and_reload, "_trigger_ip") else "unknown"

    try:
        p = subprocess.Popen(
            [sys.executable, "-u", "train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines = []
        # Stream stdout line by line so we can log epoch progress live
        for line in p.stdout:
            line = line.rstrip()
            stdout_lines.append(line)
            last_retrain_log = "\n".join(stdout_lines)
            if "Epoch" in line:
                log_activity(f"Retraining — {line.strip()}", ip=ip)

        p.wait()
        stderr_out = p.stderr.read()
        if stderr_out:
            last_retrain_log += "\n" + stderr_out

        if p.returncode == 0:
            try:
                time.sleep(0.3)
                model, ck      = load()
                retrain_status = "done"
                log_activity("Model retrain completed successfully", ip=ip)
            except Exception as e:
                retrain_status    = "failed"
                last_retrain_log += f"\nReload failed: {e}"
                log_activity(f"Model reload failed after retrain: {e}", ip=ip)
        else:
            retrain_status = "failed"
            err_lines = [l.strip() for l in stderr_out.splitlines() if l.strip()]
            short_err = err_lines[-1] if err_lines else "unknown error"
            log_activity(f"Model retrain failed — {short_err}", ip=ip)

    except Exception as e:
        retrain_status    = "failed"
        last_retrain_log += f"\nException: {e}"
        log_activity(f"Retrain exception: {e}", ip=ip)


@app.route("/retrain", methods=["POST"])
def retrain():
    global retrain_status
    client_ip = get_client_ip()
    if retrain_status == "running":
        return jsonify({"status": "already_running"}), 409
    log_activity("Model retrain initiated", ip=client_ip)
    _retrain_and_reload._trigger_ip = client_ip
    t = threading.Thread(target=_retrain_and_reload, daemon=True)
    t.start()
    return jsonify({"status": "started"})


# ── Utility routes ────────────────────────────────────────────────────────────

@app.route("/download-backup")
def download_backup():
    with zipfile.ZipFile("backup.zip", "w") as z:
        z.write("model.pth")
        z.write("data.json")
    return send_file("backup.zip", as_attachment=True, download_name="wyntr_backup.zip")


@app.route("/check-pattern", methods=["POST"])
def check_pattern():
    payload = request.json
    pattern = payload.get("pattern", "")
    tag     = payload.get("tag", "")
    try:
        with open("data.json") as f:
            file_data = json.load(f)
        for intent in file_data.get("intents", []):
            if intent.get("tag") == tag:
                patterns = intent.get("patterns", [])
                if any(p.lower() == pattern.lower() for p in patterns):
                    return jsonify({"exists": True})
        return jsonify({"exists": False})
    except Exception as e:
        return jsonify({"exists": False, "error": str(e)})


@app.route("/stats")
def stats():
    with open("data.json") as f:
        data = json.load(f)
    vocab   = ck.get("vocab", [])
    samples = sum(
        len(i.get("patterns", [])) + sum(1 for _ in i.get("pairs", []))
        for i in data.get("intents", [])
    )
    return jsonify({
        "vocab_size":     len(vocab),
        "num_tags":       len(set(i["tag"] for i in data.get("intents", []))),
        "samples":        samples,
        "retrain_status": retrain_status,
        "retrain_log":    last_retrain_log[:4000],
    })


@app.route("/logs", methods=["GET", "OPTIONS"])
def logs():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    return jsonify({"logs": activity_logs[-50:] if activity_logs else []})


@app.route("/weights")
def weights():
    vocab = ck.get("vocab", [])
    return jsonify({
        "vocab":      vocab,
        "vocab_size": len(vocab),
        "model_type": "seq2seq",
    })


if __name__ == "__main__":
    log_activity("Server started (seq2seq mode)")
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)