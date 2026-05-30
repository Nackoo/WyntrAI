# server.py — run alongside index.html
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import torch, json, random, os
from model import ChatNet
from utils import bag_of_words
import threading
import subprocess
import time
import sys
import zipfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def load():
    ck = torch.load("model.pth", weights_only=True)
    net = ChatNet(ck["input_size"], ck["hidden_size"], ck["output_size"])
    net.load_state_dict(ck["model_state"])
    net.eval()
    return net, ck

net, ck = load()
retrain_status = 'idle'  # 'idle' | 'running' | 'done' | 'failed'
last_retrain_log = ''
activity_logs = []  # stores recent activity logs

def log_activity(message):
    """Add an activity log entry."""
    global activity_logs
    from datetime import datetime
    timestamp = datetime.now().strftime('%H:%M:%S')
    entry = {'time': timestamp, 'msg': message}
    activity_logs.append(entry)
    if len(activity_logs) > 100:  # keep last 100 logs
        activity_logs.pop(0)
    print(f"[LOG] {message}")

# ── Serve the frontend ──────────────────────────────────────────
@app.route("/")
def index():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global net, ck
    sentence = request.json["sentence"]
    bow = bag_of_words(sentence, ck["all_words"])
    import torch as t
    x = t.tensor(bow, dtype=t.float32)
    with t.no_grad():
        acts = net.get_layer_activations(x)
    probs = acts["output_probs"]
    idx = probs.index(max(probs))
    tag = ck["tags"][idx]
    confidence = max(probs)
    with open("data.json") as f:
        data = json.load(f)
    response = next(
        (random.choice(i["responses"]) for i in data["intents"] if i["tag"] == tag),
        "I don't understand yet."
    )
    log_activity(f"Predicted: '{sentence[:30]}...' -> {tag} ({confidence*100:.0f}%)")
    return jsonify({"tag": tag, "confidence": confidence, "probs": probs,
                    "activations": acts, "response": response,
                    "all_words": ck["all_words"], "tags": ck["tags"]})


@app.route('/learn', methods=['POST', 'OPTIONS'])
def learn():
    """Auto-learn: add user message as a pattern to the predicted intent tag."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    payload = request.json or {}
    pattern = payload.get('pattern')
    tag = payload.get('tag')
    if not pattern or not tag:
        return jsonify({'error': 'pattern and tag required'}), 400
    try:
        with open('data.json') as f:
            data = json.load(f)
        # find or create tag
        found = False
        for intent in data['intents']:
            if intent['tag'] == tag:
                if pattern not in intent.get('patterns', []):
                    intent.setdefault('patterns', []).append(pattern)
                    found = True
                    log_activity(f"Auto-learned: '{pattern[:30]}...' for tag '{tag}'")
                break
        if not found and len(data['intents']) > 0:
            log_activity(f"Tag '{tag}' not found for auto-learn")
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({'status': 'ok'})
    except Exception as e:
        log_activity(f"Learn error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/teach', methods=['POST'])
def teach():
    payload = request.json or {}
    pattern = payload.get('pattern')
    response_text = payload.get('response')
    tag = payload.get('tag')
    if not pattern or not response_text:
        return jsonify({'error': 'pattern and response required'}), 400
    with open('data.json') as f:
        data = json.load(f)
    # find or create tag
    if tag:
        for intent in data['intents']:
            if intent['tag'] == tag:
                intent.setdefault('patterns', []).append(pattern)
                intent.setdefault('responses', []).append(response_text)
                break
        else:
            data['intents'].append({'tag': tag, 'patterns': [pattern], 'responses': [response_text]})
    else:
        # append to a generic 'misc' tag if exists or create new
        if data['intents']:
            data['intents'][0].setdefault('patterns', []).append(pattern)
            data['intents'][0].setdefault('responses', []).append(response_text)
        else:
            data['intents'].append({'tag': 'new', 'patterns': [pattern], 'responses': [response_text]})
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)
    log_activity(f"Manual teach: '{pattern[:30]}...' for tag '{tag}'")
    return jsonify({'status': 'ok'})


def _retrain_and_reload():
    # run train.py synchronously and reload model when done
    global retrain_status, last_retrain_log, net, ck
    retrain_status = 'running'
    last_retrain_log = ''
    try:
        p = subprocess.run([sys.executable, 'train.py'], check=False, capture_output=True, text=True)
        last_retrain_log = (p.stdout or '') + '\n' + (p.stderr or '')
        if p.returncode == 0:
            # try reloading model
            try:
                time.sleep(0.3)
                net, ck = load()
                retrain_status = 'done'
            except Exception as e:
                retrain_status = 'failed'
                last_retrain_log += f"\nReload failed: {e}"
        else:
            retrain_status = 'failed'
    except Exception as e:
        retrain_status = 'failed'
        last_retrain_log += f"\nException: {e}"


@app.route('/retrain', methods=['POST'])
def retrain():
    # start background retrain
    global retrain_status
    if retrain_status == 'running':
        return jsonify({'status': 'already_running'}), 409
    t = threading.Thread(target=_retrain_and_reload, daemon=True)
    t.start()
    return jsonify({'status': 'started'})
    
@app.route('/download-backup')
def download_backup():
    with zipfile.ZipFile('backup.zip', 'w') as z:
        z.write('model.pth')
        z.write('data.json')

    return send_file(
        'backup.zip',
        as_attachment=True,
        download_name='wyntr_backup.zip'
    )

@app.route('/check-pattern', methods=['POST'])
def check_pattern():
    data = request.json
    pattern = data.get('pattern', '')
    tag = data.get('tag', '')

    try:
        with open('data.json') as f:
            file_data = json.load(f)

        # Check if pattern already exists in the specified tag
        for intent in file_data.get('intents', []):
            if intent.get('tag') == tag:
                patterns = intent.get('patterns', [])
                # Case-insensitive check
                if any(p.lower() == pattern.lower() for p in patterns):
                    return jsonify({'exists': True})

        return jsonify({'exists': False})
    except Exception as e:
        log_activity(f"Check pattern error: {str(e)}")
        return jsonify({'exists': False, 'error': str(e)})

@app.route('/stats')
def stats():
    with open('data.json') as f:
        data = json.load(f)
    vocab = ck.get('all_words') if ck else []
    tags = ck.get('tags') if ck else []
    samples = sum(len(i.get('patterns', [])) for i in data.get('intents', []))
    return jsonify({'vocab_size': len(vocab), 'num_tags': len(tags), 'samples': samples, 'retrain_status': retrain_status, 'retrain_log': last_retrain_log[:4000]})

@app.route('/logs', methods=['GET', 'OPTIONS'])
def logs():
    """Return recent activity logs."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    return jsonify({'logs': activity_logs[-50:] if activity_logs else []})

@app.route("/weights")
def weights():
    sd = {k: v.tolist() for k, v in net.state_dict().items()}
    return jsonify({**sd, "all_words": ck["all_words"], "tags": ck["tags"]})

if __name__ == "__main__":
    log_activity("Server started")
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)