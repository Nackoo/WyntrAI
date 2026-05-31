# server.py — run alongside index.html
from flask import Flask, jsonify, request, send_file, redirect
from flask_cors import CORS
import torch, json, random, os
from model import ChatNet
from utils import bag_of_words
import threading
import subprocess
import time
import sys
import zipfile
import requests
from bs4 import BeautifulSoup
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

def get_unknown_words(sentence, vocabulary):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())

    stopwords = {
        "what","who","when","where","why","how",
        "is","are","was","were",
        "the","a","an",
        "of","to","in","on","for",
        "do","does","did",
        "can","could","would","should",
        "my","your","their","our",
        "his","her","i","you",
        "we","they","he","she","it"
    }

    unknown = []

    for word in words:

        if len(word) < 3:
            continue

        if word in stopwords:
            continue

        if word not in vocabulary:
            unknown.append(word)

    return unknown

def web_search(query):
    try:
        q = query.lower().strip()

        for prefix in [
            "what is ",
            "what's ",
            "who's ",
            "explain ",
            "meaning of ",
            "what are ",
            "who is ",
            "who are ",
            "tell me about ",
            "define ",
            "how do i"
        ]:
            if q.startswith(prefix):
                q = q[len(prefix):]
                break

        q = q.rstrip("?.!,")

        headers = {
            "User-Agent": "WyntrAI/1.0 (Educational Chatbot)"
        }

        search_url = (
            "https://en.wikipedia.org/w/api.php"
            "?action=opensearch"
            "&limit=1"
            "&namespace=0"
            "&format=json"
            "&search="
            + requests.utils.quote(q)
        )

        search_response = requests.get(
            search_url,
            timeout=10,
            headers=headers
        )

        if search_response.status_code != 200:
            log_activity(f"Wikipedia search failed: {search_response.status_code}")
            return None

        results = search_response.json()

        if len(results[1]) == 0:
            log_activity("No Wikipedia results found")
            return None

        title = results[1][0]

        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title)
        )

        summary_response = requests.get(
            summary_url,
            timeout=10,
            headers=headers
        )

        if summary_response.status_code != 200:
            log_activity(f"Wikipedia summary failed: {summary_response.status_code}")
            return None

        data = summary_response.json()

        if data.get("extract"):
            return data["extract"]

        return None

    except Exception as e:
        log_activity(f"Search error: {e}")
        return None

    return None

def contains_unknown_words(sentence, vocabulary):
    """
    Detect words that do not exist in the model vocabulary.
    """
    words = re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())

    unknown = []

    for word in words:
        if word not in vocabulary:
            unknown.append(word)

    return len(unknown) > 0, unknown

# ── Serve the frontend ──────────────────────────────────────────
ACCESS_PASSWORD = "ithadbetterbetonight" 
authenticated_ips = set()

@app.route("/")
def index():
    client_ip = request.remote_addr

    # 1. Check if they just submitted the password via query param
    provided_pw = request.args.get("pw")
    if provided_pw == ACCESS_PASSWORD:
        authenticated_ips.add(client_ip)
        return redirect("/")  # Redirect to clear the password from the URL bar

    # 2. Check if this computer is already authenticated in memory
    if client_ip in authenticated_ips:
        return send_file("index.html")

    # 3. Otherwise, return a simple login prompt instead of the app
    return '''
        <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
        <body style="background:#000;color:#7c6bff;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;font-family:sans-serif;margin:0;padding:20px;box-sizing:border-box;">
        <h1 style="margin-bottom:20px;letter-spacing:-1px;font-size:clamp(24px, 7vw, 32px);">Wyntr<span style="color:#fff;">AI</span> Access</h1>
        <form method="GET" style="display:flex;flex-direction:column;gap:14px;align-items:center;width:100%;max-width:320px;">
            <p style="line-height:1.5;color:#6b6b8a;margin:0 0 10px 0;text-align:center;font-size:14px;">Authorization is required to prevent malicious requests that might exhaust server resources or ruin the AI model.</p>
            <input type="password" name="pw" placeholder="Enter Password" autofocus style="padding:14px;border-radius:10px;border:1px solid #2f3336;background:#080808;color:#fff;outline:none;width:100%;text-align:center;font-size:16px;box-sizing:border-box;">
            <button type="submit" style="padding:14px 24px;background:#7c6bff;color:#fff;border:none;border-radius:10px;cursor:pointer;font-weight:bold;width:100%;font-size:14px;box-sizing:border-box;">Unlock Interface</button>
        </form>
        </body>
    ''', 401

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
        (
            random.choice(i["responses"])
            for i in data["intents"]
            if i["tag"] == tag
        ),
        "I don't understand yet."
    )

    # --------------------------------------------------
    # Internet Search Logic
    # --------------------------------------------------

    search_triggers = [
        "what is",
        "what are",
        "what was",
        "what were",

        "who is",
        "who are",
        "who was",

        "when is",
        "when was",

        "where is",
        "where are",

        "why is",
        "why are",
        "why does",
        "why do",

        "how is",
        "how are",
        "how do",
        "how does",
        "how can",
        "how to",

        "define",
        "meaning of",
        "tell me about",
        "explain",

        "difference between",
        "compare",
        "versus",
        "vs"
    ]

    lower_sentence = sentence.lower()

    is_factual_query = any(
        trigger in lower_sentence
        for trigger in search_triggers
    )

    unknown_words = get_unknown_words(
        sentence,
        ck["all_words"]
    )

    question_words = (
        "what",
        "who",
        "when",
        "where",
        "why",
        "how"
    )

    is_question = lower_sentence.startswith(question_words)

    should_search = (
        is_question and
        (
            len(unknown_words) > 0
            or confidence < 0.95
            or tag == "confused"
        )
    )

    if should_search:

        log_activity(
            f"Web search triggered. "
            f"Confidence={confidence:.2f}, "
            f"Tag={tag}, "
            f"Unknown={unknown_words}, "
            f"Question={is_question}"
        )

        search_result = web_search(sentence)

        log_activity(f"Search result: {repr(search_result)}")

        if search_result:
            response = search_result
            tag = "internet_search"

    log_activity(
        f"Predicted: '{sentence[:30]}...' "
        f"-> {tag} ({confidence*100:.0f}%)"
    )

    return jsonify({
        "tag": tag,
        "confidence": confidence,
        "probs": probs,
        "activations": acts,
        "response": response,
        "all_words": ck["all_words"],
        "tags": ck["tags"]
    })


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
        tag_found = False

        for intent in data['intents']:
            if intent['tag'] == tag:
                tag_found = True

                if pattern not in intent.get('patterns', []):
                    intent.setdefault('patterns', []).append(pattern)
                    log_activity(f"Auto-learned: '{pattern[:30]}...' for tag '{tag}'")

                break

        if not tag_found:
            data['intents'].append({
                'tag': tag,
                'patterns': [pattern],
                'responses': []
            })

            log_activity(f"Created new tag '{tag}' with pattern '{pattern[:30]}...'")
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