from flask import Flask, jsonify, request, send_file, session
from flask_cors import CORS
import torch, os

from model import Encoder, Decoder, Seq2Seq
from utils import sentence_to_indices, indices_to_sentence, normalize_contractions

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-prod")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def load():
    ck = torch.load("model.pth", weights_only=False, map_location="cpu")

    vocab_size      = ck["vocab_size"]
    embed_dim       = ck["embed_dim"]
    hidden_size     = ck["hidden_size"]
    num_layers      = ck["num_layers"]
    dropout         = ck["dropout"]
    dim_feedforward = ck.get("dim_feedforward", 4 * embed_dim)

    encoder = Encoder(
        vocab_size      = vocab_size,
        embed_dim       = embed_dim,
        hidden_size     = hidden_size,
        num_layers      = num_layers,
        dropout         = dropout,
        dim_feedforward = dim_feedforward,
    )
    decoder = Decoder(
        vocab_size      = vocab_size,
        embed_dim       = embed_dim,
        hidden_size     = hidden_size,
        num_layers      = num_layers,
        dropout         = dropout,
        dim_feedforward = dim_feedforward,
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

# ---------------------------------------------------------------------------
# In-memory history store keyed by session id.
# Each entry: {"user": str, "bot": str}
# We keep only the last 1 turn per session (sufficient for the fallback logic).
# ---------------------------------------------------------------------------
_history: dict[str, list[dict]] = {}

MIN_TOKENS = 2  # fewer tokens than this → sentence is "too short / uninformative"

def _is_informative(text: str) -> bool:
    """Return True if the text carries enough tokens to be useful context."""
    return len(text.split()) >= MIN_TOKENS

def _resolve_input(current: str, history: list[dict]) -> str:
    """
    Context-window fallback logic:
      1. Use current prompt if informative.
      2. Else prepend previous user turn if informative.
      3. Else prepend previous bot turn if informative.
      4. Else use current prompt as-is.
    """
    if _is_informative(current):
        return current

    if history:
        last = history[-1]
        prev_user = last.get("user", "")
        prev_bot  = last.get("bot", "")

        if _is_informative(prev_user):
            return f"{prev_user} {current}".strip()

        if _is_informative(prev_bot):
            return f"{prev_bot} {current}".strip()

    # Nothing useful in history — fall back to the raw current prompt
    return current

# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model, ck

    # --- session id (sent by client, or fall back to Flask session) ---
    data       = request.json
    session_id = data.get("session_id") or session.get("sid") or "default"
    session["sid"] = session_id

    raw_sentence = data["sentence"]
    sentence     = normalize_contractions(raw_sentence)
    vocab        = ck["vocab"]
    temperature  = float(data.get("temperature", 0.7))
    beam_width   = int(data.get("beam_width",    3))
    max_len      = int(data.get("max_len", 50))

    # --- resolve input with context fallback ---
    history        = _history.get(session_id, [])
    resolved       = _resolve_input(sentence, history)
    used_context   = resolved != sentence  # flag for debugging / frontend

    src_indices = sentence_to_indices(resolved, ck["vocab"], ck.get("w2i"))

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

    response = indices_to_sentence(output_indices, vocab)

    if not response.strip():
        response = "I couldn't generate anything."

    # --- update history (keep only last 1 turn) ---
    _history[session_id] = [{"user": sentence, "bot": response}]

    return jsonify({
        "tag":          "generated",
        "confidence":   1.0,
        "response":     response,
        "used_context": used_context,   # optional — remove if frontend doesn't need it
        "probs":        [],
        "activations":  {},
        "all_words":    vocab,
        "tags":         [],
    })

@app.route("/reset", methods=["POST"])
def reset_history():
    """Clear conversation history for the current session."""
    data       = request.json or {}
    session_id = data.get("session_id") or session.get("sid") or "default"
    _history.pop(session_id, None)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)