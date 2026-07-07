from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
import torch, os, re, json

from model import Encoder, Decoder, Seq2Seq
from utils import sentence_to_indices, indices_to_sentence, normalize_contractions

app = Flask(__name__)
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


def determine_context_routing(user_text, history):
    """
    Evaluates whether the input requires historical context.
    Returns: (context_to_include, context_type)
    """
    if not history:
        return None, "current"
        
    user_clean = user_text.strip()
    user_lower = user_clean.lower().rstrip('.!?')
    user_words = user_lower.split()
    
    def get_turn_text(idx):
        if abs(idx) <= len(history):
            msg = history[idx]
            return msg.get("content", "") if isinstance(msg, dict) else str(msg)
        return ""

    ctx_1 = get_turn_text(-1)  # Bot's last response
    
    # Simple, high-precision dependency triggers (No grammar mutations needed!)
    question_starters = {"why", "how", "what", "where", "who", "when", "which", "would", "could", "should", "can"}
    is_question = user_clean.endswith('?') or (user_words and user_words[0] in question_starters)
    
    # Short words like "why", "how", "yes", "no" absolutely need context
    if len(user_words) <= 3:
        return ctx_1, "history"
        
    # Checking for explicit continuation signs or pronouns pointing backward
    clean_user_words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in user_words]
    backward_anchors = {"that", "this", "it", "those", "these", "them"}
    
    if any(word in backward_anchors for word in clean_user_words) and is_question:
        return ctx_1, "history"
        
    return None, "current"
    

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model, ck

    raw_sentence = request.json.get("sentence", "")
    history      = request.json.get("history", [])
    temperature  = float(request.json.get("temperature", 0.85))
    beam_width   = int(request.json.get("beam_width", 1))
    max_len      = int(request.json.get("max_len", 50))
    vocab        = ck["vocab"]
    w2i          = ck.get("w2i")

    # Determine if we need to track history
    context_sentence, ctx_source = determine_context_routing(raw_sentence, history)

    # 1. Tokenize primary user prompt
    sentence_clean = normalize_contractions(raw_sentence)
    src_indices = sentence_to_indices(sentence_clean, vocab, w2i)

    # 2. If context is related, cleanly attach it using our structural SEP token
    if context_sentence and ctx_source == "history":
        context_clean = normalize_contractions(context_sentence)
        context_indices = sentence_to_indices(context_clean, vocab, w2i)
        
        # Structure: [User Tokens] + [SEP] + [Context Tokens]
        src_indices = src_indices + [4] + context_indices  # 4 is our SEP_IDX

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

    return jsonify({
        "tag":        "generated",
        "confidence": 1.0,
        "response":   response,
        "ctx_source": ctx_source,
        "enriched":   indices_to_sentence(src_indices, vocab) # Debug look
    })

@app.route("/predict_stream", methods=["POST"])
def predict_stream():
    """
    Same generation as /predict, but streams one Server-Sent Event per
    decoding step, exposing the model's real per-step token probabilities
    (its 'raw mathematical vector thoughts') as it decodes — plus a final
    event carrying the finished response.
    """
    global model, ck

    raw_sentence = request.json.get("sentence", "")
    history      = request.json.get("history", [])
    temperature  = float(request.json.get("temperature", 0.85))
    beam_width   = int(request.json.get("beam_width", 1))
    max_len      = int(request.json.get("max_len", 50))
    vocab        = ck["vocab"]
    w2i          = ck.get("w2i")

    context_sentence, ctx_source = determine_context_routing(raw_sentence, history)

    sentence_clean = normalize_contractions(raw_sentence)
    src_indices = sentence_to_indices(sentence_clean, vocab, w2i)

    if context_sentence and ctx_source == "history":
        context_clean = normalize_contractions(context_sentence)
        context_indices = sentence_to_indices(context_clean, vocab, w2i)
        src_indices = src_indices + [4] + context_indices  # 4 is our SEP_IDX

    def word(idx):
        return vocab[idx] if 0 <= idx < len(vocab) else "<UNK>"

    def event_stream():
        if not src_indices:
            payload = {
                "done": True,
                "tokens": [],
                "response": "I didn't catch that.",
                "ctx_source": ctx_source,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            return

        src_tensor = torch.tensor([src_indices], dtype=torch.long)

        for event in model.generate_stream(
            src_tensor, max_len=max_len, temperature=temperature, beam_width=beam_width
        ):
            if event.get("done"):
                output_indices = event["tokens"]
                event["response"] = indices_to_sentence(output_indices, vocab)
                event["ctx_source"] = ctx_source
                event["enriched"] = indices_to_sentence(src_indices, vocab)
                yield f"data: {json.dumps(event)}\n\n"
                break

            if "candidates" in event:
                for c in event["candidates"]:
                    c["word"] = word(c["idx"])
            if "beams" in event:
                for b in event["beams"]:
                    for c in b["top"]:
                        c["word"] = word(c["idx"])

            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)