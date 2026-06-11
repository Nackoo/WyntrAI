from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import torch, os, re

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

def enrich_user_input(user_text, history):
    """
    Dynamically references prior conversation context to enrich brief or 
    uninformative responses into grammatically sound sentences.
    """
    if not history:
        return user_text, "current"
        
    user_clean = user_text.strip().lower()
    user_words = user_clean.rstrip('.!?').split()
    
    # Heuristic 1: If it's an explicit question, it's informative. Skip fusion!
    if user_text.strip().endswith('?'):
        return user_text, "current"
        
    # Heuristic 2: If it already contains a subject/verb combo, it's a complete statement.
    structural_verbs = {
        "is", "are", "am", "was", "were", "can", "could", "will", "would", 
        "do", "does", "did", "have", "has", "had", "go", "get", "like", "want"
    }
    if len(user_words) >= 3 and any(w in structural_verbs for w in user_words):
        return user_text, "current"
        
    # Extract the text content of the last turn
    last_msg = history[-1]
    last_turn_text = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
    context_clean = last_turn_text.strip()
    
    # Pronoun POV transformation map
    pronoun_map = {
        "your": "my", "you": "i", "yours": "mine", "yourself": "myself",
        "my": "your", "i": "you", "mine": "yours", "myself": "yourself",
        "u": "i", "ur": "my"
    }
    
    def invert_pov(text_str):
        words = text_str.split()
        inverted = []
        for w in words:
            clean_w = re.sub(r'[^a-zA-Z\']', '', w).lower()
            if clean_w in pronoun_map:
                inv = pronoun_map[clean_w]
                if w[0].isupper():
                    inv = inv.capitalize()
                inverted.append(inv)
            else:
                inverted.append(w)
        return " ".join(inverted)

    # Contextual variants classification
    yes_variants = {"yes", "yeah", "yep", "yup", "sure", "correct", "ok", "okay"}
    no_variants = {"no", "nope", "nah", "not"}

    # 1. HANDLE CONFIRMATIONS (e.g., Bot: "Is it raining?" -> User: "yes")
    if user_clean.rstrip('.!?') in yes_variants:
        clean_ctx = re.sub(r'^(yo|hey|hi|hello|please)\s+', '', context_clean, flags=re.IGNORECASE).rstrip('?')
        return f"{user_text.strip()}, {invert_pov(clean_ctx).lower()}", "history"

    # 2. HANDLE NEGATIONS (e.g., Bot: "yo enough for today" -> User: "no")
    elif user_clean.rstrip('.!?') in no_variants:
        clean_ctx = re.sub(r'^(yo|hey|hi|hello|please)\s+', '', context_clean, flags=re.IGNORECASE).rstrip('?')
        inv_ctx = invert_pov(clean_ctx).lower()
        
        if "enough" in inv_ctx:
            return f"{user_text.strip()}, it's not {inv_ctx}", "history"
        return f"{user_text.strip()}, it is not the case that {inv_ctx}", "history"

    # 3. HANDLE UNINFORMATIVE SLOT-FILLING (e.g., Bot: "What day is your exam?" -> User: "wednesday")
    else:
        # Only slot-fill if the bot actually posed a question
        if not last_turn_text.strip().endswith('?'):
            return user_text, "current"
            
        # Strip interrogative prefix tokens to isolate the predicate
        q_lead_ins = {"what", "when", "where", "which", "who", "why", "how", "day", "time", "date"}
        filtered_words = [w for w in context_clean.rstrip('?').split() if w.lower() not in q_lead_ins]
        
        inverted_core = invert_pov(" ".join(filtered_words))
        if inverted_core:
            return f"{user_text.strip()} {inverted_core.lower()}", "history"

    return user_text, "current"

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model, ck

    raw_sentence = request.json.get("sentence", "")
    history      = request.json.get("history", [])
    temperature  = float(request.json.get("temperature", 0.7))
    beam_width   = int(request.json.get("beam_width", 3))
    max_len      = int(request.json.get("max_len", 50))
    vocab        = ck["vocab"]

    # Dynamic evaluation and structural context fusion
    enriched_sentence, ctx_source = enrich_user_input(raw_sentence, history)

    sentence = normalize_contractions(enriched_sentence)
    src_indices = sentence_to_indices(sentence, ck["vocab"], ck.get("w2i"))

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

    return jsonify({
        "tag":        "generated",
        "confidence": 1.0,
        "response":   response,
        "probs":      [],
        "activations": {},
        "all_words":  vocab,
        "tags":       [],
        "ctx_source": ctx_source,
        "enriched":   enriched_sentence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)