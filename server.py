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
    Universally links user responses back to the previous bot turn unless
    a semantic topic pivot is detected using the model's Encoder.
    """
    if not history:
        return user_text, "current"
        
    user_clean = user_text.strip()
    user_lower = user_clean.lower().rstrip('.!?')
    
    # Extract the text content of the last turn
    last_msg = history[-1]
    last_turn_text = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
    
    # Strip basic greetings and fillers from the context string to avoid pollution
    context_clean = re.sub(r'^(yo|hey|hi|hello|greetings|please)\s*,?\s*', '', last_turn_text.strip(), flags=re.IGNORECASE)
    context_clean = context_clean.rstrip('.!?')
    
    if not context_clean:
        return user_text, "current"

    # -------------------------------------------------------------------------
    # NEW: SEMANTIC TOPIC PIVOT DETECTION (Reusing model.pth Encoder)
    # -------------------------------------------------------------------------
    def get_sentence_embedding(text):
        token_indices = sentence_to_indices(normalize_contractions(text), ck["vocab"], ck.get("w2i"))
        if not token_indices:
            return None
        tensor = torch.tensor([token_indices], dtype=torch.long)
        with torch.no_grad():
            # Encoder output shape: (1, seq_len, embed_dim)
            memory = model.encoder(tensor)
            # Mean pool along seq_len dimension to get a single (embed_dim) vector
            return memory.mean(dim=1).squeeze(0)

    user_emb = get_sentence_embedding(user_clean)
    ctx_emb = get_sentence_embedding(context_clean)

    if user_emb is not None and ctx_emb is not None:
        # Calculate cosine similarity between current phrase and context
        similarity = torch.nn.functional.cosine_similarity(user_emb, ctx_emb, dim=0).item()
        
        # NOTE: Adjust this threshold (0.25 - 0.35) based on your specific training weights
        if similarity < 0.28:
            # Topic has drifted significantly. Do not fuse history context!
            return user_text, "current"
    # -------------------------------------------------------------------------

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

    inverted_context = invert_pov(context_clean)
    
    yes_variants = {"yes", "yeah", "yep", "yup", "sure", "correct", "ok", "okay"}
    no_variants = {"no", "nope", "nah", "not"}

    # CATEGORY 1: SHORT CONFIRMATIONS (e.g., "yes")
    if user_lower in yes_variants:
        return f"{user_clean}, {inverted_context.lower()}", "history"

    # CATEGORY 2: SHORT NEGATIONS (e.g., "no")
    if user_lower in no_variants:
        if "enough" in inverted_context.lower():
            return f"{user_clean}, it's not {inverted_context.lower()}", "history"
        return f"{user_clean}, it is not the case that {inverted_context.lower()}", "history"

    # CATEGORY 3: SINGLE WORD SLOT-FILLING (e.g., "wednesday")
    if len(user_clean.split()) == 1:
        q_lead_ins = {"what", "when", "where", "which", "who", "why", "how", "day", "time", "date"}
        filtered_words = [w for w in inverted_context.split() if w.lower() not in q_lead_ins]
        
        aux_verb = "is"
        for v in ["is", "are", "was", "were", "has", "have", "do", "does", "did"]:
            if v in [w.lower() for w in filtered_words]:
                aux_verb = v
                filtered_words = [w for w in filtered_words if w.lower() != v]
                break
                
        remaining_core = " ".join(filtered_words).strip()
        if remaining_core:
            return f"{user_clean} {aux_verb} {remaining_core.lower()}", "history"

    # CATEGORY 4: UNIVERSAL STRUCTURAL LINKING (e.g., "why would i?", "what's on your mind?")
    question_starters = {
        "why", "how", "what", "where", "who", "when", "which",
        "would", "could", "should", "can", "will", "shall",
        "is", "are", "am", "was", "were", "do", "does", "did"
    }
    
    user_words = user_lower.split()
    is_question = user_clean.endswith('?') or (user_words and user_words[0] in question_starters)
    base_text = user_clean.rstrip('?.!')

    if is_question:
        return f"{base_text} when you mentioned \"{inverted_context.lower()}\"?", "history"
    else:
        return f"{base_text} regarding \"{inverted_context.lower()}\"", "history"

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