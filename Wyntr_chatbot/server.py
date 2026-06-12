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
    Universally links user responses back to history.
    Traverses multi-turn structures and uses strict context-dependency routing
    to ensure long new topics pass through untouched, while long follow-ups fuse perfectly.
    """
    if not history:
        return user_text, "current"
        
    user_clean = user_text.strip()
    user_lower = user_clean.lower().rstrip('.!?')
    user_words = user_lower.split()
    
    def get_turn_text(idx):
        if abs(idx) <= len(history):
            msg = history[idx]
            txt = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            txt = re.sub(r'^(yo|hey|hi|hello|greetings|please)\s*,?\s*', '', txt.strip(), flags=re.IGNORECASE)
            return txt.rstrip('.!?').strip()
        return ""

    ctx_1 = get_turn_text(-1)  # 1 step away (Bot's last response)
    ctx_2 = get_turn_text(-2)  # 2 steps away (User's previous message)
    ctx_3 = get_turn_text(-3)  # 3 steps away (Bot's response before that)

    context_clean = ctx_1 if ctx_1 else "conversation"

    # Pronoun POV transformation map including contractions
    pronoun_map = {
        "your": "my", "you": "i", "yours": "mine", "yourself": "myself",
        "my": "your", "i": "you", "mine": "yours", "myself": "yourself",
        "u": "i", "ur": "my",
        "i'm": "you're", "you're": "i'm", "i've": "you've", "you've": "i've",
        "i'll": "you'll", "you'll": "i'll", "i'd": "you'd", "you'd": "i'd"
    }
    
    def invert_pov(text_str):
        words = text_str.split()
        inverted = []
        for w in words:
            clean_w = re.sub(r"[^a-zA-Z']", '', w).lower()
            if clean_w in pronoun_map:
                inv = pronoun_map[clean_w]
                if w[0].isupper():
                    inv = inv.capitalize()
                inverted.append(inv)
            else:
                inverted.append(w)
        return " ".join(inverted)

    # DYNAMIC LINGUISTIC TENSE SHIFTER
    def dynamic_past_tense(text):
        words = text.split()
        transformed = []
        irregular_past = {
            "going": "went", "doing": "did", "making": "made", 
            "thinking": "thought", "running": "ran", "eating": "ate",
            "seeing": "saw", "buying": "bought", "coming": "came",
            "having": "had", "getting": "got", "speaking": "spoke"
        }
        for w in words:
            clean = re.sub(r"[^a-zA-Z]", "", w).lower()
            if clean in irregular_past:
                w = re.sub(clean, irregular_past[clean], w, flags=re.IGNORECASE)
            elif clean.endswith("ing") and len(clean) > 4:
                stem = clean[:-3]
                past_form = stem[:-1] + "ied" if stem.endswith("y") else stem + "ed"
                w = re.sub(clean, past_form, w, flags=re.IGNORECASE)
            transformed.append(w)
        return " ".join(transformed)

    yes_variants = {"yes", "yeah", "yep", "yup", "sure", "correct", "ok", "okay", "indeed"}
    no_variants = {"no", "nope", "nah", "not"}

    # RULE A: MULTI-TURN ADMISSION TRACKING (e.g., "so you admit it")
    if user_lower.startswith("so you admit") or user_lower.startswith("you admit"):
        target_claim = ctx_1
        if ctx_1.lower() in yes_variants or len(ctx_1.split()) <= 1:
            if ctx_3:
                target_claim = ctx_3
                
        inverted_claim = invert_pov(target_claim)
        base_admit = re.sub(r'\s+(it|that)$', '', user_clean, flags=re.IGNORECASE)
        return f"{base_admit} {inverted_claim.lower()}", "history"

    # RULE B: MULTI-TURN VERB CONTEXT STITCHING (e.g., "you did?")
    if user_lower in {"you did", "you did?", "did you?", "did you"}:
        if ctx_1.lower().startswith(("no, but", "yes, but", "but")) and ctx_2:
            action_words = [w for w in ctx_2.lower().split() if w not in {"are", "you", "do", "did", "is", "can"}]
            if action_words:
                raw_action = " ".join(action_words)
                dynamic_action = dynamic_past_tense(raw_action)
                return f"{user_clean} {dynamic_action} {ctx_1.lower()}", "history"

    # Standard Conversational Pipelines
    inverted_context = invert_pov(context_clean)

    def convert_to_declarative(text):
        words = text.split()
        if len(words) < 2:
            return text
        first_w = words[0].lower()
        second_w = words[1].lower()
        helpers = {"do", "does", "did", "are", "is", "was", "were", "can", "could", "would", "should"}
        pronouns = {"i", "you", "it", "we", "they", "he", "she"}
        if first_w in helpers and second_w in pronouns:
            if first_w in {"are", "is", "was", "were"} and second_w == "i":
                return f"i am {' '.join(words[2:])}"
            return f"{words[1]} {words[0]} {' '.join(words[2:])}"
        return text

    declarative_context = convert_to_declarative(inverted_context)
    
    # CATEGORY 1: SHORT CONFIRMATIONS
    if user_lower in yes_variants:
        return f"{user_clean}, {declarative_context.lower()}", "history"

    if user_lower in no_variants:
        # Strip out leading filler agreement flags (like "yes", "ok") from the old context
        clean_decl = re.sub(r'^(yes|yeah|yep|yup|ok|okay|sure|indeed)\s*,?\s*', '', declarative_context, flags=re.IGNORECASE).strip()
        
        # Case A: Dynamic contraction negation ("it's cool" -> "it's not cool")
        if clean_decl.lower().startswith("it's "):
            negated = "it's not " + clean_decl[5:]
            return f"{user_clean}, {negated.lower()}", "history"
            
        # Case B: Standard declarative negation ("it is cool" -> "it is not cool")
        elif clean_decl.lower().startswith("it is "):
            negated = "it is not " + clean_decl[6:]
            return f"{user_clean}, {negated.lower()}", "history"
            
        # Case C: First person negation ("i am fine" -> "i am not fine")
        elif clean_decl.lower().startswith("i am "):
            negated = "i am not " + clean_decl[5:]
            return f"{user_clean}, {negated.lower()}", "history"
        
        # Fallback safety checks
        if "not" in clean_decl.lower() or "no" in clean_decl.lower():
            return f"{user_clean}, {clean_decl.lower()}", "history"
            
        return f"{user_clean}, it's not true that {clean_decl.lower()}", "history"

    # CATEGORY 3: SINGLE-WORD INTERROGATIVE FOLLOW-UPS
    if len(user_words) == 1 and user_lower in {"how", "why"}:
        ctx_words = inverted_context.lower().split()
        if ctx_words and ctx_words[0] == "you're":
            return f"{user_clean} are you {' '.join(ctx_words[1:])}?", "history"
        return f"{user_clean} so when you said \"{inverted_context.lower()}\"?", "history"

    # HIGH-PRECISION CONTEXT DEPENDENCY ENGINE (Category 4 Refinement)
    question_starters = {"why", "how", "what", "where", "who", "when", "which", "would", "could", "should", "can"}
    is_question = user_clean.endswith('?') or (user_words and user_words[0] in question_starters)
    base_text = user_clean.rstrip('?.!')

    # Extract alphanumeric lower tokens for precise filtering
    clean_user_words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in user_words]

    # Explicit phrases signaling follow-ups or meta-dialogue
    continuation_signals = {
        "tell me more", "explain why", "elaborate on", "why so", "in what way", 
        "are you sure", "prove it", "not really", "makes sense", "what do you mean"
    }
    has_continuation_signal = any(signal in user_lower for signal in continuation_signals)

    # Conversational speech feedback anchors
    feedback_markers = {
        "said", "mentioned", "stated", "claimed", "told", "asked", "implied",
        "disagree", "agree", "contradict", "conflict", "earlier", "previous", "before"
    }
    has_feedback_marker = any(word in feedback_markers for word in clean_user_words)
    
    # Question target evaluation
    has_why_how = "why" in clean_user_words or "how" in clean_user_words
    targets_bot_persona = any(p in clean_user_words for p in {"you", "your", "yours", "yourself"}) or "you're" in user_words

    # BASELINE STRUCTURAL ROUTING DECISION
    is_context_dependent = False

    if len(user_words) > 6:
        # Long inputs must meet strict criteria to prove they are tracking past context
        if has_feedback_marker:
            is_context_dependent = True
        elif has_why_how and targets_bot_persona:
            is_context_dependent = True
        elif "mean" in clean_user_words or "meaning" in clean_user_words:
            is_context_dependent = True
        elif has_continuation_signal:
            is_context_dependent = True
    else:
        # Short phrases (<= 6 words) use standard open-ended structural hooks
        backward_anchors = {"that", "this", "it", "those", "these", "them", "there", "here"}
        has_backward_anchor = any(word in backward_anchors for word in clean_user_words)
        is_context_dependent = has_backward_anchor or has_continuation_signal or targets_bot_persona or is_question

    # EXECUTE ROUTING STRATEGY
    if not is_context_dependent:
        return user_text, "current"  # <-- PASSED THROUGH CLEANLY (Topic Shift Guardrail)

    # If verified dependent, execute context fusion formatting
    if is_question:
        return f"{base_text} when you said \"{inverted_context.lower()}\"?", "history"
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
