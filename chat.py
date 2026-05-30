"""
chat.py — Interactive chatbot with a built-in teaching mode.

Usage:
    python chat.py

Commands while chatting:
    /teach   — enter teaching mode (add a new pattern → response pair)
    /retrain — retrain the model on updated data.json
    /stats   — show training stats
    /quit    — exit
"""

import json
import random
import subprocess
import sys

import torch

from model import ChatNet
from utils import bag_of_words

CONFIDENCE_THRESHOLD = 0.60


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    try:
        checkpoint = torch.load("model.pth", weights_only=True)
    except FileNotFoundError:
        print("⚠  model.pth not found. Run  python train.py  first.")
        sys.exit(1)

    net = ChatNet(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=checkpoint["output_size"],
    )
    net.load_state_dict(checkpoint["model_state"])
    net.eval()

    return net, checkpoint["all_words"], checkpoint["tags"]


def predict(net, sentence, all_words, tags):
    """Return (tag, confidence, activations_dict)."""
    bow = bag_of_words(sentence, all_words)
    x = torch.tensor(bow, dtype=torch.float32)

    with torch.no_grad():
        activations = net.get_layer_activations(x)
        probs = torch.tensor(activations["output_probs"])

    confidence, idx = torch.max(probs, dim=0)
    return tags[idx.item()], confidence.item(), activations


def get_response(tag):
    with open("data.json") as f:
        data = json.load(f)
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "…"


# ── Teaching mode ─────────────────────────────────────────────────────────────

def teach_mode():
    print("\n── Teaching mode ──────────────────────────────────────────")
    print("Teach me a new pattern and the response I should give.\n")

    pattern = input("  Pattern (what the user says): ").strip()
    if not pattern:
        print("  (cancelled)\n")
        return

    print("\nChoose a response tag:")
    with open("data.json") as f:
        data = json.load(f)

    tags = [i["tag"] for i in data["intents"]]
    for n, t in enumerate(tags, 1):
        print(f"  [{n}] {t}")
    print(f"  [{len(tags)+1}] Create new tag")

    choice = input("\n  Your choice: ").strip()

    try:
        choice_int = int(choice)
    except ValueError:
        print("  Invalid choice. Cancelled.\n")
        return

    if choice_int == len(tags) + 1:
        new_tag = input("  New tag name: ").strip().lower().replace(" ", "_")
        response = input("  Response for this tag: ").strip()
        data["intents"].append({
            "tag": new_tag,
            "patterns": [pattern],
            "responses": [response],
        })
        print(f"  ✓ Created new tag '{new_tag}' with 1 pattern.")
    elif 1 <= choice_int <= len(tags):
        tag = tags[choice_int - 1]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                intent["patterns"].append(pattern)
                print(f"  ✓ Added pattern to tag '{tag}'.")
                break
    else:
        print("  Invalid choice. Cancelled.\n")
        return

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

    print("  Saved to data.json. Use /retrain to update the model.\n")


# ── Stats ─────────────────────────────────────────────────────────────────────

def show_stats(all_words, tags):
    with open("data.json") as f:
        data = json.load(f)
    total_patterns = sum(len(i["patterns"]) for i in data["intents"])
    print(f"\n── Stats ──────────────────────────────────────────────────")
    print(f"  Tags          : {len(tags)}")
    print(f"  Vocabulary    : {len(all_words)} words")
    print(f"  Total patterns: {total_patterns}")
    print()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   PyTorch Chatbot  —  chat.py        ║")
    print("║   /teach  /retrain  /stats  /quit    ║")
    print("╚══════════════════════════════════════╝\n")

    net, all_words, tags = load_model()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Bot: Goodbye!")
            break

        elif user_input.lower() == "/teach":
            teach_mode()

        elif user_input.lower() == "/retrain":
            print("Retraining…")
            result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
            print(result.stdout[-500:] if result.stdout else "")
            if result.returncode == 0:
                net, all_words, tags = load_model()
                print("Bot: Model updated and reloaded!\n")
            else:
                print("Bot: Retraining failed. Check train.py output above.\n")

        elif user_input.lower() == "/stats":
            show_stats(all_words, tags)

        else:
            tag, confidence, activations = predict(net, user_input, all_words, tags)
            if confidence >= CONFIDENCE_THRESHOLD:
                response = get_response(tag)
                print(f"Bot [{tag} {confidence*100:.0f}%]: {response}\n")
            else:
                print(f"Bot [unsure {confidence*100:.0f}%]: I don't know how to respond to that yet.")
                print("  → Type /teach to add this pattern!\n")


if __name__ == "__main__":
    main()