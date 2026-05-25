"""
server.py — Discovery Bank AI Assistant
Optimised for Render.com free tier:
  - Gunicorn-friendly (single worker, longer timeout)
  - Request size limits to prevent abuse
  - Groq API key fully server-side (never in browser)

Run locally:
  pip install -r requirements.txt
  cp .env.example .env   →  add GROQ_API_KEY
  python server.py

Deploy on Render:
  Build:  pip install -r requirements.txt
  Start:  gunicorn server:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
"""

import os
import requests
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, origins="*")

# ── Config ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", 800))
PORT         = int(os.environ.get("PORT", 3000))
FLASK_ENV    = os.environ.get("FLASK_ENV", "production")

# Free tier limits — keeps usage within Render + Groq free tiers
MAX_MSG_LENGTH  = 1000   # chars per user message
MAX_HISTORY     = 10     # max messages kept in context
MAX_BODY_SIZE   = 32_000 # bytes — reject huge payloads

# NOTE: No self-ping keepalive — Render free tier bans service-initiated traffic.
# The app will sleep after 15 mins of inactivity on the free tier.
# First request after sleep takes ~30 seconds to wake up — this is normal.


# ══════════════════════════════════════════════════════════════════════
# STATIC FILES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main Discovery Bank single-page app."""
    # Try the final version first, fall back to index.html
    for filename in ["discovery-bank-final.html", "index.html"]:
        if os.path.exists(os.path.join(".", filename)):
            return send_from_directory(".", filename)
    return "Discovery Bank — no HTML file found. Upload discovery-bank-final.html", 404

@app.route("/<path:path>")
def static_files(path):
    """Serve static assets (CSS, JS, images)."""
    try:
        return send_from_directory(".", path)
    except Exception:
        return jsonify({"error": "File not found"}), 404


# ══════════════════════════════════════════════════════════════════════
# GROQ AI PROXY  →  POST /api/chat
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Groq AI proxy. API key stays on the server — never sent to browser.

    Request JSON:
    {
      "messages": [{"role": "user", "content": "What is my balance?"}],
      "system":   "You are the Discovery Bank AI assistant...",
      "temperature": 0.7,
      "max_tokens": 800
    }

    Response JSON:
    {
      "reply": "Your balance is R 24,380.50...",
      "model": "llama-3.3-70b-versatile",
      "usage": { "prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480 }
    }
    """

    # ── 1. Reject oversized requests (abuse protection) ─────────────
    content_length = request.content_length or 0
    if content_length > MAX_BODY_SIZE:
        return jsonify({"error": "Request too large. Please shorten your message."}), 413

    # ── 2. Check API key is configured ──────────────────────────────
    if not GROQ_API_KEY:
        return jsonify({
            "error": "GROQ_API_KEY not set. Add it to Render's Environment tab."
        }), 500

    # ── 3. Parse body ────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON body."}), 400

    messages    = body.get("messages", [])
    system_msg  = body.get("system", "")
    temperature = float(body.get("temperature", 0.7))
    max_tokens  = min(int(body.get("max_tokens", MAX_TOKENS)), MAX_TOKENS)

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    # ── 4. Sanitise and limit history ───────────────────────────────
    clean_messages = []
    for msg in messages[-MAX_HISTORY:]:  # keep only last N messages
        role    = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()

        if role not in ("user", "assistant"):
            continue
        if not content:
            continue

        # Truncate individual messages that are too long
        if len(content) > MAX_MSG_LENGTH:
            content = content[:MAX_MSG_LENGTH] + "..."

        clean_messages.append({"role": role, "content": content})

    if not clean_messages:
        return jsonify({"error": "No valid messages after sanitisation."}), 400

    # ── 5. Build Groq payload ────────────────────────────────────────
    groq_messages = []
    if system_msg:
        groq_messages.append({"role": "system", "content": system_msg})
    groq_messages.extend(clean_messages)

    payload = {
        "model":       GROQ_MODEL,
        "messages":    groq_messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    # ── 6. Call Groq API ─────────────────────────────────────────────
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=25,  # 25s — safely under Render's 30s request limit
        )

        data = resp.json()

        # Handle Groq-side errors
        if resp.status_code != 200:
            err = data.get("error", {}).get("message", "Groq API error")
            if resp.status_code == 401:
                return jsonify({"error": "Invalid Groq API key. Check GROQ_API_KEY in Render."}), 401
            if resp.status_code == 429:
                return jsonify({"error": "Rate limit reached. Please wait a moment and try again."}), 429
            return jsonify({"error": err}), resp.status_code

        # Extract reply
        reply = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
        )
        if not reply:
            return jsonify({"error": "Empty response from Groq."}), 500

        usage = data.get("usage", {})
        return jsonify({
            "reply": reply,
            "model": data.get("model", GROQ_MODEL),
            "usage": {
                "prompt_tokens":     usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens":      usage.get("total_tokens", 0),
            }
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({"error": "Groq took too long to respond. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach Groq API. Check your internet connection."}), 503
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ══════════════════════════════════════════════════════════════════════
# HEALTH CHECK  — Render pings this to know the app is alive
# ══════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "app":       "Discovery Bank AI Assistant",
        "model":     GROQ_MODEL,
        "api_ready": bool(GROQ_API_KEY),
    }), 200


@app.route("/api/models")
def models():
    return jsonify({
        "current": GROQ_MODEL,
        "available": [
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B — Recommended"},
            {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 8B  — Fastest"},
            {"id": "mixtral-8x7b-32768",      "label": "Mixtral 8x7B  — Long context"},
            {"id": "gemma2-9b-it",            "label": "Gemma 2 9B   — Lightweight"},
        ]
    }), 200


# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    debug = FLASK_ENV == "development"

    print("=" * 56)
    print("  🏦  Discovery Bank AI Assistant")
    print("=" * 56)
    print(f"  URL      → http://localhost:{PORT}")
    print(f"  Model    → {GROQ_MODEL}")
    print(f"  API Key  → {'✅ Set' if GROQ_API_KEY else '❌ NOT SET — add to .env'}")
    print(f"  Mode     → {'development' if debug else 'production'}")
    print("=" * 56)

    if not GROQ_API_KEY:
        print("\n  ⚠️  Add your Groq key to .env:")
        print("  GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx\n")
        print("  Get a free key → https://console.groq.com/keys\n")

    app.run(host="0.0.0.0", port=PORT, debug=debug)
