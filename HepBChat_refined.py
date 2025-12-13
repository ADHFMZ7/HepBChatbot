"""
Hepatitis B (Hep B) Chatbot — FastAPI (refined rule-based version with web UI)

Run locally:
    uvicorn HepBChat:app --reload

Docs (API):
    http://127.0.0.1:8000/docs

Chat UI:
    http://127.0.0.1:8000/

Notes
-----
• This is a RULE-BASED educational chatbot (not a clinician). It does not diagnose or treat.
• Answers come from a small built-in knowledge base + a CSV Q&A file.
• Intent classification uses regex patterns + keywords + example overlap (still rule-based).
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import re
import csv
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Hep B Chatbot",
    description=(
        "Simple rule-based chatbot that answers common Hepatitis B questions, "
        "offers a lightweight risk screener, and explains vaccination basics.\n\n"
        "DISCLAIMER: Educational only—NOT medical advice. If you have symptoms, a known exposure, "
        "are pregnant, or feel unwell, seek care from a licensed clinician or your local health department."
    ),
    version="0.3.0",
)

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    user_id: str = Field(..., examples=["saketh"], description="Arbitrary user/session id")
    message: str = Field(..., min_length=1)
    history: List[ChatTurn] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float

class StartRiskScreenRequest(BaseModel):
    user_id: str

class RiskAnswerRequest(BaseModel):
    user_id: str
    answer: Literal["yes", "no", "skip"]

class SessionState(BaseModel):
    user_id: str
    state: Literal["idle", "risk_screen"] = "idle"
    risk_index: int = 0
    risk_score: int = 0

# Simple in-memory session store
SESSIONS: Dict[str, SessionState] = {}

# -------------------------------------------------------------------
# Base knowledge (generic, per intent)
# -------------------------------------------------------------------

BASE_KNOWLEDGE: Dict[str, str] = {
    "greeting": (
        "Hi! I’m an educational Hepatitis B assistant. I can explain transmission, symptoms, testing, "
        "vaccination, and general treatment concepts. What would you like to know?"
    ),
    "transmission": (
        "Hepatitis B spreads through blood or certain body fluids. Common routes include: birth from an infected parent, "
        "unprotected sex, sharing needles or other drug-use equipment, needlestick injuries, and sharing items like razors "
        "that may have blood. It is NOT spread by casual contact, hugging, coughing, or sharing utensils."
    ),
    "symptoms": (
        "Many people—especially children—have no symptoms. When symptoms occur, they can include fatigue, poor appetite, "
        "nausea, right-upper-abdomen discomfort, dark urine, clay-colored stools, joint aches, and jaundice (yellow skin/eyes). "
        "Severe symptoms like confusion, easy bleeding, or severe abdominal pain are urgent—seek care immediately."
    ),
    "testing": (
        "Testing typically uses blood tests that look for viral antigens and antibodies (e.g., HBsAg, anti-HBs, anti-HBc). "
        "Your clinician may also check viral load (HBV DNA) and liver enzymes. Interpretation depends on combinations and timing—"
        "always review results with a clinician."
    ),
    "vaccination": (
        "Safe and effective vaccines protect against Hep B. Many people complete a 2-, 3-, or 4-dose series depending on the product, "
        "age, and circumstances. If you’re unsure of your status, ask about getting vaccinated or checking anti-HBs after vaccination "
        "when recommended."
    ),
    "prevention": (
        "Use condoms for sex, avoid sharing needles or injection equipment, don’t share razors or toothbrushes, cover open wounds, "
        "and ensure any tattoos/piercings are done with sterile equipment. Vaccination is the best prevention."
    ),
    "treatment": (
        "Acute infection is usually monitored for spontaneous clearance. Chronic Hep B management focuses on preventing liver damage. "
        "Clinicians may monitor labs and sometimes prescribe antiviral medicines depending on viral load, liver enzyme levels, age, and "
        "fibrosis. Regular follow-up is important."
    ),
    "window": (
        "During the early window period after exposure, some tests may be negative before turning positive. "
        "If you had a recent exposure, ask a clinician about the right timing for testing and whether post-exposure prophylaxis (PEP) applies."
    ),
    "lab_markers": (
        "Common markers: HBsAg (surface antigen), anti-HBs (surface antibody), anti-HBc (core antibody, IgM vs IgG), "
        "HBeAg/anti-HBe, and HBV DNA (viral load). Patterns help distinguish susceptibility, immunity, acute vs chronic infection, "
        "and treatment response—interpret with a clinician."
    ),
    "urgent": (
        "If you have severe abdominal pain, confusion, vomiting blood, black stools, yellowing skin/eyes with fever, or you’re pregnant "
        "with a possible exposure, seek urgent medical care or call local emergency services."
    ),
}

# -------------------------------------------------------------------
# CSV knowledge base
# -------------------------------------------------------------------

CSV_KNOWLEDGE: Dict[str, List[Dict[str, str]]] = {}

def load_knowledge_csv(path: str = "data/hepB_knowledge.csv") -> Dict[str, List[Dict[str, str]]]:
    """
    Load Q&A examples from a CSV file.

    Expected columns:
        intent,question_example,answer_summary,source

    The result is a dict: intent -> list of rows.
    """
    csv_path = Path(path)
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)


    with csv_path.open(newline="", encoding="utf-8", errors = "replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = (row.get("intent") or "").strip().lower()
            if not intent:
                continue
            entry = {
                "question_example": (row.get("question_example") or "").strip(),
                "answer_summary": (row.get("answer_summary") or "").strip(),
                "source": (row.get("source") or "").strip(),
            }
            buckets[intent].append(entry)

    return dict(buckets)

CSV_KNOWLEDGE = load_knowledge_csv()

# -------------------------------------------------------------------
# Intent patterns (regex)
# -------------------------------------------------------------------

INTENT_PATTERNS: Dict[str, List[str]] = {
    "greeting": [
        r"\b(hi|hello|hey|hey there|hi there|good (morning|afternoon|evening)|start|help)\b"
    ],

    "transmission": [
        r"\b(transmit|transmission|spread|contag|catch|get)\b",
        r"\b(bod(y|ily)[ -]?fluid|blood|semen|needles?|needle[- ]?stick|razor|toothbrush|tattoo|pierc(ing|e))\b",
        r"\b(from|through|via)\b.*\b(sex|blood|fluids?|birth|mother|sharing)\b",
    ],

    "symptoms": [
        r"\b(symptom|sign|feel|feeling|feel sick|sick|ill|unwell)\b",
        r"\b(jaundice|yellow(ing)? (eyes?|skin)|dark urine|clay[- ]?colored stools?|pale stools?|itch(ing)?|fatigue|tired(ness)?|nausea|vomit(ing)?|abdominal|stomach|right[- ]upper[- ](quadrant|abdomen)|joint (pain|aches))\b",
    ],

    "testing": [
        r"\b(test|tests|testing|screen(ing)?|check|panel|blood[\s-]?work|labs?|serolog(y|ic|ies))\b",
        r"\b(when|how soon|how long|time|schedule)\b.*\b(test|testing|screen)\b",
        r"\b(result|results)\b.*\b(test|labs?)\b",
    ],

    "vaccination": [
        r"\b(vacci(nation|nate|ne)|immuni[sz]e?|shot|shots|jab|booster|dose(s)?|series|schedule)\b",
        r"\b(2[- ]?dose|3[- ]?dose|4[- ]?dose|dose[- ]?schedule)\b",
        r"\b(does|will|how long).*\b(protect(ion)?|immunity)\b",
    ],

    "prevention": [
        r"\b(prevent|avoid|reduce|lower|protect|safe sex|condom(s)?|needle(s)?|harm[- ]reduction|sterile|don'?t share)\b",
    ],

    "treatment": [
        r"\b(treat(ment)?|manage(ment)?|care|therapy|medicines?|drugs?)\b",
        r"\b(antiviral(s)?|tenofovir|entecavir|peg[- ]?interferon|interferon|lamivudine|adefovir|telbivudine)\b",
        r"\b(cure|curable)\b",
    ],

    "window": [
        r"\b(window[- ]?period|incubation[- ]?period)\b",
        r"\b(how soon|how long|when|after|time(frame)?|timing)\b.*\b(test|testing|check)\b",
        r"\b(post[- ]?exposure|PEP|exposure)\b",
    ],

    "lab_markers": [
        r"\b(HBsAg|anti[- ]?HBs|anti[- ]?HBc|HBeAg|anti[- ]?HBe|HBV DNA|viral[- ]?load)\b",
        r"\b(surface (antigen|antibody)|core antibody|e[- ]?antigen|antibodies|markers?|serologic markers?)\b",
        r"\b(titer|titre|quant(itative)?|qual(itative)?)\b",
    ],

    "urgent": [
        r"\b(emergency|urgent(ly)?|go to (the )?hospital|ER|call 911|seek care)\b",
        r"\b(severe (abdominal|stomach) pain|vomit(ing)? blood|bleeding|black stools?|melaena|confus(ion)?|faint(ing)?|jaundice with fever|yellow(ing)? with fever|pregnan(t|cy).*(exposure|hep))\b",
    ],
}

# Extra vague language patterns
INTENT_PATTERNS["prevention"].extend([
    r"avoid getting|avoid infection|reduce risk|how not to get|stay safe|protect myself|stop hep b"
])
INTENT_PATTERNS["symptoms"].extend([
    r"not feel(ing)? well|feel bad|tired all the time|liver pain|signs of infection|what happens when you get|side effects"
])
INTENT_PATTERNS["treatment"].extend([
    r"how to (treat|cure)|doctor do for hep b|care plan|medication|medicine for hep b|recovery|healing|get better|therapy for hep b"
])
INTENT_PATTERNS["window"].extend([
    r"how long (before|after)|when to get tested|after exposure|post exposure|after being exposed|when will it show up|how long until test positive|after risk event"
])
INTENT_PATTERNS["lab_markers"].extend([
    r"test result|lab result|blood report|meaning of|interpret|report says|antigen|antibody|marker|hbv level|what does this mean"
])
INTENT_PATTERNS["urgent"].extend([
    r"immediately|right now|emergency room|ER visit|need to see doctor now|serious condition"
])

# -------------------------------------------------------------------
# Text utilities
# -------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "is", "are", "in", "on", "with", "about",
    "what", "how", "does", "do", "can", "i", "you", "it", "be", "get", "from", "when", "if",
    "my", "your", "this", "that", "b", "hep", "hepatitis",
}

def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[’']", "'", t)
    t = re.sub(r"[-–—]", "-", t)
    t = re.sub(r"[^a-z0-9\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> List[str]:
    t = normalize(text)
    return [tok for tok in t.split() if tok and tok not in STOPWORDS]

# -------------------------------------------------------------------
# Build intent config from CSV
# -------------------------------------------------------------------

INTENT_CONFIG: Dict[str, Dict[str, Any]] = {}

def build_intent_config() -> None:
    """
    Build a richer config per intent:
      - patterns
      - examples (question_example)
      - keywords (tokens from examples + patterns)
    """
    intents = set(BASE_KNOWLEDGE.keys()) | set(INTENT_PATTERNS.keys()) | set(CSV_KNOWLEDGE.keys())

    for intent in intents:
        patterns = INTENT_PATTERNS.get(intent, [])
        examples: List[str] = []
        keywords: set[str] = set()

        for entry in CSV_KNOWLEDGE.get(intent, []):
            q = entry.get("question_example") or ""
            if q:
                examples.append(q)
                for tok in tokenize(q):
                    keywords.add(tok)

        for pat in patterns:
            for w in re.findall(r"[A-Za-z]{3,}", pat):
                tok = w.lower()
                if tok not in STOPWORDS:
                    keywords.add(tok)

        INTENT_CONFIG[intent] = {
            "patterns": patterns,
            "examples": examples,
            "keywords": keywords,
        }

build_intent_config()

# -------------------------------------------------------------------
# NLU: classify & answer
# -------------------------------------------------------------------

def classify_intent_v2(text: str) -> tuple[str, float]:
    """
    Rule-based scoring:
      * +2 per matching regex pattern
      * +1 per keyword hit
      * +0.8 * (max token overlap) with any example question
    Returns (intent, confidence in [0,1]).
    """
    if not text.strip():
        return "greeting", 0.0

    text_norm = normalize(text)
    tokens = tokenize(text)
    token_set = set(tokens)

    best_intent = "greeting"
    best_score = 0.0

    for intent, cfg in INTENT_CONFIG.items():
        patterns: List[str] = cfg.get("patterns", [])
        examples: List[str] = cfg.get("examples", [])
        keywords: set[str] = cfg.get("keywords", set())

        pat_hits = sum(1 for p in patterns if re.search(p, text_norm, flags=re.IGNORECASE))
        kw_hits = len(token_set & keywords)

        ex_overlap = 0.0
        for ex in examples:
            ex_tokens = set(tokenize(ex))
            overlap = len(ex_tokens & token_set)
            if overlap > ex_overlap:
                ex_overlap = overlap

        score = 2.0 * pat_hits + 1.0 * kw_hits + 0.8 * ex_overlap

        if score > best_score:
            best_score = score
            best_intent = intent

    if best_score <= 0:
        return "greeting", 0.0

    confidence = min(1.0, best_score / 8.0)
    return best_intent, confidence

def pick_csv_answer(intent: str, user_text: str) -> Optional[str]:
    """
    Choose the best CSV answer for a given intent by matching the
    user_text to the question_example with highest token overlap.
    """
    entries = CSV_KNOWLEDGE.get(intent)
    if not entries:
        return None

    user_tokens = set(tokenize(user_text))
    best_entry: Optional[Dict[str, str]] = None
    best_score = -1

    for entry in entries:
        q = entry.get("question_example") or ""
        overlap = len(user_tokens & set(tokenize(q)))
        if overlap > best_score:
            best_score = overlap
            best_entry = entry

    if not best_entry:
        return None

    ans = (best_entry.get("answer_summary") or "").strip()
    src = (best_entry.get("source") or "").strip()
    if ans and src:
        return f"{ans} (Source: {src})"
    return ans or None

def response_for_intent(intent: str, user_text: str) -> str:
    csv_answer = pick_csv_answer(intent, user_text)
    if csv_answer:
        base = csv_answer
    else:
        base = BASE_KNOWLEDGE.get(intent, BASE_KNOWLEDGE["greeting"])

    disclaimer = (
        "\n\n— This is educational information, not medical advice. "
        "For personal guidance, please consult a clinician."
    )
    return base + disclaimer

# -------------------------------------------------------------------
# Risk screen
# -------------------------------------------------------------------

RISK_QUESTIONS: List[str] = [
    "Have you ever had a needlestick, shared needles, or used injection drugs?",
    "Have you had unprotected sex with a partner whose Hep B status you don’t know?",
    "Were you born in or have you lived for years in a region with higher Hep B prevalence?",
    "Has a clinician ever told you that your liver enzymes were high or that you have a liver disease?",
    "Are you currently pregnant or planning pregnancy?",
]

RISK_TIPS = (
    "Based on screening answers, consider asking a clinician about Hep B testing (HBsAg, anti-HBs, anti-HBc) "
    "and vaccination if you’re not immune. Avoid sharing needles or razors, use condoms, and keep wounds covered."
)

# -------------------------------------------------------------------
# API Routes
# -------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # If there's an active risk screen, remind user to finish it
    session = SESSIONS.get(req.user_id)
    if session and session.state == "risk_screen":
        intent, conf = classify_intent_v2(req.message)
        reply = (
            "You have a risk screen in progress. Reply to /risk/answer with 'yes', 'no', or 'skip' "
            "to continue, or call /risk/stop to end it."
        )
        return ChatResponse(reply=reply, intent="risk_screen", confidence=conf)

    # Normal chat
    intent, conf = classify_intent_v2(req.message)
    text = response_for_intent(intent, req.message)

    if intent in {"transmission", "prevention", "testing"}:
        text += (
            "\n\nIf you’d like, I can run a 5-question anonymous risk screen. "
            "POST your user_id to /risk/start and then answer each question at /risk/answer."
        )

    return ChatResponse(reply=text, intent=intent, confidence=conf)

@app.post("/risk/start", response_model=SessionState)
def start_risk_screen(req: StartRiskScreenRequest) -> SessionState:
    state = SessionState(user_id=req.user_id, state="risk_screen", risk_index=0, risk_score=0)
    SESSIONS[req.user_id] = state
    return state

@app.post("/risk/answer", response_model=Dict[str, Any])
def answer_risk(req: RiskAnswerRequest) -> Dict[str, Any]:
    session = SESSIONS.get(req.user_id)
    if not session or session.state != "risk_screen":
        raise HTTPException(status_code=400, detail="No active risk screen. Call /risk/start first.")

    if req.answer == "yes":
        session.risk_score += 1

    session.risk_index += 1

    if session.risk_index < len(RISK_QUESTIONS):
        next_q = RISK_QUESTIONS[session.risk_index]
        return {
            "status": "next",
            "question_index": session.risk_index,
            "question": next_q,
            "progress": f"{session.risk_index}/{len(RISK_QUESTIONS)}",
            "session": session.model_dump(),
        }
    else:
        total = session.risk_score
        if total >= 3:
            level = "elevated"
        elif total == 2:
            level = "moderate"
        else:
            level = "low"

        SESSIONS[req.user_id] = SessionState(user_id=req.user_id)
        return {
            "status": "done",
            "score": int(total),
            "level": level,
            "guidance": RISK_TIPS,
            "disclaimer": "Educational only. Not medical advice.",
        }

@app.post("/risk/stop", response_model=Dict[str, str])
def stop_risk(user_id: str) -> Dict[str, str]:
    SESSIONS[user_id] = SessionState(user_id=user_id)
    return {"status": "stopped"}

@app.get("/examples")
def examples() -> Dict[str, List[str]]:
    return {
        "ask": [
            "How does Hep B spread?",
            "What are typical symptoms?",
            "What tests should I ask for?",
            "What do HBsAg and anti-HBs mean?",
            "Should I get the vaccine?",
            "What is the window period after exposure?",
            "I have severe abdominal pain and yellow eyes—what do I do?",
        ]
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# -------------------------------------------------------------------
# Web UI route (simple chat website)
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def web_chat() -> str:
    # Single-page HTML + JS UI that talks to /chat
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Hep B Chatbot</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                margin: 0;
                background: #f3f4f6;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .chat-container {
                background: #ffffff;
                width: 90%;
                max-width: 800px;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                padding: 16px 20px;
                background: #1d4ed8;
                color: white;
            }
            .chat-header h1 {
                margin: 0;
                font-size: 1.25rem;
            }
            .chat-header p {
                margin: 4px 0 0;
                font-size: 0.85rem;
                opacity: 0.9;
            }
            #messages {
                padding: 16px;
                flex: 1;
                overflow-y: auto;
                background: #f9fafb;
            }
            .msg {
                margin-bottom: 12px;
                max-width: 80%;
                padding: 10px 12px;
                border-radius: 10px;
                font-size: 0.95rem;
                line-height: 1.4;
                white-space: pre-wrap;
            }
            .msg.user {
                margin-left: auto;
                background: #1d4ed8;
                color: white;
                border-bottom-right-radius: 2px;
            }
            .msg.bot {
                margin-right: auto;
                background: #e5e7eb;
                border-bottom-left-radius: 2px;
            }
            .input-row {
                display: flex;
                gap: 8px;
                padding: 12px;
                border-top: 1px solid #e5e7eb;
                background: #ffffff;
            }
            .input-row input {
                flex: 1;
                padding: 10px 12px;
                border-radius: 999px;
                border: 1px solid #d1d5db;
                font-size: 0.95rem;
                outline: none;
            }
            .input-row input:focus {
                border-color: #1d4ed8;
                box-shadow: 0 0 0 1px #1d4ed8;
            }
            .input-row button {
                border: none;
                border-radius: 999px;
                padding: 10px 18px;
                font-size: 0.95rem;
                background: #1d4ed8;
                color: white;
                cursor: pointer;
            }
            .input-row button:disabled {
                opacity: 0.6;
                cursor: wait;
            }
            .status {
                font-size: 0.75rem;
                padding: 0 16px 8px;
                color: #6b7280;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>Hepatitis B Educational Chatbot</h1>
                <p>Ask about transmission, symptoms, testing, vaccines, and more. Not medical advice.</p>
            </div>
            <div id="messages"></div>
            <div class="status" id="status"></div>
            <form class="input-row" id="chat-form">
                <input type="text" id="user-input" placeholder="Type your question here..." autocomplete="off" />
                <button type="submit" id="send-btn">Send</button>
            </form>
        </div>

        <script>
            const messagesEl = document.getElementById("messages");
            const formEl = document.getElementById("chat-form");
            const inputEl = document.getElementById("user-input");
            const statusEl = document.getElementById("status");
            const sendBtn = document.getElementById("send-btn");

            const USER_ID = "web-user"; // static user id for the browser session

            function addMessage(text, role) {
                const div = document.createElement("div");
                div.className = "msg " + (role === "user" ? "user" : "bot");
                div.textContent = text;
                messagesEl.appendChild(div);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }

            async function sendMessage(text) {
                sendBtn.disabled = true;
                statusEl.textContent = "Thinking...";
                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            user_id: USER_ID,
                            message: text,
                            history: []
                        })
                    });

                    if (!res.ok) {
                        throw new Error("Server error: " + res.status);
                    }

                    const data = await res.json();
                    addMessage(data.reply, "bot");
                    statusEl.textContent = `Intent: ${data.intent} (confidence ${Math.round(data.confidence * 100)}%)`;
                } catch (err) {
                    console.error(err);
                    addMessage("Sorry, something went wrong contacting the server.", "bot");
                    statusEl.textContent = "Error talking to backend.";
                } finally {
                    sendBtn.disabled = false;
                }
            }

            formEl.addEventListener("submit", function (e) {
                e.preventDefault();
                const text = inputEl.value.trim();
                if (!text) return;
                addMessage(text, "user");
                inputEl.value = "";
                sendMessage(text);
            });

            // Initial greeting
            addMessage("Hi, I’m a Hepatitis B educational chatbot. How can I help?", "bot");
        </script>
    </body>
    </html>
    """

