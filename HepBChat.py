"""
Hepatitis B (Hep B) Chatbot — FastAPI starter
Author: (your name)
License: MIT

Run locally:
    uvicorn main:app --reload

Docs:
    http://127.0.0.1:8000/docs

Notes
-----
• This is a RULE-BASED educational chatbot (not a clinician). It does not diagnose or treat.
• Keep answers high‑level and encourage users to seek professional care for urgent issues.
• You can grow the knowledge base by editing KNOWLEDGE_BASE and INTENT_PATTERNS below.
• If you later want embeddings/RAG, you can swap the response() function for a retriever call.

Dependencies
------------
    pip install fastapi uvicorn "pydantic>=2"

"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import re

app = FastAPI(
    title="Hep B Chatbot",
    description=(
        "Simple rule‑based chatbot that answers common Hepatitis B questions, "
        "offers a lightweight risk screener, and explains vaccination basics.\n\n"
        "DISCLAIMER: Educational only—NOT medical advice. If you have symptoms, a known exposure, "
        "are pregnant, or feel unwell, seek care from a licensed clinician or your local health department."
    ),
    version="0.1.0",
)

# -------------------------------
# Models
# -------------------------------
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

# In‑memory store for simple sessions. Replace with Redis in production.
SESSIONS: Dict[str, SessionState] = {}

# -------------------------------
# Knowledge base (short, neutral, educational)
# -------------------------------
KNOWLEDGE_BASE: Dict[str, str] = {
    "greeting": (
        "Hi! I’m an educational Hepatitis B assistant. I can explain transmission, symptoms, testing, "
        "vaccination, and general treatment concepts. What would you like to know?"
    ),
    "transmission": (
        "Hepatitis B spreads through blood or certain body fluids. Common routes include: birth from an infected parent, "
        "unprotected sex, sharing needles or other drug‑use equipment, needlestick injuries, and sharing items like razors "
        "that may have blood. It is NOT spread by casual contact, hugging, coughing, or sharing utensils."
    ),
    "symptoms": (
        "Many people—especially children—have no symptoms. When symptoms occur, they can include fatigue, poor appetite, "
        "nausea, right‑upper‑abdomen discomfort, dark urine, clay‑colored stools, joint aches, and jaundice (yellow skin/eyes). "
        "Severe symptoms like confusion, easy bleeding, or severe abdominal pain are urgent—seek care immediately."
    ),
    "testing": (
        "Testing typically uses blood tests that look for viral antigens and antibodies (e.g., HBsAg, anti‑HBs, anti‑HBc). "
        "Your clinician may also check viral load (HBV DNA) and liver enzymes. Interpretation depends on combinations and timing—" 
        "always review results with a clinician."
    ),
    "vaccination": (
        "Safe and effective vaccines protect against Hep B. Many people complete a 2‑, 3‑, or 4‑dose series depending on the product, age, and circumstances. "
        "If you’re unsure of your status, ask about getting vaccinated or checking anti‑HBs after vaccination when recommended."
    ),
    "prevention": (
        "Use condoms for sex, avoid sharing needles or injection equipment, don’t share razors or toothbrushes, cover open wounds, and ensure any tattoos/piercings "
        "are done with sterile equipment. Vaccination is the best prevention."
    ),
    "treatment": (
        "Acute infection is usually monitored for spontaneous clearance. Chronic Hep B management focuses on preventing liver damage. "
        "Clinicians may monitor labs and sometimes prescribe antiviral medicines depending on viral load, liver enzyme levels, age, and fibrosis. "
        "Regular follow‑up is important."
    ),
    "window": (
        "During the early window period after exposure, some tests may be negative before turning positive. "
        "If you had a recent exposure, ask a clinician about the right timing for testing and whether post‑exposure prophylaxis (PEP) applies."
    ),
    "lab_markers": (
        "Common markers: HBsAg (surface antigen), anti‑HBs (surface antibody), anti‑HBc (core antibody, IgM vs IgG), "
        "HBeAg/anti‑HBe, and HBV DNA (viral load). Patterns help distinguish susceptibility, immunity, acute vs chronic infection, and treatment response—" 
        "interpret with a clinician."
    ),
    "urgent": (
        "If you have severe abdominal pain, confusion, vomiting blood, black stools, yellowing skin/eyes with fever, or you’re pregnant with a possible exposure, "
        "seek urgent medical care or call local emergency services."
    ),
}

# -------------------------------
# Load extended knowledge base from CSV
# -------------------------------
import csv
from pathlib import Path

def load_knowledge_csv(path="data/hepB_knowledge.csv"):
    kb = {}
    csv_path = Path(path)
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                intent = row["intent"].strip().lower()
                answer = f"{row['answer_summary']} (Source: {row['source']})"
                kb[intent] = answer
    return kb

# Extend your existing knowledge base
KNOWLEDGE_BASE.update(load_knowledge_csv())


# Patterns to match intents; tweak/extend as needed.
INTENT_PATTERNS: Dict[str, List[str]] = {
    "greeting": [r"\b(hi|hello|hey|start|help)\b"],
    "transmission": [r"transmit|spread|contag|how.*(get|catch)", r"bod(y|ily) fluid|blood"],
    "symptoms": [r"symptom|sign|feel|jaundice|yellow"],
    "testing": [r"test|screen|blood work|lab|serolog"],
    "vaccination": [r"vacci|immuni|shot|dose|schedule"],
    "prevention": [r"prevent|avoid|protect|safe sex|condom|needle"],
    "treatment": [r"treat|medicine|antiviral|terato|tenofovir|entecavir"],
    "window": [r"window period|how soon|early test|PEP|post-?exposure"],
    "lab_markers": [r"HBsAg|anti-?HBs|anti-?HBc|HBeAg|HBV DNA|marker|antibody"],
    "urgent": [r"emergency|urgent|severe|vomit(ing)? blood|black stool|confus|pregnan"],
}

# -------------------------------
# NLU helpers
# -------------------------------
def classify_intent(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    best_intent = "greeting"
    best_score = 0.0
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pat in patterns:
            if re.search(pat, text_lower):
                score += 1
        if score > best_score:
            best_intent, best_score = intent, float(score)
    # normalize a rough confidence: cap by number of patterns
    max_pats = max((len(p) for p in INTENT_PATTERNS.values()))
    confidence = min(1.0, best_score / max(1, max_pats))
    return best_intent, confidence


def response_for_intent(intent: str) -> str:
    base = KNOWLEDGE_BASE.get(intent, KNOWLEDGE_BASE["greeting"])  # fallback
    disclaimer = ("\n\n— This is educational info, not medical advice. For personal guidance, please consult a clinician.")
    return base + disclaimer

# -------------------------------
# Risk screen (very lightweight)
# -------------------------------
RISK_QUESTIONS: List[str] = [
    "Have you ever had a needlestick, shared needles, or used injection drugs?",
    "Have you had unprotected sex with a partner whose Hep B status you don’t know?",
    "Were you born in or have you lived for years in a region with higher Hep B prevalence?",
    "Has a clinician ever told you that your liver enzymes were high or that you have a liver disease?",
    "Are you currently pregnant or planning pregnancy?",
]

RISK_TIPS = (
    "Based on screening answers, consider asking a clinician about Hep B testing (HBsAg, anti‑HBs, anti‑HBc) and vaccination if you’re not immune. "
    "Avoid sharing needles or razors, use condoms, and keep wounds covered."
)


# -------------------------------
# Routes
# -------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    intent, conf = classify_intent(req.message)

    # If the user is in an ongoing risk screen, redirect them
    session = SESSIONS.get(req.user_id)
    if session and session.state == "risk_screen":
        reply = (
            "You have a risk screen in progress. Reply to /risk/answer with 'yes', 'no', or 'skip' to continue, "
            "or call /risk/stop to end it."
        )
        return ChatResponse(reply=reply, intent="risk_screen", confidence=conf)

    text = response_for_intent(intent)

    # Offer the risk screen when relevant
    if intent in {"transmission", "prevention", "testing"}:
        text += ("") + (
            "\n\nIf you’d like, I can run a 5‑question anonymous risk screen. "
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

    # Score: yes = 1, no = 0, skip = 0
    ans = req.answer
    if ans == "yes":
        session.risk_score += 1
    # advance
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
        # Finish
        total = session.risk_score
        level = "low"
        if total >= 3:
            level = "elevated"
        elif total == 2:
            level = "moderate"
        # Reset session
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
    SESSIONS[user_id] = SessionState(user_id=user_id)  # reset
    return {"status": "stopped"}


# --------------- Utilities ---------------
@app.get("/examples")
def examples() -> Dict[str, List[str]]:
    return {
        "ask": [
            "How does Hep B spread?",
            "What are typical symptoms?",
            "What tests should I ask for?",
            "What do HBsAg and anti‑HBs mean?",
            "Should I get the vaccine?",
            "What is the window period after exposure?",
            "I have severe abdominal pain and yellow eyes—what do I do?",
        ]
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
