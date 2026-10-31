import os
import httpx
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

SYSTEM = (
    "Tu es Ceus Mike AI, un assistant IA tout-en-un créé par un jeune développeur "
    "nommé CEUS Mike Rolph Cedny. Réponds dans la langue de l'utilisateur "
    "(français par défaut), de façon claire, utile et bien organisée, sans longueurs "
    "inutiles. Si on te demande de générer une image ou une vidéo, explique que ces "
    "modules seront bientôt disponibles. N'invente jamais de faits : dis-le quand "
    "tu ne sais pas."
)
MODES = {
    "code": " Mode code : donne le code dans un bloc, puis une courte explication.",
    "doc": " Mode document : aide à résumer, traduire, corriger ou analyser le texte fourni.",
}

app = FastAPI(title="Ceus Mike AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: List[Msg]
    mode: str = "chat"


@app.get("/")
def accueil():
    return {"status": "ok", "name": "Ceus Mike AI"}


@app.post("/chat")
async def chat(body: ChatIn):
    if not GROQ_API_KEY:
        raise HTTPException(500, "La clé GROQ_API_KEY manque sur le serveur.")

    msgs = body.messages[-20:]
    if not msgs or msgs[-1].role != "user":
        raise HTTPException(400, "Le dernier message doit venir de l'utilisateur.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM + MODES.get(body.mode, "")}]
        + [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content[:8000]}
            for m in msgs
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
            )
    except httpx.HTTPError:
        raise HTTPException(502, "Impossible de joindre le service d'IA.")

    if r.status_code == 429:
        raise HTTPException(429, "Trop de demandes. Réessayez dans un instant.")
    if r.status_code != 200:
        raise HTTPException(502, "Le service d'IA a renvoyé une erreur.")

    try:
        reply = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        reply = "Je n'ai pas pu répondre à cette demande."
    return {"reply": reply}
