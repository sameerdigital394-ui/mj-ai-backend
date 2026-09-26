from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """You are MJ AI, a personal AI secretary, software development assistant, and business assistant.

For EVERY request, respond in this exact format:

PLAN:
1. ...
2. ...

ACTIONS:
- ...
- ...

RESULT:
<the actual finished work — code, content, report, plan, proposal>

Rules:
- Do the work. Do not ask questions.
- Website request → output full HTML/CSS.
- Android app request → output full Kotlin code.
- SEO request → output full report.
- Marketing request → output full campaign.
- Never say "I cannot". Always deliver something useful.
"""

class Ask(BaseModel):
    prompt: str

@app.post("/ask")
def ask(a: Ask):
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": a.prompt}
        ],
        temperature=0.7,
    )
    return {"result": r.choices[0].message.content}

@app.get("/")
def root():
    return {"status": "MJ AI running"}
