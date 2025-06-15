import os
import json
import logging
import sqlite3
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_TOKEN = f"Bearer {RAW_TOKEN}"
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai"
DB_PATH = "knowledge_base.db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_URL = f"{AIPROXY_URL}/v1/embeddings"
COMPLETION_URL = f"{AIPROXY_URL}/v1/chat/completions"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def get_embedding(text):
    headers = {"Authorization": AIPROXY_TOKEN, "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text}
    async with aiohttp.ClientSession() as session:
        async with session.post(EMBEDDING_URL, headers=headers, json=payload) as response:
            data = await response.json()
            if "data" not in data or not data["data"]:
                raise ValueError("❌ Embedding API failed: no data")
            return data["data"][0]["embedding"]

async def find_similar_content(query_embedding, conn, top_k=8):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    scored = []
    for row in rows:
        _, url, content, emb_json, _ = row
        emb = json.loads(emb_json)
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, url, content))
    scored.sort(reverse=True)
    return scored[:top_k]

async def query_openai_with_context(context, question):
    headers = {"Authorization": AIPROXY_TOKEN, "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are a helpful virtual TA for the TDS course. Use only the provided context. Always respond in JSON with 'answer' and 'links'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.2}
    async with aiohttp.ClientSession() as session:
        async with session.post(COMPLETION_URL, headers=headers, json=payload) as response:
            data = await response.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError("❌ OpenAI API failed: no choices")
            return data["choices"][0]["message"]["content"]

def clean_gpt_response(text: str) -> dict:
    try:
        if text.strip().startswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        parsed = json.loads(text)
        for link in parsed.get("links", []):
            if not link.get("text"):
                link["text"] = link.get("url", "Reference")
        return {"answer": parsed.get("answer", "⚠️ No answer generated."), "links": parsed.get("links", [])}
    except Exception:
        return {"answer": text.strip(), "links": []}

@app.post("/api/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    try:
        query_embedding = await get_embedding(request.question)
        conn = sqlite3.connect(DB_PATH)
        top_chunks = await find_similar_content(query_embedding, conn)
        context = "\n\n".join(chunk[2] for chunk in top_chunks)

        raw = await query_openai_with_context(context, request.question)
        parsed = clean_gpt_response(raw)

        fallback_links = []
        seen = set()
        for chunk in top_chunks:
            url = chunk[1]
            if "discourse.onlinedegree.iitm.ac.in" in url and url not in seen:
                seen.add(url)
                fallback_links.append({"url": url, "text": "Refer to this discussion."})
            if len(fallback_links) == 2:
                break

        return QueryResponse(
            answer=parsed["answer"] or "⚠️ No answer generated.",
            links=parsed["links"] or fallback_links
        )
    except Exception as e:
        logger.error("❌ Exception in /api/: %s", e)
        return QueryResponse(answer="⚠️ Failed to get an answer.", links=[])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)