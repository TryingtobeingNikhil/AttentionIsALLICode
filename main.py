import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import generate_text

app = FastAPI(title="Tiny GPT API")

# Enable CORS for frontend/deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    max_tokens: int = Field(default=100, ge=1, le=500)


@app.get("/")
def home():
    return {"status": "Tiny GPT running 🚀", "docs": "/docs"}


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        text = generate_text(req.prompt, req.max_tokens)
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)