from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import aiofiles

from app.utils.extractText import extract_text
from app.utils.token_counter import count_tokens
from app.models.led_billsum import summarize_led_billsum
from app.models.led_judgment import summarize_led_judgment
from app.models.pegasus import summarize_pegasus

app = FastAPI(title="Legal Summarization API", version="1.0.0")

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Legal Summarization API",
            "status": "operational",
            "version": "1.0.0",
            "endpoints": {
                "root": "GET /",
                "health": "GET /health",
                "summarize_bills_text": "POST /summarize/led_billsum",
                "summarize_judgment_text": "POST /summarize/led_judgment",
                "summarize_file": "POST /summarize/file/{category}"
            }
        }
    )

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "service": "legal-summarization-api"})

@app.post("/summarize/led_billsum")
async def summarize_bills_text(payload: TextInput):
    text = payload.text
    token_len = count_tokens(text)

    if token_len < 3000:
        result = summarize_pegasus(text)
    else:
        result = summarize_led_billsum(text)

    return {
        "summary": result["summary"],
        "model_used": result.get("model_used")
    }


@app.post("/summarize/led_judgment")
async def summarize_judgment_text(payload: TextInput):
    result = summarize_led_judgment(payload.text)
    return {
        "summary": result["summary"],
        "model_used": result.get("model_used")
    }

@app.post("/summarize/file/{category}")
async def summarize_file(category: str, file: UploadFile = File(...)):

    filename = file.filename or "uploaded"

    if not filename.lower().endswith((".txt", ".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    async with aiofiles.open(tmp_path, "wb") as out_file:
        contents = await file.read()
        await out_file.write(contents)

    try:
        extracted = extract_text(tmp_path)

        if category == "other":
            return {
                "summary": "🚧 This feature is currently under construction.",
                "model_used": None
            }

        elif category == "bills":
            token_len = count_tokens(extracted)
            if token_len < 3000:
                result = summarize_pegasus(extracted)
            else:
                result = summarize_led_billsum(extracted)

        elif category == "judgements":
            result = summarize_led_judgment(extracted)

        else:
            raise HTTPException(status_code=400, detail="Unsupported category")

        return {
            "summary": result["summary"],
            "model_used": result.get("model_used")
        }

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)