import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.utils.chunk_pegasus import chunk_and_summarize

MODEL = "Anurag33Gaikwad/legal-pegasus-billsum-summarization"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

def summarize_pegasus(text):
    return {"summary": chunk_and_summarize(text, tokenizer, model),
            "model_used": "legal-pegasus"}
