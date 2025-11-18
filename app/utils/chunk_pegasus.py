import torch
import re
from transformers import PreTrainedTokenizer, PreTrainedModel


def smart_sentence_split(text: str):
    """More accurate sentence splitting than .split('. ')"""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    # Split at ., ?, ! followed by space + capital letter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_and_summarize(
    text: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    target_length: int = 350,
    max_tokens: int = 1024
):

    sentences = smart_sentence_split(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = len(tokenizer(sentence).input_ids)

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    summaries = []
    num_chunks = len(chunks)

    per_chunk_budget = max(target_length // max(1, num_chunks), 45)

    for i, chunk in enumerate(chunks):
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens
        )

        with torch.no_grad():
            ids = model.generate(
                inputs.input_ids,
                num_beams=6,
                max_length=per_chunk_budget,
                min_length=per_chunk_budget // 2,
                repetition_penalty=1.12,
                no_repeat_ngram_size=4,
                early_stopping=True
            )

        summary_text = tokenizer.decode(ids[0], skip_special_tokens=True)
        summaries.append(summary_text)

    combined_summary = " ".join(summaries)

    if len(tokenizer(combined_summary).input_ids) < 800:
        with torch.no_grad():
            final_ids = model.generate(
                tokenizer(combined_summary, return_tensors="pt").input_ids,
                num_beams=4,
                max_length=260,
                min_length=140,
                repetition_penalty=1.10,
                no_repeat_ngram_size=4,
                early_stopping=True
            )
        combined_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)

    return combined_summary
