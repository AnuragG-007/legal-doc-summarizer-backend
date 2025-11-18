import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
from rouge_score import rouge_scorer
from app.utils.chunk_led_judgment import adaptive_chunk

MODEL = "Anurag33Gaikwad/legal-led-judgment-summarizer-16384"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = LEDForConditionalGeneration.from_pretrained(MODEL)

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def build_global_attention(enc):
    """Improves LED attention stability on long legal texts."""
    ids = enc["input_ids"]
    global_mask = torch.zeros_like(ids)

    global_mask[:, 0] = 1

    period_id = tokenizer.encode(".", add_special_tokens=False)[0]
    positions = (ids == period_id).nonzero(as_tuple=False)

    for p in positions:
        global_mask[p[0], p[1]] = 1

    return global_mask


def summarize_one(text):
    """Summarizes a single chunk without truncation or early stopping."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    global_mask = build_global_attention(enc)

    with torch.no_grad():
        ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            global_attention_mask=global_mask,
            max_length=650,
            min_length=260,
            num_beams=8,
            no_repeat_ngram_size=6,
            repetition_penalty=1.20,
            length_penalty=1.05,
            early_stopping=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize_led_judgment(text):
    """Full document legal judgment summarizer."""
    chunks = adaptive_chunk(text, tokenizer)
    scored = []

    for ch in chunks:
        summ = summarize_one(ch)
        score = scorer.score(ch, summ)["rougeL"].fmeasure
        scored.append((summ, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    combined = " ".join([s for s, _ in scored[:5]])

    enc = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    )

    global_mask = build_global_attention(enc)

    with torch.no_grad():
        final_out = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            global_attention_mask=global_mask,
            max_length=1400,
            min_length=450,
            num_beams=12,
            no_repeat_ngram_size=8,
            repetition_penalty=1.17,
            length_penalty=1.05,
            early_stopping=False
            pad_token_id=tokenizer.eos_token_id
        )

    final_summary = tokenizer.decode(final_out[0], skip_special_tokens=True)

    return {
        "summary": final_summary,
        "model_used": "legal-led-judgment"
    }
