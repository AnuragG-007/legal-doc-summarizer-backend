import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
from rouge_score import rouge_scorer
from app.utils.chunk_led_billsum import adaptive_chunk

MODEL = "Anurag33Gaikwad/legal-led-billsum-summarization"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = LEDForConditionalGeneration.from_pretrained(MODEL)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def summarize_one(text):

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    global_mask = torch.zeros_like(enc["input_ids"]); global_mask[:,0] = 1

    with torch.no_grad():
        ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            global_attention_mask=global_mask,
            max_length=512,
            min_length=200,
            num_beams=8,
            no_repeat_ngram_size=5,
            repetition_penalty=1.20,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize_led_billsum(text):
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

    global_mask = torch.zeros_like(enc["input_ids"])
    global_mask[:,0] = 1

    with torch.no_grad():
        ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            global_attention_mask=global_mask,

            max_length=950,
            min_length=256,

            num_beams=12,
            no_repeat_ngram_size=5,
            repetition_penalty=1.15,
            length_penalty=0.92,
            early_stopping=True,

            pad_token_id=tokenizer.eos_token_id
        )

    return {
        "summary": tokenizer.decode(ids[0], skip_special_tokens=True),
        "model_used": "legal-led-billsum"
    }
