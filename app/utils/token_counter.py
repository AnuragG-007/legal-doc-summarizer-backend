from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Anurag33Gaikwad/legal-pegasus-billsum-summarization")

def count_tokens(text: str) -> int:
    return len(tokenizer(text)["input_ids"])
