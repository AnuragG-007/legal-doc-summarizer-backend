MAX_LED=10000
def adaptive_chunk(text, tokenizer, token_budget_large=MAX_LED):
    paragraphs = text.split("\n")
    chunks=[]; current=""
    for para in paragraphs:
        if not para.strip(): continue
        block=(current+"\n"+para).strip()
        if len(tokenizer.encode(block))<token_budget_large:
            current=block
        else:
            if current.strip(): chunks.append(current)
            current=para
    if current.strip(): chunks.append(current)
    return chunks
