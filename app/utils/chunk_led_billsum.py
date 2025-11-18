MAX_LEN=10000
def adaptive_chunk(text, tokenizer, token_budget_large=MAX_LEN):
    paras = text.split("\n")
    chunks, current = [], ""
    for p in paras:
        if not p.strip(): continue
        temp = (current+"\n"+p).strip()
        if len(tokenizer.encode(temp)) < token_budget_large:
            current = temp
        else:
            if current.strip(): chunks.append(current)
            current = p
    if current.strip(): chunks.append(current)
    return chunks
