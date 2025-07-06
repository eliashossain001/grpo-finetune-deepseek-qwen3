from datasets import load_dataset
import numpy as np
import re

def extract_hash_answer(text):
    return text

def prepare_dataset(tokenizer, system_prompt, max_seq_length):
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })

    tokenized = dataset.map(lambda x: {
        "tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)
    }, batched=True)
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    max_prompt_len = int(np.quantile(tokenized["L"], 0.9))
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= max_prompt_len)[0])
    return dataset, max_prompt_len + 1
