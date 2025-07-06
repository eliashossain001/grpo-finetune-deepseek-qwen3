# model/save_model.py

from unsloth import FastLanguageModel

def save_model(model, tokenizer, path="model"):
    print(f"[INFO] Saving model to {path}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
