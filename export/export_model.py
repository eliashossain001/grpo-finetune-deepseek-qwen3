# export/export_model.py

from unsloth import FastLanguageModel

def export_lora_adapter(model, tokenizer, output_path="grpo_lora"):
    print("[INFO] Saving LoRA adapter to:", output_path)
    model.save_lora(output_path)

def export_gguf(model, tokenizer, path="model", quantization_method="q4_k_m"):
    print(f"[INFO] Saving GGUF model with quantization: {quantization_method}")
    model.save_pretrained_gguf(path, tokenizer, quantization_method=quantization_method)

def export_merged_model(model, tokenizer, output_path="model", mode="merged_16bit"):
    print(f"[INFO] Saving merged model as: {mode}")
    model.save_pretrained_merged(output_path, tokenizer, save_method=mode)
