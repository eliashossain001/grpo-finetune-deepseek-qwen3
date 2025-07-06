# === model/load_model.py (UPDATED) ===
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/DeepSeek-R1-0528-Qwen3-8B"  # *avoid 4‑bit variant to prevent duplicate layer names*
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32


def load_model_tokenizer():
    """Load DeepSeek R1 with Unsloth **without** vLLM during training.

    Setting `fast_inference=False` disables vLLM backend, fixing the
    `ValueError: Duplicate layer name` crash seen when loading the 4‑bit
    checkpoint with vLLM.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,       # use 16‑bit weights + LoRA for training
        fast_inference=False,     # 🔑 disable vLLM during training
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.7,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    system_prompt = (
        "You are given a problem."
        "Think about the problem and provide your working out."
        "You must think in Bahasa Indonesia."
    )

    return model, tokenizer, system_prompt, MAX_SEQ_LENGTH
