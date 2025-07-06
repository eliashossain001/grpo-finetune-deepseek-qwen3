from vllm import SamplingParams
from trl import GRPOConfig

def get_grpo_config(tokenizer, max_seq_length, max_prompt_length):
    max_completion_length = max_seq_length - max_prompt_length

    return GRPOConfig(
        vllm_sampling_params=SamplingParams(
            min_p=0.1, top_p=1.0, top_k=-1, seed=3407, stop=[tokenizer.eos_token], include_stop_str_in_output=True,
        ),
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=100,
        save_steps=100,
        report_to="none",
        output_dir="outputs",
    )
