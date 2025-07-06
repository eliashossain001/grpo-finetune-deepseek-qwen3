from model.load_model import load_model_tokenizer
from data.preprocess import prepare_dataset
from rewards.answer_check import check_answer, check_numbers
from rewards.format_check import match_format_exactly, match_format_approximately
from rewards.language_check import format_and_language_reward_func
from configs.grpo_config import get_grpo_config
from trl import GRPOTrainer

def run_grpo_training():
    model, tokenizer, system_prompt, max_seq_length = load_model_tokenizer()
    dataset, max_prompt_len = prepare_dataset(tokenizer, system_prompt, max_seq_length)
    config = get_grpo_config(tokenizer, max_seq_length, max_prompt_len)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
            format_and_language_reward_func,
        ],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
