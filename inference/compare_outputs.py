# inference/compare_outputs.py

from model.load_model import load_model_and_tokenizer
from utils.langid_utils import get_lang
from datasets import load_dataset
from vllm import SamplingParams

def compare_with_and_without_lora(system_prompt, dataset_path="open-r1/DAPO-Math-17k-Processed"):
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset(dataset_path, "en", split="train").shuffle(seed=3407).select(range(20))

    with_lora_id_count = 0
    without_lora_id_count = 0

    print("Comparing language usage with and without LoRA on 20 samples:")
    print("=" * 60)

    sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)

    for i, sample in enumerate(dataset):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]},
        ]

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        output_with_lora = model.fast_generate(text, sampling_params=sampling_params, lora_request=model.load_lora("grpo_lora"))[0].outputs[0].text
        output_without_lora = model.fast_generate(text, sampling_params=sampling_params, lora_request=None)[0].outputs[0].text

        if get_lang(output_with_lora) == 'id':
            with_lora_id_count += 1
        if get_lang(output_without_lora) == 'id':
            without_lora_id_count += 1

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/20 samples...")

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"With LoRA - Indonesian: {with_lora_id_count}/20")
    print(f"Without LoRA - Indonesian: {without_lora_id_count}/20")
    print(f"Improvement: +{with_lora_id_count - without_lora_id_count}")
