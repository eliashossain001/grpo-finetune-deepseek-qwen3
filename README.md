# GRPO Fine-Tuning on DeepSeek-R1-0528-Qwen3-8B

This repository provides an end-to-end lightweight fine-tuning pipeline using GRPO (Guided Reward Preference Optimization) on the `DeepSeek-R1-0528-Qwen3-8B` model, enhanced with **Unsloth** for parameter-efficient training.

## Features

- GRPO training with custom reward functions
- Parameter-efficient fine-tuning (LoRA) using Unsloth
- Inference-ready with GGUF/merged/int4 export support
- Model evaluation via prompt completion comparisons
- Easily adaptable to your own dataset (just replace `data/train.json` with the required format)

---

## 📁 Project Structure
```
reasoning_finetune/
├── configs/                   # Configuration scripts for GRPO training
│   └── grpo_config.py         
│
├── data/                      # Dataset handling and preprocessing
│   └── preprocess.py
│
├── model/                     # Model loading, tokenizer setup, and save utilities
│   ├── load_model.py
│   └── save_model.py
│
├── rewards/                   # Reward functions for GRPO
│   ├── answer_check.py
│   ├── format_check.py
│   └── language_check.py
│
├── train_module/              # Training logic
│   └── train.py
│
├── inference/                 # Inference and comparison script
│   └── compare_outputs.py
│
├── export/                    # Scripts to export LoRA/GGUF/merged weights
│   └── export_model.py
│
├── outputs/                   # Training checkpoints (excluded via .gitignore)
│
├── utils/                     # Utility scripts (regex, langid wrapper, etc.)
│   └── langid_utils.py
│
├── requirements.txt           # All dependencies listed here
├── main.py                    # Entry point for model training
└── README.md                  # To be added
``
---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🧑‍🏫 Usage

### 1. Prepare Dataset
Place your dataset in the `data/` folder. The format should be:

```json
[
  {
    "instruction": "Your prompt here",
    "input": "Additional context if needed",
    "output": "Expected reasoning response"
  }
]
```

### 2. Run Training

```bash
python main.py
```

### 3. Inference

```bash
python inference/compare_outputs.py
```

---

## 💾 Export Options
Supports multiple formats:

- `merged_16bit` / `merged_4bit`
- `lora` adapter-only saving
- `gguf` export for llama.cpp compatibility (q4_k_m, q5_k_m, q8_0, f16, etc.)

Run `export/export_model.py` and specify your desired export method.



## 🖼️ Sample Output

| Prompt | Output Before Finetuning | Output After Finetuning |
|--------|---------------------------|--------------------------|
| What is 17 + 25? | "Let me think... maybe 30?" | "The answer is 42." |

---

## 📊 Badges

![Model](https://img.shields.io/badge/Model-DeepSeek--Qwen3--8B-blue)
![Framework](https://img.shields.io/badge/Framework-Unsloth%20%2B%20TRL-lightgrey)
![License](https://img.shields.io/github/license/eliashossain001/reasoning-finetune)
![GPU Used](https://img.shields.io/badge/GPU-A100%2040GB-green)

---

## 💡 Notes
- This pipeline is lightweight and suitable for most consumer GPUs (LoRA+16bit).
- GRPO allows the use of multiple reward functions to guide fine-tuning.
- To train on your own dataset, simply replace the JSON file in `data/` and ensure it follows the structure shown above.

---

## 📜 License
MIT License.

---

## 🙋‍♂️ Acknowledgements
- [Unsloth](https://github.com/unslothai/unsloth)
- [DeepSeek](https://huggingface.co/deepseek-ai)
- [VLLM](https://github.com/vllm-project/vllm)
- [TRL by HuggingFace](https://github.com/huggingface/trl)

---



## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD in Progress | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)


Happy fine-tuning! 🎯
