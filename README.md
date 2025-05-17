# 🧠 Qwen2-1.5B Fine-Tuning with LoRA on Alpaca

> Fine-tuning an instruction-following LLM (Qwen2-1.5B-Instruct) using LoRA and 4-bit quantization on the Alpaca dataset — optimized for efficient training under limited compute.

---

## 🚀 Project Summary

This project demonstrates a complete pipeline to fine-tune a modern Large Language Model (LLM) using:
- 💡 **LoRA (Low-Rank Adaptation)** for parameter-efficient tuning
- 🧮 **4-bit quantization (BitsAndBytes)** for reduced memory usage
- 🐑 **Alpaca dataset** for instruction-following tasks
- 📊 **ROUGE evaluation** to measure model response quality

Designed for real-world LLM applications on a compute budget. Ideal for education, experimentation, and deployment prep.

---

## 🧰 Technologies Used

| Component      | Description                                  |
|----------------|----------------------------------------------|
| `Qwen2-1.5B`   | Open LLM with instruction-following ability |
| `LoRA (peft)`  | Efficient fine-tuning (r=8, α=16)            |
| `4-bit BnB`    | Quantized model with NF4 + double quant      |
| `Datasets`     | Alpaca (`instruction`, `input`, `output`)   |
| `Evaluate`     | ROUGE score (rouge1, rouge2, rougeL, rougeLsum) |

---

## 🏁 Training Highlights

- Model: `Qwen/Qwen2-1.5B-Instruct` (4-bit)
- Epochs: 3
- Batch size: 24
- Max length: 512 tokens
- Masking: `-100` for loss ignoring padding
- Optimizer: `AdamW + Linear LR scheduler`
- Mixed precision: `autocast` + `GradScaler`
- Logging: `tqdm` + `train.log`

---

## 📊 Evaluation Result

Using 1,000 samples from the OpenAssistant dataset:

| Metric       | Score  |
|--------------|--------|
| ROUGE-1      | 0.2746 |
| ROUGE-2      | 0.1182 |
| ROUGE-L      | 0.1899 |
| ROUGE-Lsum   | 0.2121 |

🔎 Indicates that the model has learned the instruction-following pattern, though improvements can be made with more epochs or larger dataset slices.

---

## 💡 Key Skills Demonstrated

- [x] Efficient fine-tuning of LLMs under resource constraints
- [x] Instruction-following prompt formatting (Alpaca-style)
- [x] 4-bit quantization with BitsAndBytes
- [x] Prompt engineering and masking for causal LM
- [x] Metric-based evaluation (ROUGE)

