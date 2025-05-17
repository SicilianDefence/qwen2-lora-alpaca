import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate
import logging
from tqdm import tqdm
from datetime import datetime

# ========== SETUP LOGGING ========== #
os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/eval_rouge_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
print = lambda *args, **kwargs: logger.info(" ".join(map(str, args)))

# ========== LOAD TOKENIZER & BASE MODEL + LoRA ADAPTER ========== #
base_model = "Qwen/Qwen2-1.5B-Instruct"
adapter_path = "checkpoints/latest"

# ✅ Load tokenizer dari adapter (agar vocab cocok)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

# ✅ Load base model + quant
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))

# ✅ Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# ========== LOAD DATASET ========== #
print("📥 Loading dataset OpenAssistant...")
eval_dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # Ganti dengan dataset yang sesuai

# ========== FORMAT ALPACA-STYLE PROMPT ========== #
def format_like_alpaca(example):
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output_text = example["output"].strip()

    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{instruction}"
    if input_text:
        prompt += f"\n{input_text}"
    prompt += "\n<|assistant|>\n"

    return {
        "prompt": prompt.strip(),
        "reference": output_text
    }

formatted_eval = eval_dataset.map(format_like_alpaca)

# ========== GENERATE OUTPUT ========== #
def generate_output(example):
    inputs = tokenizer(example["prompt"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "<|assistant|>" in output:
        output = output.split("<|assistant|>")[-1].strip()
    return {"generated": output}

print("🚀 Generating output...")
generated_results = formatted_eval.map(generate_output)

# ========== HITUNG ROUGE ========== #
print("🧪 Evaluating ROUGE...")
rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=generated_results["generated"],
    references=generated_results["reference"]
)

print("📊 ROUGE Evaluation Result:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

print("✅ Evaluasi selesai. Hasil log disimpan di:", log_filename)
