import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from datasets import load_dataset
from tqdm import tqdm
import time
import logging

# ========== SETUP LOGGING ==========
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
print = lambda *args, **kwargs: logging.info(" ".join(map(str, args)))

# ========== SETUP CUDA & ENV ==========
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=' + os.environ.get('CONDA_PREFIX', '')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("CUDA available:", torch.cuda.is_available())
print("Jumlah GPU:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Nama GPU:", torch.cuda.get_device_name(0))
    print("Memory GPU (MB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2))
else:
    print("GPU tidak terdeteksi. Pastikan driver & CUDA Toolkit sudah benar.")

# ========== LOAD DATA ==========
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_ultrachat_clean(example):
    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{example['instruction'].strip()}\n"
    if example["input"]:
        prompt += f"{example['input'].strip()}\n"
    prompt += "<|assistant|>\n"
    return {
        "prompt": prompt.strip(),
        "output": example["output"].strip()
    }

formatted_dataset = dataset.map(format_ultrachat_clean, remove_columns=dataset.column_names)

# ========== LOAD MODEL ==========
student_id = "Qwen/Qwen2-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

student_tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
student_model = AutoModelForCausalLM.from_pretrained(
    student_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

student_model = prepare_model_for_kbit_training(student_model, use_gradient_checkpointing=True)

if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token or "[PAD]"
student_model.resize_token_embeddings(len(student_tokenizer))

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
student_model = get_peft_model(student_model, lora_config)
student_model.config.use_cache = False
student_model = torch.compile(student_model, mode="reduce-overhead")

# ========== TOKENIZATION ==========
def tokenize_for_supervised(batch):
    full_texts = [p + o for p, o in zip(batch["prompt"], batch["output"])]
    tokenized = student_tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    labels = [
        [token if mask else -100 for token, mask in zip(input_ids, attn)]
        for input_ids, attn in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = formatted_dataset.map(
    tokenize_for_supervised,
    batched=True,
    num_proc=4,
    remove_columns=["prompt", "output"]
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ========== TRAINING SETUP ==========
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=24,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
num_epochs = 3
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

scaler = GradScaler()

student_model.train()
student_model.to("cuda")

os.makedirs("checkpoints", exist_ok=True)

# ========== TRAINING LOOP ==========
for epoch in range(num_epochs):
    print(f"🚀 Memulai Epoch {epoch+1}/{num_epochs}")
    start = time.time()
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for step, batch in enumerate(loop):
        input_ids = batch["input_ids"].to("cuda", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
        labels = batch["labels"].to("cuda", non_blocking=True)

        with autocast():
            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())
        print(f"[Epoch {epoch+1}] Step {step+1}: Loss = {loss.item():.4f}")

    duration = (time.time() - start) / 60
    print(f"✅ Epoch {epoch+1} selesai dalam {duration:.2f} menit.")
    
    # Simpan model
    checkpoint_dir = "checkpoints/latest"
    os.makedirs(checkpoint_dir, exist_ok=True)
    student_model.save_pretrained(checkpoint_dir)
    student_tokenizer.save_pretrained(checkpoint_dir)
    print(f"📦 Model disimpan (replace) di {checkpoint_dir}")
    print(f"✅ Selesai Epoch {epoch+1}/{num_epochs} dalam {duration:.2f} menit.")
