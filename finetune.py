from pathlib import Path
from random import randrange
import json
import torch.serialization
import numpy.core.multiarray
import importlib

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer
import matplotlib.pyplot as plt

torch.serialization.add_safe_globals(
    [numpy.core.multiarray._reconstruct]
)

dataset_path   = "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/data/train_out.csv"
model_id       = "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/meditron-7b"          
results_dir    = Path("/scratch/network/jh3258/meditron-fine-tune-rare-diseases/results")
results_dir.mkdir(parents=True, exist_ok=True)

num_epochs     = 20
batch_size     = 6
grad_acc_steps = 2
seed           = 42

lora_confs = {
    "rank4": LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM"),
    "rank8": LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM"),
}

dataset = load_dataset("csv", data_files=dataset_path)["train"].shuffle(seed=seed)

def format_instruction(sample):
    if sample["category"] == "RAREDISEASE":
        prompt = (
            "### Task:\nExtract the exact name or names of rare diseases from the "
            "input text and output them in a list.\n### Definition:\nRare diseases "
            "are defined as diseases that affect a small number of people compared "
            "to the general population.\n### Input Text: "
        )
    elif sample["category"] == "DISEASE":
        prompt = (
            "### Task:\nExtract the exact name or names of diseases from the input "
            "text and output them in a list.\n### Definition:\nDiseases are defined "
            "as abnormal conditions resulting from various causes, such as "
            "infection, inflammation, environmental factors, or genetic defect, "
            "and characterized by an identifiable group of signs, symptoms, or both."
            "\n### Input Text: "
        )
    elif sample["category"] == "SYMPTOM":
        prompt = (
            "### Task:\nExtract the exact name or names of symptoms from the input "
            "text and output them in a list.\n### Definition:\nSymptoms are defined "
            "as physical or mental problems that cannot be measured from tests or "
            "observed by a doctor.\n### Input Text: "
        )
    elif sample["category"] == "SIGN":
        prompt = (
            "### Task:\nExtract the exact name or names of signs from the input "
            "text and output them in a list.\n### Definition:\nSigns are defined as "
            "physical or mental problems that can be measured from tests or "
            "observed by a doctor.\n### Input Text: "
        )
    else:
        prompt = "### Input Text: "

    return f"""{prompt}
{sample['context']}
### Output:
{sample['response']}
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_losses = {}  

for rank_name, peft_cfg in lora_confs.items():
    print(f"\n=== Training with LoRA {rank_name} ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_cfg)

    output_dir = results_dir / f"finetuned_{rank_name}"
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True,      
        report_to="none",       
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_cfg,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=args,
    )
    
    ckpt = output_dir / "checkpoint-240"
    if ckpt.exists():
        print(f"Resuming from {ckpt}")
        trainer.train(resume_from_checkpoint=str(ckpt))
    else:
        trainer.train()
    
    trainer.save_model()
    
    adapter_dir   = results_dir / f"{rank_name}/adapter"
    tokenizer_dir = results_dir / f"{rank_name}/tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(adapter_dir)        
    tokenizer.save_pretrained(tokenizer_dir)  

    epochs, losses = [], []
    for entry in trainer.state.log_history:
        if "loss" in entry and "epoch" in entry:
            epochs.append(entry["epoch"])
            losses.append(entry["loss"])

    lora_losses[rank_name] = {"epochs": epochs, "losses": losses}

    with open(results_dir / f"loss_history_{rank_name}.json", "w") as f:
        json.dump(lora_losses[rank_name], f, indent=2)

files = {
    "LoRA rank 4": results_dir / "loss_history_rank4.json",
    "LoRA rank 8": results_dir / "loss_history_rank8.json",
}

lora_losses = {}
for label, fp in files.items():
    with open(fp) as f:
        lora_losses[label] = json.load(f)

plt.figure(figsize=(10, 6))
for label, data in lora_losses.items():
    plt.plot(data["epochs"], data["losses"], marker="o", label=label)

plt.title("Loss vs Epoch for Meditron Fine-Tuning")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.grid(True)

plot_path = results_dir / "loss_vs_epoch_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {plot_path}")
