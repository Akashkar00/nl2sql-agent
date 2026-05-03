"""
Training entrypoint. Run from Kaggle notebook (or locally if you have GPU).

Usage:
    python -m src.train --config configs/training_config.yaml

Reads config, loads Unsloth-quantized base model, applies QLoRA, trains, pushes to HF Hub.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Late imports — Unsloth must be imported before transformers in some envs
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # === Load model ===
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["base_model"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # === Apply LoRA ===
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg["lora"]["random_state"],
    )

    # === Load dataset ===
    train_path = cfg["data"]["train_path"]
    eval_path = cfg["data"]["eval_path"]

    data_files = {"train": train_path}
    if Path(eval_path).exists():
        data_files["eval"] = eval_path
    dataset = load_dataset("json", data_files=data_files)

    # === W&B ===
    if cfg.get("wandb"):
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
        report_to = "wandb"
    else:
        report_to = "none"

    # === Training args ===
    t = cfg["training"]
    hub_kwargs = {}
    if cfg.get("hub", {}).get("push_to_hub"):
        hub_repo = os.getenv("HF_HUB_REPO")
        if not hub_repo:
            raise RuntimeError("HF_HUB_REPO env var required when push_to_hub=true")
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": hub_repo,
            "hub_strategy": cfg["hub"].get("hub_strategy", "every_save"),
        }

    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        seed=t["seed"],
        run_name=cfg.get("wandb", {}).get("run_name"),
        report_to=report_to,
        **hub_kwargs,
    )

    # === Trainer ===
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        dataset_text_field=cfg["data"]["text_field"],
        max_seq_length=cfg["model"]["max_seq_length"],
        args=training_args,
        packing=False,
    )

    print(f"training on {len(dataset['train'])} examples")
    trainer.train()

    # Final save + push
    trainer.save_model(t["output_dir"])
    if cfg.get("hub", {}).get("push_to_hub"):
        trainer.push_to_hub()
        print(f"final adapter pushed to {os.environ['HF_HUB_REPO']}")


if __name__ == "__main__":
    main()
