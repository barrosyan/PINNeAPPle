"""LoRA fine-tuning of a local causal LM on data logged via
``conversation_store.ConversationStore``.

Scope, stated plainly: this fine-tunes a local Hugging-Face-format causal
LM's weights (via ``transformers``+``peft``, optional deps -- ``pip
install "pinneapple[finetune]"``). It does **not** fine-tune an Ollama
model directly -- Ollama itself does not do full/LoRA weight fine-tuning
(its ``Modelfile`` mechanism customises a system prompt/imports an
already-trained adapter, it doesn't train one). The realistic local
pipeline is: (1) fine-tune a HF-format base model here, on your logged
conversations; (2) if you want to *serve* the result via Ollama
afterwards, convert it to GGUF with ``llama.cpp``'s own conversion tooling
and ``ollama create`` from it -- both are separate, well-documented steps
outside this module's scope (GGUF conversion is its own significant tool,
not something to silently wrap).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from .conversation_store import ConversationStore


@dataclass
class FinetuneConfig:
    base_model: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "./pinneapple_finetuned"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    max_seq_length: int = 1024


def prepare_dataset(store: ConversationStore, out_jsonl: str, *, module: Optional[str] = None) -> str:
    """Export logged (prompt, response) pairs to a JSONL file, ready for
    :func:`finetune_lora`. Thin wrapper over ``ConversationStore
    .export_jsonl`` -- kept as a separate step (not folded into
    ``finetune_lora``) so the dataset can be inspected/filtered/versioned
    before spending any GPU time on it."""
    return store.export_jsonl(out_jsonl, module=module)


def finetune_lora(dataset_jsonl: str, cfg: FinetuneConfig):
    """Run a real LoRA fine-tune (``transformers.Trainer`` + ``peft
    .LoraConfig``) of ``cfg.base_model`` on the (prompt, response) pairs in
    ``dataset_jsonl`` (as written by :func:`prepare_dataset`).

    Requires the ``finetune`` extra (``transformers``, ``peft``,
    ``datasets``) -- raises a clear ``ImportError`` naming it if missing,
    same pattern as every other optional-dependency bridge in this
    package.

    Returns the path to the saved LoRA adapter (``cfg.output_dir``).
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
            Trainer, TrainingArguments,
        )
    except ImportError as e:
        raise ImportError(
            "pinneapple_llm.finetune requires the 'finetune' extra: "
            'pip install "pinneapple[finetune]" (transformers, peft, datasets).'
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    dataset = load_dataset("json", data_files=dataset_jsonl, split="train")

    def _format(example):
        text = f"{example['prompt']}\n\n{example['response']}{tokenizer.eos_token}"
        enc = tokenizer(text, truncation=True, max_length=cfg.max_seq_length, padding="max_length")
        enc["labels"] = list(enc["input_ids"])
        return enc

    tokenized = dataset.map(_format, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "pinneapple_finetune_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    return cfg.output_dir
