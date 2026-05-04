"""
fine_tune.py - Fine-tune Whisper on Azerbaijani speech datasets.

Part B of the az-stt-intern project.

Recommended usage with Google FLEURS Azerbaijani:

    python part_b/fine_tune.py \
        --model_name openai/whisper-tiny \
        --dataset_name google/fleurs \
        --language az_az \
        --train_samples 50 \
        --eval_samples 10 \
        --output_dir results/whisper_az_finetuned
"""

import argparse
import inspect
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import evaluate

# Allow importing utils.py from part_a when this script is run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part_a"))
from utils import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on an Azerbaijani ASR dataset."
    )
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
    parser.add_argument("--dataset_name", type=str, default="google/fleurs")
    parser.add_argument(
        "--language",
        type=str,
        default="az_az",
        help="Dataset config. Use 'az_az' for google/fleurs or 'az' for Common Voice.",
    )
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/whisper_az_finetuned",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    return parser.parse_args()


def whisper_language_from_dataset_config(language: str) -> str:
    """
    Convert dataset config names to Whisper language names.

    FLEURS Azerbaijani uses 'az_az', while Whisper expects 'azerbaijani'.
    """
    mapping = {
        "az": "azerbaijani",
        "az_az": "azerbaijani",
        "azerbaijani": "azerbaijani",
    }
    return mapping.get(language, language)


def get_reference_text(example: Dict) -> str:
    """
    Extract reference transcription from different ASR dataset formats.
    """
    for col in ["sentence", "transcription", "raw_transcription", "text"]:
        if col in example and example[col]:
            return str(example[col])

    raise KeyError(
        f"No reference text column found. Available columns: {list(example.keys())}"
    )


def get_results_dir(output_dir: str) -> str:
    parts = os.path.normpath(output_dir).split(os.sep)
    if "results" in parts:
        return "results"
    return os.path.dirname(output_dir) or "."


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input features and labels for Whisper seq2seq training.
    """

    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if labels.shape[1] > 0 and (
            labels[:, 0] == self.decoder_start_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch: Dict, processor: WhisperProcessor) -> Dict:
    """
    Extract log-mel input features and tokenize the target transcription.
    """
    audio = batch["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)

    batch["input_features"] = processor.feature_extractor(
        array,
        sampling_rate=16000,
        return_tensors="np",
    ).input_features[0]

    reference_text = get_reference_text(batch)
    reference_text = normalize_text(reference_text, remove_punctuation=False)
    batch["labels"] = processor.tokenizer(reference_text).input_ids

    return batch


def make_compute_metrics(processor: WhisperProcessor, wer_metric):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(
            label_ids == -100,
            processor.tokenizer.pad_token_id,
            label_ids,
        )

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_text(text) for text in pred_str]
        label_str = [normalize_text(text) for text in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def plot_training_curves(log_history: List[Dict], output_path: str):
    train_loss_steps = []
    train_loss_vals = []
    eval_loss_steps = []
    eval_loss_vals = []
    eval_wer_steps = []
    eval_wer_vals = []

    for entry in log_history:
        step = entry.get("step")

        if "loss" in entry and "eval_loss" not in entry:
            train_loss_steps.append(step)
            train_loss_vals.append(entry["loss"])

        if "eval_loss" in entry:
            eval_loss_steps.append(step)
            eval_loss_vals.append(entry["eval_loss"])

        if "eval_wer" in entry:
            eval_wer_steps.append(step)
            eval_wer_vals.append(entry["eval_wer"])

    n_plots = 1 + int(bool(eval_loss_vals)) + int(bool(eval_wer_vals))
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    if train_loss_vals:
        axes[ax_idx].plot(train_loss_steps, train_loss_vals, label="Train Loss")
        axes[ax_idx].set_title("Training Loss")
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].legend()
        ax_idx += 1

    if eval_loss_vals:
        axes[ax_idx].plot(eval_loss_steps, eval_loss_vals, label="Validation Loss")
        axes[ax_idx].set_title("Validation Loss")
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].legend()
        ax_idx += 1

    if eval_wer_vals:
        axes[ax_idx].plot(eval_wer_steps, eval_wer_vals, label="Validation WER")
        axes[ax_idx].set_title("Validation WER")
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("WER")
        axes[ax_idx].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[*] Training curves saved to: {output_path}")


def get_eval_strategy_kwargs() -> Dict[str, str]:
    """
    Support both older and newer Transformers versions.
    """
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)

    if "eval_strategy" in sig.parameters:
        return {"eval_strategy": "epoch"}

    return {"evaluation_strategy": "epoch"}


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    whisper_language = whisper_language_from_dataset_config(args.language)

    print(f"[*] Device: {device} | FP16: {use_fp16}")
    print(f"[*] Model: {args.model_name}")
    print(f"[*] Dataset: {args.dataset_name} [{args.language}]")
    print(f"[*] Whisper language: {whisper_language}")
    print(f"[*] Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = get_results_dir(args.output_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("\n[*] Loading processor and model...")

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=whisper_language,
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = whisper_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print(f"\n[*] Loading dataset: {args.dataset_name} [{args.language}]")

    raw_train = load_dataset(
        args.dataset_name,
        args.language,
        split="train",
        trust_remote_code=True,
    )

    raw_eval = load_dataset(
        args.dataset_name,
        args.language,
        split="validation",
        trust_remote_code=True,
    )

    train_n = min(args.train_samples, len(raw_train))
    eval_n = min(args.eval_samples, len(raw_eval))

    raw_train = raw_train.select(range(train_n))
    raw_eval = raw_eval.select(range(eval_n))

    print(f"    Train samples: {train_n}")
    print(f"    Eval samples : {eval_n}")

    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=16000))
    raw_eval = raw_eval.cast_column("audio", Audio(sampling_rate=16000))

    print("\n[*] Preparing features...")

    train_dataset = raw_train.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=raw_train.column_names,
        desc="Train feature extraction",
    )

    eval_dataset = raw_eval.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=raw_eval.column_names,
        desc="Eval feature extraction",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")
    compute_metrics = make_compute_metrics(processor, wer_metric)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="epoch",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        **get_eval_strategy_kwargs(),
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Transformers compatibility: older versions use tokenizer=,
    # newer versions prefer processing_class=.
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("\n[*] Starting fine-tuning...")
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"\n[✓] Model and processor saved to: {args.output_dir}")

    log_history = trainer.state.log_history

    metrics_path = os.path.join(results_dir, "training_metrics.csv")
    pd.DataFrame(log_history).to_csv(metrics_path, index=False)
    print(f"[*] Training metrics saved to: {metrics_path}")

    curves_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(log_history, curves_path)

    print("\n" + "=" * 60)
    print("  FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Train samples : {train_n}")
    print(f"  Eval samples  : {eval_n}")
    print(f"  Epochs        : {args.num_train_epochs}")
    print(f"  Best model    : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
