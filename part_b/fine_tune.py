"""
fine_tune.py - Fine-tune Whisper on Azerbaijani Common Voice.

Part B of the az-stt-intern project.

Usage:
    python part_b/fine_tune.py \
        --model_name openai/whisper-tiny \
        --dataset_name mozilla-foundation/common_voice_17_0 \
        --language az \
        --train_samples 200 \
        --eval_samples 50 \
        --output_dir results/whisper_az_finetuned
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# Allow importing utils from part_a
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part_a"))
from utils import normalize_text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Azerbaijani Common Voice."
    )
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_17_0",
    )
    parser.add_argument("--language", type=str, default="az")
    parser.add_argument(
        "--train_samples",
        type=int,
        default=200,
        help="Number of training samples (default: 200)",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=50,
        help="Number of validation samples (default: 50)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/whisper_az_finetuned",
        help="Directory to save fine-tuned model and checkpoints",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input_features and labels to the same length within each batch.
    Replaces padding token id (-100) so the loss ignores them.
    """

    def __init__(self, processor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        # Separate inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad audio features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present (Whisper adds it during generation)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(batch: dict, processor, language: str) -> dict:
    """
    Feature extraction and tokenization for a single dataset sample.
    Called via dataset.map().
    """
    audio = batch["audio"]
    array = np.array(audio["array"], dtype=np.float32)

    # Extract log-mel spectrogram features
    batch["input_features"] = processor.feature_extractor(
        array, sampling_rate=16000, return_tensors="np"
    ).input_features[0]

    # Tokenize the reference sentence
    sentence = normalize_text(batch["sentence"], remove_punctuation=False)
    batch["labels"] = processor.tokenizer(sentence).input_ids

    return batch


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def make_compute_metrics(processor, wer_metric):
    """Factory function to create compute_metrics closure."""

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 (padding) with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize before computing WER
        pred_str = [normalize_text(p) for p in pred_str]
        label_str = [normalize_text(l) for l in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Training curve plotting
# ---------------------------------------------------------------------------

def plot_training_curves(log_history: list[dict], output_path: str):
    """
    Generate and save training curves from Trainer log history.

    Args:
        log_history: trainer.state.log_history list of dicts.
        output_path: Path to save the PNG file.
    """
    train_loss_steps = []
    train_loss_vals = []
    eval_loss_steps = []
    eval_loss_vals = []
    eval_wer_steps = []
    eval_wer_vals = []

    for entry in log_history:
        step = entry.get("step", None)
        if "loss" in entry and "eval_loss" not in entry:
            train_loss_steps.append(step)
            train_loss_vals.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss_steps.append(step)
            eval_loss_vals.append(entry["eval_loss"])
        if "eval_wer" in entry:
            eval_wer_steps.append(step)
            eval_wer_vals.append(entry["eval_wer"])

    n_plots = 1 + (1 if eval_loss_vals else 0) + (1 if eval_wer_vals else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # Training loss
    if train_loss_vals:
        axes[ax_idx].plot(train_loss_steps, train_loss_vals, label="Train Loss", color="steelblue")
        axes[ax_idx].set_title("Training Loss")
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].legend()
        ax_idx += 1

    # Validation loss
    if eval_loss_vals:
        axes[ax_idx].plot(eval_loss_steps, eval_loss_vals, label="Val Loss", color="darkorange")
        axes[ax_idx].set_title("Validation Loss")
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].legend()
        ax_idx += 1

    # Validation WER
    if eval_wer_vals:
        axes[ax_idx].plot(eval_wer_steps, eval_wer_vals, label="Val WER", color="seagreen")
        axes[ax_idx].set_title("Validation WER")
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("WER")
        axes[ax_idx].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Training curves saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    print(f"[*] Device: {device} | FP16: {use_fp16}")
    print(f"[*] Model: {args.model_name}")
    print(f"[*] Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.dirname(args.output_dir) if "results" not in args.output_dir else "results"
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Load processor and model
    # -------------------------------------------------------------------
    print("\n[*] Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Set language for generation
    model.config.forced_decoder_ids = None  # We handle this via processor
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # -------------------------------------------------------------------
    # Load and prepare datasets
    # -------------------------------------------------------------------
    print(f"\n[*] Loading dataset: {args.dataset_name} [{args.language}]")

    raw_train = load_dataset(
        args.dataset_name, args.language, split="train", trust_remote_code=True
    )
    raw_eval = load_dataset(
        args.dataset_name, args.language, split="validation", trust_remote_code=True
    )

    # Subsample
    train_n = min(args.train_samples, len(raw_train))
    eval_n = min(args.eval_samples, len(raw_eval))
    raw_train = raw_train.select(range(train_n))
    raw_eval = raw_eval.select(range(eval_n))

    print(f"    Train samples: {train_n} | Eval samples: {eval_n}")

    # Resample audio to 16kHz
    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=16000))
    raw_eval = raw_eval.cast_column("audio", Audio(sampling_rate=16000))

    # Feature extraction + tokenization
    print("[*] Preparing features (this may take a few minutes)...")
    train_dataset = raw_train.map(
        prepare_dataset,
        fn_kwargs={"processor": processor, "language": args.language},
        remove_columns=raw_train.column_names,
        desc="Train feature extraction",
    )
    eval_dataset = raw_eval.map(
        prepare_dataset,
        fn_kwargs={"processor": processor, "language": args.language},
        remove_columns=raw_eval.column_names,
        desc="Eval feature extraction",
    )

    # -------------------------------------------------------------------
    # Data collator and metrics
    # -------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")
    compute_metrics = make_compute_metrics(processor, wer_metric)

    # -------------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------------
    # Handle eval_strategy vs evaluation_strategy (Transformers >=4.41 renamed it)
    import transformers
    trainer_kwargs_extra = {}
    try:
        from transformers import TrainingArguments as _TA
        import inspect
        sig = inspect.signature(_TA.__init__)
        if "eval_strategy" in sig.parameters:
            trainer_kwargs_extra["eval_strategy"] = "epoch"
        else:
            trainer_kwargs_extra["evaluation_strategy"] = "epoch"
    except Exception:
        trainer_kwargs_extra["evaluation_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
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
        report_to="none",  # Disable wandb/tensorboard
        **trainer_kwargs_extra,
    )

    # -------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    print("\n[*] Starting fine-tuning...")
    train_result = trainer.train()

    # Save best model
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n[✓] Model saved to: {args.output_dir}")

    # -------------------------------------------------------------------
    # Save training metrics CSV
    # -------------------------------------------------------------------
    log_history = trainer.state.log_history
    metrics_path = os.path.join(results_dir, "training_metrics.csv")
    metrics_df = pd.DataFrame(log_history)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[*] Training metrics saved to: {metrics_path}")

    # -------------------------------------------------------------------
    # Plot training curves
    # -------------------------------------------------------------------
    curves_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(log_history, curves_path)

    # Summary
    print("\n" + "=" * 50)
    print("  FINE-TUNING COMPLETE")
    print("=" * 50)
    print(f"  Train samples : {train_n}")
    print(f"  Eval samples  : {eval_n}")
    print(f"  Epochs        : {args.num_train_epochs}")
    print(f"  Best model    : {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
