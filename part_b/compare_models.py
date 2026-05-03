"""
compare_models.py - Side-by-side comparison of baseline vs fine-tuned Whisper.

Evaluates BOTH models on the EXACT SAME test subset to ensure a fair comparison.
Saves a CSV summary and a bar chart.

Part B of the az-stt-intern project.

Usage:
    python part_b/compare_models.py \
        --baseline_model openai/whisper-small \
        --finetuned_model results/whisper_az_finetuned \
        --dataset_name mozilla-foundation/common_voice_17_0 \
        --language az \
        --test_samples 50 \
        --output_dir results
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part_a"))
from utils import compute_wer_cer, get_audio_duration, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs fine-tuned Whisper model on the same test subset."
    )
    parser.add_argument(
        "--baseline_model", type=str, default="openai/whisper-small",
        help="HF model ID for baseline Whisper"
    )
    parser.add_argument(
        "--finetuned_model", type=str, default="results/whisper_az_finetuned",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mozilla-foundation/common_voice_17_0"
    )
    parser.add_argument("--language", type=str, default="az")
    parser.add_argument("--test_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def load_model(model_path: str, device: str):
    """Load a Whisper model and processor from a HF ID or local directory."""
    print(f"  Loading: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor


def evaluate_model(
    model,
    processor,
    dataset,
    language: str,
    device: str,
    model_label: str,
) -> dict:
    """
    Run evaluation loop on a dataset and return aggregate metrics.

    Returns:
        dict with keys: model, average_wer, average_cer, num_samples
    """
    wers, cers = [], []

    for sample in tqdm(dataset, desc=f"  Evaluating [{model_label}]"):
        reference_raw = sample.get("sentence", "")
        audio = sample["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        if sr != 16000:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)

        try:
            inputs = processor(array, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
            with torch.no_grad():
                pred_ids = model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids
                )
            prediction_raw = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"\n  [!] Error: {e}")
            prediction_raw = ""

        ref = normalize_text(reference_raw)
        hyp = normalize_text(prediction_raw)
        wer, cer = compute_wer_cer(ref, hyp)
        wers.append(wer)
        cers.append(cer)

    return {
        "model": model_label,
        "average_wer": round(float(np.mean(wers)), 4),
        "average_cer": round(float(np.mean(cers)), 4),
        "num_samples": len(wers),
    }


def plot_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Save a grouped bar chart comparing WER and CER across models."""
    models = comparison_df["model"].tolist()
    wers = [v * 100 for v in comparison_df["average_wer"].tolist()]
    cers = [v * 100 for v in comparison_df["average_cer"].tolist()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, wers, width, label="WER (%)", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, cers, width, label="CER (%)", color="darkorange", alpha=0.85)

    ax.set_title("Baseline vs Fine-tuned Model: WER & CER Comparison", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with values
    for bar in bars1:
        ax.annotate(
            f"{bar.get_height():.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        ax.annotate(
            f"{bar.get_height():.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Comparison chart saved to: {output_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Load dataset ONCE — both models evaluate on the exact same subset
    # -------------------------------------------------------------------
    print(f"\n[*] Loading test dataset: {args.dataset_name} [{args.language}]")
    dataset = load_dataset(
        args.dataset_name,
        args.language,
        split="test",
        trust_remote_code=True,
    )
    total = min(args.test_samples, len(dataset))
    dataset = dataset.select(range(total))
    print(f"    {total} samples loaded.\n")

    comparison_rows = []

    # -------------------------------------------------------------------
    # Baseline model
    # -------------------------------------------------------------------
    print("[*] Evaluating BASELINE model...")
    baseline_model, baseline_processor = load_model(args.baseline_model, device)
    baseline_results = evaluate_model(
        baseline_model, baseline_processor, dataset, args.language, device,
        model_label="Baseline (whisper-small)"
    )
    comparison_rows.append(baseline_results)

    # Free GPU memory before loading the next model
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Fine-tuned model
    # -------------------------------------------------------------------
    print("\n[*] Evaluating FINE-TUNED model...")
    ft_model, ft_processor = load_model(args.finetuned_model, device)
    ft_results = evaluate_model(
        ft_model, ft_processor, dataset, args.language, device,
        model_label="Fine-tuned (whisper-tiny)"
    )
    comparison_rows.append(ft_results)

    del ft_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Save comparison CSV
    # -------------------------------------------------------------------
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[*] Comparison CSV saved to: {csv_path}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    chart_path = os.path.join(args.output_dir, "wer_cer_comparison.png")
    plot_comparison(comparison_df, chart_path)

    # -------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<35} {'Avg WER':>8} {'Avg CER':>8} {'Samples':>8}")
    print("  " + "-" * 56)
    for _, row in comparison_df.iterrows():
        print(
            f"  {row['model']:<35} "
            f"{row['average_wer']*100:>7.2f}% "
            f"{row['average_cer']*100:>7.2f}% "
            f"{row['num_samples']:>8}"
        )
    print("=" * 60)

    # WER improvement
    if len(comparison_rows) == 2:
        baseline_wer = comparison_rows[0]["average_wer"]
        ft_wer = comparison_rows[1]["average_wer"]
        delta = (baseline_wer - ft_wer) / baseline_wer * 100
        direction = "improvement" if delta > 0 else "regression"
        print(f"\n  WER {direction}: {abs(delta):.1f}% relative")

    print(f"\n[✓] Done.\n")


if __name__ == "__main__":
    main()
