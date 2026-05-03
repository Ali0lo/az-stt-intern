"""
evaluate_finetuned.py - Evaluate a fine-tuned Whisper model on Azerbaijani Common Voice.

Part B of the az-stt-intern project.

Usage:
    python part_b/evaluate_finetuned.py \
        --model_path results/whisper_az_finetuned \
        --dataset_name mozilla-foundation/common_voice_17_0 \
        --language az \
        --split test \
        --max_samples 50 \
        --output_dir results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Allow importing utils from part_a
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part_a"))
from utils import compute_wer_cer, format_results_table, get_audio_duration, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on Azerbaijani Common Voice."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/whisper_az_finetuned",
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_17_0",
    )
    parser.add_argument("--language", type=str, default="az")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def transcribe(model, processor, sample: dict, language: str, device: str) -> str:
    """Run inference on one audio sample."""
    audio = sample["audio"]
    array = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]

    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)

    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Loading fine-tuned model from: {args.model_path}")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()

    print(f"[*] Loading dataset: {args.dataset_name} [{args.language}] split={args.split}")
    dataset = load_dataset(
        args.dataset_name,
        args.language,
        split=args.split,
        trust_remote_code=True,
    )
    total = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(total))
    print(f"    {total} samples loaded.")

    results = []
    for idx, sample in enumerate(tqdm(dataset, total=total, desc="Evaluating fine-tuned")):
        reference_raw = sample.get("sentence", "")
        duration = get_audio_duration(sample)

        try:
            prediction_raw = transcribe(model, processor, sample, args.language, device)
        except Exception as e:
            print(f"\n[!] Error on sample {idx}: {e}")
            prediction_raw = ""

        reference = normalize_text(reference_raw)
        prediction = normalize_text(prediction_raw)
        wer, cer = compute_wer_cer(reference, prediction)

        results.append(
            {
                "id": idx,
                "reference": reference,
                "prediction": prediction,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "audio_duration": round(duration, 2),
            }
        )

    df = pd.DataFrame(results)
    output_csv = os.path.join(args.output_dir, "finetuned_results.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8")

    avg_wer = df["wer"].mean()
    avg_cer = df["cer"].mean()

    print("\n" + "=" * 55)
    print("  FINE-TUNED MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Model      : {args.model_path}")
    print(f"  Samples    : {total}")
    print(f"  Avg WER    : {avg_wer * 100:.2f}%")
    print(f"  Avg CER    : {avg_cer * 100:.2f}%")
    print("=" * 55)

    best = df.nsmallest(5, "wer").to_dict("records")
    print("\n  BEST 5 SAMPLES:")
    print(format_results_table(best))

    worst = df.nlargest(5, "wer").to_dict("records")
    print("\n  WORST 5 SAMPLES:")
    print(format_results_table(worst))

    print(f"\n[✓] Results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
