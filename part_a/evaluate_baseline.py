"""
evaluate_baseline.py - Baseline Whisper ASR evaluation on Azerbaijani Common Voice.

Part A of the az-stt-intern project.

Usage:
    python part_a/evaluate_baseline.py \
        --model_name openai/whisper-small \
        --dataset_name mozilla-foundation/common_voice_17_0 \
        --language az \
        --split test \
        --max_samples 50 \
        --output_dir results
"""

import argparse
import os
import sys

# Allow importing from part_a even when script is run from repo root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils import compute_wer_cer, format_results_table, get_audio_duration, normalize_text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper baseline model on Azerbaijani Common Voice."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="Hugging Face model ID for Whisper (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_17_0",
        help="Hugging Face dataset ID",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="az",
        help="Language/config name for Common Voice dataset (default: az)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results CSV (default: results/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model_and_processor(model_name: str, device: str):
    """Load Whisper model and processor from Hugging Face."""
    print(f"\n[*] Loading model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"    Model loaded on device: {device}")
    return model, processor


def transcribe_sample(
    model,
    processor,
    sample: dict,
    language: str,
    device: str,
) -> str:
    """
    Run Whisper inference on a single audio sample.

    Args:
        model:     Loaded WhisperForConditionalGeneration model.
        processor: Loaded WhisperProcessor.
        sample:    Dataset row containing 'audio' dict with 'array' and 'sampling_rate'.
        language:  Target language code (e.g. 'az').
        device:    'cuda' or 'cpu'.

    Returns:
        Transcribed text string.
    """
    audio = sample["audio"]
    array = audio["array"].astype(np.float32)
    sr = audio["sampling_rate"]

    # Resample to 16kHz if needed (Common Voice is usually already 48kHz → need resample)
    if sr != 16000:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)

    # Prepare input features
    inputs = processor(
        array,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    # Force Azerbaijani language decoding
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "baseline_results.csv")

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    print(f"\n[*] Loading dataset: {args.dataset_name} (config={args.language}, split={args.split})")
    print(f"    This may take a moment on first run (downloading audio files)...")

    dataset = load_dataset(
        args.dataset_name,
        args.language,
        split=args.split,
        trust_remote_code=True,
    )

    # Limit to max_samples
    total = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(total))
    print(f"    Loaded {total} samples from '{args.split}' split.")

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    model, processor = load_model_and_processor(args.model_name, device)

    # -------------------------------------------------------------------
    # Inference loop
    # -------------------------------------------------------------------
    print(f"\n[*] Running inference on {total} samples...")
    results = []

    for idx, sample in enumerate(tqdm(dataset, total=total, desc="Evaluating")):
        reference_raw = sample.get("sentence", "")
        duration = get_audio_duration(sample)

        try:
            prediction_raw = transcribe_sample(model, processor, sample, args.language, device)
        except Exception as e:
            print(f"\n    [!] Error on sample {idx}: {e}")
            prediction_raw = ""

        # Normalize both reference and prediction
        reference = normalize_text(reference_raw, remove_punctuation=True)
        prediction = normalize_text(prediction_raw, remove_punctuation=True)

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

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[*] Results saved to: {output_csv}")

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------
    avg_wer = df["wer"].mean()
    avg_cer = df["cer"].mean()

    print("\n" + "=" * 60)
    print("  BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model        : {args.model_name}")
    print(f"  Dataset      : {args.dataset_name} [{args.language}]")
    print(f"  Split        : {args.split}")
    print(f"  Samples      : {total}")
    print(f"  Average WER  : {avg_wer * 100:.2f}%")
    print(f"  Average CER  : {avg_cer * 100:.2f}%")
    print("=" * 60)

    # Best 5 (lowest WER)
    best = df.nsmallest(5, "wer").to_dict("records")
    print("\n  BEST 5 SAMPLES (lowest WER):")
    print(format_results_table(best, n=5))

    # Worst 5 (highest WER)
    worst = df.nlargest(5, "wer").to_dict("records")
    print("\n  WORST 5 SAMPLES (highest WER):")
    print(format_results_table(worst, n=5))

    print(f"\n[✓] Done. Results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
