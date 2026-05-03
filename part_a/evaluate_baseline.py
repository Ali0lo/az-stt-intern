"""
evaluate_baseline.py - Baseline Whisper ASR evaluation on Azerbaijani speech datasets.

Part A of the az-stt-intern project.

Recommended usage with Google FLEURS Azerbaijani:

    python part_a/evaluate_baseline.py \
        --model_name openai/whisper-small \
        --dataset_name google/fleurs \
        --language az_az \
        --split test \
        --max_samples 50 \
        --output_dir results
"""

import argparse
import os
import sys

# Allow importing utils.py when script is run from repo root
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
        description="Evaluate a Whisper baseline model on an Azerbaijani ASR dataset."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="Hugging Face Whisper model ID. Default: openai/whisper-small",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="google/fleurs",
        help="Hugging Face dataset ID. Default: google/fleurs",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="az_az",
        help=(
            "Dataset config/language. "
            "Use 'az_az' for google/fleurs or 'az' for Common Voice."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on. Default: test",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate. Default: 50",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results CSV. Default: results",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def whisper_language_from_dataset_config(language: str) -> str:
    """
    Convert dataset config names to Whisper language names.

    Dataset configs and Whisper decoding language names are not always the same.

    Examples:
    - FLEURS Azerbaijani: az_az
    - Common Voice Azerbaijani: az
    - Whisper expected name: azerbaijani
    """
    mapping = {
        "az": "azerbaijani",
        "az_az": "azerbaijani",
        "azerbaijani": "azerbaijani",
    }
    return mapping.get(language, language)


def get_reference_text(sample: dict) -> str:
    """
    Extract reference transcription from different dataset formats.

    Common Voice usually uses:
    - sentence

    FLEURS usually uses:
    - transcription
    - raw_transcription
    """
    possible_columns = [
        "sentence",
        "transcription",
        "raw_transcription",
        "text",
    ]

    for col in possible_columns:
        if col in sample and sample[col]:
            return str(sample[col])

    raise KeyError(
        f"No reference text column found. Available columns: {list(sample.keys())}"
    )


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
    """
    audio = sample["audio"]
    array = audio["array"].astype(np.float32)
    sampling_rate = audio["sampling_rate"]

    # Whisper expects 16 kHz audio
    if sampling_rate != 16000:
        import librosa

        array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=16000,
        )
        sampling_rate = 16000

    inputs = processor(
        array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device)

    whisper_language = whisper_language_from_dataset_config(language)

    # Important:
    # The dataset config can be "az_az", but Whisper expects "azerbaijani".
    model.generation_config.language = whisper_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=128,
            do_sample=False,
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return transcription.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "baseline_results.csv")

    print(
        f"\n[*] Loading dataset: {args.dataset_name} "
        f"(config={args.language}, split={args.split})"
    )
    print("    This may take a moment on first run...")

    dataset = load_dataset(
        args.dataset_name,
        args.language,
        split=args.split,
        trust_remote_code=True,
    )

    total = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(total))

    print(f"    Loaded {total} samples from '{args.split}' split.")

    model, processor = load_model_and_processor(args.model_name, device)

    print(f"\n[*] Running inference on {total} samples...")

    results = []

    for idx, sample in enumerate(tqdm(dataset, total=total, desc="Evaluating")):
        try:
            reference_raw = get_reference_text(sample)
        except Exception as e:
            print(f"\n[!] Could not read reference for sample {idx}: {e}")
            reference_raw = ""

        duration = get_audio_duration(sample)

        try:
            prediction_raw = transcribe_sample(
                model=model,
                processor=processor,
                sample=sample,
                language=args.language,
                device=device,
            )
        except Exception as e:
            print(f"\n[!] Error during inference on sample {idx}: {e}")
            prediction_raw = ""

        # Debug first few samples
        if idx < 3:
            print("\n" + "-" * 60)
            print(f"Sample {idx}")
            print(f"Reference raw : {reference_raw}")
            print(f"Prediction raw: {repr(prediction_raw)}")
            print("-" * 60)

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

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n[*] Results saved to: {output_csv}")

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

    best = df.nsmallest(5, "wer").to_dict("records")
    print("\n  BEST 5 SAMPLES (lowest WER):")
    print(format_results_table(best, n=5))

    worst = df.nlargest(5, "wer").to_dict("records")
    print("\n  WORST 5 SAMPLES (highest WER):")
    print(format_results_table(worst, n=5))

    print(f"\n[✓] Done. Results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()