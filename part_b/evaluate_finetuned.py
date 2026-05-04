"""
evaluate_finetuned.py - Evaluate a fine-tuned Whisper model on Azerbaijani speech datasets.

Part B of the az-stt-intern project.

Recommended usage with Google FLEURS Azerbaijani:

    python part_b/evaluate_finetuned.py \
        --model_path results/whisper_az_finetuned \
        --dataset_name google/fleurs \
        --language az_az \
        --split test \
        --max_samples 50 \
        --output_dir results
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Allow importing utils.py from part_a when this script is run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part_a"))
from utils import compute_wer_cer, format_results_table, get_audio_duration, normalize_text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on an Azerbaijani ASR dataset."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="results/whisper_az_finetuned",
        help="Path to the fine-tuned Whisper model.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="google/fleurs",
        help="Hugging Face dataset ID.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="az_az",
        help=(
            "Dataset config/language. Use 'az_az' for google/fleurs "
            "or 'az' for Common Voice."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def whisper_language_from_dataset_config(language: str) -> str:
    """
    Convert dataset config names to Whisper language names.

    FLEURS Azerbaijani uses:
    - az_az

    Common Voice Azerbaijani usually uses:
    - az

    Whisper expects:
    - azerbaijani
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

    Common Voice usually uses:
    - sentence

    FLEURS usually uses:
    - transcription
    - raw_transcription
    """
    for col in ["sentence", "transcription", "raw_transcription", "text"]:
        if col in example and example[col]:
            return str(example[col])

    raise KeyError(
        f"No reference text column found. Available columns: {list(example.keys())}"
    )


def load_model_and_processor(model_path: str, whisper_language: str, device: str):
    """Load fine-tuned Whisper model and processor."""
    print(f"\n[*] Loading fine-tuned model from: {model_path}")
    print(f"[*] Whisper language: {whisper_language}")

    processor = WhisperProcessor.from_pretrained(
        model_path,
        language=whisper_language,
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model.generation_config.language = whisper_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print(f"    Model loaded on device: {device}")
    return model, processor


def transcribe_sample(
    model,
    processor,
    sample: Dict,
    whisper_language: str,
    device: str,
) -> str:
    """
    Run Whisper inference on one audio sample.
    """
    audio = sample["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = audio["sampling_rate"]

    # Audio is usually already cast to 16 kHz, but keep this safe.
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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_language = whisper_language_from_dataset_config(args.language)

    print(f"[*] Using device: {device}")
    print(f"[*] Dataset language/config: {args.language}")
    print(f"[*] Whisper language: {whisper_language}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "finetuned_results.csv")

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    print(
        f"\n[*] Loading dataset: {args.dataset_name} "
        f"(config={args.language}, split={args.split})"
    )

    dataset = load_dataset(
        args.dataset_name,
        args.language,
        split=args.split,
        trust_remote_code=True,
    )

    total = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(total))

    # Ensure audio is resampled to 16 kHz for Whisper.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"    Loaded {total} samples from '{args.split}' split.")

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    model, processor = load_model_and_processor(
        model_path=args.model_path,
        whisper_language=whisper_language,
        device=device,
    )

    # -------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------
    print(f"\n[*] Running fine-tuned model inference on {total} samples...")

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
                whisper_language=whisper_language,
                device=device,
            )
        except Exception as e:
            print(f"\n[!] Error during inference on sample {idx}: {e}")
            prediction_raw = ""

        # Debug first few samples to avoid fake 0% WER/CER from empty text.
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

    # -------------------------------------------------------------------
    # Save and summarize
    # -------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    avg_wer = df["wer"].mean()
    avg_cer = df["cer"].mean()

    print("\n" + "=" * 60)
    print("  FINE-TUNED MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model      : {args.model_path}")
    print(f"  Dataset    : {args.dataset_name} [{args.language}]")
    print(f"  Split      : {args.split}")
    print(f"  Samples    : {total}")
    print(f"  Avg WER    : {avg_wer * 100:.2f}%")
    print(f"  Avg CER    : {avg_cer * 100:.2f}%")
    print("=" * 60)

    best = df.nsmallest(5, "wer").to_dict("records")
    print("\n  BEST 5 SAMPLES:")
    print(format_results_table(best, n=5))

    worst = df.nlargest(5, "wer").to_dict("records")
    print("\n  WORST 5 SAMPLES:")
    print(format_results_table(worst, n=5))

    print(f"\n[✓] Results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
