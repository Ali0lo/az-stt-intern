"""
utils.py - Text normalization utilities for Azerbaijani ASR evaluation.

Preserves Azerbaijani-specific characters:
  ə, ı, ö, ü, ç, ş, ğ
"""

import re
import unicodedata


# Azerbaijani alphabet letters (lowercase) that must be preserved
AZ_SPECIAL_CHARS = set("əıöüçşğ")


def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Normalize text for ASR evaluation.

    Steps:
      1. Unicode NFC normalization
      2. Lowercase
      3. Optionally remove punctuation (preserving Azerbaijani letters)
      4. Collapse multiple whitespace to single space
      5. Strip leading/trailing whitespace

    Args:
        text: Raw text string.
        remove_punctuation: If True, remove punctuation characters while
                            preserving Azerbaijani special letters.

    Returns:
        Normalized text string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1: Unicode normalization (NFC = composed form, standard for Azerbaijani)
    text = unicodedata.normalize("NFC", text)

    # Step 2: Lowercase (handles Turkish/Azerbaijani dotted/dotless i correctly via Python)
    text = text.lower()

    # Step 3: Optionally remove punctuation
    if remove_punctuation:
        text = _remove_punctuation_preserve_az(text)

    # Step 4: Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Step 5: Strip
    text = text.strip()

    return text


def _remove_punctuation_preserve_az(text: str) -> str:
    """
    Remove punctuation from text while preserving Azerbaijani special characters.

    Standard string.punctuation would strip ğ, ş, etc. if they happen to be
    in unusual encodings. We use Unicode category-based filtering instead.

    Args:
        text: Lowercase text.

    Returns:
        Text with punctuation removed.
    """
    result = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep letters (L*), numbers (N*), spaces (Zs), and AZ special chars explicitly
        if cat.startswith("L") or cat.startswith("N") or cat == "Zs" or ch == " ":
            result.append(ch)
        elif ch in AZ_SPECIAL_CHARS:
            # Explicit safeguard (should already be caught by 'L*' category)
            result.append(ch)
        # Everything else (punctuation, symbols) is dropped

    return "".join(result)


def compute_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER).

    Uses the `jiwer` library for reliable computation.

    Args:
        reference:  Ground truth transcript (already normalized).
        hypothesis: ASR model output (already normalized).

    Returns:
        Tuple of (wer, cer) as floats in range [0, ∞).
        Values can exceed 1.0 when insertions are many.
    """
    import jiwer

    # jiwer handles empty strings gracefully; WER=1.0 if ref is non-empty and hyp is empty
    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)

    return wer, cer


def get_audio_duration(sample: dict) -> float:
    """
    Extract audio duration in seconds from a Common Voice dataset sample.

    Common Voice stores audio as a dict with keys 'array' and 'sampling_rate'.

    Args:
        sample: Dataset row with 'audio' key.

    Returns:
        Duration in seconds, or 0.0 if not computable.
    """
    try:
        audio = sample["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]
        return len(array) / sr
    except (KeyError, ZeroDivisionError, TypeError):
        return 0.0


def format_results_table(rows: list[dict], n: int = 5) -> str:
    """
    Format a list of result rows as a simple ASCII table for terminal output.

    Args:
        rows: List of dicts with keys: id, reference, prediction, wer, cer.
        n:    Number of rows to show.

    Returns:
        Formatted string.
    """
    lines = []
    header = f"{'ID':<6} {'WER':>6} {'CER':>6}  {'Reference':<35} {'Prediction':<35}"
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows[:n]:
        ref = row["reference"][:33] + ".." if len(row["reference"]) > 35 else row["reference"]
        hyp = row["prediction"][:33] + ".." if len(row["prediction"]) > 35 else row["prediction"]
        lines.append(
            f"{str(row['id']):<6} {row['wer']:>6.3f} {row['cer']:>6.3f}  {ref:<35} {hyp:<35}"
        )

    return "\n".join(lines)
