# az-stt-intern — Azerbaijani Automatic Speech Recognition

An end-to-end ASR evaluation and fine-tuning pipeline for the Azerbaijani language, built with [OpenAI Whisper](https://github.com/openai/whisper) and the [Mozilla Common Voice 17.0](https://commonvoice.mozilla.org/) dataset.

Submitted as part of an AI Engineer Internship technical assessment.

---

## 📋 Project Overview

This project demonstrates a complete ASR pipeline in three parts:

| Part | Description |
|------|-------------|
| **A** | Baseline evaluation of `whisper-small` on Azerbaijani test speech |
| **B** | Fine-tuning `whisper-tiny` on a small Azerbaijani training subset |
| **C** | Analytical report (in Azerbaijani) covering results, challenges, and future work |

The goal is **not** to achieve state-of-the-art accuracy but to build a clean, reproducible, and technically sound pipeline suitable for an internship submission.

---

## 📂 Repository Structure

```
az-stt-intern/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── part_a/
│   ├── evaluate_baseline.py     ← Baseline Whisper evaluation script
│   └── utils.py                 ← Shared text normalization and metrics utilities
├── part_b/
│   ├── fine_tune.py             ← Fine-tuning pipeline using Seq2SeqTrainer
│   ├── evaluate_finetuned.py    ← Evaluate the saved fine-tuned model
│   └── compare_models.py        ← Side-by-side WER/CER comparison + chart
├── results/                     ← All generated outputs land here
│   └── .gitkeep
└── report.md                    ← Analytical report in Azerbaijani
```

---

## 🗂 Dataset

**Mozilla Common Voice 17.0 — Azerbaijani (`az`)**

| Property | Value |
|----------|-------|
| HF Dataset ID | `mozilla-foundation/common_voice_17_0` |
| Language config | `az` |
| License | CC-0 |
| Audio format | MP3 (resampled to 16 kHz at runtime) |
| Approximate train hours | ~14h (varies by version) |

The dataset is crowdsourced and contains natural speech with varying microphone quality, accent diversity, and background noise — all of which make ASR on Azerbaijani particularly challenging.

> **Note:** Due to time and GPU constraints, this project uses small subsets (50 test samples, 200 train + 50 validation samples). Full dataset evaluation would require more compute.

---

## 🤖 Model Choice and Justification

| Role | Model | Reason |
|------|-------|--------|
| Baseline | `openai/whisper-small` | Best accuracy/speed trade-off for zero-shot Azerbaijani; ~244M params |
| Fine-tuning | `openai/whisper-tiny` | Fits comfortably on free Colab GPU; fast iteration; ~39M params |

**Why Whisper?**
- Multilingual pre-training covers Azerbaijani out-of-the-box
- Publicly available in multiple sizes
- Well-integrated with Hugging Face `transformers`
- Decoder supports forced language tokens (e.g., `<|az|>`)

---

## ⚙️ Setup

### Prerequisites

- Python 3.9+
- pip
- (Optional but recommended) CUDA-capable GPU

### Install dependencies

```bash
pip install -r requirements.txt
```

> On Google Colab with GPU runtime, all dependencies install in ~2 minutes.

---

## 🅰 Part A — Baseline Evaluation

Evaluates `whisper-small` (zero-shot) on 50 samples from the Common Voice test split.

```bash
python part_a/evaluate_baseline.py \
  --model_name openai/whisper-small \
  --dataset_name mozilla-foundation/common_voice_17_0 \
  --language az \
  --split test \
  --max_samples 50 \
  --output_dir results
```

**Outputs:**
- `results/baseline_results.csv` — per-sample WER, CER, reference, prediction, duration
- Terminal: average WER%, average CER%, best/worst 5 samples

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `openai/whisper-small` | HF model ID |
| `--dataset_name` | `mozilla-foundation/common_voice_17_0` | HF dataset ID |
| `--language` | `az` | Common Voice language config |
| `--split` | `test` | Dataset split |
| `--max_samples` | `50` | Number of samples to evaluate |
| `--output_dir` | `results` | Output directory |

---

## 🅱 Part B — Fine-tuning

### Step 1: Fine-tune

Fine-tunes `whisper-tiny` on 200 training samples for 3 epochs.

```bash
python part_b/fine_tune.py \
  --model_name openai/whisper-tiny \
  --dataset_name mozilla-foundation/common_voice_17_0 \
  --language az \
  --train_samples 200 \
  --eval_samples 50 \
  --output_dir results/whisper_az_finetuned
```

**Outputs:**
- `results/whisper_az_finetuned/` — saved model + processor
- `results/training_metrics.csv` — per-step training log
- `results/training_curves.png` — loss and WER curves

### Step 2: Evaluate fine-tuned model

```bash
python part_b/evaluate_finetuned.py \
  --model_path results/whisper_az_finetuned \
  --dataset_name mozilla-foundation/common_voice_17_0 \
  --language az \
  --split test \
  --max_samples 50 \
  --output_dir results
```

**Outputs:** `results/finetuned_results.csv`

### Step 3: Compare baseline vs fine-tuned

```bash
python part_b/compare_models.py \
  --baseline_model openai/whisper-small \
  --finetuned_model results/whisper_az_finetuned \
  --dataset_name mozilla-foundation/common_voice_17_0 \
  --language az \
  --test_samples 50 \
  --output_dir results
```

**Outputs:**
- `results/comparison.csv` — aggregate WER/CER per model
- `results/wer_cer_comparison.png` — grouped bar chart

---

## 📊 Results

> ⚠️ Fill in these values after running the scripts.

| Model | Avg WER | Avg CER | Samples |
|-------|---------|---------|---------|
| whisper-small (baseline) | `[FILL]`% | `[FILL]`% | 50 |
| whisper-tiny (fine-tuned) | `[FILL]`% | `[FILL]`% | 50 |

---

## 📏 WER and CER Explained

**Word Error Rate (WER)**

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total words in reference.
WER measures how many words the model got wrong at the word level. Lower is better.

**Character Error Rate (CER)**

Same formula but applied at the character level. CER is often more informative for agglutinative languages (like Azerbaijani) where a single wrong morpheme suffix can inflate WER disproportionately.

---

## ⚠️ Limitations

- **Small subset**: 50 test / 200 train samples are far too few for robust conclusions. Results may have high variance.
- **Whisper tiny for fine-tuning**: The tiny model has limited capacity. A meaningful improvement over the much larger `whisper-small` baseline is unlikely.
- **No data augmentation**: Speed perturbation, noise injection, and SpecAugment could improve generalization.
- **Azerbaijani is low-resource**: Whisper has seen significantly less Azerbaijani data during pre-training than English, Spanish, or French.
- **Common Voice quality**: Crowdsourced audio varies widely in microphone quality and background noise.

---

## 📄 Report

See [`report.md`](report.md) for the full analytical report in Azerbaijani.

---

## 🛠 Environment

Tested on:
- Python 3.10
- PyTorch 2.1
- Transformers 4.38
- Google Colab (T4 GPU, ~15 GB VRAM)

---

## 📝 License

This project is for educational/internship purposes only.
Mozilla Common Voice dataset is licensed under CC-0.
Whisper model weights are licensed under Apache 2.0.
