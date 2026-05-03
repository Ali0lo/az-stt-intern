# az-stt-intern — Azərbaycan Dili üçün Nitq Tanıma Sistemi

Bu layihə Azərbaycan dili üçün avtomatik nitq tanıma (ASR — Automatic Speech Recognition) pipeline-ının qurulmasını göstərir. Layihədə əvvəlcə hazır `Whisper` modeli ilə baza qiymətləndirmə aparılır, daha sonra kiçik Azərbaycan dili datası üzərində fine-tuning cəhdi edilir və nəticələr WER/CER metrikaları ilə müqayisə olunur.

Layihə AI Engineer Intern texniki tapşırığı üçün hazırlanmışdır.

---

## 1. Layihənin Qısa İzahı

Tapşırığın məqsədi Azərbaycan dili üçün işlək ASR pipeline qurmaqdır.

Layihə üç hissədən ibarətdir:

| Hissə | Təsvir |
|------|--------|
| **A — Baza tətbiq** | Hazır `openai/whisper-small` modeli ilə Azərbaycan dili audio nümunələrinin transkripsiyası və WER/CER hesablanması |
| **B — Fine-tuning cəhdi** | `openai/whisper-tiny` modelinin kiçik Azərbaycan dili datası üzərində fine-tune edilməsi |
| **C — Analitik hesabat** | Azərbaycan dilində nəticələrin, çətinliklərin və yaxşılaşdırma yollarının izahı |

Əsas məqsəd state-of-the-art nəticə almaq deyil, texniki cəhətdən düzgün, işlək və təkrar yaradıla bilən ASR pipeline təqdim etməkdir.

---

## 2. Repository Strukturu

```text
az-stt-intern/
├── README.md
├── requirements.txt
├── part_a/
│   ├── evaluate_baseline.py
│   └── utils.py
├── part_b/
│   ├── fine_tune.py
│   ├── evaluate_finetuned.py
│   └── compare_models.py
├── results/
│   ├── baseline_results.csv
│   └── .gitkeep
└── report.md
```

---

## 3. Dataset

İlkin tapşırıqda Mozilla Common Voice Azerbaijani dataseti tövsiyə olunmuşdu. Lakin son dəyişikliklərə görə Common Voice-un bəzi Hugging Face versiyaları standart `datasets.load_dataset()` üsulu ilə problemsiz yüklənmir.

Bu səbəbdən layihədə alternativ və reproduksiya oluna bilən dataset kimi **Google FLEURS Azerbaijani** istifadə edilmişdir.

| Xüsusiyyət | Dəyər |
|-----------|-------|
| Dataset | Google FLEURS |
| Hugging Face ID | `google/fleurs` |
| Dil konfiqurasiyası | `az_az` |
| Split | `train`, `validation`, `test` |
| Audio sampling rate | Runtime zamanı 16 kHz-ə çevrilir |

Bu dataset Azərbaycan dili üçün oxunmuş nitq nümunələrindən ibarətdir və ASR qiymətləndirməsi üçün uyğundur.

---

## 4. İstifadə Olunan Model və Parametrlər

### Baza model

| Parametr | Dəyər |
|---------|-------|
| Model | `openai/whisper-small` |
| İstifadə məqsədi | Zero-shot ASR qiymətləndirməsi |
| Dataset | `google/fleurs` |
| Language config | `az_az` |
| Split | `test` |
| Test nümunə sayı | 3 ilkin test / 50 final test üçün nəzərdə tutulub |
| Metrikalar | WER, CER |

### Fine-tuning modeli

| Parametr | Dəyər |
|---------|-------|
| Model | `openai/whisper-tiny` |
| İstifadə məqsədi | Azərbaycan dili datası üzərində fine-tuning cəhdi |
| Dataset | `google/fleurs` |
| Language config | `az_az` |
| Train nümunə sayı | 200 |
| Validation nümunə sayı | 50 |
| Epoch sayı | 3 |
| Learning rate | `1e-5` |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Checkpoint seçimi | Ən aşağı validation WER əsasında |

`whisper-small` baza qiymətləndirmə üçün seçilmişdir, çünki multilingual ASR üçün daha güclü modeldir. Fine-tuning üçün isə `whisper-tiny` istifadə olunur, çünki daha yüngüldür və Google Colab kimi məhdud GPU resurslarında daha rahat işləyir.

---

## 5. WER/CER Nəticələri

Aşağıdakı nəticələr `google/fleurs` datasetinin `az_az` konfiqurasiyasında `test` splitindən götürülmüş **50 nümunə** üzərində hesablanmışdır.

| Model | Dataset | Split | Nümunə sayı | Ortalama WER | Ortalama CER |
|------|---------|-------|-------------:|--------------:|--------------:|
| `openai/whisper-small` baseline | `google/fleurs` (`az_az`) | `test` | 50 | **51.01%** | **15.29%** |

### Qısa nəticə analizi

50 nümunəlik baseline test nəticəsində `openai/whisper-small` modeli üçün orta **WER 51.01%**, orta **CER isə 15.29%** olmuşdur.

Bu nəticə göstərir ki, model ümumi cümlə strukturunu müəyyən qədər tanıya bilir, lakin Azərbaycan dilində söz səviyyəsində hələ nəzərəçarpacaq səhvlər edir. CER göstəricisinin WER-dən xeyli aşağı olması onu göstərir ki, model bir çox hallarda sözü tamamilə itirmir, daha çox fonetik və yazılış baxımından yaxın formalar yaradır.

### Ən yaxşı 5 nümunə

| ID | WER | CER |
|----|----:|----:|
| 9  | 25.0% | 14.5% |
| 11 | 26.3% | 8.4% |
| 0  | 27.3% | 7.7% |
| 34 | 30.0% | 5.4% |
| 39 | 30.0% | 8.6% |

### Ən pis 5 nümunə

| ID | WER | CER |
|----|----:|----:|
| 7  | 75.0% | 21.8% |
| 5  | 71.7% | 35.2% |
| 21 | 71.7% | 38.4% |
| 25 | 71.4% | 36.1% |
| 26 | 68.8% | 19.1% |

---
## 6. Fine-tuning Nəticəsinin Müqayisəsi

Fine-tuning mərhələsində `openai/whisper-tiny` modeli kiçik train subset üzərində öyrədilir və daha sonra eyni test nümunələri üzərində baza model ilə müqayisə olunur.

| Model | Fine-tuning statusu | Test nümunə sayı | WER | CER |
|------|----------------------|-----------------:|----:|----:|
| `openai/whisper-small` | Fine-tune edilməyib, zero-shot baseline | 3 | **35.70%** | **7.67%** |
| `openai/whisper-small` | Fine-tune edilməyib, zero-shot baseline | 50 | Hələ icra edilməyib | Hələ icra edilməyib |
| `openai/whisper-tiny` | 200 train nümunəsi üzərində fine-tune edilir | 50 | Hələ icra edilməyib | Hələ icra edilməyib |

### Gözlənilən müşahidə

Fine-tuned `whisper-tiny` modelinin `whisper-small` baseline modelindən mütləq daha yaxşı nəticə verməsi gözlənilmir. Bunun əsas səbəbi baza modelin daha böyük olması, fine-tuning datasının isə çox kiçik seçilməsidir. Buna baxmayaraq, bu mərhələ fine-tuning pipeline-ının texniki olaraq düzgün qurulduğunu göstərmək üçün vacibdir.

Fine-tuning nəticələri aşağıdakı fayllarda saxlanılır:

| Fayl | Təsvir |
|------|--------|
| `results/finetuned_results.csv` | Fine-tuned modelin test nəticələri |
| `results/comparison.csv` | Baseline və fine-tuned modelin WER/CER müqayisəsi |
| `results/training_metrics.csv` | Training loss və validation metrikaları |
| `results/training_curves.png` | Loss və validation WER qrafiki |
| `results/wer_cer_comparison.png` | Baseline və fine-tuned modelin vizual müqayisəsi |

---

## 7. Kodu İşə Salmaq üçün Addımlar

### 7.1. Mühit yaratmaq

```bash
conda create -n az-stt python=3.10 -y
conda activate az-stt
```

### 7.2. Asılılıqları quraşdırmaq

```bash
pip install -r requirements.txt
```

### 7.3. Baza modeli qiymətləndirmək

Sürətli test üçün:

```bash
python part_a/evaluate_baseline.py \
  --model_name openai/whisper-small \
  --dataset_name google/fleurs \
  --language az_az \
  --split test \
  --max_samples 3 \
  --output_dir results
```

Final baseline nəticəsi üçün:

```bash
python part_a/evaluate_baseline.py \
  --model_name openai/whisper-small \
  --dataset_name google/fleurs \
  --language az_az \
  --split test \
  --max_samples 50 \
  --output_dir results
```

### 7.4. Fine-tuning işə salmaq

```bash
python part_b/fine_tune.py \
  --model_name openai/whisper-tiny \
  --dataset_name google/fleurs \
  --language az_az \
  --train_samples 200 \
  --eval_samples 50 \
  --output_dir results/whisper_az_finetuned
```

### 7.5. Fine-tuned modeli qiymətləndirmək

```bash
python part_b/evaluate_finetuned.py \
  --model_path results/whisper_az_finetuned \
  --dataset_name google/fleurs \
  --language az_az \
  --split test \
  --max_samples 50 \
  --output_dir results
```

### 7.6. Baza və fine-tuned modeli müqayisə etmək

```bash
python part_b/compare_models.py \
  --baseline_model openai/whisper-small \
  --finetuned_model results/whisper_az_finetuned \
  --dataset_name google/fleurs \
  --language az_az \
  --test_samples 50 \
  --output_dir results
```

---

## 8. Çıxış Faylları

| Fayl | Təsvir |
|------|--------|
| `results/baseline_results.csv` | Baza model üçün reference, prediction, WER, CER |
| `results/finetuned_results.csv` | Fine-tuned model üçün reference, prediction, WER, CER |
| `results/comparison.csv` | Baza və fine-tuned modelin ümumi WER/CER müqayisəsi |
| `results/training_metrics.csv` | Fine-tuning zamanı loss və validation metrikaları |
| `results/training_curves.png` | Training loss və validation WER qrafiki |
| `results/wer_cer_comparison.png` | Baza və fine-tuned model müqayisə qrafiki |

---

## 9. Metrikalar

### WER — Word Error Rate

WER söz səviyyəsində səhv nisbətini ölçür:

```text
WER = (Substitutions + Deletions + Insertions) / Reference sözlərinin sayı
```

WER nə qədər aşağıdırsa, modelin söz səviyyəsində transkripsiya keyfiyyəti bir o qədər yaxşıdır.

### CER — Character Error Rate

CER simvol səviyyəsində səhv nisbətini ölçür. Azərbaycan dili kimi aqqlütinativ dillərdə CER əlavə olaraq faydalıdır, çünki bir şəkilçi və ya hərf səhvi WER-i çox artıra bilər, amma CER transkripsiyanın real yaxınlığını daha incə göstərir.

---

## 10. Məhdudiyyətlər

- İlkin nəticə yalnız 3 nümunə üzərində hesablanmışdır.
- Daha etibarlı qiymətləndirmə üçün 50 və ya daha çox test nümunəsi istifadə edilməlidir.
- Fine-tuning kiçik dataset hissəsi ilə aparılır, buna görə nəticələr yüksək variasiyalı ola bilər.
- `whisper-tiny` modeli `whisper-small` modelindən daha kiçikdir, buna görə fine-tuned nəticə baseline-dan zəif ola bilər.
- FLEURS oxunmuş nitq datasıdır; real danışıq, səs-küy və dialekt şəraitində nəticələr fərqli ola bilər.
- Azərbaycan dilində açıq və keyfiyyətli ASR datası məhduddur.

---

## 11. Analitik Hesabat

Ətraflı analiz üçün baxın:

```text
report.md
```

Hesabatda aşağıdakılar izah olunur:

- Texniki problemlər və həll yolları
- Azərbaycan dilinin ASR üçün yaratdığı çətinliklər
- WER/CER nəticələrinin təhlili
- Modelin tipik səhvləri
- Production səviyyəsinə keçid üçün görülməli işlər
- Növbəti yaxşılaşdırma addımları

---

## 12. Lisenziya

Bu layihə təhsil və internship texniki qiymətləndirməsi üçün hazırlanmışdır.

- Google FLEURS dataset: CC-BY-4.0
- Whisper model çəkiləri: Apache 2.0
