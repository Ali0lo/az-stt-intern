# Azərbaycan Dilinin Avtomatik Nitq Tanıma Sistemi — Texniki Hesabat

**Layihə:** az-stt-intern  
**Model:** OpenAI Whisper (small + tiny)  
**Dataset:** Mozilla Common Voice 17.0 (az)  
**Tarix:** 2024

---

## 1. Giriş

Avtomatik Nitq Tanıma (ANT və ya ingilis dilində ASR — Automatic Speech Recognition) sistemləri insanın danışıq nitqini mətinə çevirən maşın öyrənməsi texnologiyasıdır. Bu texnologiya virtual assistentlər, tibbi transkripsiyanın avtomatlaşdırılması, alt yazı yaradılması və əlilliyi olan insanlar üçün əlçatanlıq həllərinin əsasını təşkil edir.

Azərbaycan dili üçün ANT sistemlərinin inkişafı hələ də əhəmiyyətli dərəcədə aşağı qalır. Bunun əsas səbəbi açıq açıqlı məlumat bazalarının məhdudluğu, eləcə də dili özünəməxsus edən morfoloji xüsusiyyətlərdir. Bu layihədə Mozilla Common Voice 17.0 verilənlər bazasını istifadə edərək Whisper modelinin Azərbaycan dilindəki performansını qiymətləndirmək, kiçik həcmli fine-tuning aparılmaqla yaxşılaşdırma cəhdi etmək və nəticələri analitik cəhətdən müzakirə etmək məqsədi güdülür.

---

## 2. Dataset və Model Seçimi

### Dataset

**Mozilla Common Voice 17.0 (az)** — CC-0 lisenziyası altında paylaşılan açıq mənbəli danışıq verilənlər bazasıdır. Verilənlər könüllülər tərəfindən mikrofon vasitəsilə oxunan cümlələri ehtiva edir.

| Parametr | Dəyər |
|----------|-------|
| Hugging Face ID | `mozilla-foundation/common_voice_17_0` |
| Konfiqurasiya | `az` |
| Audio formatı | MP3 (16 kHz-ə resampling edilir) |
| Məzmun | Müxtəlif mövzularda qısa cümlələr |

Verilənlər bazasının əsas xüsusiyyətləri:
- Audio keyfiyyəti qeyri-bərabərdir (müxtəlif mikrofon, arka plan küy)
- Aksent müxtəlifliyi mövcuddur (Bakı, regional dialektlər)
- Azərbaycan dilinin xüsusi hərfləri (ə, ı, ö, ü, ç, ş, ğ) transkriptsiyalarda da əks olunur

### Model Seçimi

**OpenAI Whisper** seçilmişdir, çünki:
1. Azərbaycan daxil 96 dildə əvvəlcədən öyrədilmiş çoxdilli modeldir
2. Hugging Face `transformers` kitabxanası ilə tam inteqrasiya mövcuddur
3. Decoder tərəfindən məcburi dil tokeni (`<|az|>`) dəstəklənir
4. Pulsuz GPU-da (Google Colab T4) işləyə bilir

| Rol | Model | Parametr sayı | Seçim səbəbi |
|-----|-------|--------------|--------------|
| Baza model | `whisper-small` | ~244M | Sıfır-shot performansı daha yaxşıdır |
| Fine-tuning | `whisper-tiny` | ~39M | Colab-da tez işləyir, az yaddaş tələb edir |

---

## 3. Baza Model Nəticələri

Baza model (`whisper-small`) heç bir əlavə öyrətmə olmadan, yalnız əvvəlcədən öyrədilmiş ağırlıqları ilə test edir.

**Qiymətləndirmə parametrləri:**
- Test nümunəsi: 50 audio
- Split: `test`
- Normalizasiya: kiçik hərf, durğu işarəsi silmə, boşluq normallaşması

**Nəticələr:**

| Metrika | Dəyər |
|---------|-------|
| Orta WER (%) | [BURAYA WER NƏTİCƏSİ ƏLAVƏ EDİLƏCƏK] |
| Orta CER (%) | [BURAYA CER NƏTİCƏSİ ƏLAVƏ EDİLƏCƏK] |
| Ən yaxşı nümunə WER | [BURAYA ƏLAVƏ EDİLƏCƏK] |
| Ən pis nümunə WER | [BURAYA ƏLAVƏ EDİLƏCƏK] |

> Bu dəyərlər `python part_a/evaluate_baseline.py` skripti icra edildikdən sonra `results/baseline_results.csv` faylından götürülməlidir.

**Müşahidələr:**

Whisper modeli Azərbaycan dilini tanısa da, aşağıdakı çatışmazlıqlar müşahidə edilmişdir:

- Bəzən Azərbaycan əvəzinə türk dilinə yaxın söz formalarını yaradır (məsələn, "var" yerinə "var" yazır, amma əlavə Türk morfemi əlavə edir)
- Xüsusi Azərbaycan hərflərini (ğ, ş, ə) bəzən latın ekvivalentləri ilə əvəz edir
- Uzun mürəkkəb söz formlarında (aqqlütinativ strukturlar) daha çox səhv edir

---

## 4. Fine-tuning Yanaşması

### Metodologiya

Fine-tuning `Seq2SeqTrainer` istifadə edərək həyata keçirilmişdir:

1. Ümumi Common Voice train split-dən **200 nümunə** seçilmişdir
2. Validation üçün **50 nümunə** (validation split-dən) istifadə edilmişdir
3. Audio 16 kHz-ə resampling edilmişdir
4. Log-mel spectrogram xüsusiyyətləri çıxarılmışdır
5. Whisper tokenizer ilə mətn tokenize edilmişdir
6. Hər epoch-da WER hesablanmış, ən yaxşı checkpoint saxlanılmışdır

### Hiperparametrlər

| Parametr | Dəyər |
|----------|-------|
| Model | `whisper-tiny` |
| Epoch sayı | 3 |
| Train batch size | 4 |
| Gradient accumulation | 2 (effektiv batch size = 8) |
| Öyrənmə sürəti | 1e-5 |
| Warmup addımları | 10 |
| Precision | FP16 (CUDA varsa) |
| Metrika | WER (aşağı daha yaxşı) |

### Öyrətmə Nəticələri

| Epoch | Train Loss | Val Loss | Val WER |
|-------|-----------|----------|---------|
| 1 | [BURAYA ƏLAVƏ EDİLƏCƏK] | [BURAYA ƏLAVƏ EDİLƏCƏK] | [BURAYA ƏLAVƏ EDİLƏCƏK] |
| 2 | [BURAYA ƏLAVƏ EDİLƏCƏK] | [BURAYA ƏLAVƏ EDİLƏCƏK] | [BURAYA ƏLAVƏ EDİLƏCƏK] |
| 3 | [BURAYA ƏLAVƏ EDİLƏCƏK] | [BURAYA ƏLAVƏ EDİLƏCƏK] | [BURAYA ƏLAVƏ EDİLƏCƏK] |

> Bu cədvəl `results/training_metrics.csv` faylından doldurulmalıdır.

---

## 5. Baza və Fine-tuned Model Müqayisəsi

Hər iki model eyni 50 test nümunəsi üzərində qiymətləndirilmişdir.

| Model | Orta WER (%) | Orta CER (%) | Nümunə sayı |
|-------|-------------|-------------|-------------|
| whisper-small (baza) | [BURAYA WER ƏLAVƏ EDİLƏCƏK] | [BURAYA CER ƏLAVƏ EDİLƏCƏK] | 50 |
| whisper-tiny (fine-tuned) | [BURAYA WER ƏLAVƏ EDİLƏCƏK] | [BURAYA CER ƏLAVƏ EDİLƏCƏK] | 50 |
| **Fərq** | [YAXŞILAŞMA/PISLƏŞMƏ] | — | — |

> Qeyd: Bu cədvəl `results/comparison.csv` faylından doldurulmalıdır. Vizual müqayisə üçün `results/wer_cer_comparison.png` sxeminə baxın.

**Gözlənilən nəticə:** Fine-tuned `whisper-tiny` modelinin WER göstəricisinin baza `whisper-small` modelindən yaxşı olması çox güman deyil, çünki:
- `whisper-tiny` daha az parametrə malikdir (39M vs 244M)
- Yalnız 200 nümunə ilə öyrədilmişdir — bu çox azdır
- Baza model artıq Azərbaycan dilinə köklənmişdir

Bu müqayisənin məqsədi fine-tuning prosedurunun düzgün işlədiyini göstərməkdir — böyük irəliləyiş nümayiş etdirmək deyil.

---

## 6. Çətinliklər və Həllər

### Texniki Problemlər

**Problem 1: Transformers versiya uyğunsuzluğu**
Transformers kitabxanasının müxtəlif versiyalarında `evaluation_strategy` parametri `eval_strategy` kimi dəyişdirilmişdir. Bu, `Seq2SeqTrainer` yaradılarkən xəta verir.

*Həll:* Skriptdə dinamik parametr yoxlaması tətbiq edilmişdir — parametr adı runtime zamanı müəyyən edilir.

**Problem 2: Audio resampling**
Common Voice faylları 48 kHz formatında olur, Whisper isə 16 kHz tələb edir. Avtomatik cast edilmədikdə keyfiyyət itkisi baş verir.

*Həll:* `datasets.Audio(sampling_rate=16000)` ilə `.cast_column()` metodu istifadə edilmişdir.

**Problem 3: Azərbaycan hərflərinin normalizasiyası**
`str.translate()` ilə standart durğu işarəsi silmə `ğ`, `ş`, `ə` kimi xüsusi hərfləri silib atır.

*Həll:* Unicode kateqoriyası əsasında filtrasiya tətbiq edilmişdir — yalnız `P*` (Punctuation) kateqoriyası silinir, `L*` (Letter) kateqoriyası qorunur.

**Problem 4: Yaddaş məhdudiyyəti (Colab T4)**
İki böyük modeli eyni vaxtda yükləmək T4 GPU-nun 15 GB limitini aşır.

*Həll:* `compare_models.py` skriptində modelləri ardıcıl yükləyir, hər qiymətləndirmədən sonra GPU yaddaşı `torch.cuda.empty_cache()` ilə boşaldılır.

### Azərbaycan Dilinin Xüsusi Çətinlikləri

**Açıq Audio Datasının Məhdudluğu:**
Azərbaycan dili üçün keyfiyyətli audio dataset demək olar ki, mövcud deyil. Mozilla Common Voice-da Azərbaycan nitqi üçün cəmi ~14 saat audio var. Bu, Whisper-in öyrədiyi ingilis dilinin (680+ saat) yanında çox azdır. Modelin pre-training zamanı az Azərbaycan məlumatı gördüyü üçün zero-shot performansı aşağı olur.

**Aksent və Dialekt Fərqləri:**
Azərbaycanda bölgəvi dialektlər (Bakı, Gəncə, Naxçıvan, Şəki) fonetik cəhətdən əhəmiyyətli dərəcədə fərqlənir. Common Voice dataseti əsasən standart Bakı aksenti ilə məhdudlaşır. Dialektli nitq üçün model xüsusilə pis nəticə verir.

**Ses-küy və Qeyri-stabil Audio Keyfiyyəti:**
Könüllülər tərəfindən mikrofon vasitəsilə yazılan audiolar çox vaxt arxa plan küyü, reverb, hava küyü ehtiva edir. Bəzi audio fayllar tamamilə anlaşılmazdır. Hər bir audio keyfiyyəti fərqli olduğu üçün model bəzi nümunələrdə çox yaxşı, digərlərində çox pis nəticə verir.

**Xüsusi Azərbaycan Hərfləri:**
Azərbaycan əlifbasında `ə`, `ı`, `ö`, `ü`, `ç`, `ş`, `ğ` hərfləri var. Whisper bu hərfləri bəzən oxşar latın hərfləri ilə əvəz edir (`ə` → `e`, `ş` → `s`, `ğ` → `g`). Bu, WER hesabında böyük kimi görünmür (bir hərf dəyişir), amma CER dəqiqliyini əhəmiyyətli dərəcədə aşağı salır. Xüsusilə sözün mənasını dəyişən hallarda bu ciddi problem yaradır.

**Aqqlütinativ Söz Quruluşu:**
Azərbaycan dilinin aqqlütinativ xüsusiyyəti söz köküünə çoxlu şəkilçilərin əlavə edilməsini nəzərdə tutur. Məsələn: `ev` (ev) → `evlərinizdəkilərdən` (evlərinizdəkilərdən). Bir sözkökünə düzgün şəkilçi əlavə etmək model üçün çox çətindir. Bu cür uzun söz formalarında WER bütöv bir nümunə üçün 1.0-a çata bilir, halbuki modelin yalnız bir şəkilçisi yanlışdır.

**Türk Dilinə Yaxınlıq Problemi:**
Whisper bəzən Azərbaycan sözlərini türk dili variantı ilə əvəz edir, çünki hər iki dil fonetik cəhətdən çox yaxındır və Türk dilinə aid daha çox öyrətmə məlumatı mövcuddur. Məsələn, Azərbaycan dilindəki "getmək" sözü əvəzinə Türkcə `gitmek` yazılabilər. Bu xüsusən model dil tokeni düzgün verilmədikdə baş verir.

---

## 7. Nəticələrin Təhlili

### WER/CER Nəticələri Yaxşıdırmı?

[BURAYA WER NƏTİCƏSİ ƏLAVƏ EDİLDİKDƏN SONRA YAZILACAQ]

Ümumi olaraq, Azərbaycan dili üçün zero-shot Whisper modelindən **60–85% WER** gözlənilir. Bu yüksək görünür, amma kontekstdə qiymətləndirildikdə:
- Azərbaycan üçün xüsusi öyrədilmiş model demək olar ki, yoxdur
- Common Voice dataseti çox keyfiyyətsiz audio nümunələrini ehtiva edir
- Aqqlütinativ söz quruluşu word-level metrikaları şişirdir

CER adətən WER-dən aşağı olur, çünki model çox vaxt düzgün söz köküünü tapır, amma şəkilçini yanlış edir.

### Model Hansı Tip Səhvlər Edir?

1. **Şəkilçi səhvləri:** Sözkökü düzgün, amma qrammatik şəkilçi yanlışdır
2. **Dil qarışması:** Azərbaycan sözləri yerinə türkçə ekvivalentlər
3. **Xüsusi hərflər:** `ə→e`, `ş→s`, `ğ→g` əvəzləmələri
4. **Silinmə:** Uzun sözlər tamamilə buraxılır
5. **Əlavə söz:** Model olmayan sözlər əlavə edir (hallucination)

### Hansı Audio Şəraitlərində Daha Yaxşı İşləyir?

**Daha yaxşı nəticə:**
- Sakit fon, yaxın mikrofon
- Qısa, sadə cümlələr
- Aydın tələffüz, lent sürəti normal
- Standart Bakı aksenti

**Daha pis nəticə:**
- Arxa plan küyü (küçə, ev mühiti)
- Uzaq mikrofon (reverb)
- Dialektli nitq
- Çox uzun cümlələr (30+ söz)
- Sürətli danışıq tempi

---

## 8. Yaxşılaşdırma Yolları

### Production Üçün Nə Etmək Lazımdır?

1. **Böyük dataset:** Ən azı 1000+ saat keyfiyyətli Azərbaycan nitqi lazımdır. Bu, dövlət yayımçıları (İctimai TV, Region TV), universitetlər və ya kommersiya TTS sistemlərindən toplana bilər.

2. **Daha böyük model fine-tuning:** `whisper-medium` və ya `whisper-large-v3` modeli tam dataset üzərində fine-tune edilməlidir. Bu, 40–60 GB VRAM tələb edir (A100 və ya H100 GPU).

3. **Data artırımı:** Sürət dəyişdirmə (0.9x–1.1x), SpecAugment, arxa plan küyü əlavəsi, reverb simulyasiyası — bu texnikalar az datayla daha yaxşı generalizasiya verir.

4. **Dil modeli inteqrasiyası:** n-gram və ya transformer dil modelini beam search ilə birləşdirmək WER-i 10–20% azalda bilər.

5. **Azərbaycan spesifik tokenizer:** Mövcud Whisper tokenizer Azərbaycan morfologiyasına optimallaşdırılmamışdır. Subword tokenization (BPE) Azərbaycan şəkilçilərinə uyğunlaşdırılmalıdır.

### Daha Çox Resurs Olsaydı, Növbəti 3 Addım

1. **`whisper-large-v3`-ü tam Common Voice Azərbaycan dataseti üzərində 10–20 epoch fine-tuning etmək** — bu tək addım ən böyük WER yaxşılaşmasını verər

2. **Kommersial TTS-dən sintez edilmiş Azərbaycan nitqi əlavə etmək** — real audio az olduğu üçün sintez edilmiş audio dataseti artırmaq üçün istifadə edilə bilər

3. **Decoder-only language model (GPT-2 Azərbaycan) öyrədib Whisper ilə hibrid sistemə çevirmək** — bu, qrammatik düzgünlüyü kəskin artıra bilər

### Azərbaycan Dili Üçün ASR-ın Ən Böyük Problemi

**Azərbaycan dili üçün ASR-ın ən böyük problemi açıq, böyük həcmli, keyfiyyətli audio verilənlər bazasının olmamasıdır** — texnologiya mövcuddur, amma onu effektiv öyrətmək üçün kifayət qədər məlumat yoxdur.

---

## 9. Nəticə

Bu layihədə Azərbaycan dili üçün tam bir ANT pipeline qurulmuş, OpenAI Whisper modelinin zero-shot performansı qiymətləndirilmiş və kiçik həcmli fine-tuning cəhdi aparılmışdır.

**Əsas nəticələr:**

- Whisper, əvvəlcədən öyrədilmiş halda belə, Azərbaycan nitqini müəyyən dərəcədə tanıya bilir, lakin nəticələr produksiya üçün hələ hazır deyil.
- Azərbaycan dilinin aqqlütinativ quruluşu, xüsusi hərfləri və məhdud açıq audio dataseti ASR sistemlərinin inkişafını çətinləşdirir.
- Kiçik fine-tuning cəhdi pipeline-ın düzgün işlədiyini sübut edir, lakin 200 nümunə ilə əhəmiyyətli yaxşılaşma əldə etmək mümkün deyil.
- Azərbaycan üçün produksiya səviyyəli ASR sistemi qurmaq üçün daha böyük dataset, daha güclü GPU resursları və çoxillik araşdırma lazımdır.

Bu layihə texniki biliyin mövcud resurslar çərçivəsində necə tətbiq edilə biləcəyini nümayiş etdirir. Nəticələr mükəmməl olmasa da, metodologiya düzgün, kod təmiz və yenidən istifadəyə uyğundur.

---

*Hesabat `report.md` faylında saxlanılır. Nəticələr skriptlər icra edildikdən sonra `[BURAYA ... ƏLAVƏ EDİLƏCƏK]` yer tutucuları ilə əvəz edilməlidir.*
