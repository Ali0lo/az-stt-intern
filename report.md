# Azərbaycan Dili üçün Nitq Tanıma Sistemi — Analitik Hesabat

**Layihə:** `az-stt-intern`  
**Tapşırıq:** AI Engineer Intern texniki tapşırığı  
**Mövzu:** Azərbaycan dili üçün Automatic Speech Recognition (ASR) pipeline  
**Tarix:** 2026  

---

## 1. Giriş

Bu layihənin məqsədi Azərbaycan dili üçün işlək avtomatik nitq tanıma (ASR — Automatic Speech Recognition) pipeline qurmaqdır. Layihə çərçivəsində hazır çoxdilli Whisper modelindən istifadə edərək baza transkripsiya nəticələri əldə edilmiş, daha sonra kiçik Azərbaycan dili datası üzərində fine-tuning cəhdi aparılmışdır.

Tapşırığın əsas məqsədi state-of-the-art nəticə əldə etmək deyil, ASR pipeline-ının texniki baxımdan düzgün qurulmasını göstərməkdir. Buna görə layihədə dataset yükləmə, audio preprocessing, model inference, WER/CER hesablanması, fine-tuning, validation WER izlənməsi və nəticələrin müqayisəsi mərhələləri həyata keçirilmişdir.

---

## 2. Dataset və Model Seçimi

İlkin tapşırıqda Mozilla Common Voice Azerbaijani datasetindən istifadə tövsiyə olunmuşdu. Lakin praktiki icra zamanı Common Voice-un bəzi Hugging Face versiyaları standart `datasets.load_dataset()` workflow-u ilə problemsiz yüklənmədi. Bu səbəbdən tapşırıqda icazə verilən alternativ dataset seçimi əsasında **Google FLEURS Azerbaijani** datasetindən istifadə edildi.

| Xüsusiyyət | Dəyər |
|-----------|-------|
| Dataset | Google FLEURS |
| Hugging Face ID | `google/fleurs` |
| Dil konfiqurasiyası | `az_az` |
| İstifadə olunan split | `train`, `validation`, `test` |
| Audio preprocessing | 16 kHz sampling rate |

FLEURS datasetində Azərbaycan dili üçün oxunmuş nitq nümunələri mövcuddur. Bu dataset ASR sistemlərinin test edilməsi və kiçik fine-tuning təcrübələri üçün uyğundur.

Model seçimi aşağıdakı kimi aparılmışdır:

| Məqsəd | Model | Səbəb |
|-------|-------|-------|
| Baza qiymətləndirmə | `openai/whisper-small` | Daha güclü multilingual zero-shot ASR modeli |
| Fine-tuning | `openai/whisper-tiny` | Daha yüngül model, məhdud resurslarda daha rahat fine-tune olunur |

`whisper-small` baza model kimi seçilmişdir, çünki Azərbaycan dili üçün hazır multilingual transkripsiya imkanı verir. Fine-tuning üçün isə `whisper-tiny` istifadə edilmişdir, çünki CPU və ya pulsuz GPU mühitlərində daha sürətli işləyir.

---

## 3. Baza Model Nəticələri

Baza qiymətləndirmə mərhələsində `openai/whisper-small` modeli `google/fleurs` datasetinin `az_az` konfiqurasiyasından götürülmüş 50 test nümunəsi üzərində yoxlanılmışdır.

| Model | Dataset | Test nümunə sayı | Ortalama WER | Ortalama CER |
|------|---------|-----------------:|--------------:|--------------:|
| `openai/whisper-small` | `google/fleurs` (`az_az`) | 50 | **51.01%** | **15.29%** |

Nəticələr göstərir ki, `whisper-small` modeli Azərbaycan dili üçün cümlə strukturunu müəyyən qədər tanıya bilir, lakin söz səviyyəsində hələ də ciddi səhvlər edir. Bununla belə, CER göstəricisinin WER-dən xeyli aşağı olması modelin bir çox hallarda sözləri tamamilə itirmədiyini, daha çox fonetik və yazılış baxımından yaxın transkripsiyalar yaratdığını göstərir.

Ən yaxşı nümunələrdə WER təxminən 25–30% aralığında olmuşdur. Ən pis nümunələrdə isə WER 68–75% aralığına qədər yüksəlmişdir. Bu fərq audio keyfiyyəti, cümlə uzunluğu, fonetik mürəkkəblik və Azərbaycan dilinə məxsus səslərin tanınması ilə bağlı ola bilər.

---

## 4. Fine-tuning Yanaşması

Fine-tuning mərhələsində `openai/whisper-tiny` modeli Google FLEURS Azərbaycan dili datasından götürülmüş kiçik subset üzərində öyrədilmişdir.

İstifadə olunan parametrlər:

| Parametr | Dəyər |
|---------|-------|
| Model | `openai/whisper-tiny` |
| Dataset | `google/fleurs` |
| Dil konfiqurasiyası | `az_az` |
| Train nümunə sayı | 50 |
| Validation nümunə sayı | 10 |
| Epoch sayı | 3 |
| Learning rate | `1e-5` |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Checkpoint seçimi | Ən aşağı validation WER əsasında |

Fine-tuning zamanı validation WER hər epoch üzrə izlənmişdir:

| Epoch | Validation WER |
|------:|---------------:|
| 1 | 96.26% |
| 2 | 90.37% |
| 3 | 91.44% |

Ən yaxşı validation WER epoch 2-də əldə edilmişdir. Epoch 3-də WER bir qədər pisləşmişdir, bu da kiçik dataset səbəbindən modelin stabil ümumiləşdirmə aparmadığını və overfitting riskinin olduğunu göstərir.

---

## 5. Baza və Fine-tuned Model Müqayisəsi

Fine-tuning tamamlandıqdan sonra fine-tuned `whisper-tiny` modeli eyni test splitindən götürülmüş 50 nümunə üzərində qiymətləndirilmişdir.

| Model | Status | Test nümunə sayı | Ortalama WER | Ortalama CER |
|------|--------|-----------------:|--------------:|--------------:|
| `openai/whisper-small` | Zero-shot baseline | 50 | **51.01%** | **15.29%** |
| `openai/whisper-tiny` | 50 train nümunəsi ilə fine-tune edilmiş | 50 | **84.66%** | **34.61%** |

Fine-tuned model baza modeldən daha zəif nəticə göstərmişdir. Bu nəticə gözlənilməz deyil, çünki müqayisə edilən modellərin ölçüləri fərqlidir: baseline üçün daha böyük `whisper-small`, fine-tuning üçün isə daha kiçik `whisper-tiny` istifadə edilmişdir. Bundan əlavə, fine-tuning yalnız 50 train nümunəsi ilə aparılmışdır ki, bu da modelin Azərbaycan dili üçün yaxşı ümumiləşdirmə öyrənməsi üçün kifayət deyil.

Bu mərhələnin əsas dəyəri nəticənin mütləq yaxşılaşması deyil, fine-tuning pipeline-ının işləməsini göstərməkdir: data hazırlığı, feature extraction, tokenization, training, validation WER izlənməsi, checkpoint saxlanması və nəticələrin baza model ilə müqayisəsi uğurla həyata keçirilmişdir.

---

## 6. Çətinliklər və Həllər

### 6.1. Dataset yükləmə problemi

Əsas çətinliklərdən biri Mozilla Common Voice datasetinin bəzi Hugging Face versiyalarının standart `load_dataset()` ilə yüklənməməsi oldu. Bu problem layihənin əvvəlində Common Voice əvəzinə alternativ dataset seçilməsini tələb etdi.

**Həll:** Tapşırıqda “və ya istənilən dataset” icazəsi olduğu üçün Google FLEURS Azerbaijani datasetindən istifadə edildi. Bu dataset Hugging Face üzərindən yükləndi və `az_az` konfiqurasiyası ilə işlədildi.

### 6.2. Dataset language və Whisper language fərqi

FLEURS datasetində Azərbaycan dili konfiqurasiyası `az_az` kimi verilir. Lakin Whisper modeli decoding üçün `az_az` yox, `azerbaijani` language adını gözləyir. Bu fərq əvvəlcə boş prediction-lara və səhv nəticələrə səbəb oldu.

**Həll:** Kodda ayrıca mapping funksiyası əlavə edildi:

```text
az_az -> azerbaijani
az -> azerbaijani
```

Bu düzəlişdən sonra model normal transkripsiya yaratmağa başladı.

### 6.3. Reference column fərqi

Common Voice datasetində reference mətn adətən `sentence` column-da olur. FLEURS datasetində isə reference mətn `transcription` və ya `raw_transcription` column-larında ola bilər.

**Həll:** Kodda universal reference extraction funksiyası yazıldı. Funksiya aşağıdakı column-ları ardıcıllıqla yoxlayır:

```text
sentence
transcription
raw_transcription
text
```

Bu yanaşma kodu həm Common Voice, həm də FLEURS üçün daha çevik edir.

### 6.4. Məhdud resurs problemi

Fine-tuning CPU mühitində aparıldığı üçün dataset ölçüsü kiçik saxlanıldı. 50 train və 10 validation nümunəsi ilə training texniki olaraq mümkün oldu, lakin performans baxımından kifayət qədər güclü nəticə vermədi.

**Həll:** Kiçik subset istifadə edildi, epoch sayı məhdud saxlanıldı və validation WER izlənərək ən yaxşı checkpoint seçildi.

---

## 7. Azərbaycan Dilinin ASR üçün Yaratdığı Çətinliklər

Azərbaycan dili ASR üçün bir neçə səbəbə görə çətinlik yaradır:

1. **Low-resource problem:** Azərbaycan dili üçün açıq və keyfiyyətli labeled speech data ingilis, fransız və ya ispan dili ilə müqayisədə daha azdır.
2. **Aqqlütinativ quruluş:** Azərbaycan dilində sözlər şəkilçilər vasitəsilə çox müxtəlif formalara düşə bilir. Bu, WER-i artırır.
3. **Xüsusi hərflər:** `ə`, `ı`, `ö`, `ü`, `ç`, `ş`, `ğ` kimi hərflər model tərəfindən bəzən qarışdırılır.
4. **Fonetik yaxınlıqlar:** `x/q`, `ə/e`, `ı/i`, `ç/c` kimi yaxın səslər model üçün qarışıqlıq yarada bilər.
5. **Aksent və danışıq fərqləri:** Müxtəlif bölgələrdən olan danışanların tələffüz fərqləri model performansına təsir edə bilər.
6. **Audio şəraiti:** Mikrofon keyfiyyəti, fon səs-küyü və danışıq sürəti nəticələri dəyişir.

---

## 8. Nəticələrin Təhlili

Baza model üçün WER 51.01%, CER isə 15.29% olmuşdur. Bu nəticə production səviyyəsi üçün kifayət qədər yaxşı deyil, lakin zero-shot multilingual model üçün məqbul başlanğıc nəticə hesab edilə bilər.

CER-in WER-dən xeyli aşağı olması vacib müşahidədir. Bu onu göstərir ki, model çox vaxt cümləni tamamilə səhv yaratmır, lakin sözlərin yazılışında və fonetik formasında səhvlər edir.

Modelin tipik səhvləri:

- fonetik oxşar sözlərin qarışdırılması;
- Azərbaycan dilinə məxsus hərflərin səhv yazılması;
- rəqəm və tarixlərin fərqli formada transkripsiyası;
- uzun cümlələrdə bəzi sözlərin buraxılması;
- bəzi sözlərin türk dilinə və ya fonetik oxşar formaya yaxınlaşdırılması.

Fine-tuned `whisper-tiny` modelinin nəticəsi daha zəif olmuşdur: WER 84.66%, CER 34.61%. Bu, kiçik dataset və kiçik model ölçüsü ilə izah olunur. Bu nəticə fine-tuning prosesinin uğursuz olduğu anlamına gəlmir; əksinə, kiçik datasetlə fine-tuning-in kifayət etmədiyini göstərən praktik eksperimentdir.

---

## 9. Hansı Audio Şəraitlərində Model Daha Yaxşı və Daha Pis İşləyir?

Model daha yaxşı işləyir:

- aydın və sakit mühitdə yazılmış audio nümunələrində;
- qısa və sadə cümlələrdə;
- standart tələffüzə yaxın nitqdə;
- az sayda xüsusi ad və rəqəm olan cümlələrdə.

Model daha pis işləyir:

- uzun və mürəkkəb cümlələrdə;
- rəqəmlər, tarixlər və xüsusi adlar olan nümunələrdə;
- fonetik baxımdan yaxın sözlərin çox olduğu cümlələrdə;
- qeyri-standart tələffüz və səs-küy olan audioda;
- Azərbaycan dilinə məxsus səslərin çox olduğu ifadələrdə.

---

## 10. Yaxşılaşdırma Yolları

Bu pipeline-ı production səviyyəsinə çatdırmaq üçün aşağıdakı addımlar vacibdir:

1. **Daha böyük Azərbaycan dili datasetindən istifadə:** Minlərlə saatlıq müxtəlif danışanlardan toplanmış audio və dəqiq transkripsiyalar lazımdır.
2. **Daha böyük Whisper modelini fine-tune etmək:** `whisper-small`, `whisper-medium` və ya daha böyük modellər Azərbaycan dili üçün fine-tune edilə bilər.
3. **Data augmentation:** Noise injection, speed perturbation və SpecAugment kimi metodlarla modelin robustluğu artırıla bilər.
4. **Text normalization:** Rəqəmlər, tarixlər, xüsusi adlar və durğu işarələri üçün daha güclü normalization pipeline qurulmalıdır.
5. **Dialect və accent coverage:** Müxtəlif bölgələrdən danışanların nitqi datasetə daxil edilməlidir.
6. **Evaluation genişləndirilməsi:** Yalnız WER/CER deyil, səhv tiplərinə görə ayrıca analiz də aparılmalıdır.

Daha çox resurs olsaydı, növbəti 3 addım belə olardı:

1. Daha böyük və balanslaşdırılmış Azərbaycan dili speech dataset toplamaq və ya mövcud datasetləri birləşdirmək.
2. `whisper-small` və ya `whisper-medium` modelini GPU üzərində daha uzun fine-tune etmək.
3. Rəqəm, tarix, xüsusi ad və Azərbaycan hərfləri üçün ayrıca post-processing və normalization sistemi qurmaq.

---

## 11. Azərbaycan dili üçün ASR-ın ən böyük problemi

Azərbaycan dili üçün ASR-ın ən böyük problemi yüksək keyfiyyətli, müxtəlif aksentləri və real audio şəraitlərini əhatə edən böyük labeled speech datasetlərinin məhdud olmasıdır.

---

## 12. Nəticə

Bu layihədə Azərbaycan dili üçün işlək ASR pipeline qurulmuşdur. Baza model kimi `openai/whisper-small` istifadə edilmiş və 50 test nümunəsi üzərində WER 51.01%, CER 15.29% nəticəsi əldə edilmişdir. Fine-tuning mərhələsində `openai/whisper-tiny` modeli 50 train nümunəsi üzərində 3 epoch öyrədilmiş və test nəticəsi WER 84.66%, CER 34.61% olmuşdur.

Fine-tuned model baza modeldən zəif nəticə göstərsə də, layihə texniki baxımdan əsas tələbləri yerinə yetirir: dataset hazırlığı, inference, WER/CER hesablanması, fine-tuning cəhdi, validation WER izlənməsi, checkpoint saxlanması və nəticələrin müqayisəsi həyata keçirilmişdir.

Gələcəkdə daha böyük dataset, daha güclü model, GPU resursları və daha yaxşı text normalization ilə Azərbaycan dili üçün daha keyfiyyətli ASR sistemi qurmaq mümkündür.
