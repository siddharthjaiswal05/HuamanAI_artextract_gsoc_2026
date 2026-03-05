# Task 1 — WikiArt Classification
## Style, Genre, and Artist Recognition with CNN + BiLSTM + Attention

---

## What This Task Does

Given a painting image, the model predicts three things at once: what artistic style the painting belongs to (e.g. Impressionism, Baroque), what genre it is (e.g. portrait, landscape, still life), and who painted it. These are three separate classification problems, each trained independently on the same base architecture.

---

## Dataset

**Source:** WikiArt — a large collection of fine art images  
**Kaggle paths:**
- Images: `/kaggle/input/datasets/steubk/wikiart`
- CSVs: `/kaggle/input/datasets/siddinfinity/wikidatescsv`

The dataset comes with pre-defined train and val CSV files. Since there is no separate test split, the val set is divided 50/50 into a validation set (used during training to pick the best checkpoint) and a held-out test set (used only once at the end for final evaluation).

| Task | Classes | Train | Val | Test |
|------|---------|-------|-----|------|
| Style | 27 | 56,487 | 12,082 | 12,106 |
| Genre | 10 | 45,080 | 9,642 | 9,657 |
| Artist | 23 | 13,344 | 2,853 | 2,853 |

There is no leakage between train and val/test splits for artist and genre. Style has 8 images that appear in both train and style splits, which is negligible across 80,675 images.

---

## EDA Findings

### Class Imbalance

Class imbalance is the most important data property affecting model design.

**Style** is the hardest task: 27 classes with a maximum-to-minimum class count ratio of 129x. The largest class has 8,890 images; the smallest has 69. This extreme imbalance means a model that only predicts the majority class would still get roughly 15% accuracy, but would be useless on rare styles. Weighted cross-entropy loss is required.

**Genre** has 10 classes with a 7.4x imbalance ratio — much more manageable.

**Artist** has 23 classes with a 3.9x ratio — the most balanced of the three tasks.

Class weights are computed as `total / (n_classes * class_count)` for each class and saved as `.pt` files. The style weight range is [0.235, 30.320], meaning the rarest style is given 130 times more loss weight than the most common one.

### Image Properties

All images were sampled from the style training set (800 images):

- Mean width: 1,636 px | Mean height: 1,646 px
- Width std: 366 px — images vary significantly in size
- Mean aspect ratio: 1.06 (slightly wider than tall on average)
- All images are resized to 224x224 during training

### Color Patterns

Average pixel values were computed per style class. Darker styles like High Renaissance and Baroque have mean pixel values below 0.35 across all channels. Lighter styles like Impressionism and Naive Art sit above 0.5. This color signal alone partially separates styles and is part of what the model learns in early convolutional layers.

### Cross-task Overlap

A large portion of images appear in multiple task splits, which supports future multi-task learning:

| Pair | Shared images |
|------|--------------|
| Artist and Genre | 11,292 |
| Artist and Style | 13,226 |
| Genre and Style | 44,870 |
| All three | 11,274 |

---

## Model Architecture

The model is called CNNBiLSTMClassifier. The core idea is to treat the spatial grid of CNN features as a sequence and process it with a recurrent network, so the model can capture how different regions of a painting relate to each other rather than pooling everything into a single vector.

### Step 1: CNN Feature Extraction

The backbone is **EfficientNet-B3** pretrained on ImageNet. Its output at the final convolutional layer is a feature map of shape `(batch, 1536, 7, 7)` — 1,536 channels over a 7x7 spatial grid.

A 1x1 convolution reduces the channels from 1,536 to 512:

```
Input image: (B, 3, 224, 224)
  -> EfficientNet-B3 features
  -> (B, 1536, 7, 7)
  -> Conv2d(1536, 512, kernel=1) + BatchNorm + GELU
  -> (B, 512, 7, 7)
```

### Step 2: Reshape to Sequence

The 7x7 spatial grid is flattened into a sequence of 49 positions, each with a 512-dimensional feature vector. This lets the BiLSTM treat spatial positions in the painting as time steps:

```
(B, 512, 7, 7)  ->  reshape  ->  (B, 49, 512)
```

The reason to use a sequence model here is that art style is often about composition — whether the sky is in the top half, whether figures dominate the foreground, how light distributes across the canvas. A BiLSTM can model these spatial relationships across the 49 grid positions in both directions.

### Step 3: Bidirectional LSTM

A 2-layer BiLSTM processes the 49-step sequence:

```
Input:  (B, 49, 512)
BiLSTM: hidden=256, layers=2, bidirectional=True
Output: (B, 49, 512)   -- 256 forward + 256 backward
```

The bidirectional design means each position is informed by context from both the left and right of the sequence, which is appropriate since spatial position in a painting has no natural left-to-right order.

### Step 4: Self-Attention

A learned attention mechanism assigns a scalar weight to each of the 49 sequence positions:

```
scores  = Linear(512, 1)(lstm_output)        -> (B, 49)
weights = softmax(scores)                    -> (B, 49)
context = sum(lstm_output * weights)         -> (B, 512)
```

This produces a single 512-dimensional context vector that is a weighted combination of all spatial positions. Positions that are more discriminative for the predicted class get higher weights. The attention map can be visualized as a heatmap over the original image to see where the model is looking.

### Step 5: Classification Head

```
context: (B, 512)
  -> LayerNorm
  -> Dropout(0.3)
  -> Linear(512, 512) + GELU
  -> Dropout(0.15)
  -> Linear(512, n_classes)
  -> logits: (B, n_classes)
```

**Total parameters: 14,915,652**

---

## Training

### Two-Phase Training

Training uses a two-phase strategy to avoid overwriting pretrained ImageNet weights too early.

**Phase 1 — 5 epochs, backbone frozen:**  
Only the channel reducer, BiLSTM, attention, and classifier head are trained. The EfficientNet backbone weights are frozen. This lets the new components stabilize before touching the backbone. Learning rate: 3e-4.

**Phase 2 — 10 epochs, backbone partially unfrozen:**  
The last 3 blocks of EfficientNet are unfrozen. Fine-tuning with a lower learning rate of 5e-5 allows the CNN to adapt its learned features toward art classification. The first blocks stay frozen because they learn general low-level features (edges, textures) that are useful across all domains.

| Phase | Trainable params |
|-------|-----------------|
| Phase 1 (frozen) | ~4.2 million |
| Phase 2 (unfrozen last 3 blocks) | ~12.7 million |

### Other Training Details

- Optimizer: AdamW with weight decay 1e-4
- Scheduler: Cosine annealing learning rate decay
- Loss: CrossEntropyLoss with per-class weights (addresses imbalance)
- Mixed precision: torch.cuda.amp (FP16 forward pass, FP32 gradient accumulation)
- Gradient clipping: max norm 1.0
- GPU: Tesla P100-PCIE-16GB

### Data Augmentation (training only)

- Random resized crop (scale 0.7–1.0)
- Random horizontal flip
- Random vertical flip (p=0.1)
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation up to 15 degrees

Validation and test use center crop only, no augmentation.

### NaN Loss

A single NaN validation loss occurred at P1 E04 for style and P1 E05 for genre. This is caused by numerical overflow in float16 with extreme class weights (up to 30x). The issue was isolated to the loss metric — accuracy and F1 continued computing correctly and checkpointing was unaffected. All final results are valid.

---

## Validation Results

Best validation macro-F1 per task (from the best checkpoint saved during training):

| Task | Best Val F1 | Best Val Acc |
|------|-------------|--------------|
| Style | 0.5223 | — |
| Genre | 0.7251 | — |
| Artist | 0.7699 | — |

### Training Curve Observations

For all three tasks, validation loss decreases consistently across both phases. Phase 2 shows a clear jump when the backbone is unfrozen — this is most visible for style, where F1 goes from 0.392 at the end of Phase 1 to 0.522 at the end of Phase 2. Validation accuracy consistently stays above training accuracy in early epochs, which indicates the model is not overfitting.

### Hardest Classes per Task

**Style — 5 hardest classes (val F1):**

| Class | F1 |
|-------|----|
| New_Realism | 0.188 |
| Fauvism | 0.285 |
| Post_Impressionism | 0.370 |
| Expressionism | 0.378 |
| Mannerism_Late_Renaissance | 0.411 |

New Realism is the hardest because it strongly overlaps visually with contemporary photography and Realism. Fauvism and Post-Impressionism are frequently confused with each other and with Impressionism because they share loose brushwork and bright colors.

**Genre — 5 hardest classes (val F1):**

| Class | F1 |
|-------|----|
| illustration | 0.588 |
| sketch_and_study | 0.622 |
| genre_painting | 0.624 |
| nude_painting | 0.672 |
| cityscape | 0.731 |

Illustration and sketch_and_study are hard because they represent media rather than content — an illustration can depict any subject. Genre painting (scenes of everyday life) visually overlaps with portrait, landscape, and religious painting.

**Artist — 5 hardest classes (val F1):**

| Class | F1 |
|-------|----|
| Salvador_Dali | 0.438 |
| Boris_Kustodiev | 0.623 |
| Pablo_Picasso | 0.653 |
| Ilya_Repin | 0.667 |
| Martiros_Saryan | 0.670 |

Salvador Dali is the hardest artist to classify. His style spans Surrealism, Realism, and abstract work — there is no single visual signature. Picasso is similarly hard because his output spans Cubism, his Blue and Rose periods, and later figurative work that looks unlike his most famous paintings.

---

## Final Test Results

The test set was evaluated once, after all training was complete, using the best checkpoint for each task.

| Task | Classes | Test Images | Test Accuracy | Test Macro-F1 |
|------|---------|-------------|---------------|---------------|
| Style | 27 | 12,106 | 52.4% | 0.5296 |
| Genre | 10 | 9,657 | 75.2% | 0.7256 |
| Artist | 23 | 2,853 | 77.1% | 0.7527 |

### Style Classification (27 classes)

Full per-class results on the test set:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Abstract_Expressionism | 0.54 | 0.63 | 0.58 | 417 |
| Action_painting | 0.40 | 0.71 | 0.51 | 14 |
| Analytical_Cubism | 0.48 | 0.81 | 0.60 | 16 |
| Art_Nouveau | 0.54 | 0.46 | 0.50 | 646 |
| Baroque | 0.56 | 0.51 | 0.53 | 633 |
| Color_Field_Painting | 0.69 | 0.73 | 0.71 | 242 |
| Contemporary_Realism | 0.32 | 0.50 | 0.39 | 72 |
| Cubism | 0.53 | 0.59 | 0.56 | 330 |
| Early_Renaissance | 0.56 | 0.72 | 0.63 | 209 |
| Expressionism | 0.45 | 0.31 | 0.37 | 1,009 |
| Fauvism | 0.22 | 0.56 | 0.31 | 140 |
| High_Renaissance | 0.40 | 0.51 | 0.45 | 201 |
| Impressionism | 0.64 | 0.64 | 0.64 | 1,904 |
| Mannerism_Late_Renaissance | 0.36 | 0.62 | 0.46 | 191 |
| Minimalism | 0.65 | 0.76 | 0.70 | 200 |
| Naive_Art_Primitivism | 0.48 | 0.64 | 0.55 | 359 |
| New_Realism | 0.12 | 0.38 | 0.18 | 47 |
| Northern_Renaissance | 0.57 | 0.72 | 0.64 | 383 |
| Pointillism | 0.58 | 0.80 | 0.67 | 76 |
| Pop_Art | 0.48 | 0.58 | 0.52 | 220 |
| Post_Impressionism | 0.39 | 0.35 | 0.37 | 966 |
| Realism | 0.54 | 0.47 | 0.50 | 1,608 |
| Rococo | 0.49 | 0.73 | 0.59 | 313 |
| Romanticism | 0.60 | 0.38 | 0.46 | 1,040 |
| Symbolism | 0.52 | 0.46 | 0.49 | 663 |
| Synthetic_Cubism | 0.47 | 0.66 | 0.55 | 32 |
| Ukiyo_e | 0.76 | 0.92 | 0.83 | 175 |

Ukiyo_e (Japanese woodblock-influenced painting) achieves the highest F1 of 0.83 because it has a very distinct visual signature — flat color planes, strong outlines, and specific subject matter — that does not overlap with European styles. Color_Field_Painting (F1 0.71) and Minimalism (F1 0.70) also perform well for similar reasons: they are visually distinct.

The lowest performers are styles that occupy visual middle ground between neighboring movements. New_Realism (F1 0.18) has only 47 test samples and overlaps significantly with Realism and Contemporary_Realism. Fauvism (F1 0.31) is frequently predicted as Post_Impressionism or Expressionism, which share the same expressive brushwork.

### Genre Classification (10 classes)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| abstract_painting | 0.84 | 0.91 | 0.87 | 745 |
| cityscape | 0.68 | 0.82 | 0.74 | 687 |
| genre_painting | 0.70 | 0.57 | 0.63 | 1,595 |
| illustration | 0.48 | 0.63 | 0.55 | 284 |
| landscape | 0.89 | 0.79 | 0.84 | 1,979 |
| nude_painting | 0.60 | 0.79 | 0.68 | 286 |
| portrait | 0.86 | 0.79 | 0.82 | 2,101 |
| religious_painting | 0.74 | 0.71 | 0.72 | 975 |
| sketch_and_study | 0.51 | 0.75 | 0.60 | 588 |
| still_life | 0.73 | 0.88 | 0.80 | 417 |

Abstract_painting (F1 0.87) and landscape (F1 0.84) are the easiest genres. Portrait (F1 0.82) and still_life (F1 0.80) are also strong. Genre_painting is the hardest because it depicts everyday scenes that visually resemble landscapes, portraits, and cityscapes simultaneously — the category is defined by social content, not visual form.

### Artist Classification (23 classes)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Albrecht_Durer | 0.77 | 0.83 | 0.80 | 124 |
| Boris_Kustodiev | 0.52 | 0.60 | 0.55 | 94 |
| Camille_Pissarro | 0.83 | 0.77 | 0.80 | 133 |
| Childe_Hassam | 0.73 | 0.80 | 0.76 | 83 |
| Claude_Monet | 0.89 | 0.75 | 0.81 | 200 |
| Edgar_Degas | 0.77 | 0.79 | 0.78 | 91 |
| Eugene_Boudin | 0.76 | 0.92 | 0.83 | 83 |
| Gustave_Dore | 0.90 | 0.91 | 0.91 | 113 |
| Ilya_Repin | 0.66 | 0.56 | 0.60 | 81 |
| Ivan_Aivazovsky | 0.91 | 0.93 | 0.92 | 86 |
| Ivan_Shishkin | 0.78 | 0.79 | 0.79 | 78 |
| John_Singer_Sargent | 0.61 | 0.76 | 0.68 | 118 |
| Marc_Chagall | 0.82 | 0.68 | 0.74 | 115 |
| Martiros_Saryan | 0.67 | 0.74 | 0.70 | 86 |
| Nicholas_Roerich | 0.86 | 0.80 | 0.83 | 272 |
| Pablo_Picasso | 0.62 | 0.67 | 0.64 | 114 |
| Paul_Cezanne | 0.72 | 0.55 | 0.62 | 86 |
| Pierre_Auguste_Renoir | 0.90 | 0.85 | 0.87 | 210 |
| Pyotr_Konchalovsky | 0.74 | 0.79 | 0.76 | 138 |
| Raphael_Kirchner | 0.89 | 0.86 | 0.87 | 77 |
| Rembrandt | 0.86 | 0.75 | 0.80 | 116 |
| Salvador_Dali | 0.44 | 0.39 | 0.42 | 71 |
| Vincent_van_Gogh | 0.74 | 0.86 | 0.80 | 284 |

Ivan_Aivazovsky (F1 0.92) is the easiest artist to identify — he painted almost exclusively seascapes with a recognizable luminous sky treatment. Gustave_Dore (F1 0.91) is highly distinctive due to his monochrome engraving style. Salvador_Dali (F1 0.42) is the hardest because his work spans wildly different visual styles across his career.

---

## Attention Map Analysis

The attention maps show which spatial positions the BiLSTM focused on when making a prediction. These were generated by passing the attention weights back through the spatial dimensions and overlaying them on the input image.

Observations from the visualizations:

- For portraits and figures, attention concentrates on the face and upper body. In the genre task, a portrait of a woman in a colorful hat correctly predicted as "portrait" with confidence 0.99 showed attention tightly centered on the face.
- For landscapes, attention spreads across the horizon line and sky boundary, which is often where the most compositionally distinctive information sits.
- For still life paintings, attention focused on the central objects (a silver jug and plate) and correctly predicted the class with confidence 1.00.
- For Impressionist paintings, attention was more distributed across the canvas, which makes sense since Impressionism is identified by texture across the entire surface rather than localized features.
- Misclassified examples showed attention on ambiguous regions. A Northern_Renaissance painting was predicted as Early_Renaissance with confidence 0.31 — the low confidence correctly signals uncertainty.

---

## Outlier Analysis

The notebook identifies two types of interesting examples after test evaluation:

**High-confidence wrong predictions** are paintings the model got wrong but was very sure about. Examples from the style task include a Cubism painting classified as Analytical_Cubism with confidence 1.000, a Post_Impressionism painting classified as Contemporary_Realism with confidence 0.993, and a Cubism painting classified as Synthetic_Cubism with confidence 0.996. These are not really model failures — they are cases where the visual boundary between classes is genuinely unclear. Analytical and Synthetic Cubism are sub-movements of Cubism, and Post-Impressionism and Realism share many surface properties.

**Low-confidence correct predictions** are paintings the model got right but was uncertain about. These are typically works that sit at stylistic boundaries — a High_Renaissance painting classified correctly at confidence 0.148, and a Post_Impressionism painting at confidence 0.163. These are valuable for understanding where style categories blur in practice.

---

## Saved Outputs

| File | Description |
|------|-------------|
| best_style.pth | Best style model checkpoint |
| best_genre.pth | Best genre model checkpoint |
| best_artist.pth | Best artist model checkpoint |
| style_class_weights.pt | Per-class loss weights for style |
| genre_class_weights.pt | Per-class loss weights for genre |
| artist_class_weights.pt | Per-class loss weights for artist |
| style_test_split.csv | Held-out test set for style |
| genre_test_split.csv | Held-out test set for genre |
| artist_test_split.csv | Held-out test set for artist |
| outliers_style.csv | Full confidence scores for style test set |
| outliers_genre.csv | Full confidence scores for genre test set |
| outliers_artist.csv | Full confidence scores for artist test set |
| 01_class_distributions.png | Class count bar charts for all three tasks |
| 02_style_samples.png | Sample images per style class |
| 03_image_sizes.png | Width, height, and aspect ratio distributions |
| 04_color_palette.png | Average color per style with color swatches |
| 05_training_curves.png | Loss, accuracy, and F1 curves for all tasks |
| 06_val_confusion_*.png | Validation confusion matrices |
| 07_val_perclass_f1_*.png | Per-class F1 bar charts (val) |
| 08_attention_*.png | Attention map overlays (6 examples per task) |
| 09_outliers_*_wrong.png | High-confidence incorrect predictions |
| 09_outliers_*_boundary.png | Low-confidence correct predictions |
| 10_test_confusion_*.png | Final test confusion matrices |

---

## Possible Improvements

**Multi-task training:** 11,274 images appear in all three task splits simultaneously. A shared backbone with three classification heads trained jointly would allow each task to benefit from the supervision signal of the others. Style and genre are correlated (abstract painting is almost always a modern style) and this correlation is currently unexploited.

**Handling extreme imbalance differently:** The weight range for style (0.235 to 30.320) is very wide. Oversampling rare classes or using focal loss would provide an alternative way to address the 129x imbalance without relying entirely on loss weighting.

**Longer training for style:** The style F1 curve had not fully plateaued by epoch 15. More epochs with a lower learning rate would likely improve performance on the hardest classes (Fauvism, New_Realism, Post_Impressionism).

**Test-time augmentation:** Averaging predictions across multiple crops and flips of the same image at test time consistently improves accuracy by 1-3% without any additional training.

**Cross-task feature transfer:** The artist and genre models both reach higher accuracy than style. The intermediate feature representations from these tasks could be used to warm-start or regularize the style model, since style often correlates with artist and time period.

---

## Requirements

```
torch
torchvision
pandas
numpy
matplotlib
seaborn
scikit-learn
Pillow
opencv-python
tqdm
```

GPU is required for reasonable training time. Each task takes approximately 15 epochs on a Tesla P100, with style taking the longest due to dataset size (about 4 hours total across all three tasks).
