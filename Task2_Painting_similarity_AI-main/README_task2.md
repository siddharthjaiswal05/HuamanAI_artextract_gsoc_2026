# Task 2: Painting Similarity Search
## National Gallery of Art Open Data Program


## Abstract

This task explores image similarity search across the National Gallery of Art painting collection. Given a query painting, the system retrieves the most visually similar works from the gallery. The approach uses dual deep feature extraction, a convolutional backbone for local texture and composition, and a self supervised Vision Transformer for semantic style, combined with a fast cosine similarity index. The system operates on 3,544 paintings spanning 1240 to 2021 and achieves a mean average precision of 0.4817 using art style as ground truth, representing a significant improvement over the VGG16/ResNet50 baseline from the reference tutorial.


## Dataset

**Source:** The National Gallery of Art Open Data Program (Creative Commons Zero)  
**Kaggle path:** `/kaggle/input/the-national-gallery-of-art-open-data-pro/opendata-main/data/`

The dataset is a relational collection of CSV files. The key tables used in this task are:

| File | Rows | Purpose |
|------|------|---------|
| objects.csv | 138,698 | All NGA artworks. Filter by `classification == 'Painting'` gives 4,260 records |
| published_images.csv | 106,609 | IIIF image URLs. Joins to objects via `depictstmsobjectid` (not `objectid`) |
| objects_terms.csv | 394,957 | Subject, style, theme, technique tags per artwork |
| objects_constituents.csv | 803,134 | Links artworks to artists via constituentid |
| constituents.csv | 26,234 | Artist names, nationality, dates |
| objects_dimensions.csv | 210,770 | Physical dimensions per artwork |

**Important schema note:** `published_images.csv` does not contain an `objectid` column. The foreign key linking images to artworks is `depictstmsobjectid`.


## EDA Findings

### Collection Overview

| Metric | Value |
|--------|-------|
| Paintings (classification filter) | 4,260 |
| Paintings with published IIIF images | 3,821 |
| Successfully downloaded thumbnails | 3,544 |
| Year range | 1240 to 2021 |
| Unique artists | 1,194 |
| Unique nationalities | 43 |
| Paintings with art style labels | 2,544 (66.6%) |

### Art Styles Distribution (ground truth for evaluation)

The `visualbrowserstyle` field in `objects_terms.csv` provides clean categorical style labels. These are used as ground truth for Precision@K evaluation, two paintings are considered relevant if they share the same style.

| Style | Count |
|-------|-------|
| Renaissance | 404 |
| Baroque | 395 |
| Realist | 349 |
| Naive | 285 |
| Abstract Expressionist | 284 |
| Impressionist | 281 |
| Romantic | 168 |
| Post-Impressionist | 115 |
| Rococo | 73 |
| Gothic | 61 |
| Minimalist | 58 |
| German Expressionist | 35 |

### Image Properties

Full resolution originals in the NGA collection are very high quality. Thumbnails (200px) are served via the IIIF API and are sufficient for feature extraction.

| Property | Value |
|----------|-------|
| Full-res width (mean) | 10,491 px |
| Full-res height (mean) | 10,467 px |
| Orientation — Portrait | 1,846 (52%) |
| Orientation — Landscape | 1,763 (49%) |
| Orientation — Square | 212 (6%) |

### Key EDA Insights

**Color patterns by nationality:** Dutch and Flemish paintings trend darker (consistent with chiaroscuro technique in Baroque and Golden Age periods). American and French paintings show warmer average RGB values.

**Timeline concentration:** The majority of the collection dates to 1600–1900, with American artists most active between 1800 and 1950 and Dutch/Flemish artists peaking in the 1600s.

**Portrait subset correction:** The tutorial used `element == 'painted surface'` in `objects_dimensions.csv` to identify portraits, which incorrectly matched 98.9% of all paintings (the field describes a physical property, not content). The correct approach is to filter `objects_terms` by portrait-related terms (portrait, figure, bust, head, self-portrait), which identifies 206 paintings (5.4%) as figurative works.


## Approach

### Comparison with Tutorial Baseline

| Aspect | Tutorial | This Work |
|--------|----------|-----------|
| Dataset size | 155 paintings | 3,544 paintings |
| Feature extractor | VGG16 / ResNet50 (frozen) | EfficientNet-B3 + DINO ViT-S/8 |
| Embedding dimension | ~4,096 (VGG) | 1,920 (1536 + 384) |
| Similarity search | Brute-force cosine | FAISS IndexFlatIP |
| Face detection | MTCNN (poor on paintings) | Portrait subset via objects_terms |
| Evaluation | SSIM + RMSE (pixel-level) | Precision@K + mAP + SSIM + RMSE |

### Feature Extraction

Two models are combined to capture complementary aspects of visual similarity:

**EfficientNet B3** (1536-d): A convolutional network pretrained on ImageNet. The final global average pool output captures local texture, brushstroke patterns, color distributions, and spatial composition. Well-suited to discriminating between paintings that differ in surface appearance.

**DINO ViT-S/8** (384 d): A Vision Transformer trained with self supervised learning using the DINO objective. The CLS token embedding captures semantic and structural properties, object parts, pose, and style without color bias. DINO patch attention has been shown to naturally segment foreground objects in paintings even without fine-tuning.

The two embeddings are concatenated to form a 1,920-dimensional vector, then L2 normalized so that cosine similarity equals inner product.

```
Image (200px thumbnail, center-cropped to 224x224)
    |
    |--- EfficientNet-B3 features  --> 1536-d
    |--- DINO ViT-S/8 features     -->  384-d
    |
    Concatenate --> 1920-d
    L2-normalize
    |
    FAISS IndexFlatIP (cosine search)
    |
    Top-K results
```

### Similarity Index

FAISS `IndexFlatIP` performs exact inner product search on unit-normalized vectors, equivalent to exact cosine similarity. For the 3,544-painting collection this is fast enough for real time querying. For larger collections (100k+), `IndexIVFFlat` with approximate nearest neighbor search would be appropriate.


## Results

### Quantitative Evaluation

**Ground truth:** Two paintings are considered relevant if they share the same `visualbrowserstyle` label (e.g. both Impressionist).  
**Query set:** 200 randomly sampled paintings with style labels.

| Metric | Score |
|--------|-------|
| Precision@1 | 0.4650 |
| Precision@5 | 0.4010 |
| Precision@10 | 0.3765 |
| Precision@20 | 0.3405 |
| mAP | 0.4817 |

### Precision@10 per Art Style

| Style | P@10 | Queries |
|-------|------|---------|
| Abstract Expressionist | 0.639 | 28 |
| Renaissance | 0.545 | 20 |
| German Expressionist | 0.500 | 1 |
| Baroque | 0.475 | 28 |
| Minimalist | 0.457 | 7 |
| Gothic | 0.400 | 2 |
| Naive | 0.320 | 25 |
| Impressionist | 0.269 | 29 |
| Romantic | 0.238 | 13 |
| Realist | 0.228 | 32 |
| Post-Impressionist | 0.189 | 9 |
| Rococo | 0.160 | 5 |

Abstract Expressionist retrieves most accurately, likely because its distinct non-representational visual features (flat color fields, gestural marks) differ sharply from other styles. Realist and Post Impressionist score lower because their surface appearance overlaps with neighboring styles.

### Comparison with Tutorial Baseline

| Metric | Tutorial (VGG16/ResNet50) | This Work |
|--------|--------------------------|-----------|
| Average SSIM | ~0.171 | 0.2793 |
| Average RMSE | ~0.271 | 0.2410 |
| Precision@10 | Not reported | 0.3765 |
| mAP | Not reported | 0.4817 |

SSIM improved by 63% and RMSE decreased by 11% compared to the reported tutorial numbers. The introduction of Precision@K and mAP provides a semantically meaningful evaluation that pixel level metrics cannot capture.

### t-SNE Embedding Visualization

A t-SNE projection of 2,331 embeddings (paintings with style labels) was computed via PCA reduction to 50 dimensions (explaining 59.1% of variance) followed by t-SNE with perplexity 35. The resulting 2D map shows visible clustering by art style, particularly for Abstract Expressionist, Gothic, and Renaissance works. Neighboring clusters (Baroque and Renaissance, Realist and Romantic) reflect genuine stylistic continuity in art history.


## Implementation

### Requirements

```
faiss-cpu
umap-learn
scikit-image
torch
torchvision
Pillow
pandas
numpy
matplotlib
seaborn
tqdm
scikit-learn
```

Install:
```
pip install faiss-cpu umap-learn scikit-image -q
```

DINO ViT is loaded via `torch.hub`:
```python
torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
```

### Running the Notebook

1. Add the NGA dataset on Kaggle: search "The National Gallery of Art Open Data Pro" and add to the notebook.
2. Enable GPU accelerator (Tesla P100 was used in this run).
3. Run all cells in order. The notebook handles download, feature extraction, index construction, evaluation, and visualization in a single pass.

### Outputs

| File | Description |
|------|-------------|
| paintings_clean.csv | Full metadata for 3,821 paintings with local paths |
| portraits_subset.csv | 206 figurative/portrait paintings identified via term tags |
| features.npy | L2 normalized embedding matrix, shape (3544, 1920) |
| object_ids.npy | Corresponding objectid array |
| painting_similarity.index | FAISS IndexFlatIP binary index |
| 01_dataset_overview.png | 6-panel distribution charts |
| 02_sample_grid.png | 6x6 random sample grid |
| 03_image_properties.png | Size and orientation analysis |
| 04_color_by_nationality.png | Average RGB per nationality with color swatches |
| 05_subject_terms.png | Term, theme, and style distributions |
| 06_style_samples.png | 3 example paintings per art style |
| 07_timeline.png | Paintings over time by nationality |
| 08_precision_per_style.png | P@10 bar chart per style |
| 09_retrieval_*.png | Visual retrieval grids for 4 query styles |
| 10_portrait_retrieval_*.png | Portrait/figure similarity demos |
| 11_tsne_by_style.png | t-SNE embedding colored by art style |
| 12_tsne_by_nationality.png | t-SNE embedding colored by nationality |


## Possible Improvements

**Fine tuning on art data:** Both EfficientNet and DINO are pretrained on natural images. Fine tuning on a labeled art dataset (e.g. WikiArt, which is used in Task 1) would likely improve retrieval quality, particularly for styles with low P@10 such as Rococo and Post-Impressionist.

**Higher resolution inputs:** Thumbnails are 200px. Requesting the IIIF API at 400px or 600px would provide richer texture information, particularly for detail sensitive styles.

**Cross task features:** Task 1 trains a classifier on WikiArt style labels. The penultimate layer of that model could serve as an art domain specific feature extractor for Task 2, replacing or augmenting the ImageNet-pretrained models.

**Approximate search scaling:** For the full 101,854 image NGA collection (all object types), replacing `IndexFlatIP` with `IndexIVFFlat` or `IndexHNSWFlat` would allow sub-millisecond queries.

**Multispectral and infrared data:** As noted in the tutorial, NGA has published some multispectral imaging data. Incorporating non-visible-spectrum features could reveal compositional underdrawings and enable a hidden image discovery mode, which the tutorial identifies as a promising research direction.


## Citation

National Gallery of Art Collection Dataset  
https://github.com/NationalGalleryOfArt/opendata  
Released under Creative Commons Zero (CC0)
