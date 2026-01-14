# AncientEgyptianEmbeddings
The Egyptians believed the most significant thing you could do in your life was die.


# Ancient Egyptian Sentence Embeddings

**Fine-tuning LaBSE for Cross-lingual Semantic Similarity in Ancient Egyptian**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This project adapts the Language-Agnostic BERT Sentence Embeddings (LaBSE) model to Ancient Egyptian hieroglyphic transliteration, enabling cross-lingual semantic search and information retrieval for Egyptological research.

## Project Overview

Ancient Egyptian, despite being extensively documented, remains a low-resource language for modern NLP. This work addresses two key challenges:

1. **Tokenization**: Standard multilingual tokenizers fail on Egyptian due to unique morphological markers (`.t` feminine suffix, compound hyphens) and special transliteration characters (ḫ, ḏ, ꜥ, ꜣ)
2. **Cross-lingual Alignment**: Pre-trained models lack Egyptian in their vocabulary and training data

### Key Contributions

- ✅ **Custom WordPiece Tokenizer**: 0% UNK rate on Ancient Egyptian (vs high UNK rate for LaBSE original)
- ✅ **Vocabulary Expansion**: Added 761 Egyptian-specific tokens to LaBSE (501,153 → 501,914)
- ✅ **Fine-tuned Model**: Trained on 23,400 Egyptian-German parallel pairs from Old Kingdom texts
- ✅ **Public Dataset**: Processed TLA corpus with German translations ready for NLP research

## Results

### Tokenization Quality
| Tokenizer | UNK Rate | Fertility |
|-----------|----------|-----------|
| LaBSE Original | Variable (high) | 2.3-2.5 |
| **Custom WordPiece** | **0.0%** | **2.31** |

### Cross-lingual Similarity
- **Parallel pairs** (correct translations): **0.557** average similarity
- **Non-parallel pairs** (unrelated sentences): **0.116** average similarity
- **Best example**: `mw.t =f` (his mother) ↔ "seine Mutter" = **0.740** similarity

### Training
- **Validation loss reduction**: 66.5% (0.2797 → 0.0937) in 6 epochs
- **Training time**: ~9 minutes on single GPU
- **No overfitting**: Generalization achieved on 21,060 training pairs

##  Repository Structure

```
ancient-egyptian-embeddings/
├── data/
│   ├── README.md                         
│   └── processed/
│       ├── altes_reich_lemmas_NORMALIZED.txt   # 26,450 sentences, 221,194 tokens
│       ├── corpus_lemmatized_filtered.txt      # Cleaned for Word2Vec
│       ├── corpus_lemmatized_full.parquet      # Full corpus with metadata
│       └── training_pairs.csv                   # 100,819 parallel pairs
│
├── notebooks/
│   ├── 01_data_preparation.ipynb          # Corpus processing from TLA dump
│   ├── 02_corpus_analysis.ipynb           # Statistical analysis
│   ├── 03_word2vec_experiments.ipynb      # Static embeddings baseline
│   ├── 04_tokenizer_development.ipynb     # Custom WordPiece training
│   ├── 05_labse_finetuning.ipynb         # Model fine-tuning
│   └── 06_evaluation.ipynb                # Similarity analysis & visualization
│
├── models/
│   ├── custom_egyptian_tokenizer.json     # Trained WordPiece tokenizer (4K vocab)
│   ├── LaBSE_egyptian_expanded/           # LaBSE with expanded vocabulary
│   └── LaBSE_egyptian_finetuned/          # Final fine-tuned model
│
│
├── report/
│   ├── report.tex                          # LaTeX source
│   ├── report.pdf                          # Compiled report
│   ├── lit.bib                             # Bibliography
│   └── figures/                            # Generated visualizations
│
├── results/
│   ├── tokenizer_comparison.csv            # Tokenization benchmarks
│   ├── training_progress.png               # Loss curves
│   └── all_models_comparison.html          # Interactive embedding visualization
│
└── README.md                               
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/valennikolaeva/ancient-egyptian-embeddings.git
cd ancient-egyptian-embeddings

# Install dependencies
pip install -r requirements.txt
```

### Using the Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
model_path = "./models/LaBSE_egyptian_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def encode_sentence(text):
    """Encode sentence to embedding"""
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded)
    # Mean pooling
    embeddings = output[0].mean(dim=1)
    # Normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

# Example: Egyptian sentence
egy_text = "ḥm-nṯr n Ptḥ"
egy_embedding = encode_sentence(egy_text)

# German translation
ger_text = "Priester des Ptah"
ger_embedding = encode_sentence(ger_text)

# Compute similarity
similarity = torch.nn.functional.cosine_similarity(egy_embedding, ger_embedding)
print(f"Similarity: {similarity.item():.4f}")  # Expected: ~0.66
```

### Tokenization Only

```python
from tokenizers import Tokenizer

# Load custom Egyptian tokenizer
tokenizer = Tokenizer.from_file("./models/custom_egyptian_tokenizer.json")

# Test
text = "mw.t =f ḥm-nṯr n Ptḥ"
encoding = tokenizer.encode(text)
print(f"Tokens: {encoding.tokens}")
print(f"No UNK: {'[UNK]' not in encoding.tokens}")  # True
```

## Dataset

### Source
Data from **Thesaurus Linguae Aegyptiae** (TLA) project:
- **Institution**: Berlin-Brandenburg Academy of Sciences and Humanities
- **URL**: https://edoc.bbaw.de/frontdoor/index/index/docId/2919
- **Additional source**: `thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium` on Hugging Face

### Statistics
- **Corpus**: 26,450 sentences, 221,194 tokens, 7,270 unique lemmas
- **Parallel pairs**: 100,819 Egyptian-German sentence pairs (after processing: 23,400 high-quality pairs)
- **Period**: Altes Reich (Old Kingdom, ca. 2686-2181 BCE)
- **Avg sentence length**: 8.36 tokens (Egyptian), 12.31 tokens (German)

See [data/README.md](data/README.md) for detailed documentation.

## Methodology

### 1. Tokenizer Development
- **Architecture**: WordPiece (BERT-compatible)
- **Vocabulary**: 4,000 tokens trained on Egyptian corpus
- **Key feature**: Zero UNK tokens on Egyptian while maintaining reasonable fertility

### 2. Vocabulary Expansion
- Identified 761 new Egyptian tokens not in LaBSE vocabulary
- Expanded embedding matrix from 501,153 → 501,914 tokens
- Initialized new embeddings via sampling from normal distribution parameterized by existing embeddings

### 3. Fine-tuning
- **Objective**: Bidirectional translation ranking loss (contrastive)
- **Data**: 21,060 train / 2,340 validation pairs
- **Training**: 6 epochs, ~9 minutes on GPU
- **Best validation loss**: 0.0937

## Evaluation

### Qualitative Assessment
Model successfully distinguishes related vs unrelated sentence pairs:

| Egyptian | German | Similarity |
|----------|--------|------------|
| `ḥm-nṯr n Ptḥ` | Priester des Ptah (priest of Ptah) | 0.660 ✓ |
| `mw.t =f` | seine Mutter (his mother) | 0.740 ✓ |
| `ḥm-nṯr n Ptḥ` | der König ging nach Süden (king went south) | 0.191 ✓ |
| `jri̯ =f pr` | Priester des Amun (priest of Amun) | 0.002 ✓ |

### Limitations
- **Word Sense Induction**: Limited success with polysemous words due to corpus size and formulaic context
- **Quantitative metrics**: Formal retrieval evaluation (P@k, MRR) pending standardized test set
- **Temporal coverage**: Old Kingdom only; Middle/Late Egyptian require additional data

