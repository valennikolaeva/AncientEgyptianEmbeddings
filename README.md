The Egyptians believed the most significant thing you could do in your life was die.

# Ancient Egyptian Contextual Embeddings 𓇓𓏏𓈖𓇋𓇋𓀭

## Repository Structure

```bash
├── data/
│   └── processed/             # Cleaned and normalized corpora (.txt, .parquet)
│       ├── altes_reich_lemmas_NORMALIZED.txt
│       ├── corpus_lemmatized_full.parquet
│       └── training_pairs.parquet
├── notebooks/                 # Step-by-step research pipeline
│   ├── 01_data_preparation.ipynb
│   ├── 02_word2vec_experiments.ipynb
│   ├── 03_wsi_word2vec.ipynb
│   └── 04_tokenizer_and_LaBSE.ipynb
├── reports/                   # Visualizations and exported analysis
│   ├── all_models_comparison.html
│   ├── labse_semantic_analysis_2.pdf
│   ├── word2vec_semantic_analysis.pdf
│   └── *_umap.html            # Interactive UMAP visualizations 
└── README.md                  # Main project documentation

```

This repository contains the code and methodology for fine-tuning **LaBSE (Language-Agnostic BERT Sentence Embedding)** on Ancient Egyptian (Old Kingdom) texts. The project addresses the challenges of **Word Sense Induction (WSI)** and cross-lingual semantic alignment.

## Project Overview

Ancient Egyptian presents a unique challenge due to extreme polysemy. Static vector models (like Word2Vec) often fail to distinguish between different roles of the same lemma (e.g., *pr* as "house" vs. "to go forth"). 

This project leverages a **Transformer-based architecture** to generate dynamic, context-aware embeddings. By fine-tuning LaBSE on a parallel Egyptian-German corpus, we demonstrate that deep learning models can capture the nuanced syntactic and semantic structures of a dead language.

### Key Results
* **Recall@1:** 71.3% (accuracy in finding the correct German translation).
* **Recall@5:** 90.3%.
* **WSI Performance:** Adjusted Rand Index (ARI) for the lemma *mw* (water) improved from **0.07** (Word2Vec) to **0.54** (LaBSE).
* **Zero OOV:** Custom WordPiece tokenizer reduced the **UNK rate from 11% to 0%**.

---

##  Technical Implementation

### 1. Custom Tokenizer
Standard multilingual tokenizers do not support Egyptological Unicode characters (e.g., `ꜣ, ꜥ, ḫ, ṯ`). 
- **Solution:** Trained a custom **WordPiece tokenizer** on the Old Kingdom corpus.
- **Vocab Size:** 6,000 tokens.
- **Fertility:** 1.33 (minimal subword splitting per lemma).

### 2. Fine-tuning Logic
- **Model:** LaBSE (Language-Agnostic BERT).
- **Loss:** Contrastive Loss (Multiple Negatives Ranking Loss).
- **Optimization:** AdamW with linear warmup.
- **Data:** ~28,000 parallel sentences from the *Thesaurus Linguae Aegyptiae* (TLA).

### 3. Feature Extraction (WSI)
To preserve maximum semantic detail, representations are extracted from the **11th layer (penultimate)** of the encoder. We apply mean-pooling over subword tokens to form the final contextual lemma vector.


