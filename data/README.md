
This directory contains the Ancient Egyptian corpus data used for training sentence embeddings and tokenizer.

## Data Source

All data originates from the **Thesaurus Linguae Aegyptiae (TLA)** project:
- **URL**: https://edoc.bbaw.de/frontdoor/index/index/docId/2919
- **Institution**: Berlin-Brandenburg Academy of Sciences and Humanities
- **Content**: Digital corpus of Ancient Egyptian texts from various periods
- **License**: Open access for research purposes



### Primary Source: TLA Corpus Dump
- `corpus.zip` - Complete TLA corpus in JSON format
- `vocabulary.zip` - Lemma dictionary with word IDs and transliterations

### Secondary Source: Hugging Face Dataset
- **Repository**: `thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium`
- **File**: `train.jsonl`
- **Usage**: Additional Old Kingdom texts to expand corpus coverage

## Directory Structure

```
data/   
└── processed/
    ├── altes_reich_lemmas_NORMALIZED.txt     # Normalized lemmatized corpus
    ├── corpus_lemmatized_filtered.txt        # Filtered version for Word2Vec
    ├── corpus_lemmatized_full.parquet        # Complete processed corpus
    └── training_pairs.parquet                # Egyptian-German parallel sentences
```

## Processing Pipeline

### Step 1: Raw Data Extraction
The TLA corpus dump contains:
- Hierarchical JSON files with texts, sentences, and word-level annotations
- Each word has: surface form (`wChar`), lemma ID (`lKey`), flexion code (`flexCode`)
- Translations at sentence level in the `translation` field

### Step 2: Lemma Normalization
**Script**: [`01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb)

Process:
1. Load lemma dictionary from `vocabulary.zip`
2. Create mapping: `lemma_id → lemma_transliteration`
3. For each text unit → sentence → word:
   - Extract surface form
   - Look up lemma using `lKey`
   - Fall back to surface form if lemma not found
4. Normalize transliteration (consistent encoding of special characters)

**Output**: `altes_reich_lemmas_NORMALIZED.txt`
- One sentence per line
- Lemmatized and normalized
- Used for Word2Vec and tokenizer training

### Step 3: Parallel Corpus Creation
**Script**: Shown in the code above (sentence-translation pair extraction)

Process:
1. Parse all JSON files from corpus
2. For each sentence with translation:
   - Extract lemmatized Egyptian sentence
   - Extract German translation
   - Create pair record with metadata
3. Filter by quality criteria:
   - Minimum sentence length ≥ 3 lemmas
   - Minimum translation length ≥ 5 words
   - Non-empty translations

**Output**: `training_pairs.parquet`

### Step 4: Filtered Versions

**`corpus_lemmatized_filtered.txt`**:
- Additional quality filtering applied
- Removed fragmentary texts
- Removed sentences with uncertain readings (marked with `[...]` or `?`)
- Only thesaurus data 
- Used for Word2Vec training (cleaner data)

**`corpus_lemmatized_full.parquet`**:
- Complete processed corpus with all metadata
- Includes: text IDs, sentence IDs, lemmas, surface forms, flexion codes
- Efficient columnar format for analysis
- Contains both translated and non-translated sentences

## File Descriptions

### `altes_reich_lemmas_NORMALIZED.txt`
- **Content**: Lemmatized Ancient Egyptian sentences
- **Normalization**: Consistent Unicode representation of Egyptian characters (ḫ, ḏ, ꜥ, ꜣ, etc.)
- **Usage**: Data for Word2Vec experiments


### `corpus_lemmatized_filtered.txt`
- **Content**: Subset of normalized corpus after quality filtering - **TLA Thesaurus only**
- **Filtering criteria**:
  - Remove fragmentary texts (marked with `[...]`)
  - Remove uncertain readings (marked with `?`)
  - Minimum 3 words per sentence
- **Usage**: Data for Word2Vec experiments

### `corpus_lemmatized_full.parquet`
- **Content**: Complete processed corpus with metadata from **both TLA dump and Hugging Face dataset**
- **Columns**:
  - `text_id`: TLA text unit identifier
  - `sentence_id`: Sentence identifier
  - `lemmas`: List of lemmas
  - `lemma_ids`: TLA lemma IDs
  - `surface_forms`: List of surface forms
  - `flexion_codes`: Morphological codes
  - `lemmas_text`: Space-separated lemmas
  - `surface_text`: Space-separated surface forms
- **Usage**: Full corpus analysis, metadata extraction


### `training_pairs.parquet`
- **Columns**:
  - `text_id`: Source text identifier
  - `sentence_id`: Sentence identifier
  - `lemmatized_sentence`: Egyptian sentence (lemmatized)
  - `translation`: German translation
  - `word_count`: Number of words in Egyptian sentence
  - Additional metadata columns
- **Usage**: Fine-tuning LaBSE on Egyptian-German parallel data


## Corpus Statistics

### Overall Corpus (altes_reich_lemmas_NORMALIZED.txt)
- **Total sentences**: 26,450
- **Total tokens**: 221,194
- **Unique lemmas**: 7,270
- **Average sentence length**: 8.36 tokens
- **Sentence length range**: 2-276 tokens
- **Period**: Altes Reich (Old Kingdom, ca. 2686-2181 BCE)
- **Sources**: TLA Thesaurus + Hugging Face premium dataset

### Filtered Corpus (corpus_lemmatized_filtered.txt)
- **Total sentences**: 20,492 (77.5% of normalized)
- **Total tokens**: 181,967 (82.3% of normalized)
- **Unique lemmas**: 5,950
- **Source**: TLA Thesaurus only

### Parallel Corpus (training_pairs.parquet)
- **Parallel sentences**: 28,066
- **Egyptian tokens**: 192,925
- **German tokens**: 311,402
- **Average Egyptian sentence length**: 6.87 tokens
- **Average German sentence length**: 11.10 tokens

### Most Frequent Lemmas
1. `=k` (2nd person suffix pronoun) - 12,059 occurrences (5.45%)
2. `=f` (3rd person masc. suffix pronoun) - 10,849 occurrences (4.90%)
3. `n` (preposition "to, for") - 8,992 occurrences (4.07%)
4. `m` (preposition "in, with") - 8,200 occurrences (3.71%)
5. `1...n` (number placeholder) - 8,045 occurrences (3.64%)
6. `Ppy` (royal name Pepi) - 3,420 occurrences (1.55%)
7. `r` (preposition "to, at") - 3,128 occurrences (1.41%)
8. `n.j` (genitive particle) - 2,517 occurrences (1.14%)
9. `pn` (demonstrative "this") - 2,293 occurrences (1.04%)
10. `ꜥ` (body part "arm") - 2,123 occurrences (0.96%)

## Data Quality Notes

⸢...⸣ — Doubtful or restored reading. (Used when characters are damaged or unclear but can be identified with some uncertainty).

[...] — Lacuna restored by the editor. (Text lost due to damage, but reconstructed based on context).

{...} — Text suppressed by the editor. (Characters believed to be an error or superfluous in the original source).

〈...〉 — Text added by the editor. (Additions made to correct an omission or error in the original text).


### Known Issues
1. **Fragmentary texts**: Some texts are partially preserved; marked with `[...]`
2. **Uncertain readings**: Egyptologists mark uncertain transliterations with `?`
3. **Translation granularity**: Some translations are at text level, not sentence level
4. **Morphological complexity**: Egyptian morphology not fully captured in linear text

### Preprocessing Decisions
1. **Hyphens preserved**: Compounds like `ḥm-nṯr` (prophet) kept as single unit
2. **Feminine markers preserved**: `.t` suffix retained (e.g., `nfr.t`)
3. **Determinatives excluded**: Semantic classifiers not included in transliteration
4. **Normalization**: All special characters converted to consistent Unicode

