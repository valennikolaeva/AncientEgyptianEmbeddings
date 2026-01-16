# ё Research Notebooks

In this directory, you will find the step-by-step computational pipeline for the Ancient Egyptian semantic analysis. The notebooks are numbered to reflect the logical order of the research process.

## Pipeline Overview

1. **`01_data_preparation.ipynb`** * **Purpose:** Cleaning and normalizing the raw corpus from the Thesaurus Linguae Aegyptiae (TLA).  
   * **Key Tasks:** Standardizing Unicode transliterations, filtering fragmented texts, and creating parallel sentence pairs for the fine-tuning stage.

2. **`02_word2vec_experiments.ipynb`** * **Purpose:** Establishing a baseline using static embeddings.  
   * **Key Tasks:** Training a Skip-gram Word2Vec model on the Old Kingdom corpus and evaluating initial lemma similarities.

3. **`03_wsi_word2vec.ipynb`** * **Purpose:** Word Sense Induction (WSI) using the static baseline.  
   * **Key Tasks:** Clustering Word2Vec vectors for polysemous lemmas and calculating ARI/Silhouette scores to identify "meaning collapse."

4. **`04_tokenizer_and_LaBSE.ipynb`** * **Purpose:** The core transformer-based approach.  
   * **Key Tasks:** * Training the custom WordPiece tokenizer.
     * Fine-tuning the LaBSE model with Contrastive Loss.
     * Extracting contextualized embeddings from the 11th layer.
     * Final evaluation and comparison against the baseline.

##  Evaluation Metrics Used
* **ARI (Adjusted Rand Index):** To measure clustering accuracy against expert Lemma IDs.
* **Silhouette Score:** To evaluate the spatial separation of semantic "senses."
* **Recall@1 / Recall@5:** For cross-lingual retrieval performance.

