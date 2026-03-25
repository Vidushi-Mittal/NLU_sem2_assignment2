# NLU_sem2_assignment2
# Natural Language Understanding - Assignment 2 (NLU-A2)

**Author:** Vidushi Mittal  
**Roll Number:** M25MAC014  

This repository contains the implementation for two separate Natural Language Processing (NLP) tasks: learning domain-specific word embeddings and character-level sequence generation using Recurrent Neural Networks (RNNs).

---

## 🛠️ Prerequisites & Dependencies

To run the scripts, ensure you have Python 3.8+ installed along with the following libraries:

`pip install torch torchvision torchaudio numpy nltk bs4 requests wordcloud matplotlib gensim PyMuPDF faker`

*(Note: You will also need to download the necessary NLTK corpora, which the scripts handle automatically via `nltk.download()`).*

---

## 📄 Problem 1: Learning Word Embeddings from IIT Jodhpur Data
**Files:** `M25MAC014_prob1.py`, `M25MAC014_prob1.ipynb`

### Objective
To construct a domain-specific text corpus by scraping the official IIT Jodhpur website and PDF documents, and to train and evaluate Word2Vec embeddings (CBOW and Skip-gram) on this corpus.

### Key Features & Workflow
1. **Data Collection:** Scrapes textual data from URLs and dynamically loaded academic PDF brochures using `BeautifulSoup` and `PyMuPDF`.
2. **Preprocessing Pipeline:** Applies regex cleaning, lowercasing, and tokenization to remove boilerplate and non-alphabetical characters.
3. **Model Training:** * Custom PyTorch implementations of CBOW and Skip-gram.
   * Optimized C-backend implementations using `Gensim` `Word2Vec`.
4. **Evaluation & Visualization:**
   * Extracts top-N frequent words and specific 300-dimensional embedding vectors.
   * Performs semantic evaluations including Cosine Similarity (Nearest Neighbors) and Word Analogies (e.g., `UG : BTech :: PG : ?`).
   * Visualizes the high-dimensional embedding space using **t-SNE** 2D projections.

### How to Run
`python M25MAC014_prob1.py`

*Outputs: Corpus statistics, WordCloud, Top-10 frequent words, semantic similarity results, and t-SNE plot visualizations.*

---

## 📄 Problem 2: Character-Level Name Generation Using RNN Variants
**Files:** `M25MAC014_prob2.py`, `M25MAC014_prob2.ipynb`

### Objective
To design, implement from scratch, and compare different sequence models for character-level synthetic Indian name generation.

### Key Features & Workflow
1. **Dataset Generation:** Utilizes the `Faker` library (`en_IN` locale) to dynamically generate a clean, unique training dataset of 1,000 Indian names.
2. **Model Architectures (PyTorch):**
   * **Vanilla RNN:** A standard recurrent neural network.
   * **Bidirectional LSTM (BLSTM):** Processes sequences forward and backward to capture deep phonetic dependencies.
   * **Attention + RNN (GRU):** Utilizes a dot-product attention mechanism to weight context dynamically.
3. **Hyperparameter Tuning:** Implements a Grid Search to find the optimal configuration across Hidden Sizes (64, 128, 256), Learning Rates (0.001, 0.005), and Layers (1, 2).
4. **Quantitative Evaluation:** Evaluates each model based on:
   * **Novelty %:** How many generated names are entirely new (not in the training set).
   * **Diversity:** The ratio of unique names generated over the total generated batch.

### How to Run
`python M25MAC014_prob2.py`

*Outputs: Dataset generation confirmation, parameter counts for all models, real-time grid-search training logs, and the final benchmark table comparing Loss, Novelty, Diversity, and sampled name generations.*

---

## 📊 Summary of Findings
* **Problem 1:** The `Gensim` inbuilt models significantly outperformed the from-scratch implementations in capturing academic semantics (e.g., successfully clustering degree programs and research agencies).
* **Problem 2:** The **Attention + RNN** proved to be the most robust architecture, balancing a low training loss with high structural realism, avoiding the "mode collapse" (repetition) heavily observed in the BLSTM model.
