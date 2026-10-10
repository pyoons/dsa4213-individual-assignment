# DSA4213 Assignment 1: Language Models on Medical Abstracts

This project focuses on medical abstracts. Specifically, I ask the question: do large general-purpose embeddings (GloVe) bear embeddings trained on a small in-domain corpus? Where do differences e.g. frequent vs rare words appear between these models?

## Repo structure 

medlm/
├── README.md
├── requirements.txt
├── configs/            # one YAML per run
├── data/               # processed splits + vocab
├── src/
│   ├── data.py         # loading, tokenizing, vocab, batching
│   ├── ngram.py
│   ├── models.py       # RNN, LSTM, Transformer
│   ├── train.py        # shared training loop
│   ├── evaluate.py     # perplexity, per-token losses, timing
│   ├── generate.py
│   ├── embeddings.py   # word2vec training, GloVe loading, coverage
│   └── downstream.py   # Part III features + classifier
├── results/            # JSON logs, per-token loss arrays, figures
└── notebooks/analysis.ipynb


## Data

| Resource | Source | License / citation |
|---|---|---|
| Medical Abstracts TC corpus | <https://github.com/sebischair/Medical-Abstracts-TC-Corpus> (same data as `TimSchopf/medical_abstracts` on Hugging Face) | CC BY-SA 3.0; Schopf, Braun & Matthes (2023), NLPIR '22 |
| GloVe 6B, 200d (Wikipedia 2014 + Gigaword 5) | gensim-data release `glove-wiki-gigaword-200` (identical vectors to Stanford `glove.6B.200d.txt`) | Pennington, Socher & Manning (2014) |


Download guide:

```bash
mkdir -p data/raw data/glove
git clone --depth 1 https://github.com/sebischair/Medical-Abstracts-TC-Corpus.git /tmp/matc
cp /tmp/matc/medical_tc_*.csv data/raw/
curl -L -o data/glove/glove-wiki-gigaword-200.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-200/glove-wiki-gigaword-200.gz
```

## Setup and commands

```bash
pip install -r requirements.txt

# Step 1: compare hyphen handling against GloVe, then build processed data
python -m src.prepare_data --config configs/data.yaml --coverage-check   # -> results/coverage_check.json
python -m src.prepare_data --config configs/data.yaml                    # -> data/processed/

# Step 2: interpolated trigram
python -m src.ngram --config configs/ngram.yaml                          # -> results/ngram/
```

