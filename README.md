# dsa4213-individual-assignment

Repo structure 

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