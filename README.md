# DSA4213 Assignment 1: Language Models on Medical Abstracts

This project focuses on medical abstracts. Specifically, do large general-purpose embeddings (GloVe) beat embeddings trained on a small in-domain corpus? Where do differences e.g. frequent vs rare words appear between these models?

All reported results were produced on a single Google Colab Tesla T4 GPU. To reproduce them on Colab, copy this repository into Google Drive, set Runtime type to T4 GPU and run the commands below in order. Run all training and evaluation in one session. 

## Running on Google Colab

1. Copy this repository into Google Drive. The notebooks expect it at `MyDrive/DSA4213/dsa4213-individual-assignment`; if you put it elsewhere, change the `PROJECT` path in the notebook's Drive-mount cell.
2. Download the data into `data/raw/` and `data/glove/` (see Setup).
3. Open the notebooks in order and set *Runtime → Change runtime type → T4 GPU*:

| Notebook | Runs | Pre-requisites |
|---|---|---|
| `notebooks/dsa4213_part_1.ipynb` | data preparation, trigram, RNN/LSTM/Transformer training, evaluation, generation, checks, Part I analysis |
| `notebooks/dsa4213_part_2.ipynb` | Skip-gram and GloVe embedding tables, both Part II LSTMs, evaluation, generation, Part II analysis | Part I outputs (the Random LSTM is reused) |
| `notebooks/dsa4213_part_3.ipynb` | feature extraction from the frozen LMs, classifiers, Part III analysis | Part I and Part II models |


## Repository structure

```
dsa4213-individual-assignment/
├── configs/                      # one YAML file per step (all hyperparameters and paths)
│   ├── data.yaml                 # tokenisation, vocabulary, splits, block length
│   ├── ngram.yaml                # interpolated trigram (λ grid, add-k)
│   ├── part1_rnn.yaml            # Part I RNN
│   ├── part1_lstm.yaml           # Part I LSTM (also the Part II "Random" setting)
│   ├── part1_transformer.yaml    # Part I causal Transformer
│   ├── generate.yaml             # Part I matched-decoding generation
│   ├── part2_embeddings.yaml     # Skip-gram training and GloVe loading
│   ├── part2_lstm_skipgram.yaml  # Part II LSTM with frozen Skip-gram table
│   ├── part2_lstm_glove.yaml     # Part II LSTM with frozen GloVe table
│   ├── part2_generate.yaml       # Part II matched-decoding generation
│   └── part3.yaml                # Part III representations, pooling, classifier
├── data/                         # large files (*.gz, *.kv, *.npy) stored with Git LFS
│   ├── raw/                      # Medical Abstracts CSVs
│   ├── glove/                    # GloVe 6B 200d
│   ├── processed/                # splits, vocabulary, token blocks (generated)
│   └── embeddings/               # Skip-gram and GloVe embedding matrices (generated)
├── notebooks/                    # Google Colab notebooks used to produce the reported results
│   ├── dsa4213_part_1.ipynb      # setup and Part I
│   ├── dsa4213_part_2.ipynb      # Part II
│   └── dsa4213_part_3.ipynb      # Part III
├── results/
│   ├── part1_results/            # trigram, RNN/LSTM/Transformer logs, evaluation, generation, checks, figures
│   ├── part2_results/            # embedding report, Skip-gram/GloVe LSTM logs, evaluation, generation, figures
│   └── part3_results/            # classification results, per-class F1, error analysis, figures
├── src/
│   ├── __init__.py
│   ├── utils.py                  # seeding, config loading, hardware logging
│   ├── data.py                   # tokeniser, vocabulary, 128-token blocks, word-type buckets
│   ├── data_preparation.py       # builds leakage-free splits and processed data
│   ├── ngram.py                  # interpolated trigram: fitting, λ tuning, scoring
│   ├── embeddings.py             # GloVe loading and embedding-matrix construction
│   ├── build_embeddings.py       # trains Skip-gram, builds Part II embedding tables
│   ├── models.py                 # RNN, LSTM and Transformer LMs
│   ├── train.py                  # training loop (AdamW, warm-up + cosine, clipping, early stopping)
│   ├── evaluate.py               # test CE/perplexity, parameter counts, throughput
│   ├── generate.py               # matched-decoding text generation and metrics
│   ├── checks.py                 # causality, padding, alignment and leakage checks
│   ├── analyze_results.py        # Part I analysis: bootstrap CIs, word-type and context breakdowns
│   ├── analyze_part2.py          # Part II analysis: convergence, coverage, nearest neighbours
│   ├── downstream.py             # Part III feature extraction and classifiers
│   └── analyze_part3.py          # Part III CIs, LM-vs-downstream ranking, error analysis
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

The scripts write each run to `results/<run_name>/` (for example `results/part1_lstm/`). For submission, the outputs were grouped into `results/part1_results/`, `results/part2_results/` and `results/part3_results/`. Re-running the commands writes to the per-run folders again.

## Data

| Resource | Source | License / citation |
|---|---|---|
| Medical Abstracts TC corpus | <https://github.com/sebischair/Medical-Abstracts-TC-Corpus> (same data as `TimSchopf/medical_abstracts` on Hugging Face) | CC BY-SA 3.0; Schopf, Braun & Matthes (2023), NLPIR '22 |
| GloVe 6B, 200d (Wikipedia 2014 + Gigaword 5) | gensim-data release `glove-wiki-gigaword-200` (identical vectors to Stanford `glove.6B.200d.txt`) | Pennington, Socher & Manning (2014) |

All data files, including the raw corpus, GloVe and the generated embedding matrices, are committed to this repository. Large files (`*.gz`, `*.kv`, `*.npy`) are stored with [Git LFS](https://git-lfs.com/), as set in `.gitattributes`. Install Git LFS before cloning, otherwise these files download as small text pointers instead of the real data:

```bash
git lfs install
git clone https://github.com/pyoons/dsa4213-individual-assignment.git
cd dsa4213-individual-assignment
git lfs pull      # only needed if the repository was cloned before Git LFS was installed
```

Both datasets are redistributed under their original terms and must be cited as above. To download them from the original sources instead:

```bash
mkdir -p data/raw data/glove
git clone --depth 1 https://github.com/sebischair/Medical-Abstracts-TC-Corpus.git /tmp/matc
cp /tmp/matc/medical_tc_*.csv data/raw/
curl -L -o data/glove/glove-wiki-gigaword-200.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-200/glove-wiki-gigaword-200.gz
```

### Install and prepare the data

```bash
pip install -r requirements.txt
python -m src.data_preparation --config configs/data.yaml --coverage-check   # -> results/coverage_check.json
python -m src.data_preparation --config configs/data.yaml                    # -> data/processed/
```

## Part I: Language model training and comparison

Four next-token LMs are trained on the same data and scored on the same 585,134 test targets (padding excluded).

### Interpolated trigram

The model is P(w|u,v) = λ₃P_ML(w|u,v) + λ₂P_ML(w|v) + λ₁P_add-k(w), with k = 0.01.
- When a history was never seen in training, its weight is redistributed to the remaining components, so the result stays a normalised distribution.
- History is taken from inside the current 128-token block, matching the context available to the neural models.
- The λ values are tuned on validation with a 0.05 grid, giving λ₁, λ₂, λ₃ = 0.20, 0.45, 0.35.

### Neural models

| | RNN | LSTM | Transformer |
|---|---|---|---|
| Input embedding | 17,645 × 200, random init, trainable | same | same, plus learned positions (128 × 200) |
| Core | 2 layers, tanh, hidden 256 | 2 layers, hidden 256 | 2 pre-LN layers, 4 heads, FFN 800, GELU, **causal mask** |
| Output | Linear 256 → V | Linear 256 → V | LayerNorm + Linear 200 → V |
| Core (non-embedding) params | 249k | 995k | 966k |
| Total params | 8.31M | 9.06M | 8.07M |

- **Embedding dim 200** matches GloVe 6B 200d, so that Part I LSTM can be reused as the Part II "Random" setting.
- **No weight tying**, so freezing the input table in Part II leaves the output layer trainable.
- **Context**: every model sees at most one 128-token block. Recurrent state starts at zero for each block, and Transformer attention stays inside the block.
- **Training procedure** (`configs/part1_*.yaml`):
  - AdamW, lr 1e-3, weight decay 0.01
  - 200 warm-up steps, then cosine decay to 10% of the learning rate
  - gradient clipping at 1.0, batch size 64 blocks, dropout 0.3
  - at most 40 epochs, early stopping with patience 5 on validation cross-entropy (CE). Run 1 used 20 epochs; every model was still improving at epoch 20, so the budget was doubled
  - the best-validation checkpoint is the one evaluated
- **Loss**: `CrossEntropyLoss(ignore_index=<pad>)`, averaged over real target tokens.

### Commands

On Colab, `notebooks/dsa4213_part_1.ipynb` runs these steps (and the data preparation in Setup). To run them from a terminal instead:

```bash
python -m src.ngram --config configs/ngram.yaml
python -m src.train --config configs/part1_rnn.yaml
python -m src.train --config configs/part1_lstm.yaml
python -m src.train --config configs/part1_transformer.yaml
python -m src.evaluate --runs part1_rnn part1_lstm part1_transformer
python -m src.generate --config configs/generate.yaml
python -m src.checks
python -m src.analyze_results
```

`--smoke` on `src.train` runs 3 batches × 2 epochs into `results/smoke/`. `--max-blocks N` on `src.evaluate` scores only N blocks. Both are for checking that the code runs; never use them for reported results.

### What each step measures

| Requirement | Where |
|---|---|
| Prediction quality | `evaluate.py`: val/test CE and perplexity on the common test targets. `analyze_results.py`: 95% bootstrap CIs over blocks, plus paired CIs for model differences |
| Cost | `train_log.json`: training time and per-epoch times. `eval.json`: parameter counts, checkpoint size and inference throughput (all test targets, batch 64, 1 warm-up + median of 3). `generation/metrics.json`: generation tokens/s. Hardware is logged in every file |
| Learning behaviour | Per-epoch train/val CE, gradient-norm mean/max and non-finite steps |
| Generated text | `generate.py`: 5 prompts (the first 8 tokens of one single-label test abstract per class). Matched decoding for all four models: T = 1.0, top-k 40, 50 new tokens, `<pad>`/`<eos>` banned, the same random stream per (prompt, sample), 5 samples per prompt. Metrics: distinct-1/2, repeated-trigram rate, `<unk>` and `<num>` rate |
| Where models differ | `analyze_results.py`: test CE by frequency bucket, by general vs. domain word, and by position inside the block |
| Correctness | `checks.py`: causality, with a negative control (the Transformer without its mask fails, as it should); padding exclusion; block alignment; no abstract in two splits |

## Part II: Embedding variants (LSTM)

The same LSTM as Part I, trained with the same seed and procedure; **only the input-embedding table changes**.

| Setting | Input table | Trainable | Run |
|---|---|---|---|
| Random | random init | yes | `results/part1_lstm` (the Part I run, reused) |
| Skip-gram | trained on the **train split** (gensim, sg=1, window 5, 5 negatives, 10 epochs) | frozen | `results/part2_lstm_skipgram` |
| GloVe | GloVe 6B 200d | frozen | `results/part2_lstm_glove` |

- **Same starting weights.** `models.make_embedding` always creates the random table first, so the random-number stream is used identically in every setting. With seed 4213, every non-embedding weight starts bit-identical to the Part I LSTM.
- **Missing words** (one rule for both sources): a vector drawn from N(0, σ²), where σ is the standard deviation of that source's covered vectors, with a fixed seed. `<pad>` is all zeros. Skip-gram covers every word except `<eos>`. GloVe covers 14,022 of 17,644 types (75.6% of rare types).
- **Frozen means the whole table.** `requires_grad=False`, and `train.py` asserts the table's checksum is unchanged after training.
- **Cost.** Skip-gram training time is logged separately (`embeddings_report.json`; ~3 min on CPU) from LM training time.

### Commands

On Colab, `notebooks/dsa4213_part_2.ipynb` runs these steps. To run them from a terminal instead:

```bash
python -m src.build_embeddings --config configs/part2_embeddings.yaml
python -m src.train --config configs/part2_lstm_skipgram.yaml
python -m src.train --config configs/part2_lstm_glove.yaml
python -m src.evaluate --runs part2_lstm_skipgram part2_lstm_glove
python -m src.generate --config configs/part2_generate.yaml
python -m src.analyze_part2
```

### What the analysis reports

- test/validation perplexity with bootstrap CIs, plus paired CIs between settings;
- trainable parameters, LM and embedding training cost;
- convergence (validation CE per epoch, epochs to reach a common level) and stability (gradient norms, clipping, train–validation gap);
- test CE by word type, including GloVe-covered vs. GloVe-missing words, and each word type's contribution to the difference from the Random setting;
- nearest neighbours of medical words in each table;
- matched-decoding generation metrics.

A fine-tuned GloVe setting (GloVe initialisation, trainable table) was not run. It is listed as a limitation in the report.

## Part III: Downstream disease-category classification

The LMs from Parts I and II are **frozen**: no language model is trained or fine-tuned here.

- **Task:** abstracts with exactly one condition label, 5 classes. The splits are the LMs' own (train 5,946 / val 661 / test 1,691), so no test abstract was seen by any LM, embedding or classifier.
- **Representations:** each LM reads an abstract in 128-token chunks, preceded by `<eos>` as in training. The top-layer hidden states become one vector by
  - **mean pooling** (the average over every real token state, excluding `<eos>` and padding), or
  - **last pooling** (the state at the abstract's final token).

  Compared: LSTM-Random, LSTM-Skip-gram, LSTM-GloVe and Transformer (mean pooling), plus LSTM-Random with last pooling. TF-IDF is a no-LM reference.
- **Classifier (identical for all):** `StandardScaler` + logistic regression. C is chosen from {0.01, 0.1, 1, 10} on validation macro-F1, then the model is evaluated once on test.

### Commands

On Colab, `notebooks/dsa4213_part_3.ipynb` runs these steps. To run them from a terminal instead:

```bash
python -m src.downstream --config configs/part3.yaml
python -m src.analyze_part3
```

### What the analysis reports

- macro-F1 with 95% bootstrap CIs, plus paired CIs against the best LM representation;
- accuracy and per-class F1;
- LM perplexity rank vs. macro-F1 rank;
- error analysis: token statistics of wrong vs. correct predictions, most frequent confusions, disagreements between representations, and the most confident mistakes.

## Attribution

- The README was written with assistance from Claude (Anthropic), then reviewed by the author.
