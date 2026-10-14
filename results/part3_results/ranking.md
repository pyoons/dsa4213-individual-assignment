| Representation   |   LM test PPL |   Test macro-F1 |   PPL rank (1 = best) |   F1 rank (1 = best) |
|:-----------------|--------------:|----------------:|----------------------:|---------------------:|
| LSTM-Skip-gram   |         88.14 |          0.6358 |                     1 |                    2 |
| Transformer      |         90.79 |          0.6583 |                     2 |                    1 |
| LSTM-Random      |         92.55 |          0.4979 |                     3 |                    4 |
| LSTM-GloVe       |        105.8  |          0.5989 |                     4 |                    3 |

Spearman correlation between LM quality (lower perplexity) and macro-F1: 0.60 (+1 = the better LM always classifies better; n = 4 models, so descriptive only).