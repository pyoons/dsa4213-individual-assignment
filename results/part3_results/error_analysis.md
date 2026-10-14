## Errors vs. correct predictions: token statistics of the test abstracts

| Representation     |   Error rate |   length (wrong) |   rare share (wrong) |   <unk> share (wrong) |   GloVe-missing share (wrong) |   length (right) |   rare share (right) |   <unk> share (right) |   GloVe-missing share (right) |
|:-------------------|-------------:|-----------------:|---------------------:|----------------------:|------------------------------:|-----------------:|---------------------:|----------------------:|------------------------------:|
| LSTM-Random        |        0.439 |          205.884 |                0.108 |                 0.027 |                         0.012 |          219.122 |                0.097 |                 0.024 |                         0.013 |
| LSTM-Skip-gram     |        0.328 |          207.427 |                0.108 |                 0.029 |                         0.012 |          216.189 |                0.099 |                 0.023 |                         0.012 |
| LSTM-GloVe         |        0.351 |          204.532 |                0.110 |                 0.026 |                         0.012 |          218.068 |                0.098 |                 0.024 |                         0.012 |
| Transformer        |        0.306 |          203.521 |                0.109 |                 0.027 |                         0.011 |          217.638 |                0.099 |                 0.024 |                         0.012 |
| LSTM-Random (last) |        0.570 |          207.245 |                0.106 |                 0.027 |                         0.011 |          221.360 |                0.097 |                 0.022 |                         0.014 |
| TF-IDF (no LM)     |        0.212 |          197.296 |                0.109 |                 0.029 |                         0.012 |          217.615 |                0.100 |                 0.024 |                         0.012 |

## Most frequent confusions (top 3 per representation)

| Representation     | True                            | Predicted                       |   Count |   Share of all errors |
|:-------------------|:--------------------------------|:--------------------------------|--------:|----------------------:|
| LSTM-Random        | nervous system diseases         | general pathological conditions |     102 |                 0.137 |
| LSTM-Random        | neoplasms                       | general pathological conditions |      97 |                 0.131 |
| LSTM-Random        | cardiovascular diseases         | general pathological conditions |      95 |                 0.128 |
| LSTM-Skip-gram     | nervous system diseases         | general pathological conditions |      77 |                 0.139 |
| LSTM-Skip-gram     | neoplasms                       | general pathological conditions |      65 |                 0.117 |
| LSTM-Skip-gram     | cardiovascular diseases         | general pathological conditions |      64 |                 0.115 |
| LSTM-GloVe         | nervous system diseases         | general pathological conditions |      69 |                 0.116 |
| LSTM-GloVe         | general pathological conditions | nervous system diseases         |      67 |                 0.113 |
| LSTM-GloVe         | general pathological conditions | neoplasms                       |      66 |                 0.111 |
| Transformer        | nervous system diseases         | general pathological conditions |      90 |                 0.174 |
| Transformer        | cardiovascular diseases         | general pathological conditions |      76 |                 0.147 |
| Transformer        | neoplasms                       | general pathological conditions |      61 |                 0.118 |
| LSTM-Random (last) | general pathological conditions | neoplasms                       |     149 |                 0.155 |
| LSTM-Random (last) | neoplasms                       | general pathological conditions |     116 |                 0.120 |
| LSTM-Random (last) | general pathological conditions | cardiovascular diseases         |     116 |                 0.120 |
| TF-IDF (no LM)     | nervous system diseases         | general pathological conditions |      60 |                 0.168 |
| TF-IDF (no LM)     | cardiovascular diseases         | general pathological conditions |      47 |                 0.131 |
| TF-IDF (no LM)     | neoplasms                       | general pathological conditions |      45 |                 0.126 |

## Disagreements between LM representations

| A              | B              |   A right, B wrong |   B right, A wrong |   both wrong |
|:---------------|:---------------|-------------------:|-------------------:|-------------:|
| LSTM-Random    | LSTM-Skip-gram |                143 |                330 |          412 |
| LSTM-Random    | Transformer    |                112 |                336 |          406 |
| LSTM-Skip-gram | Transformer    |                159 |                196 |          359 |
| LSTM-GloVe     | LSTM-Random    |                316 |                168 |          426 |
| LSTM-GloVe     | LSTM-Skip-gram |                168 |                207 |          387 |
| LSTM-GloVe     | Transformer    |                147 |                223 |          371 |

Abstracts misclassified by every mean-pooled LM representation: 254 of 1691 (15.0%).

## Most confident mistakes of Transformer

- **True:** general pathological conditions | **Transformer predicted:** neoplasms (p = 0.99) | all LMs wrong: yes
  *Metastatic tumors of unknown origin*
- **True:** general pathological conditions | **Transformer predicted:** neoplasms (p = 0.99) | all LMs wrong: yes
  *Comparison of DNA and karyotype aneuploidy in malignant lymphomas*
- **True:** general pathological conditions | **Transformer predicted:** neoplasms (p = 0.99) | all LMs wrong: yes
  *Central adenoid cystic carcinoma of the mandible*
- **True:** general pathological conditions | **Transformer predicted:** neoplasms (p = 0.99) | all LMs wrong: yes
  *Orchiectomy versus oestrogen in the treatment of advanced prostatic cancer*
- **True:** nervous system diseases | **Transformer predicted:** cardiovascular diseases (p = 0.99) | all LMs wrong: yes
  *Hemodynamic effects of nasal CPAP examined by Doppler echocardiography*
- **True:** general pathological conditions | **Transformer predicted:** neoplasms (p = 0.99) | all LMs wrong: no
  *High-risk metastatic gestational trophoblastic disease*