# Description
This is the official code for paper [Text Representation Distillation via Information Bottleneck Principle](https://arxiv.org/abs/2311.05472).


This code is based on the [SimCSE code](https://github.com/princeton-nlp/SimCSE)

# Requirements
```
transformers==4.2.1
torch==1.7.1
scipy
datasets
pandas
scikit-learn
prettytable
gradio
setuptools
```
# Training

Before training, you need to prepare the training data for both the distillation stage and the fine-tuning stage. 

### Data Format

Toy data for each stage can be found in the data folder:

- unsup_toy_sents.txt: Training sentences for the distillation stage. Each line represents a sentence.

- toy_embs.pt: Embedding of the teacher model. This is a N * D matrix, where N is the number of training sentences and D is the dimension of the teacher model.

- sup_toy_data.csv: Training data for the fine-tuning stage. Each line contains three sentences: the anchor text, the positive sentence, and the negative sentence. The sentences are separated by commas.

## distillation stage
```
bash run_unsup.sh
```

## fine-tune stage
```
bash run_sup.sh
```
# evaluation
```
bash eval.sh
``` 

## Evaluation Results
### STS

| Model                    | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R         | Avg   | Params | Dimension |
|---------------------------|-------|-------|-------|-------|-------|-------|---------------|-------|--------|-----------|
| SimCSE-RoBERTa-base   | 76.53 | 85.21 | 80.95 | 86.03 | 82.57 | 85.83 | 80.50          | 82.52 | 110M   | 768       |
| SimCSE-RoBERTalarge   | 77.46 | 87.27 | 82.36 | 86.66 | 83.93 | 86.70 | 81.95          | 83.76 | 330M   | 1024      |
| SimCSE-MiniLM            | 70.34 | 78.59 | 75.08 | 81.10 | 77.74 | 79.39 | 77.85          | 77.16 | 23M    | 384       |
| MiniLM-MSE               | 73.75 | 81.42 | 77.72 | 83.58 | 78.99 | 81.19 | 78.48          | 79.30 | 23M    | 384       |
| MiniLM-HPD               | 76.03 | 84.71 | 80.45 | 85.53 | 82.07 | 85.33 | 80.01          | 82.05 | 23M    | 128       |
| MiniLM-CRD               | 74.79 | 84.19 | 78.98 | 84.70 | 80.65 | 82.71 | 79.91          | 81.30 | 23M    | 384       |
| MiniLM-IBKD     | **76.77** | **86.13** | **81.03** | **85.66** | **82.81** | **86.14** | **81.25** | **82.69** | 23M    | 384       |

### MSMARCO


| Model               | MRR@10 | Recall@1000 | Dimension | Params |
|---------------------|--------|-------------|-----------|--------|
| CoCondenser | 38.21  | 98.40       | 768       | 110M   |
| MiniLM-sup          | 30.51  | 94.32       | 384       | 23M    |
| MiniLM-MSE          | 28.12  | 93.01       | 384       | 23M    |
| MiniLM-CRD          | 28.79  | 93.12       | 384       | 23M    |
| MiniLM-HPD    | 36.53  | 96.70       | 128       | 23M    |
| MiniLM-IBKD  | **37.49**  | **97.81**       | 384       | 23M    |

## Model Checkpoints
We have published our model checkpont on modelscope:

| Model | url |
|:------:|:------:|
| MiniLM-STS | https://modelscope.cn/models/damo/nlp_minilm_ibkd_sentence-embedding_english-sts/summary |
| MiniLM-MSMARCO | https://modelscope.cn/models/damo/nlp_minilm_ibkd_sentence-embedding_english-msmarco/summary |


