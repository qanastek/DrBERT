# DrBERT : Unsupervised Domain-Specific Language Model Pre-training on French Open Source Biomedical Literature and Clinical Cases

ddd

# 1. DrBERT models

**DrBERT** is a French RoBERTa trained on a open source corpus of French medical crawled textual data called NACHOS. Models with different amount of data from differents public and private sources are trained using the CNRS (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/jean-zay/) French supercomputer. Only the weights of the models trained using exclusively open-sources data are publicly released to prevent any personnal information leak and to follow the european GDPR laws :

| Model name | Corpus | Number of layers | Attention Heads | Embedding Dimension | Sequence Length |
| :------:       | :---: |  :---: | :---: | :---: | :---: |
| `DrBERT-7-GB-cased` | NACHOS 7 GB | 12  | 12  | 768  | 512 |
| `DrBERT-4-GB-cased` | NACHOS 4 GB | 12  | 12  | 768  | 512 |
| `DrBERT-4-GB-cased-CP` | NACHOS 4 GB | 12   | 12  | 768   | 512 |

# 2. Using DrBERT

You can use DrBERT with [Hugging Face's Transformers library](https://github.com/huggingface/transformers) as follow.

Loading the model and tokenizer :

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("qanastek/DrBERT-7GB")
model = AutoModel.from_pretrained("qanastek/DrBERT-7GB")
```

Perform the mask filling task :

```python
from transformers import pipeline 

fill_mask  = pipeline("fill-mask", model="qanastek/DrBERT-7GB", tokenizer="qanastek/DrBERT-7GB")
results = fill_mask("La patiente est atteinte d'une <mask>")
```

# 3. Pre-training DrBERT tokenizer and model from scratch by using HuggingFace Transformers Library

## 3.1 Install dependencies

```bash
accelerate @ git+https://github.com/huggingface/accelerate@66edfe103a0de9607f9b9fdcf6a8e2132486d99b
datasets==2.6.1
sentencepiece==0.1.97
protobuf==3.20.1
evaluate==0.2.2
tensorboard==2.11.0
torch >= 1.3
```

## 3.2 Download NACHOS Dataset text file

Download the full NACHOS dataset from [Zenodo]() and place it the the `from_scratch` or `continued_pretraining` directory.

## 3.3 Build your own tokenizer from scratch based on NACHOS or use an already existing one (continued pre-training only)

ddd

## 3.4 Preprocessing and tokenization of the dataset

ddd

## 3.5 Model pre-training from scratch or continue pre-training

First, change the number of GPUs `--ntasks=128` you are needing to match your computational capabilities in the shell script called `run_training.sh`. In our case, we used 128 V100 32 GB GPUs from 32 nodes of 4 GPUs (`--ntasks-per-node=4` and `--gres=gpu:4`) during 20 hours (`--time=20:00:00`).

PS: If you are using Jean Zay, you also need to change the `-A` flag to match one of your `@gpu` profile capable of running the job. You also need to move **ALL** of your datasets, tokenizer, script and outputs on the `$SCRATCH` disk space to preserve others users of suffuring of IO issues.




