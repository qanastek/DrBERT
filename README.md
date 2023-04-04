<p align="center">
  <img src="https://github.com/qanastek/DrBERT/blob/main/assets/logo.png?raw=true" alt="drawing" width="250"/>
</p>

# DrBERT: A Robust Pre-trained Model in French for Biomedical and Clinical domains

In recent years, pre-trained language models (PLMs) achieve the best performance on a wide range of natural language processing (NLP) tasks. While the first models were trained on general domain data, specialized ones have emerged to more effectively treat specific domains. 
In this paper, we propose an original study of PLMs in the medical domain on French language. We compare, for the first time, the performance of PLMs trained on both public data from the web and private data from healthcare establishments. We also evaluate different learning strategies on a set of biomedical tasks. 
Finally, we release the first specialized PLMs for the biomedical field in French, called DrBERT, as well as the largest corpus of medical data under free license on which these models are trained.

# 1. DrBERT models

**DrBERT** is a French RoBERTa trained on a open source corpus of French medical crawled textual data called NACHOS. Models with different amount of data from differents public and private sources are trained using the CNRS (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/jean-zay/) French supercomputer. Only the weights of the models trained using exclusively open-sources data are publicly released to prevent any personnal information leak and to follow the european GDPR laws :

| Model name | Corpus | Number of layers | Attention Heads | Embedding Dimension | Sequence Length | Model URL |
| :------:       | :---: |  :---: | :---: | :---: | :---: | :---: |
| `DrBERT-7-GB-cased` | NACHOS 7 GB | 12  | 12  | 768  | 512 | [HuggingFace](https://huggingface.co/Dr-BERT/DrBERT-7GB) |
| `DrBERT-4-GB-cased` | NACHOS 4 GB | 12  | 12  | 768  | 512 | [HuggingFace](https://huggingface.co/Dr-BERT/DrBERT-4GB) |
| `DrBERT-4-GB-cased-CP-CamemBERT` | NACHOS 4 GB | 12   | 12  | 768   | 512 | [HuggingFace](https://huggingface.co/Dr-BERT/DrBERT-4GB-CP-CamemBERT) |
| `DrBERT-4-GB-cased-CP-PubMedBERT` | NACHOS 4 GB | 12   | 12  | 768   | 512 | [HuggingFace](https://huggingface.co/Dr-BERT/DrBERT-4GB-CP-PubMedBERT) |

# 2. Using DrBERT

You can use DrBERT with [Hugging Face's Transformers library](https://github.com/huggingface/transformers) as follow.

Loading the model and tokenizer :

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Dr-BERT/DrBERT-7GB")
model = AutoModel.from_pretrained("Dr-BERT/DrBERT-7GB")
```

Perform the mask filling task :

```python
from transformers import pipeline 

fill_mask  = pipeline("fill-mask", model="Dr-BERT/DrBERT-7GB", tokenizer="Dr-BERT/DrBERT-7GB")
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

## 3.3 Build your own tokenizer from scratch based on NACHOS

Note : This step is required only in the case of an from scratch pre-training, if you want to do a continued pre-training you just have to download the model and the tokenizer that correspond to the model you want to continue the training from. In this case, you simply have to go to the HuggingFace Hub, select a model (for example [RoBERTa-base](https://huggingface.co/roberta-base)). Finally, you have to download the entire model / tokenizer repository by clicking on the `Use In Transformers` button and get the Git link `git clone https://huggingface.co/roberta-base`.

Build the tokenizer from scratch on your data of the file `./corpus.txt` by using `./build_tokenizer.sh`.

## 3.4 Preprocessing and tokenization of the dataset

First, replace the field `tokenizer_path` of the shell script to match the path of your tokenizer directory downloaded before using HuggingFace Git or the one you have build.

Run `./preprocessing_dataset.sh` to generate the tokenized dataset by using the givent tokenizer.

## 3.5 Model training

First, change the number of GPUs `--ntasks=128` you are needing to match your computational capabilities in the shell script called `run_training.sh`. In our case, we used 128 V100 32 GB GPUs from 32 nodes of 4 GPUs (`--ntasks-per-node=4` and `--gres=gpu:4`) during 20 hours (`--time=20:00:00`).

If you are using Jean Zay, you also need to change the `-A` flag to match one of your `@gpu` profile capable of running the job. You also need to move **ALL** of your datasets, tokenizer, script and outputs on the `$SCRATCH` disk space to preserve others users of suffuring of IO issues.

### 3.5.1 Pre-training from scratch

Once the SLURM parameters updated, you have to change name of the model architecture in the flag `--model_type="camembert"` and to update the `--config_overrides=` according to the specifications of the architecture you are trying to train. In our case, RoBERTa had a `514` sequence length, a vocabulary of `32005` (32K tokens of the tokenizer and 5 of the model architecture) tokens, the identifier of the beginning-of-sentence token (BOS) and end-of-sentence token (EOS) are respectivly `5` and `6`. Change the 

Then, go to `./from_scratch/` directory.

Run `sbatch ./run_training.sh` to send the training job in the SLURM queue.

### 3.5.2 continue pre-training

Once the SLURM parameters updated, you have to change path of the model / tokenizer you want to start from `--model_name_or_path=` / `--tokenizer_name=` to the path of the model downloaded from HuggingFace's Git in the section 3.3.

Then, go to `./continued_pretraining/` directory.

Run `sbatch ./run_training.sh` to send the training job in the SLURM queue.

# 4. Fine-tuning on a downstream task

You just need to change the name of the model to `Dr-BERT/DrBERT-7GB` in any of the examples given by HuggingFace's team [here](https://huggingface.co/docs/transformers/tasks/sequence_classification).

# Citation BibTeX

```bibtex
@misc{labrak2023drbert,
      title={DrBERT: A Robust Pre-trained Model in French for Biomedical and Clinical domains}, 
      author={Yanis Labrak and Adrien Bazoge and Richard Dufour and Mickael Rouvier and Emmanuel Morin and Béatrice Daille and Pierre-Antoine Gourraud},
      year={2023},
      eprint={2304.00958},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


