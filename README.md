# Intergroup Bias in NFL comments

This repository contains all code, data and notebooks for the paper ["Do they mean ‘us’? Interpreting Referring Expressions in Intergroup Bias"](https://arxiv.org/abs/2406.17947).

## Data

All data is in the `data/` folder. We release our annotated data in two forms, in accordance with the Reddit Terms of Service:

- `gold_data.tsv` contains the expert annotated data that was used for fine-tuning and few-shot prompting models in the paper. This contains 1499 comments annotated for intergroup referring expressions (in-group, out-group, other).
- `ann_data.tsv` contains the crowd-sourced annotations on the same set of comments in the test set. 

We also release metadata on our larger raw dataset that we perform analysis on. 


## Code

Explanations (with and without win probability) were generated using GPT-4o with the script `explanations-gpt.py` and the prompt `explanations.txt` and `explanations-wp.txt`. `fewshot-gpt.py` prompts GPT-4o with different 


We finetuned [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using the [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) framework &mdash; follow the instructions on the repo to setup a virtual environment for model fine-tuning and development. `llama.yml` lists our finetuning configuration. `infer_llama.py` performs inference with quantization and LoRA (if necessary) and writes the model outputs, and predicted tagged sentences to the model directory.
