# LATEX MATHEMATICAL LANGUAGE MODEL

## Overview
This project aims to train a language model on a mathematical book written in Latex. The goal of the model is to generate text that follows the Latex syntax and resembles mathematical content. By training the model on such data, we aim to explore the LSTM and Universal Transformer ability to capture dependencies and reproduce the syntactic structures of Latex. 

## Dataset
The dataset is obtained from the open-source textbook on algebraic stacks ["The stacks"](https://github.com/stacks/stacks-project)

A few processed datasets can be found in this repository for easiness:
- data.tex: Complete filtered dataset (main)
- data_eos.tex: data.tex with marks for the main blocks (used for the experiment with sequence lengths)
- data_masked.tex: data.tex with formulas and texts masked (used for fine-tuning the model)
- data_small.tex: small dataset (used for quick tests)

### [Optional] Create dataset
By default creates data.tex file.

To include block tags, use --add_eos.
```bash
python data/preprocess_data.py [--add_eos] [--output_file data_eos.tex]
```

### [Optional] Simplify dataset
Choose method of simplification.
```bash
python data/simplify_dataset.py  [--mask_formulas] [--mask_text] [--merge_math_commands]
```


## Model architecture
The baseline model architecture used in this project is a multilayer LSTM. 
The other architecture used is [Universal Transformer](https://arxiv.org/abs/1807.038199)

## Usage

### Train

python train.py [--data_path data/algebra.tex]  [--disable_wandb]

For a quicker train test, use algebra.tex file. Default uses the cleaned data.tex file

wandb environment is disabled with --disable_wandb on command line

### Prompt
python prompt.py --model_path model_YYYYMMDD-HHMMSS.pth --vocab_path vocab_mappings_YYYYMMDD-HHMMSS.pkl --initial_str "Given a matrix A," --predict_len 200 --temperature 0.8

### Installation

### Training

### Testing

## Acknowledgments


The initial idea for this project was inspired by Andrej Karpathy's blog post ["The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
