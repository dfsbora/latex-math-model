# LATEX MATHEMATICAL LANGUAGE MODEL

## Overview
This project aims to train an RNN language model on a mathematical book written in Latex. The goal of the model is to generate text that follows the Latex syntax and resembles mathematical content. By training the model on such data, we aim to explore the RNN's ability to capture short-term dependencies and reproduce the syntactic structures of Latex. 

## Dataset
The dataset is obtained from the open-source textbook on algebraic stacks ["The stacks"](https://github.com/stacks/stacks-project)

## Model architecture
The primary model architecture used in this project is a recurrent neural network (RNN).

## Usage

### Get data

python data/preprocess_data.py [--add_miscellany True]

To include miscellaneous chapters, set --add_miscellany to True.

### Train

python train.py [--data_path data/algebra.tex]

For a quicker train test, use algebra.tex file. Default uses the cleaned data.tex file

### Prompt
python prompt.py --model_path model_YYYYMMDD-HHMMSS.pth --vocab_path vocab_mappings_YYYYMMDD-HHMMSS.pkl --initial_str "Given a matrix A," --predict_len 200 --temperature 0.8

### Installation

### Training

### Testing

## Acknowledgments

The initial idea for this project was inspired by Andrej Karpathy's blog post ["The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness)
