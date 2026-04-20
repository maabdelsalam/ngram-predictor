# N-Gram Predictor

This project implements an N-Gram model for text prediction.
## Project File Paths
TRAIN_RAW_DIR=data/raw/train/
EVAL_RAW_DIR=data/raw/eval/
TRAIN_TOKENS=data/processed/train_tokens.txt
EVAL_TOKENS=data/processed/eval_tokens.txt
MODEL=data/model/model.json
VOCAB=data/model/vocab.json
UNK_THRESHOLD=3
TOP_K=3
NGRAM_ORDER=4
## Project Structure

- `config/`: Configuration files
- `data/`: Data files
- `src/`: Source code
- `tests/`: Test files
- `main.py`: Entry point

## Installation

1. Install dependencies: `pip install -r requirements.txt`

## Usage

Run `python main.py` to start the application.