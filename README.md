# N-Gram Predictor

This project implements an N-Gram model for text prediction.
## Project File Paths
TRAIN_RAW_DIR=data/raw/train/\
EVAL_RAW_DIR=data/raw/eval/\
TRAIN_TOKENS=data/processed/train_tokens.txt\
EVAL_TOKENS=data/processed/eval_tokens.txt\
MODEL=data/model/model.json\
VOCAB=data/model/vocab.json\
UNK_THRESHOLD=3\
TOP_K=3\
NGRAM_ORDER=4\
## Project Structure

- `config/`: Configuration files
- `data/`: Data files
- `src/`: Source code
- `tests/`: Test files
- `main.py`: Entry point
- `ui/app.py`: streamlit app

## Requirements

1. Install dependencies: `pip install -r requirements.txt`

## Usage

Run `python main.py [--step <all dataprep model inference>]` to start the application.\
you can choose which step to run\
--step all      : runs all steps\
--step dataprep : runs data prep and token file generation only\
--step model    : runs model generation only\
--step inference: runs prediction only