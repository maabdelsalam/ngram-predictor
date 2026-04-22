#!/usr/bin/env python3

"""
Main entry point for the N-Gram Predictor.
"""
import os
from dotenv import load_dotenv
import argparse

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="N-Gram Text Predictor")
    parser.add_argument("--train", help="Train the model")
    parser.add_argument("--predict", help="Predict next word")
    args = parser.parse_args()
    load_dotenv(dotenv_path="config/.env")

    Normalizer(os.getenv("TRAIN_RAW_DIR"),os.getenv("TRAIN_TOKENS"))
    ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))

    if args.train:
        # Train logic
        pass
    elif args.predict:
        # Predict logic
        pass
    else:
        print("Use --train or --predict")

if __name__ == "__main__":
    main()