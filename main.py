#!/usr/bin/env python3

"""
Main entry point for the N-Gram Predictor.
"""

import argparse

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="N-Gram Text Predictor")
    parser.add_argument("--train", help="Train the model")
    parser.add_argument("--predict", help="Predict next word")
    args = parser.parse_args()

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