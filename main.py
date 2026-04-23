#!/usr/bin/env python3

"""
Main entry point for the N-Gram Predictor.
"""

import os
from dotenv import load_dotenv
import argparse
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="N-Gram Text Predictor")
    #parser.add_argument("--train", help="Train the model")
    #parser.add_argument("--predict", help="Predict next word")
    parser.add_argument("--step", choices=["dataprep","model","inference","all"], default="all", help="Step to execute: dataprep, model, inference, or all")
    args = parser.parse_args()
    load_dotenv(dotenv_path="config/.env")

        
    try:
            
        if args.step in ["dataprep"]:
            normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))
            print("\nData Prep Done...")
            
        if args.step in ["model"]:
            ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))
            print("\nModel Training Done...")
            
        if args.step in ["inference"]:
            normalizer = Normalizer()
            ngram_model = NGramModel()
            ngram_model.load(os.getenv("MODELS"), os.getenv("VOCAB"))
            predictor = Predictor(ngram_model, normalizer)
            while True:
                try:
                    user_input = input("Enter your input text (quit or ctrl-c to exit): ")
                    if user_input.lower() == "quit":
                        print("Exiting...")
                        break
                    predictions = predictor.predict_next(user_input, int(os.getenv("TOP_K")))
                    print(predictions)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break

        if args.step in ["all"]:
            normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))
            ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))
            predictor = Predictor(ngram_model, normalizer)
            print("\nData Prep and Model Training Done...")
            while True:
                try:
                    user_input = input("Enter your input text (quit or ctrl-c to exit): ")
                    if user_input.lower() == "quit":
                        print("Exiting...")
                        break
                    predictions = predictor.predict_next(user_input, int(os.getenv("TOP_K")))
                    print(predictions)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
            

    except KeyboardInterrupt:
        print("\nExiting...")

    ''' if args.step in ["dataprep", "all"]:
        normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))
    if args.step in ["model", "all"]:
        ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))
    if args.step in ["inference", "all"]:
        predictor = Predictor(ngram_model, normalizer)
        predictions = predictor.predict_next("the adventure of", int(os.getenv("TOP_K")))
        print(predictions)'''

    

if __name__ == "__main__":
    '''load_dotenv(dotenv_path="config/.env")
    print("Enter your input text (quit or ctrl-c to exit):")
    #while input is not "quit" or ctrl-c not pressed, keep accepting input and predicting next word:
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == "quit":
                print("Exiting...")
                break
            main()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            '''
    main()
    