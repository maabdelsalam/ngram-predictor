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
    ''' The main function serves as the entry point for the N-Gram Predictor application. It uses argparse to allow users to specify which step of the process they want to execute: data preparation, model training, inference, or all steps. Based on the user's choice, it initializes the necessary components (Normalizer, NGramModel, Predictor) and executes the corresponding functions. The function also includes error handling for keyboard interrupts to allow for graceful exits during execution. This modular approach allows users to run specific parts of the pipeline without needing to execute the entire process each time.'''
    parser = argparse.ArgumentParser(description="N-Gram Text Predictor")
    #parser.add_argument("--train", help="Train the model")
    #parser.add_argument("--predict", help="Predict next word")
    parser.add_argument("--step", choices=["dataprep","model","inference","all"], default="all", help="Step to execute: dataprep, model, inference, or all")
    args = parser.parse_args()
    load_dotenv(dotenv_path="config/.env")

        
    try:
            
        if args.step in ["dataprep"]:
            print("\nStarting Data Preparation...")
            normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))
            print("\nData Prep Done...")
            
        if args.step in ["model"]:
            print("\nStarting Model Training...")
            ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODEL"))
            print("\nModel Training Done...")
            
        if args.step in ["inference"]:
            normalizer = Normalizer()
            ngram_model = NGramModel()
            ngram_model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
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
            print("\nStarting Data Preparation and Model Training...")
            normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))
            ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODEL"))
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
        ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODEL"))
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
    