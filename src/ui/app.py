'''UI component for the N-gram Predictor application.'''

import sys

from dotenv import load_dotenv
import streamlit as st

import os   
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from model.ngram_model import NGramModel
from data_prep.normalizer import Normalizer
from inference.predictor import Predictor

class PredictorUI:
    def __init__(self):
        pass

    def run(self):
        #streamlit UI code goes here
        st.title("N-Gram Text Predictor")
        st.write("Enter your input text and click the button to get predictions for the next word.")
        user_input = st.text_input("Input Text")
        if user_input:
            # if an error occus display message
            try:
                predictions = self.predict_next(user_input)
                st.write("Predicted Next Words:")
                st.write(str(predictions))
            except Exception as e:
                st.error(f"Error occurred while predicting next words. Please ensure the model is trained and loaded correctly.")
        if st.button("Data Preparation"):
            # Call the data preparation function and display results
            st.write("Starting data preparation...")
            self.prepare_data()
            st.write("Data preparation completed.")
        if st.button("Train Model"):
            # Call the model training function and display results
            st.write("Starting model training...")
            self.train_model()
            st.write("Model training completed.")
        #if st.button("Predict Next Word"):
            # Call the predictor function and display results
            
        
    def prepare_data(self):
        normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))

    def train_model(self):
        ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))

    def predict_next(self, text):
        normalizer = Normalizer()
        ngram_model = NGramModel()
        ngram_model.load(os.getenv("MODELS"), os.getenv("VOCAB"))
        predictor = Predictor(ngram_model, normalizer)
        predictions = predictor.predict_next(text, int(os.getenv("TOP_K")))
        # st.write(f"Raw predictions (including <UNK>): {predictions}")
        #remove <unk> token from predictions
        predictions = [pred for pred in predictions if pred != '<UNK>'] 
        # st.write(f"Predictions after removing <UNK>: {predictions}")
        #while predicctions is empty try again with lower ngram order until we get predictions or reach unigram
        while not predictions and ngram_model.n > 1:
            ngram_model.n -= 1
            predictions = predictor.predict_next(text, int(os.getenv("TOP_K")))
            predictions = [pred for pred in predictions if pred != '<UNK>']
        return predictions
        

if __name__ == "__main__":
    
    ui = PredictorUI()
    load_dotenv(dotenv_path="config/.env")
    ui.run()    