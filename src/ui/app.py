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
    ''' The PredictorUI class is responsible for providing a user interface for the N-Gram Predictor application using Streamlit. It includes methods for running the UI, preparing data, training the model, and making predictions. The run method sets up the Streamlit interface, allowing users to input text and receive predictions for the next word. It also includes buttons for triggering data preparation and model training processes. The class handles user interactions and displays results or error messages as needed, providing a seamless experience for users to interact with the n-gram predictor.'''
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
        ''' The prepare_data method initializes the Normalizer class with the specified training raw directory and token file. This method is responsible for preparing the data by normalizing the raw text data and saving the tokenized output to a file. It serves as a crucial step in the data preparation process, ensuring that the input data is in the correct format for model training. The method can be triggered from the UI to allow users to easily prepare their data before training the model.'''
        normalizer = Normalizer(os.getenv("TRAIN_RAW_DIR"), os.getenv("TRAIN_TOKENS"))

    def train_model(self):
        ''' The train_model method initializes the NGramModel class with the specified token file, vocabulary file, and model file. It calls the init_model method to build the vocabulary and count/probability tables for the n-grams, effectively training the model. This method can be triggered from the UI to allow users to easily train their model after preparing the data. It ensures that the model is built and ready for making predictions based on the prepared data.'''
        ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODEL"))

    def predict_next(self, text):
        ''' The predict_next method takes the input text, initializes the Normalizer and NGramModel classes, loads the model and vocabulary, and uses the Predictor class to generate predictions for the next word. It normalizes the input text, maps out-of-vocabulary words to <UNK>, and retrieves the probability distribution for the next word based on the n-gram model. The method returns the top-k predicted next words, handling cases where predictions may be empty by backing off to lower n-gram orders until it reaches unigrams or finds predictions. This method is essential for providing the core functionality of generating predictions based on user input in the UI.'''
        normalizer = Normalizer()
        ngram_model = NGramModel()
        ngram_model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
        predictor = Predictor(ngram_model, normalizer)
        predictions = predictor.predict_next(text, int(os.getenv("TOP_K")))
        # st.write(f"Raw predictions (including <UNK>): {predictions}")
        #remove <unk> token from predictions
        # predictions = [pred for pred in predictions if pred != '<UNK>'] 
        # # st.write(f"Predictions after removing <UNK>: {predictions}")
        # #while predicctions is empty try again with lower ngram order until we get predictions or reach unigram
        # while not predictions and ngram_model.n > 1:
        #     ngram_model.n -= 1
        #     predictions = predictor.predict_next(text, int(os.getenv("TOP_K")))
        #     predictions = [pred for pred in predictions if pred != '<UNK>']
        return predictions
        

if __name__ == "__main__":
    
    ui = PredictorUI()
    load_dotenv(dotenv_path="config/.env")
    ui.run()    