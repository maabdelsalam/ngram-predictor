'''This module defines the Predictor class, which is responsible for accepting a pre-loaded NGramModel and Normalizer via the constructor, normalizing input text, and returning the top-k predicted next words sorted by probability.'''
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


from dotenv import load_dotenv
from model.ngram_model import NGramModel
from data_prep.normalizer import Normalizer

''' The Predictor class is designed to take a pre-loaded NGramModel and Normalizer, and provide functionality to predict the next word(s) based on input text. It includes methods for normalizing the input text, mapping out-of-vocabulary words to a special <UNK> token, and generating predictions based on the n-gram model's probability distribution. The predict_next method returns the top-k predicted next words sorted by probability, while also handling cases where predictions may be empty by backing off to lower n-gram orders.'''
class Predictor:
    def __init__(self, model,normalizer):
        self.model = model
        self.normalizer = normalizer
        self.n = int(os.getenv("NGRAM_ORDER", 3))

    ''' The normalize_input method takes the raw input text and applies normalization steps using the Normalizer class. It ensures that the input is processed in a way that is consistent with how the training data was normalized, which helps the model make accurate predictions. The method also handles context length by returning only the last N-1 tokens if the input exceeds the n-gram order, ensuring that the model receives the appropriate context for prediction.'''
    def normalize_input(self, text):
        # Implement input normalization logic using the Normalizer class
        text_norm=self.normalizer.normalize(text)
        context_length = len(text_norm.split())
        if context_length < self.n:
            return text_norm
        else:
            #return the last N-1 tokens of the normalized text as context for prediction
            return ' '.join(text_norm.split()[-(self.n-1):])

    ''' The map_oov method takes the normalized input context and maps any out-of-vocabulary (OOV) words to a special <UNK> token. This is done by checking each token in the context against the model's vocabulary and replacing any token that is not found with <UNK>. This allows the model to handle unseen words during prediction without breaking the input format.'''
    def map_oov(self, context):
        # Map out-of-vocabulary words to <UNK>
        tokens = context.split()
        mapped_tokens = [token if token in self.model.vocab else "<UNK>" for token in tokens]
        return ' '.join(mapped_tokens)
    
    ''' The predict_next method takes the raw input text and the number of top predictions to return (top_k). It normalizes the input text, maps OOV words to <UNK>, and then uses the n-gram model to get a probability distribution for the next word. The method sorts the predictions by probability and returns the top-k predicted next words. If no predictions are found, it backs off to lower n-gram orders until it reaches unigrams or finds predictions. This ensures that the method can still provide predictions even when the context is not fully covered by the higher-order n-grams.'''
    def predict_next(self, text, top_k):
        # Normalize the input text
        normalized_text = self.normalize_input(text)
        # Map OOV words to <UNK>
        mapped_context = self.map_oov(normalized_text)
        # Get the probability distribution for the next word
        prob_dist = self.model.lookup(mapped_context)
        # Sort the predictions by probability and return the top-k
        sorted_predictions = dict(sorted(prob_dist.items(), key=lambda item: item[1], reverse=True))
        if top_k > len(sorted_predictions):
            top_k = len(sorted_predictions)
        return list(sorted_predictions.keys())[:top_k]
    
if __name__ == "__main__":
    # Example usage
    load_dotenv(dotenv_path="config/.env")
    print(os.getenv("TRAIN_TOKENS"))
    print(os.getenv("VOCAB"))
    print(os.getenv("MODEL"))
    normalizer=Normalizer(os.getenv("TRAIN_RAW_DIR_test"), os.getenv("TRAIN_TOKENS"))
    ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODEL"))
    predictor = Predictor(ngram_model, normalizer)
    predictions = predictor.predict_next("it IS the adventure batee5", int(os.getenv("TOP_K")))
    print(predictions)