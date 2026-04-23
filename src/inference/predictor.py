'''This module defines the Predictor class, which is responsible for accepting a pre-loaded NGramModel and Normalizer via the constructor, normalizing input text, and returning the top-k predicted next words sorted by probability.'''
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


from dotenv import load_dotenv
from model.ngram_model import NGramModel
from data_prep.normalizer import Normalizer

class Predictor:
    def __init__(self, model,normalizer):
        self.model = model
        self.normalizer = normalizer
        self.n = int(os.getenv("NGRAM_ORDER", 3))

    def normalize_input(self, text):
        # Implement input normalization logic using the Normalizer class
        text_norm=self.normalizer.normalize(text)
        context_length = len(text_norm.split())
        if context_length < self.n:
            return text_norm
        else:
            #return the last N-1 tokens of the normalized text as context for prediction
            return ' '.join(text_norm.split()[-(self.n-1):])

    
    def map_oov(self, context):
        # Map out-of-vocabulary words to <UNK>
        tokens = context.split()
        mapped_tokens = [token if token in self.model.vocab else "<UNK>" for token in tokens]
        return ' '.join(mapped_tokens)
    
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
        #return dict(list(sorted_predictions.items())[:top_k])
    
if __name__ == "__main__":
    # Example usage
    load_dotenv(dotenv_path="config/.env")
    print(os.getenv("TRAIN_TOKENS"))
    print(os.getenv("VOCAB"))
    print(os.getenv("MODELS"))
    normalizer=Normalizer(os.getenv("TRAIN_RAW_DIR_test"), os.getenv("TRAIN_TOKENS"))
    ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))
    predictor = Predictor(ngram_model, normalizer)
    predictions = predictor.predict_next("it IS the adventure batee5", int(os.getenv("TOP_K")))
    print(predictions)