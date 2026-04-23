''' This module defines the Normalizer class, which is responsible for loading, cleaning, tokenizing, and saving the text data for the N-Gram model.'''
import os
import re
import string
import unicodedata
from dotenv import load_dotenv
''' The Normalizer class is responsible for loading raw text data, cleaning it by removing Project Gutenberg headers/footers, converting to lowercase, removing punctuation and numbers, normalizing whitespace, tokenizing into sentences and words, and saving the cleaned tokens to a file. It provides methods for each of these steps and can be initialized with input and output paths to perform the entire normalization process in one step.'''
class Normalizer:
    def __init__(self, input_path=None, output_path=None):
        texts=""
        if input_path and output_path:
            self.init_norm(input_path, output_path)
        
    ''' The init_norm method orchestrates the entire normalization process by loading raw text from the specified input path, cleaning it, tokenizing it into sentences, and saving the normalized tokens to the specified output path. It calls the individual methods for each step in sequence.'''
    def init_norm(self, input_path, output_path):
        texts=self.load(input_path)
        strip_text = self.strip_gutenberg(texts)
        sentences = self.sentence_tokenize(strip_text)
        self.save(sentences,output_path)

    ''' The load method reads all text files from the specified folder path and concatenates their contents into a single string. It checks if the folder exists and is not empty before attempting to read the files. If the folder is missing or empty, it raises a FileNotFoundError with an appropriate message.'''
    def load(self,folder_path):
        # Load all text files from the specified folder
        texts = ""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Input folder '{folder_path}' not found. Please check the path and try again.")
        if not os.listdir(folder_path):
            raise FileNotFoundError(f"Input folder '{folder_path}' is empty. Please add text files and try again.")
        for name in os.listdir(folder_path):
            if name.endswith(".txt"):
                with open(os.path.join(folder_path, name), 'r', encoding='utf-8') as file:
                    texts += file.read()
        return texts    
    ''' The strip_gutenberg method removes Project Gutenberg headers and footers from the input text using regular expressions. It identifies the start and end markers for the Gutenberg content and extracts only the text between these markers. It also removes any remaining marker occurrences to ensure a clean output.''' 
    def strip_gutenberg(self,text):
        # Remove Project Gutenberg headers and footers
        start_pattern = r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK(?:\s*\d*)*\s*\*{3}"
        end_pattern = r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK(?:\s*\d*)*\s*\*{3}"

        # keep only the text after the first start marker
        start_match = re.search(start_pattern, text, flags=re.IGNORECASE)
        if start_match:
            text = text[start_match.end():]

        # keep only the text before the last end marker
        end_matches = list(re.finditer(end_pattern, text, flags=re.IGNORECASE))
        if end_matches:
            text = text[:end_matches[-1].start()]

        # remove any remaining marker occurrences
        text = re.sub(start_pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(end_pattern, "", text, flags=re.IGNORECASE)

        return text.strip()
    
    ''' The lowercase method converts the input text to lowercase using the built-in lower() function. This is a common normalization step to ensure that the model treats words with different cases as the same token.'''
    def lowercase(self,text):
        # Convert text to lowercase
        return text.lower()
    ''' The remove_punctuation method removes punctuation characters from the input text while ensuring that spaces are preserved where needed. It iterates through each character in the text and checks if it is a punctuation character using the unicodedata.category function. If a punctuation character is found, it checks if the next character is not a space and adds a space if necessary. Otherwise, it simply skips the punctuation character. This helps to maintain word boundaries while cleaning the text.'''
    def remove_punctuation(self,text):
        # Remove punctuation from the text, ensuring spaces where needed
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            if unicodedata.category(c).startswith('P'):
                # If next character exists and is not a space, replace with space
                if i + 1 < len(text) and text[i + 1] != ' ':
                    result.append(' ')
                # Otherwise, remove the punctuation (don't append)
            else:
                result.append(c)
            i += 1
        return ''.join(result)

    ''' The remove_numbers method removes numeric characters from the input text using the translate() method with a translation table that maps digits to None. This is another common normalization step to ensure that numbers do not interfere with the model's understanding of word patterns.'''
    def remove_numbers(self,text):
        # Remove numbers from the text
        return text.translate(str.maketrans('', '', string.digits))
    
    ''' The remove_whitespace method removes extra whitespace from the input text using a regular expression that replaces multiple whitespace characters with a single space and then strips leading and trailing whitespace. This helps to clean up the text and ensure consistent spacing between words.'''
    def remove_whitespace(self,text):
        # Remove extra whitespace from the text
        return re.sub(r'\s+', ' ', text).strip()
    

    def normalize(self,text):
        # Apply all normalization steps to the text
        text_norm=self.lowercase(text)
        text_norm=self.remove_punctuation(text_norm)
        text_norm=self.remove_numbers(text_norm)
        text_norm=self.remove_whitespace(text_norm)
        return text_norm
    
    def sentence_tokenize(self,text):
        # Split text into sentences
        pattern = r'(?<!\b(?:Mr|Mrs|Dr|Ms|Jr|Sr|Prof|St|etc))(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(r'(?<=[.!?])\s+', text)
        merged = []
        current = ""
        for s in sentences:
            current += s + " "
            if not re.search(r'\b(?:Mr|Mrs|Dr|Ms|Jr|Sr|Prof|St|etc)\.$', s):
                merged.append(current.strip())
                current = ""
        if current:
            merged.append(current.strip())
        return merged

    '''' The word_tokenize method splits a sentence into individual words using the split() method. This is a simple tokenization approach that assumes words are separated by spaces. More advanced tokenization techniques could be implemented if needed, but this basic method is sufficient for many applications.'''
    def word_tokenize(self, sentence):
        # Split sentences into words
        return sentence.split()
    
    ''' The save method takes a list of sentences, normalizes each sentence, and saves the normalized tokens to a specified output file. It ensures that the output directory exists before writing the file. Each normalized sentence is tokenized into words and saved as a space-separated string, with each sentence on a new line in the output file.'''
    def save(self, sentences,output_path):
        # Save the normalized sentences to a file
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        tokens_per_sentence = []
        for sentence in sentences:
            normalized = self.normalize(sentence)
            if normalized:
                words = normalized.split()  # tokenize into words
                tokens_per_sentence.append(" ".join(words))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(tokens_per_sentence))

if __name__ == "__main__":
    # Example usage
    load_dotenv(dotenv_path="config/.env")
    print(os.getenv("TRAIN_RAW_DIR_test"))
    norm = Normalizer(os.getenv("TRAIN_RAW_DIR_test"), os.getenv("TRAIN_TOKENS"))
    texts = norm.load(os.getenv("TRAIN_RAW_DIR_test"))
    print("-"*20)
    print(texts)
    strip_text = norm.strip_gutenberg(texts)
    print("-"*20)
    print(strip_text)
    sentences = norm.sentence_tokenize(strip_text)
    print("-"*20)
    print(sentences)
    norm_text = [norm.normalize(s) for s in sentences if norm.normalize(s)]
    print("-"*20)
    print(norm_text)
    word_tokens = [norm.word_tokenize(s) for s in norm_text]
    print("-"*20)
    print(word_tokens)  
    norm.save(sentences,os.getenv("TRAIN_TOKENS"))


    