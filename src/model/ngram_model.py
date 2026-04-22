'''This module defines the NGramModel class, which is responsible for building, storing, and exposing n-gram probability tables and backoff lookup across all orders from 1 up to NGRAM_ORDER'''
import os
import json
import copy
from dotenv import load_dotenv

class NGramModel:
    def __init__(self,token_file=None,vocab_file=None,model_file=None):
        
        self.model = {}
        self.vocab = {}
        if token_file and vocab_file and model_file:
            self.init_model(token_file,vocab_file,model_file)
        # self.build_vocab(token_file,vocab_file)
        # self.build_counts_and_probabilities(token_file,model_file)

    def init_model(self, token_file,vocab_file,model_file):
        self.build_vocab(token_file,vocab_file)
        self.build_counts_and_probabilities(token_file,model_file)

    def build_vocab(self, token_file,vocab_file):
        # Implement vocabulary building logic
        self.vocab = {}
        #check if the token file exists        
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Token file '{token_file}' not found. Please Run Data Prep")
        #read the token file and build the vocab
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                for token in line.split():
                    if token not in self.vocab:
                        self.vocab[token] = 1  # Start counting from 1
                    else:
                        self.vocab[token] += 1
        # check if vocab_file directory exists, if not create it
        # sort the vocab by frequency and then alphabetically
        # self.vocab = dict(sorted(self.vocab.items(), key=lambda item: (-item[1], item[0])))
        vocab_dir = os.path.dirname(vocab_file)
        if vocab_dir and not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        # check if vocab_file already exists, if so, overwrite it
        if os.path.exists(vocab_file):
            print(f"Warning: Vocab file '{vocab_file}' already exists and will be overwritten.")
        # write the vocab.json file
        self.vocab = [token for token, count in self.vocab.items() if count>int(os.getenv("UNK_THRESHOLD"))]
        self.vocab.append("<UNK>")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f)
      

    def build_counts_and_probabilities(self,token_file,model_file):
        ngram_n=int(os.getenv("NGRAM_ORDER", 3))
        
        self.model["count"]={}
        self.model["prob"]={}

        for n in range(1, ngram_n+1):
            self.model["count"][str(n)] = {}
            self.model["prob"][str(n)] = {}
            
            with open(token_file, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = [token if token in self.vocab else "<UNK>" for token in line.split()]
                    for i in range(len(tokens) - n + 1):
                        if n == 1:
                            ngram = tokens[i]
                            if ngram not in self.model["count"][str(n)]:
                                self.model["count"][str(n)][ngram] = 1
                            else:
                                self.model["count"][str(n)][ngram] += 1
                            continue
                        ngram = ' '.join(tokens[i:i+n-1])
                        if ngram not in self.model["count"][str(n)]:
                            self.model["count"][str(n)][ngram] = {}
                        if tokens[i+n-1] not in self.model["count"][str(n)][ngram]:
                            self.model["count"][str(n)][ngram][tokens[i+n-1]] = 1
                        else:
                            self.model["count"][str(n)][ngram][tokens[i+n-1]] += 1
            # sort
            # self.model["count"][str(n)] = dict(sorted(self.model["count"][str(n)].items(), key=lambda item: (-item[1], item[0])))
            
            self.model["prob"][str(n)] = copy.copy(self.model["count"][str(n)])
                
            # Convert counts to probabilities
            # calculate probablity for each n-gram by dividing its count by the total count of all n-grams of previous order. for 1-gram use the total word count for probability
            for ngram in self.model["prob"][str(n)]:
                if n==1:
                    total_count = sum(self.model["count"][str(n)].values())
                    self.model["prob"][str(n)][ngram] = self.model["count"][str(n)][ngram]/total_count
                else:
                    # ngram_arr = ngram.split(' ')
                    total_count = sum(self.model["count"][str(n)][ngram].values())
                    for next_word in self.model["prob"][str(n)][ngram]:
                        self.model["prob"][str(n)][ngram][next_word] = self.model["count"][str(n)][ngram][next_word]/total_count
        model_dir = os.path.dirname(model_file)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # check if model_file already exists, if so, overwrite it
        if os.path.exists(model_file):
            print(f"Warning: model file '{model_file}' already exists and will be overwritten.")
        # write the model.json file
        
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(self.model["prob"], f)
        
    def lookup(self, context):
        ngram_n=int(os.getenv("NGRAM_ORDER", 3))
        context_length = len(context.split()) +1
        if context_length < ngram_n:
            ngram_n = context_length
        for n in range(ngram_n, 0, -1):
            if context in self.model["prob"][str(n)]:
                return self.model["prob"][str(n)][context]
            context = ' '.join(context.split(' ')[1:])
        return {}
        
    def load(self, model_path,vocab_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please Run Model Training")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file '{vocab_path}' not found. Please Run Model Training")
        self.model["count"]={}
        self.model["prob"]={}
        with open(model_path, 'r', encoding='utf-8') as f:
            model=json.load(f)
            self.model["prob"] = model
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
if __name__ == "__main__":
    load_dotenv(dotenv_path="config/.env")
    print(os.getenv("TRAIN_TOKENS"))
    print(os.getenv("VOCAB"))
    print(os.getenv("MODELS"))
    ngram_model = NGramModel(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"), os.getenv("MODELS"))
    ngram_model.build_vocab(os.getenv("TRAIN_TOKENS"), os.getenv("VOCAB"))
    ngram_model.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"),os.getenv("MODELS"))
    predicted = ngram_model.lookup("the adventure <UNK>")
    print(predicted)
    print("-"*20)
    predicted = dict(sorted(predicted.items(), key=lambda item: item[1], reverse=True))
    print(predicted)
    print("-"*20)
    ranked = {k: v for k, v in sorted(predicted.items(), key=lambda item: item[1], reverse=True)}
    print(ranked)
    print("-"*20)
    ranked_topk = dict(list(ranked.items())[:int(os.getenv("TOP_K"))])
    print(ranked_topk)
    print("-"*20)