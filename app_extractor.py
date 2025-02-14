
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import contractions
import spacy


class FeatureExtractor():
    def __init__(self, corpus, sw_list=None, ngram_range=(1,1), max_features=None, use_lemma=True,batch_size=50,use_word2vec=True):
        self.sw_list = sw_list
        self.ngram_range = ngram_range
        self.max_features = max_features  
        self.use_lemma = use_lemma
        self.use_word2vec=use_word2vec
        
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        if self.sw_list:
            self.sw_list=set(sw_list) 
        self.tokenized_corpus = self.preprocess_texts(corpus, batch_size=batch_size)
        if not self.use_word2vec: 
            preprocessed_corpus = [" ".join(tokens) for tokens in self.tokenized_corpus]
            self.vectorizer = CountVectorizer(stop_words=self.sw_list, ngram_range=self.ngram_range, max_features=self.max_features)
            self.X = self.vectorizer.fit_transform(preprocessed_corpus)
           
        
    def preprocess_texts(self, texts, batch_size):
        processed_texts = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            tokens = []
                        
            for token in doc:
               token_text= token.lemma_.lower() if self.use_lemma else token.norm_.lower()
              
               token_text = contractions.fix(token_text)
               token_text = re.sub(r"[^\w\s]", "", token_text)
               
               #Eliminar Stop Words
               if token_text not in {"not", "no","yes","never", "hardly", "barely", "only", "even", "just"} and (token.is_stop or token_text in self.sw_list):
                  continue
              
               #Eliminar espacios
               if token_text.strip() == '':
                 continue

               #Eliminar numeros
               if token_text.isnumeric():
                  continue
               #Eliminar URLs y correos electrónicos 
               if re.match(r"(https?://\S+|www\.\S+)", token_text) or re.match(r"\S+@\S+\.\S+", token_text):
                    continue 
                                            
               tokens.append(token_text)
            processed_texts.append(tokens)   
        return processed_texts  
    

class PreProcess():
     def __init__(self, use_lemma=True, use_stop_words=True):
           self.use_lemma=use_lemma
           self.use_stop_words=use_stop_words
           self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  
           if self.use_stop_words:
                self.stop_words = set(stopwords.words('english'))
                custom_stopwords = {"use", "be", "one","go","time","think","job","work"}
                self.stop_words.update(custom_stopwords) 
           else:
               self.stop_words=None    
           
     def preprocess_texts(self, texts,batch_size=500):
        processed_texts=""
        if isinstance(texts, str):
            texts = [texts]
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            tokens = []
                     
            for token in doc:
               token_text= token.lemma_.lower() if self.use_lemma else token.norm_.lower()
               
              
               token_text = contractions.fix(token_text)
               token_text = re.sub(r"[^\w\s]", "", token_text)
               
               #Eliminar Stop Words
               if token_text not in {"not", "no","yes","never", "hardly", "barely", "only", "even", "just"} and (token.is_stop or token_text in self.stop_words):
                  continue
              
               #Eliminar espacios
               if token_text.strip() == '':
                 continue

               #Eliminar numeros
               if token_text.isnumeric():
                  continue
               #Eliminar URLs y correos electrónicos 
               if re.match(r"(https?://\S+|www\.\S+)", token_text) or re.match(r"\S+@\S+\.\S+", token_text):
                    continue 
                                            
               tokens.append(token_text)
            cleaned_text = " ".join(tokens).strip()
            if cleaned_text:
               processed_texts=cleaned_text
            else:
               processed_texts="UNK"    
        return processed_texts      
             