import os
import joblib
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import spacy
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler

spacy.require_gpu()

save_path_bert="./embeddings/embeddings_bert.pkl"
save_path_word2="./embeddings/embeddings_word.pkl"
word_embeddings = np.load("./embeddings/word_embeddings_3.npy", allow_pickle=True).item()
embedding_dim = next(iter(word_embeddings.values())).shape[0] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


pos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
pos_tags_list = list(nlp.pipe_labels['tagger'])
pos_encoder.fit([[pos] for pos in pos_tags_list])
pos_dim = len(pos_encoder.categories_[0])  


stop_words = set(stopwords.words('english'))
custom_stopwords = {"use", "be", "get", "one", "would", "make"}
stop_words.update(custom_stopwords)

    

def display_histogram(data, title):
    fig = plt.figure(figsize=(10,6))
    plt.hist(data, bins=20,color='blue',edgecolor='black', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.show()
    
def map_sentiment(raiting):
    if raiting in [1,2,3]:
        return 0
    elif raiting in [4,5]:
        return 1
    else:
        return None

def remplazar_review(row):
    review = row['reviewText']
    summary = row['summary']        
    if pd.isna(review) or (isinstance(review, str) and review.strip() == '') or review==None or review=='None':
       return summary
    
    return review

def process_text(doc,n):
   
    tokens = [
        token.lemma_.lower() for token in doc 
        if not token.is_punct and not token.is_space and token.lemma_.lower() not in stop_words 
    ]
    return list(ngrams(tokens, n))

def get_ngrams(corpus, n=2,most_common=15):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    all_ngrams = []
    for doc in nlp.pipe(corpus, batch_size=200): 
        all_ngrams.extend(process_text(doc,n))
    
    freq_dist = FreqDist(all_ngrams)  
    most_common_ngrams = freq_dist.most_common(most_common)  
    return most_common_ngrams

  



def tokenize_and_vectorize(texts, max_length=250):
    batch_vectors = []
    for text in texts:
        tokens = [token.text for token in nlp(text)]
        vectors = [word_embeddings.get(word, np.zeros(embedding_dim, dtype=np.float32)) for word in tokens]
        
        if len(vectors) > max_length:
            vectors = vectors[:max_length]
        else:
            padding = np.zeros((max_length - len(vectors), embedding_dim), dtype=np.float32)
            vectors = np.vstack((vectors, padding))  

        batch_vectors.append(vectors)

   
    batch_vectors = np.array(batch_vectors, dtype=np.float32)
    joblib.dump(batch_vectors, save_path_word2)  
    print(f"Embeddings guardados en {save_path_word2}")
  
    return torch.tensor(batch_vectors, dtype=torch.float32, device=device)

def tokenize_and_vectorize_transformers(texts, batch_size=64):
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,max_length=250)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs) 

      
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        all_vectors.extend(sentence_embeddings)
    joblib.dump(all_vectors, save_path_bert)
    print(f"Embeddings guardados en {save_path_bert}")
    return np.array(all_vectors)



    


def scale_and_save_embeddings(train_embeddings, val_embeddings, scaler_path="./models/scaler.pkl"):
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        scaler = StandardScaler()
     
    train_mean_embeddings = np.array([emb.mean(axis=0) for emb in train_embeddings])
    val_mean_embeddings = np.array([emb.mean(axis=0) for emb in val_embeddings])

    if not hasattr(scaler, "mean_"): 
        scaler.fit(train_mean_embeddings)
        joblib.dump(scaler, scaler_path) 
       

   
    train_scaled = scaler.transform(train_mean_embeddings)
    val_scaled = scaler.transform(val_mean_embeddings)

    return train_scaled, val_scaled

def tokenize_and_vectorize_transformers_(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.squeeze(0).numpy() 








    