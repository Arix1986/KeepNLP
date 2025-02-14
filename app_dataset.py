
import joblib
import numpy as np
from torch.utils.data import Dataset
import torch


class SentimentDataset(Dataset):
    def __init__(self,labels, type):
        super(SentimentDataset,self).__init__()
        self.labels = labels
        self.path_train='./embeddings/embeddings_bert_train.pkl'
        self.path_valid='./embeddings/embeddings_bert_val.pkl' 
        dict={'train':self.path_train, 'val':self.path_valid}
        embeddings_path=dict[type]
        self.embeddings = joblib.load(embeddings_path)  
        self.labels = labels
    def __len__(self):
        return len(self.embeddings)
    

    def __getitem__(self, index):
        text_embedding = torch.tensor(self.embeddings[index], dtype=torch.float32) 
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        input_embeddings = text_embedding.unsqueeze(0)
        return {"input_embeddings": input_embeddings, "labels": label}



