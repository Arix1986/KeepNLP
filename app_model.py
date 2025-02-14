import joblib
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline as ImbPipeline
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC







class MModel():
    def __init__(self, df_train, df_test, random_state=42):
        np.random.seed(random_state) 
        self.x_train = np.array(df_train['reviewTextProcess'].astype(str)) 
        self.y_train = np.array(df_train['sentiment'])
        self.x_test = np.array(df_test['reviewTextProcess'].astype(str))
        self.y_test = np.array(df_test['sentiment'])
        self.random_state = random_state
        self.best_model = None 
        self.class_weights = compute_class_weight(class_weight='balanced',  classes=np.array([0, 1]), y=self.y_train)
        self.class_weight_dict = {0: self.class_weights[0], 1: self.class_weights[1]}
        self.param_grid_TF = {
            'vect__max_df': [0.9],  
            'vect__min_df': [3],  
            'vect__max_features': [9000, 10000],  
            'vect__ngram_range': [(1,3)],
            'clf__C': [0.5,0.7,0.9]  
        }
        self.param_grid_CV = {
            'vect__max_df': [0.9],  
            'vect__min_df': [3],  
            'vect__max_features': [10000],  
            'vect__ngram_range': [(1,3)],
            'clf__C': [0.001, 0.1],
            'clf__kernel':['linear'] 
            
        }
       

        self.models = [
            {
                'name': 'TF-IDF',
                'params': self.param_grid_TF,
                'pipeline': ImbPipeline([  
                    ('vect', TfidfVectorizer()),
                    ('sampling', SMOTETomek(sampling_strategy=0.7, random_state=self.random_state)), 
                    ('clf', LogisticRegression(solver='liblinear', class_weight=self.class_weight_dict, random_state=self.random_state))
                ])
            },
            {
                'name': 'CountVectorizer',
                'params': self.param_grid_CV,
                'pipeline': ImbPipeline([
                    ('vect', CountVectorizer()),
                    ('scaler', MaxAbsScaler()),
                    ('sampling', SMOTETomek(sampling_strategy=0.7, random_state=self.random_state)),  
                    ('clf', SVC(probability=True,class_weight=self.class_weight_dict, random_state=self.random_state))
                ])
            }
        ]
    def train_(self):
        best_score = -np.inf
        self.best_params = None
       
        for model in self.models:
           
            b_search = HalvingGridSearchCV(
                model['pipeline'],
                param_grid=model['params'], 
                cv=3,
                factor=10,
                scoring ='roc_auc',
                n_jobs=1,
                verbose=3,
                random_state=self.random_state
            )

            print(f"\n Entrenando modelo: {model['name']}")       
            b_search.fit(self.x_train, self.y_train)
           
            score = b_search.best_score_
            if score > best_score:
                best_score = score
                self.best_model = b_search.best_estimator_
                self.best_params = b_search.best_params_
            
            print(f" Mejor ROC-AUC para {model['name']}: {best_score:.4f}")
            joblib.dump(self.best_model, f"best_model_{model['name']}_{best_score:.4f}.pkl")
            self.evaluate_model()
        if self.best_model:
            joblib.dump(self.best_model, f"best_model_{best_score:.4f}.pkl")
            print(f"\n Mejor ROC-AUC  guardado: {best_score} ")
            

    def evaluate_model(self):
        if not self.best_model:
            print("No hay un modelo entrenado. Ejecuta `train_()` primero.")
            return
        
        print("\n Evaluando Mejor Modelo")
        y_pred = self.best_model.predict(self.x_test)

        print("\n Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Negative', 'Positive']))

        print("\n Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        

class DeepModelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5):
        super(DeepModelGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
       
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, embeddings):
       
        gru_out, _ = self.gru(embeddings)
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)
        return logits
    
    
