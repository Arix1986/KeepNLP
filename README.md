


![Arquitectura del Proyecto](./assets/estructura.png)

## 🛠️ Descripción de los Módulos  

### 🔹 `app_features.py`  
Análisis exploratorio de datos (**EDA**) con limpieza y preprocesamiento. Se examinan los **n-gramas más frecuentes**, la distribución de reseñas, y se visualizan nubes de palabras de **sentimientos positivos y negativos**.  

### 🔹 `app_extractor.py`  
Contiene las clases `FeatureExtractor` y `PreProcess`, responsables del preprocesamiento textual. Se utilizan **spaCy y NLTK** para tokenización y eliminación de **stopwords**, **contractions** para expandir contracciones, y **expresiones regulares** para limpiar texto.  

### 🔹 `app_utils.py`  
Funciones auxiliares como **precomputación de embeddings** con **BERT-base-uncased**, generación de histogramas y gráficos para evaluar distribuciones de palabras y clases.  

### 🔹 `app_model.py`  
Implementación de **MModel**, que ejecuta un pipeline con **TF-IDF + Logistic Regression** y **CountVectorizer + SVC**, utilizando **SMOTETomek** para mitigar el desbalanceo. También se integra **ImbPipeline** para mejorar la representación de la clase minoritaria.  

### 🔹 `app_word2vec.py`  
Modelo **Word2Vec** con una puntuación de **0.9097**. Se visualizan embeddings en **2D con t-SNE**, se analiza la **cardinalidad del vocabulario** y se presentan ejemplos de palabras similares.  

### 🔹 `app_dataset.py`  
Clase **Dataset** para `DeepModelGRU`, encargada de cargar los **embeddings precomputados** y devolver los tensores listos para entrenamiento.  

### 🔹 `app_training.py`  
Clase de entrenamiento para `DeepModelGRU`, implementando **PyTorch** con:
- **BCEWithLogitsLoss** como función de pérdida.  
- **AdamW** como optimizador para mejorar la convergencia.  
- **Scheduler con paciencia de 2** para ajustar dinámicamente el learning rate.  
- **GradScaler** para manejar cálculos en **punto flotante de precisión mixta**, optimizando rendimiento en **GPU**.  

### 🔹 `datasets/`  
Carpeta que contiene los **conjuntos de datos** utilizados en el análisis y entrenamiento.  

### 🔹 `embeddings/`  
Almacena los **embeddings precomputados** generados con **BERT y Word2Vec**.  

### 🔹 `model/`  
Modelos entrenados y guardados, incluyendo **MModel, DeepModelGRU y Word2Vec**.  

### 🔹 `requirements.txt`  
Lista de librerías necesarias para ejecutar el proyecto sin conflictos.  
