import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from scipy.sparse import hstack

# Carregar dados
data = pd.read_csv("news_data.csv")

# Verifique a distribuição das classes
print("Distribuição das classes:")
print(data['label'].value_counts())
print("\n")

# Função para limpar texto
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

data['text'] = data['text'].apply(clean_text)

# Função para identificar vieses no texto
def detect_bias_features(text):
    loaded_words = ['chocante', 'escândalo', 'polêmica', 'urgente','extrema direita']
    emotional_words = ['medo', 'ódio', 'esperança', 'ansiedade']
    vague_terms = ['alguns dizem', 'fontes próximas', 'especialistas afirmam']

    loaded_count = sum(text.count(word) for word in loaded_words)
    emotional_count = sum(text.count(word) for word in emotional_words)
    vague_count = sum(text.count(term) for term in vague_terms)
    
    return [loaded_count, emotional_count, vague_count]

# Adicionando as features de vieses ao dataset
data['bias_features'] = data['text'].apply(lambda x: detect_bias_features(x))

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vetorização de textos
vectorizer = TfidfVectorizer(max_features=500, stop_words=stopwords.words('portuguese'))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Preparação das features de vieses para treino e teste
X_train_bias = np.array([detect_bias_features(text) for text in X_train])
X_test_bias = np.array([detect_bias_features(text) for text in X_test])

# Concatenar as features TF-IDF com as de viés
X_train_combined = hstack([X_train_tfidf, X_train_bias])
X_test_combined = hstack([X_test_tfidf, X_test_bias])

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Predição e avaliação
y_pred = model.predict(X_test_combined)
print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred, zero_division=1))
print("\n")

# Função para predição com relatório de vieses
def predict_reliability(text):
    print("=== Análise da Notícia ===")
    print(f"Texto da notícia: {text}")
    text_clean = clean_text(text)
    text_tfidf = vectorizer.transform([text_clean])
    bias_features = np.array([detect_bias_features(text_clean)])
    text_combined = hstack([text_tfidf, bias_features])
    
    prediction = model.predict(text_combined)[0]
    resultado = "Confiável" if prediction == "confiável" else "Não Confiável"
    print(f"\nResultado da Classificação: {resultado}")
    
    # Relatório de vieses
    print("\n=== Relatório de Viés ===")
    if sum(bias_features[0]) > 0:
        print("Sinais de viés detectados:")
        print(f"  Sensacionalismo: {bias_features[0][0]} ocorrência(s)")
        print(f"  Apelo emocional: {bias_features[0][1]} ocorrência(s)")
        print(f"  Termos vagos: {bias_features[0][2]} ocorrência(s)")
    else:
        print("Nenhum viés significativo detectado.")
    
    return resultado

# Exemplo de uso
predict_reliability("Grande escândalo é revelado e gera medo na população")
