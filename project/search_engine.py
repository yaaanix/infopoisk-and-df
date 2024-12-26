# Сюда я вынесла все import'ы.
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import os 
import torch
import numpy as np
import time

df = pd.read_csv("hf://datasets/rogozinushka/psychologist_answers/psiholog_2023_12_16.csv")
df_cleaned = df.dropna()

nltk.download('stopwords')
russian_stopwords = set(stopwords.words("russian"))
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    # Приведение к нижнему регистру.
    text = text.lower()
    
    # Удаление только пунктуации (оставляем числа).
    text = re.sub(r'[^\w\s]', '', text)
    
    words = text.split()
    
    # Лемматизация и удаление стоп-слов.
    processed_words = []
    for word in words:
        if word not in russian_stopwords:
            lemma = morph.parse(word)[0].normal_form  # можно заменить на стемминг, если нужна скорость
            processed_words.append(lemma)
    
    return ' '.join(processed_words)

df_cleaned['questions_tf_idf'] = df_cleaned['question_body'].apply(preprocess_text)

df_cleaned['answers_tf_idf'] = df_cleaned['answers'].apply(preprocess_text)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_top_keywords(texts, top_n=3):
    """
    Извлекает топ-N ключевых слов из текстов с помощью TF-IDF.
    """
    if texts.empty:
        return pd.DataFrame(columns=[f'keyword_{i+1}' for i in range(top_n)])

    # Преобразование текста в векторное представление
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Извлечение ключевых слов
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = []

    for row in tfidf_matrix:
        indices = row.toarray().argsort()[0, -top_n:][::-1]
        top_keywords.append(feature_names[indices])

    # Преобразование результата в DataFrame
    return pd.DataFrame(top_keywords, columns=[f'keyword_{i+1}' for i in range(top_n)])

# Обработка колонки question_body
question_body_keywords = extract_top_keywords(df_cleaned['questions_tf_idf'].fillna(''), top_n=3)

# Обработка колонки answers
answers_keywords = extract_top_keywords(df_cleaned['answers_tf_idf'].fillna(''), top_n=3)

# Добавление ключевых слов в исходный DataFrame
df_cleaned = pd.concat([df_cleaned, question_body_keywords.add_prefix('qbody_'), answers_keywords.add_prefix('answers_')], axis=1)

# Результат
df_cleaned = df_cleaned.dropna()


with open('tfidf_vectorizer1.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('tfidf_matrix1.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('bert_embeddings1.pkl', 'rb') as f:
    bert_embeddings = pickle.load(f)

# Функция для поиска по TF-IDF.
def search_tfidf(query, tfidf_vectorizer, tfidf_matrix, top_n):
    # Преобразую запрос в векторную форму.
    query_vec = tfidf_vectorizer.transform([query])
    
    # Рассчитываю косинусную близость между запросом и документами.
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Сортирую результаты по убыванию близости.
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    return top_indices, similarities[top_indices]
# Функция для получения эмбеддингов BERT 
def get_bert_embedding_for_query(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    #print(f"Размерность эмбеддинга для запроса: {cls_embedding.shape}")
    return cls_embedding

# Функция для поиска по BERT'у
def search_bert(query_embedding, bert_embeddings, top_n):
    similarities = cosine_similarity([query_embedding], bert_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return top_indices, similarities[top_indices]

# Главная функция для выполнения поиска.
def search_engine(query, index_choice, df_cleaned, top_n):
    # Фиксирую время начала (это понадобится далее).
    start_time = time.time()
    
    if index_choice == 'tf-idf':
        print("Используем TF-IDF для поиска...")
        top_indices, top_similarities = search_tfidf(query, tfidf_vectorizer, tfidf_matrix, top_n)
    elif index_choice == 'bert':
        print("Используем BERT для поиска...")
        tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        query_embedding = get_bert_embedding_for_query(query, tokenizer, model)
        top_indices, top_similarities = search_bert(query_embedding, bert_embeddings, top_n)
    
    else:
        print("Неверный вариант индекса. Выберите 'tf-idf' или 'bert'.")
        return

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Поиск занял {elapsed_time:.4f} секунд.\n")

    print(f"Топ-{top_n} подходящих документов:")
    for i, idx in enumerate(top_indices):
        similarity = top_similarities[i]
        question = df_cleaned['question_title'].iloc[idx]
        answer = df_cleaned['answers'].iloc[idx]
        print(f"Сходство: {similarity:.4f}")
        print(f"Вопрос: {question}")
        print(f"Ответ: {answer}\n")
        

query = input("Введите текст запроса: ")
index_choice = input("Введите, какой вариант индекса использовать ('tf-idf' или 'bert'): ")
try:
    top_n = int(input('Введите количество текстов в выдаче: '))
    search_engine(query, index_choice, df_cleaned, top_n)
except ValueError:
    print("Количество текстов в выдаче должно быть числом.")

