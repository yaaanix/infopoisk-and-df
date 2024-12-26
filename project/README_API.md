# Flask API с использованием Pydantic и Enum

## Описание эндпоинтов

1. **GET `/methods`**  
Возвращает список доступных методов анализа данных.  
**Формат ответа:**  
`{"methods": ["tf-idf", "bert"]}`  

2. **GET `/corpora`**  
Возвращает информацию о корпусе, включая количество токенов и название.  
**Формат ответа:**  
`{"tokens": 123, "name": "psycho"}`  

3. **POST `/search`**  
Выполняет поиск по указанным параметрам.  

**Формат входных данных:**  
`{"query": "example query", "index_choice": "tf-idf", "df_cleaned": [{"text": "example1"}, {"text": "example2"}]}`  

**Формат ответа:**  
`{"result": [{"query": "example query", "engine": "TF-IDF", "matches": 2}]}`  

**Пример вызова:**  
`curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{"query": "example query", "index_choice": "tf-idf", "df_cleaned": [{"text": "example1"}, {"text": "example2"}]}'`

## Проект задействует файл search_engine.py, поэтому api.py и search_engine.py должны находиться в одной папке


