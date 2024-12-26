from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from enum import Enum
import search_engine

app = Flask(__name__)

class MethodsEnum(str, Enum):
    tf_idf = "tf-idf"
    bert = "bert"

class MethodsResponse(BaseModel):
    methods: list[MethodsEnum]

class CorporaResponse(BaseModel):
    tokens: int
    name: str

class SearchRequest(BaseModel):
    type: str

@app.route('/methods', methods=['GET'])
def get_methods():
    response = MethodsResponse(methods=[MethodsEnum.tf_idf, MethodsEnum.bert])
    return jsonify(response.dict()), 200

@app.route('/corpora', methods=['GET'])
def get_corpora():
    response = CorporaResponse(tokens=123, name="psycho")
    return jsonify(response.dict()), 200

@app.route('/search', methods=['GET'])
def search_endpoint():
    try:
        args = SearchRequest(type=request.args.get('type'))
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    result = search_engine.search_engine(args.type)
    response = SearchResponse(result=result)
    return jsonify(response.dict()), 200

if __name__ == '__main__':
    app.run(debug=True)




# ---------------------------------- divide ---------------------------------- #
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from enum import Enum
import pandas as pd
import search_engine

app = Flask(__name__)

class MethodsEnum(str, Enum):
    tf_idf = "tf-idf"
    bert = "bert"

class MethodsResponse(BaseModel):
    methods: list[MethodsEnum]

class CorporaResponse(BaseModel):
    tokens: int
    name: str

class SearchRequest(BaseModel):
    query: str
    index_choice: MethodsEnum
    df_cleaned: list[dict]

class SearchResponse(BaseModel):
    result: list[dict]

@app.route('/methods', methods=['GET'])
def get_methods():
    response = MethodsResponse(methods=[MethodsEnum.tf_idf, MethodsEnum.bert])
    return jsonify(response.dict()), 200

@app.route('/corpora', methods=['GET'])
def get_corpora():
    response = CorporaResponse(tokens=123, name="psycho")
    return jsonify(response.dict()), 200

@app.route('/search', methods=['POST'])
def search_endpoint():
    try:
        data = request.get_json()
        args = SearchRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    df_cleaned = pd.DataFrame(args.df_cleaned)
    result = search_engine.search_engine(args.query, args.index_choice, df_cleaned)
    response = SearchResponse(result=result)
    return jsonify(response.dict()), 200

if __name__ == '__main__':
    app.run(debug=True)
