from flask import Flask, request, render_template, redirect, url_for
import os
import fitz  # PyMuPDF
import numpy as np
import re
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

es = Elasticsearch(hosts=["http://localhost:9200"])
model = SentenceTransformer('all-MiniLM-L6-v2')


def create_index(index_name):
    if not es.indices.exists(index=index_name):
        index_config = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "clean_text": {"type": "text"},
                    "digit_text": {"type": "text"},
                    "symbol_text": {"type": "text"},
                    "embedding": {"type": "dense_vector"},
                    "file_name": {"type": "keyword"}
                }
            }
        }
        es.indices.create(index=index_name, body=index_config)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc]).strip()


def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())


def extract_digits(text):
    return " ".join(re.findall(r"\d+", text))


def extract_symbols(text):
    return " ".join(re.findall(r"[^\w\s]", text))


def convert_to_float64(embedding):
    return np.array(embedding, dtype=np.float64).tolist()


def save_to_elasticsearch(index_name, doc_id, text, file_name):
    create_index(index_name)
    clean_text = preprocess_text(text)
    digit_text = extract_digits(text)
    symbol_text = extract_symbols(text)
    embedding = convert_to_float64(model.encode(clean_text))
    doc = {
        "text": text,
        "clean_text": clean_text,
        "digit_text": digit_text,
        "symbol_text": symbol_text,
        "embedding": embedding,
        "file_name": file_name
    }
    es.index(index=index_name, id=doc_id, body=doc)


def knn_search_all_indices(query_text, k=10, score_threshold=1.2, fallback_score_threshold=0.1):
    clean_query = preprocess_text(query_text)
    query_vector = convert_to_float64(model.encode(clean_query))
    indices = es.indices.get_alias("*").keys()
    results = []
    total_hits = 0

    for idx in indices:
        if idx.startswith("."):
            continue
        try:
            response = es.search(
                index=idx,
                body={
                    "size": k,
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "should": [
                                        {"match_bool_prefix": {"clean_text": clean_query}},
                                        {"match": {"digit_text": clean_query}},
                                        {"match": {"symbol_text": clean_query}},
                                    ]
                                }
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                }
            )
            hits = response["hits"]["hits"]
            filtered_hits = [hit for hit in hits if hit["_score"] >= score_threshold]
            total_hits += len(filtered_hits)

            if len(filtered_hits) == 0:
                fallback_response = es.search(
                    index=idx,
                    body={
                        "size": k,
                        "query": {
                            "match_phrase": {
                                "text": {
                                    "query": query_text,
                                    "slop": 5
                                }
                            }
                        }
                    }
                )
                fallback_hits = fallback_response["hits"]["hits"]
                fallback_filtered = [hit for hit in fallback_hits if hit["_score"] >= fallback_score_threshold]
                total_hits += len(fallback_filtered)
                filtered_hits.extend(fallback_filtered)

            for hit in filtered_hits:
                full_text = hit["_source"]["text"]
                match_position = full_text.lower().find(clean_query.lower())
                start = max(0, match_position - 100)
                end = min(len(full_text), match_position + 200)
                context = full_text[start:end].strip().replace('\n', ' ')
                results.append({
                    "index": idx,
                    "score": hit["_score"],
                    "text": context,
                    "file_name": hit["_source"].get("file_name", "N/A")
                })

        except Exception as e:
            print(f"Skipping index {idx} due to error: {e}")

    # 🔁 Sort all results globally by score descending
    results.sort(key=lambda x: x['score'], reverse=True)

    return results, total_hits


# Dummy login credentials
FIXED_USERNAME = 'user'
FIXED_PASSWORD = 'password123'


@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == FIXED_USERNAME and password == FIXED_PASSWORD:
            return redirect(url_for('home'))
        else:
            message = "❌ Invalid username or password."
    return render_template('login.html', message=message)


@app.route('/home', methods=['GET', 'POST'])
def home():
    search_result = []
    upload_result = {}
    total_hits = 0
    message = ""

    if request.method == 'POST':
        if 'doc_id' in request.form and 'pdf_file' in request.files:
            file = request.files['pdf_file']
            doc_id = request.form['doc_id'].strip()
            if file.filename.endswith('.pdf') and doc_id:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                text = extract_text_from_pdf(file_path)
                save_to_elasticsearch(doc_id, doc_id, text, file.filename)
                message = f"✅ File uploaded and processed successfully with ID: {doc_id}"
                upload_result = {
                    "id": doc_id,
                    "text": text[:500] + "..." if len(text) > 500 else text
                }
            else:
                message = "❌ Invalid file or missing Document ID."
        elif 'search' in request.form:
            search_query = request.form['search']
            search_result, total_hits = knn_search_all_indices(search_query)

    return render_template(
        'index.html',
        search_result=search_result,
        upload_result=upload_result,
        total_hits=total_hits,
        message=message
    )


if __name__ == '__main__':
    app.run(debug=True)
