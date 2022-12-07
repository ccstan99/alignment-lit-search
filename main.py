from flask import Flask, render_template, jsonify, url_for, request
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np
import pandas as pd

app = Flask(__name__)

# PINECONE_API_KEY = os.getenv["PINECONE_API_KEY"]
PINECONE_API_KEY = "040b0588-32b2-4195-b234-63e068540253"
MODEL = 'allenai-specter'
INDEX = 'alignment-lit'
DIMS = 768
DEFAULT_QUERY = "What is AI Safety?"

def init():
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')
    model = SentenceTransformer(MODEL)
    index = pinecone.Index(INDEX)

    df = pd.read_json('arxiv_pos_list_metadata.json')
    df.set_index('paper_version', inplace = True)
    # print("df.shape", df.shape)
    return model, index, df

def search_papers(query, top_k=5):
    print("search_papers", query, top_k)
    xq = model.encode(query).tolist()
    print("model.encode.tolist()",xq[:5])
    results = index.query(xq, top_k=top_k, includeMetadata=False)
    results_list = []
    for result in results["matches"]:
        metadata = df.loc[result.id]
        print("result", result.id, result.score)
        # print("metadata",metadata)
        results_list.append({
            "id": result.id,
            "score": result.score,
            "title": str(metadata["title"]),
            "authors": str(metadata["authors"]),
            "url": str(metadata["url"]),
            "abstract": str(metadata["abstract"]),
        })
    return results_list

model, index, df = init()

@app.route("/")
def home():
    query = DEFAULT_QUERY
    return render_template("search.html", query=query, results=search_papers(query))

@app.route("/api/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        query = request.form.query
    if request.method == "GET":
        query = request.args.get("query", DEFAULT_QUERY)
    return jsonify(search_papers(query))


if __name__ == "__main__":
    app.run(debug=True)

