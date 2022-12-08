import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pinecone

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
MODEL = 'allenai-specter'
INDEX = 'alignment-lit'
NAMESPACE = 'faq'
DIMS = 768

@st.experimental_singleton(show_spinner=False)
def init():
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')
    index = pinecone.Index(INDEX)
    model = SentenceTransformer(MODEL)
    return index, model

st.header('Stampy FAQ Search')
query = st.text_input("What is your question?", value="What is AI Safety?")

with st.spinner("Initializing..."):
    index, model = init()

xq = model.encode(query).tolist()
result = index.query(xq, namespace=NAMESPACE, top_k=5, includeMetadata=True)
for item in result["matches"]:
    st.subheader("["+item["metadata"]["title"]+"]("+item["metadata"]["url"]+")")
    st.write("({0:.2f}) ".format(item["score"]), item["metadata"]["text"])

# run_search(query, model, index)
