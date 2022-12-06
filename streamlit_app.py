import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
MODEL = 'allenai-specter'
INDEX = 'alignment-lit'
DIMS = 768

@st.experimental_singleton(show_spinner=False)
def init():
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')
    model = SentenceTransformer(MODEL)
    index = pinecone.Index(INDEX)
    return model, index

st.header('Alignment Literature Search')
query_sentence = st.text_input("What is your question?", value="What is AI Safety?")

# x = st.slider("Select a number")
# option = st.selectbox(
#      'What is your favorite color?',
#      ('Blue', 'Red', 'Green'))
# options = st.multiselect(
#      'What are your favorite colors',
#      ['Green', 'Yellow', 'Red', 'Blue'],
#      ['Yellow', 'Red'])
# coffee = st.checkbox('Coffee')

with st.spinner("Initializing..."):
    model, index = init()

xq = model.encode(query_sentence).tolist()
result = index.query(xq, top_k=5, includeMetadata=True)
for item in result["matches"]:
    st.write("### ["+item["metadata"]["title"]+"]("+item["metadata"]["url"]+")")
    st.write("({0:.2f}) ".format(item["score"]), item["metadata"]["abstract"])

# run_search(query_sentence, model, index)
