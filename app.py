import streamlit as st
from annotated_text import annotated_text
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pinecone

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
RETRIEVER_MODEL = 'allenai-specter'
READER_MODEL = 'deepset/electra-base-squad2'
INDEX = 'alignment-lit'
NAMESPACE = 'faq'
DIMS = 768

@st.experimental_singleton(show_spinner=False)
def init():
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')
    index = pinecone.Index(INDEX)
    retreiver = SentenceTransformer(RETRIEVER_MODEL)
    reader = pipeline(model=READER_MODEL, task="question-answering")
    return index, retreiver, reader

st.header('Stampy FAQ Extractive Search')
query = st.text_input("What is your question?", value="What is AI Safety?")

with st.spinner("Initializing..."):
    index, retreiver, reader = init()

xq = retreiver.encode(query).tolist()
result = index.query(xq, namespace=NAMESPACE, top_k=5, includeMetadata=True)
num_found = 0
for item in result["matches"]:
    context = item["metadata"]["text"]
    answer = reader(question=query, context=context)
    score = answer["score"]
    if (score > 0.01):
        num_found += 1
        st.subheader("["+item["metadata"]["title"]+"]("+item["metadata"]["url"]+")")
        annotated_text("({0:.2f}) ".format(answer["score"]), 
        context[:answer["start"]], (answer["answer"],"","#FDFDC9"), context[answer["end"]:])

    # st.markdown("({0:.2f}) ".format(answer["score"]) +
    # context[:answer["start"]] + "***" + answer["answer"] + "***" + context[answer["end"]:],
    # unsafe_allow_html=True)

if num_found == 0:
    st.write("Sorry, no answer found")