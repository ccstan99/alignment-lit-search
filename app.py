import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pinecone
import urllib.parse

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# RETRIEVER_MODEL = 'allenai-specter'
RETRIEVER_MODEL = 'multi-qa-mpnet-base-cos-v1'
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
titles = []
for item in result["matches"]:
    context = item["metadata"]["text"]
    answer = reader(question=query, context=context)
    score = answer["score"]
    if (score > 0.001):
        title = item["metadata"]["title"]
        url = item["metadata"]["url"]
        if title in titles:
            st.write("...")
        else:
            titles.append(title)
            st.subheader("["+title+"]("+url+")")

        answer_text = answer["answer"]
        url = url + "#:~:text=" +  urllib.parse.quote(answer_text)
        st.markdown("({0:.3f}) ".format(answer["score"]) +
        context[:answer["start"]] +
        "<a href='" + url + "' style='background-color:#FDFDC9'>"+answer_text+"</a>" + 
        context[answer["end"]:],
        unsafe_allow_html=True)

if len(titles) == 0:
    st.write("Sorry, no answer found")