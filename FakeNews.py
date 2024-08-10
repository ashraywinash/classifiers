import streamlit as st
import joblib
import pandas as pd
import spacy
from sklearn.neighbors import KNeighborsClassifier


nlp = spacy.load('en_core_web_lg')
nlp.disable_pipes(['parser', 'attribute_ruler', 'ner', 'tok2vec'])
model = joblib.load('models/fake_news_detection.pkl')
labels = ["fake", "real"]


def preprocess(text, nlp_obj):
    doc = nlp_obj(text)
    final_tokens_1 = [token for token in doc if token.pos_ not in ["PUNCT","SYM","X"] and token.text not in ["’s", "n’t", "‘"]]
    final_tokens_4 = [token for token in final_tokens_1 if not token.is_stop]
    final_tokens_3 = [token.lemma_ for token in final_tokens_4]
    processedText = ' '.join(final_tokens_3)
    processedText = processedText.lower()
    return processedText

def getEmbeddings(text, nlp_obj):
    doc = nlp_obj(text)
    vect = doc.vector
    return vect

# <---------- Components Start ----------------->

st.title('Fake News Classifier')

text = st.text_area("Enter the news snippet", height=300)

if len(text) > 0:
        processed_text = preprocess(text, nlp)
        embed = getEmbeddings(processed_text, nlp).reshape((1, -1))
        pred = model.predict(embed)
        
        st.subheader(f"The news snippet is: {labels[pred[0]]}")

