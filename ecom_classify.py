import streamlit as st
import fasttext
import re

model = fasttext.load_model('models/ecommerce_classifier.bin')

def preprocess(text):
    if(isinstance(text, str)):
        pattern = r"[^\w\s]"
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = cleaned_text.lower()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    return text

# <------------ Components ------------------>

st.title("Product Classifier")

txt = st.text_area("Enter the product title : ",height=300)

if len(txt) > 0:

    txt = preprocess(txt)
    pred = model.predict(txt)
    st.subheader("The product category is : " + pred[0][0][9:],)