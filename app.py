import streamlit as st

pg = st.navigation([st.Page('ecom_classify.py', title="Ecommerce Product Classifier"), st.Page("FakeNews.py", title="Fake News Classifier")])
pg.run()