
import streamlit as st
from modules.model import load_model
import pandas as pd

st.title("🔍 Unsupervised Clustering Explorer")

model = load_model("models/kmeans_model.pkl")

uploaded_file = st.file_uploader("Upload CSV for clustering")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df['Cluster'] = preds
    st.write(df)
