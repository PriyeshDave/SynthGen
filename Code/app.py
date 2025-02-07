import streamlit as st
import pandas as pd
from data_loader import load_data
from tabular_generator import TabularDataGenerator
from text_generator import TextDataGenerator  

# Streamlit UI
st.title("Synthetic Data Generation Framework")

# Select Data Type
data_type = st.radio("Select Data Type:", ["Tabular Data", "Textual Data"])

if data_type == "Tabular Data":
    uploaded_file = st.file_uploader("Upload CSV file for structured data", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("### Uploaded Data Preview:")
            st.dataframe(df.head())

            # Initialize Tabular Generator
            tabular_gen = TabularDataGenerator(df)
            tabular_gen.run_pipeline()

elif data_type == "Textual Data":
    st.write("🚧 Textual Data Generation - Coming Soon 🚧")