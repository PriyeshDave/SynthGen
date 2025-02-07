import streamlit as st
import pandas as pd
import numpy as np
from data_visualizer import DataVisualizer
from synthetic_evaluator import SyntheticEvaluator
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN
# from sdv.tabular import TVAE
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


class TabularDataGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()



    def preprocess_data(self):
        """Encodes categorical features for model training."""
        df_encoded = self.df.copy()
        label_encoders = {}
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
        return df_encoded, label_encoders
    


    def generate_synthetic_data(self, model_type: str, epochs: int, num_records: int) -> pd.DataFrame:
        """Generates synthetic data using the selected model."""
        df_encoded, label_encoders = self.preprocess_data()
        if model_type == "CTGAN":
            model = CTGAN(epochs=epochs)
        elif model_type == 'Gaussian Copula':
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(self.df)
            model = GaussianCopulaSynthesizer(metadata)
        else:
            st.error("Currently, only CTGAN is supported.")
            return pd.DataFrame()
        
        model.fit(df_encoded)
        synthetic_data = model.sample(num_records)
        for col in self.categorical_cols:
            synth_col_values = synthetic_data[col].astype(int)
            valid_indices = synth_col_values.isin(range(len(label_encoders[col].classes_)))
            synth_col_values[~valid_indices] = np.nan
            if synth_col_values.isna().sum() > 0:
                most_frequent = self.df[col].mode()[0]
                synth_col_values.fillna(label_encoders[col].transform([most_frequent])[0], inplace=True)
            synthetic_data[col] = label_encoders[col].inverse_transform(synth_col_values.astype(int))
        return synthetic_data



    def run_pipeline(self):
        """Runs the full Streamlit UI pipeline."""
        st.write("## Data Distributions")
        visualizer = DataVisualizer(self.df)
        visualizer.plot_distributions()

        # Model Selection
        model_type = st.selectbox("Select Model for Synthetic Data Generation", ["CTGAN" , 'Gaussian Copula'])
        epochs = st.slider("Select Training Epochs", 100, 1000, 300)
        num_records = st.number_input("Enter the number of synthetic records to generate:", 
                                  min_value=1, max_value=5 * self.df.shape[0], value=self.df.shape[0], step=1)

        if st.button("Generate Synthetic Data"):
            synthetic_data = self.generate_synthetic_data(model_type, epochs, num_records)
            synthetic_data.columns = self.df.columns

            if not synthetic_data.empty:
                st.write("### Synthetic Data Preview:")
                st.dataframe(synthetic_data)

                visualizer = DataVisualizer(synthetic_data)
                visualizer.plot_distributions()


                # Evaluate
                evaluator = SyntheticEvaluator(self.df, synthetic_data)
                evaluator.evaluate()

                # Download Option
                csv = synthetic_data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Synthetic Data", csv, "synthetic_data.csv", "text/csv")
