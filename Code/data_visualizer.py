import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_distributions(self):
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if numerical_cols.any():
            st.write("### Numerical Distributions")
            for col in numerical_cols:
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))
                sns.histplot(self.df[col], kde=True, ax=axes[0])
                sns.boxplot(x=self.df[col], ax=axes[1])
                sns.violinplot(x=self.df[col], ax=axes[2])
                st.pyplot(fig)

        if categorical_cols.any():
            st.write("### Categorical Distributions")
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(x=self.df[col], order=self.df[col].value_counts().index, ax=ax)
                st.pyplot(fig)