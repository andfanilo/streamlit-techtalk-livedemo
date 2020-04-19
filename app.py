import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from vega_datasets import data

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
all_datasets = data.list_datasets()


def main():
    st.header("Vega Datasets explorer")
    st.sidebar.header("Configuration")

    chosen_dataset = st.sidebar.selectbox(
        "Choose a Vega dataset :", all_datasets, index=all_datasets.index("iris")
    )

    df, description, url = load_data(chosen_dataset)
    all_cols = df.columns.values
    numeric_cols = df.select_dtypes(include=numerics).columns.values
    obj_cols = df.select_dtypes(include=["object"]).columns.values


def load_data(name_dataset):
    metadata = getattr(data, name_dataset.replace("-", "_"))
    df = metadata()
    description = metadata.description
    url = metadata.url
    return df, description, url


@st.cache
def train_rf(df, features, target):
    X = df[features]
    y = df[target].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    main()
