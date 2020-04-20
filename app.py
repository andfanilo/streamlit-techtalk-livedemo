from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]


def main():
    df = load_data()
    target = "Survived"

    all_cols = df.columns.values
    numeric_cols = df.select_dtypes(include=numerics).columns.values
    obj_cols = df.select_dtypes(include=["object"]).columns.values

    ### START HERE ###
    pass
    ### END HERE ###


def load_data():
    return pd.read_csv("data/titanic.csv")


def read_markdown_file(path):
    return Path(path).read_text()


@st.cache
def train_rf(df, features, target):
    X = df[features].fillna(-1)
    y = df[target].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    fig, ax = plt.subplots()
    plot_confusion_matrix(clf, X_test, y_test, ax=ax)

    return clf, fig


if __name__ == "__main__":
    main()
