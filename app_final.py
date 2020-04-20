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
    st.header("Titanic: Machine Learning from Disaster")
    st.sidebar.header("Configuration")

    df = load_data()
    target = "Survived"

    all_cols = df.columns.values
    numeric_cols = df.select_dtypes(include=numerics).columns.values
    obj_cols = df.select_dtypes(include=["object"]).columns.values

    description = read_markdown_file("pages/titanic.md")
    st.image("images/titanic.jpg", width=400)
    st.markdown(f"{description}", unsafe_allow_html=True)

    if st.sidebar.checkbox("Data preview", True):
        st.subheader("Data preview")
        st.markdown(f"Shape of dataset : {df.shape[0]} rows, {df.shape[1]} columns")
        if st.checkbox("Data types"):
            st.dataframe(df.dtypes)
        if st.checkbox("Pandas Summary"):
            st.write(df.describe())
        cols_to_style = st.multiselect(
            "Choose columns to apply BG gradient", numeric_cols
        )
        st.dataframe(df.style.background_gradient(subset=cols_to_style, cmap="BuGn"))
        st.markdown("---")

    if st.sidebar.checkbox("Plot numeric column distribution", False):
        st.subheader("Plot numeric column distribution")
        with st.echo():
            col = st.selectbox("Choose a column to display", numeric_cols)
            n_bins = st.number_input("Max number of bins ?", 5, 100, 10)
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    alt.X(f"{col}:Q", bin=alt.Bin(maxbins=n_bins)),
                    alt.Y("count()"),
                    color=f"{target}:N",
                )
            )
            st.altair_chart(chart)
        st.markdown("---")

    if st.sidebar.checkbox("Classification", False):
        st.subheader("Classification")

        if st.button("Run training"):
            clf, confusion_matrix = train_rf(df, numeric_cols, target)
            st.balloons()
            st.pyplot(confusion_matrix)

        st.markdown("---")

    # st.help(pd.merge)
    # st.balloons() # at the end

    st.sidebar.header("About")
    st.sidebar.text("Made by M. Fanilo ANDRIANASOLO")
    st.sidebar.text(
        "Code : https://github.com/andfanilo/streamlit-lyondatascience-20200422"
    )


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
