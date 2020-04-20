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
    features = [c for c in df.columns.values if c != target]

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
        cols_to_style = st.multiselect("Choose columns to apply BG gradient", features)
        st.dataframe(df.style.background_gradient(subset=cols_to_style, cmap="BuGn"))
        st.markdown("---")

    if st.sidebar.checkbox("Plot distribution", False):
        st.subheader("Plot distribution")
        with st.echo():
            col = st.selectbox("Choose a column to display", features)
            with_target = st.checkbox("Separate per target ?")
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(alt.X(f"{col}:Q", bin=alt.Bin(maxbins=10)), alt.Y("count()"),)
            )
            if with_target:
                chart = chart.encode(color=f"{target}:N")
            st.altair_chart(chart)
        st.markdown("---")

    if st.sidebar.checkbox("Classification", False):
        st.subheader("Classification")

        clf, confusion_matrix = train_rf(df, features, target)
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
    df = pd.read_csv("data/titanic.csv")
    df.drop(["PassengerId"], axis=1, inplace=True)
    df.drop(["Name"], axis=1, inplace=True)
    df.drop(["Ticket"], axis=1, inplace=True)
    df.drop(["Cabin"], axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(df["Sex"].astype("category"), prefix="sex")], axis=1
    )
    df = pd.concat(
        [df, pd.get_dummies(df["Embarked"].astype("category"), prefix="embarked")],
        axis=1,
    )
    df.drop(["Sex"], axis=1, inplace=True)
    df.drop(["Embarked"], axis=1, inplace=True)

    return df


def read_markdown_file(path):
    return Path(path).read_text()


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def train_rf(df, features, target):
    st.error("Cache miss !")
    X = df[features]
    y = df[target].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf = RandomForestClassifier(max_depth=3)
    clf.fit(X_train, y_train)

    fig, ax = plt.subplots()
    plot_confusion_matrix(clf, X_test, y_test, ax=ax)

    return clf, fig


if __name__ == "__main__":
    main()
