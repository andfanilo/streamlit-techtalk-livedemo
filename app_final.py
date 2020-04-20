import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils import load_data
from utils import read_markdown_file
from utils import train_rf


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

        if st.button("Run training"):
            clf, confusion_matrix = train_rf(df, features, target)
            st.balloons()
            st.pyplot(confusion_matrix)

        st.markdown("---")

    # st.help(pd.merge)

    st.sidebar.header("About")
    st.sidebar.text("Made by M. Fanilo ANDRIANASOLO")
    st.sidebar.text(
        "Code : https://github.com/andfanilo/streamlit-lyondatascience-20200422"
    )


if __name__ == "__main__":
    main()
