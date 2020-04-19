import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]


@st.cache
def load_data():
    path = "data/titanic.csv"
    return pd.read_csv(path)


@st.cache(suppress_st_warning=True)
def process(df, col):
    st.markdown(
        f'<span style="color:red;font-size:24px">Cache miss: process(df, {col}) ran</span>',
        unsafe_allow_html=True,
    )
    return df[[col, "Survived"]]


df = load_data()
numeric_cols = np.sort(
    df.select_dtypes(include=numerics).columns.values, axis=-1, kind="mergesort"
)

st.header("Studying Titanic dataset")

col = st.selectbox("Select a column :", numeric_cols)

source = process(df, col)
chart = (
    alt.Chart(source)
    .mark_bar()
    .encode(alt.X(f"{col}:Q", bin=True), y="count()", color="Survived:N")
)
st.altair_chart(chart)
