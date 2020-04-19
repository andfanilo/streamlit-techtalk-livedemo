import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

path = "data/titanic.csv"
df = pd.read_csv(path)

chart = alt.Chart(df).mark_bar().encode(alt.X("Age:Q", bin=True), y="count()")
st.altair_chart(chart)

fig, ax = plt.subplots()
df.hist(
    bins=8,
    column="Age",
    grid=False,
    figsize=(8, 8),
    color="#86bf91",
    zorder=2,
    rwidth=0.9,
    ax=ax,
)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.distplot(df["Age"], ax=ax)
st.pyplot(fig)

fig = px.histogram(df, x="Age")
st.plotly_chart(fig)
