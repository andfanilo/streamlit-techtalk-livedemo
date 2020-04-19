import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def find_outliers(df, window, sigma):
    avg = df[variable].rolling(window=window).mean()
    residual = df[variable] - avg
    std = residual.rolling(window=window).std()
    return np.abs(residual) > std * sigma


st.header("Outlier detection in Occupancy Detection Data Set")

uploaded_file = st.file_uploader("Choose data/occupancy.csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if len(data.columns) != 7:
        st.error("This doesn't look like the correct dataset...")

    else:
        data["date"] = data.date.astype("datetime64[ns]")

        variable = "Temperature"
        window = st.slider("Window : ", 1, 60, 30)
        sigma = st.slider("Sigma : ", 1, 20, 10)

        data["is_outlier"] = find_outliers(data, window, sigma)

        chart = (
            alt.Chart(data)
            .mark_line(size=2, color="blue")
            .encode(
                alt.X("date:T", axis=alt.Axis(format="%x")),
                alt.Y(f"{variable}:Q", scale=alt.Scale(domain=[18, 24])),
            )
        )

        outlier = (
            alt.Chart(data)
            .mark_circle(color="firebrick")
            .encode(
                alt.X("date:T"),
                alt.Y(f"{variable}:Q"),
                opacity=alt.condition("datum.is_outlier", alt.value(1), alt.value(0)),
            )
        )

        st.altair_chart((chart + outlier), use_container_width=True)
