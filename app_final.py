import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils import load_data
from utils import read_markdown_file
from utils import st_shap
from utils import train_rf


def main():
    st.title("Titanic: Machine Learning from Disaster")

    df = load_data()
    target = "Survived"
    features = [c for c in df.columns.values if c != target]

    with st.beta_expander("About Titanic"):
        c1, c2 = st.beta_columns(2)
        description = read_markdown_file("pages/titanic.md")
        c1.markdown(f"{description}", unsafe_allow_html=True)
        c2.image("images/titanic.jpg")

    st.header("Data preview")
    st.caption(f"Shape of dataset : {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.describe())
    cols_to_style = st.multiselect("Choose columns to apply BG gradient", features)
    st.dataframe(df.style.background_gradient(subset=cols_to_style, cmap="BuGn"))
    st.markdown("---")

    st.header("Plot distribution")
    col = st.selectbox("Choose a column to display", features)
    with_target = st.checkbox("Separate per target ?")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=10)),
            alt.Y("count()"),
            tooltip=[col, "count()"],
        )
        .interactive()
    )
    if with_target:
        chart = chart.encode(color=f"{target}:N")
    st.altair_chart(chart, use_container_width=True)
    st.markdown("---")

    st.header("Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.pyplot(fig)
    st.markdown("---")

    st.header("Classification")
    with st.form(key="classify"):
        c1, c2 = st.beta_columns(2)
        n_estimators = c1.number_input("Choose number of trees:", 1, 1000, 100)
        max_depth = c2.number_input("Max depth:", 1, 100, 5)
        button_pressed = st.form_submit_button(label="Train model")

    if button_pressed:
        with st.spinner("Training..."):
            clf, confusion_matrix, importance_plot, force_plot = train_rf(
                df, n_estimators, max_depth
            )
            st.balloons()
            st.pyplot(confusion_matrix)
            st.pyplot(importance_plot)
            st_shap(force_plot, 400)


if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Techtalk Live demo", page_icon=":tada:")

    main()

    with st.sidebar:
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top: 0.75em;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )
