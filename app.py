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


if __name__ == "__main__":
    main()
