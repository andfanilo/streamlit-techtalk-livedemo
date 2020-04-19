import streamlit as st

st.title("Streamlit meetup at Lyon Data Science")

st.header("Hello world")

st.markdown("Streamlit is **_really_ cool** :sunglasses:")

st.text("Do you know your trigonometry formulas ?")

st.latex("\\cos (2\\theta) = \\cos^2 \\theta - \\sin^2 \\theta")

st.subheader("Let's write some Python code !")

code = """
def hello():
print("Hello, Streamlit!")
"""
st.code(code, language="python")
