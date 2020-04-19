import datetime
import pandas as pd
import streamlit as st


# button
if st.button("Say hello"):
    st.write("Why hello there")

# checkbox
if st.checkbox("I agree"):
    st.write("I agree")

# selectbox
option = st.selectbox(
    "How would you like to be contacted?", ("Email", "Home phone", "Mobile phone")
)
st.write("You selected:", option)

# multiselect
o = st.multiselect("Choose cols", ("Age", "Gender", "Income"))
st.write(f"You selected: {o}")

# slider
age = st.slider("How old are you : ", 0, 130)
st.markdown(f"I’m {age} years old :tada:")

# radio
genre = st.radio("Movie genre ?", ("Comedy", "Drama", "Doc"))
st.write(f"It's {genre} !")

# checkbox
agree = st.checkbox("Streamlit is easy !")
if agree:
    st.write("Great :ok_hand:")

# time input
t = st.time_input("Set an alarm for", datetime.time(8, 45))
st.write("Alarm is set for", t)


# file uploader
f = st.file_uploader("Choose a CSV", type="csv")
if f is not None:
    data = pd.read_csv(f)
    st.write(data)

# date_input
d = st.date_input("When's your birthday", datetime.date(2019, 7, 6))
st.write("Your birthday is:", d)

# text_area
txt = st.text_area("Text to analyze")
st.markdown(
    txt.replace("it", '<span style="color:red">it</span>'), unsafe_allow_html=True
)
# it was the best of times, it was the worst of times, it was
# the age of wisdom, it was the age of foolishness
