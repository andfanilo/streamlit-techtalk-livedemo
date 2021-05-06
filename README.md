# Streamlit Live Demo - Titanic analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/andfanilo/streamlit-techtalk-livedemo/app_final.py)

## Prerequisites

```
conda create -n streamlit python=3.7
conda activate streamlit
pip install streamlit pandas altair matplotlib seaborn plotly bokeh scikit-learn
```

## Run

```bash
streamlit run app.py
```

## Demo steps

### Part 1 - First steps 

- Create empty `app.py`
- Open terminal with `CTRL + SHIFT + ù`. Run `streamlit run app.py`
- `st.write("Hello World")`. Activate `Run on save`.
  - I like livereload.
  - Streamlit watches file and updates when change or save detected
- `st.markdown(":tada: Streamlit is **super** easy")` 
  - Tip : markdown accepts HTML/CSS `st.markdown(":tada: <span style='color:red;'>Streamlit</span> is **super** easy", unsafe_allow_html=True)`
- Pull an image of Titanic : `st.image("images/titanic.jpg", width=300)`.
- Now let's plot, first read Titanic in DF `df = pd.read_csv('data/titanic.csv')`.
  - We can `st.dataframe(df)` to see our dataframe live.
  - Tip : Style dataframe : `st.dataframe(df.style.applymap(lambda data: f"background-color: {'red' if data==0 else 'green'}", subset=['Survived']))`
- Plot Matplotlib Age distribution (built server side as static image)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
df.hist("Age", ax=ax)
st.pyplot(ax) # show that Streamlit shows error
st.pyplot(fig)
```

- Describe Altair (binding Python to Vega to graph using JSON spec)

- So to build chart, built client side not server side
```python
chart = alt.Chart(df).mark_bar().encode(x="Age", y="count()", color="Survived:N", tooltip=['Age', "count()"]).interactive()
st.altair_chart(chart)
st.altair_chart(chart, use_container_width=True)
```
- If it's hard to remember, wrap with echo to see code `with st.echo(): chart = ...`

```python
with st.echo():
  chart = alt.Chart(df).mark_bar().encode(x="Age", y="count()", color="Survived", tooltip=['Age', "count()"]).interactive()
  st.altair_chart(chart, use_container_width=True)
```

Transition to interactive : now this is very static, what if I want to plot the distribution of another column ?

- I don't remember exactly how `st.selectbox` works so `st.help(st.selectbox)`, works with any Python object.
- extract `col = "Age"` and then `st.help(st.selectbox)` then `col = st.selectbox("...", df.columns)`.
- `col = st.selectbox("...", df.columns, df.columns.values.tolist().index("Age"))`
  - put `Name`, I like `Name` :) .

### Part 2 - Interactive

- so widget state is stored inside object directly
- I like file uploader, let's use it. `f = st.file_uploader(..)` f is directly file uploaded. 
  - `None` for now, `st.write(f)` to prove it
- `if f is not None:` run the code. Then change csv files.
- If you remember sidebar, put file_uploader in sidebar `f = st.sidebar.file_uploader(..)`.
- Let's put expander to hide dataframe.
```python
with st.beta_expander("Data preview"):
    st.dataframe(df.style.background_gradient(cmap="BuGn"))
```
- say that for now multipage is `if st.sidebar.selectbox`

### Part 3 - Caching

- First let's refactor a bit to put loading data in a function `def load_data(f): return pd.read_csv(f)`.
  - say Streamlit displays your errors on screen, not in logs, when `pd.read_csv(None)`
- Add `st.warning("CACHE MISS")` in this function.
- Add `@st.cache`. Copy paste the suppress warning `@st.cache(allow_output_mutation=True, suppress_st_warning=True)`. 
- Upload different files again.

### Part 4 - Button for ML

- Import utils : `from utils import *`. Replace `load_data`.
```python
st.subheader("Classification")
c1, c2 = st.beta_columns(2)
n_estimators = c1.number_input("Choose number of trees:", 1, 1000, 100)
max_depth = c2.number_input("Max depth:", 1, 100, 5)

if st.button("Run training"):
    with st.spinner("Training en cours"):
        clf, confusion_matrix = train_rf(df, n_estimators, max_depth)
        st.balloons()
        st.pyplot(confusion_matrix)
```

## Resources

- https://pmbaumgartner.github.io/streamlitopedia/ https://github.com/pmbaumgartner/dank-data-explorer https://dank-data-explorer.herokuapp.com/
- https://datasetsformlapp.herokuapp.com/ https://www.youtube.com/watch?v=SIu2VL-RAXc
