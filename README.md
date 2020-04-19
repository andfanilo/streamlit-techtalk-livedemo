# Demo building

## Prerequisites

- conda create -n streamlit python=3.7
- conda activate streamlit
- pip install streamlit pandas altair matplotlib seaborn plotly bokeh scikit-learn vega_datasets

## Run

```bash
streamlit run app.py
# or
streamlit run https://raw.githubusercontent.com/andfanilo/streamlit-lyondatascience-20200422/master/app_final.py
```

## Demo steps

- Write description on screen + url
  - If description is None, write a `st.warning`
- New sidebar checkbox data preview
  - Print all dataframe
  - Print `df.dtypes`
  - Print `df.describe()`
  - Put checkboxes for dtypes/describe
  - Style dataframe with `.background_gradient(subset=cols_to_style, cmap="BuGn")`
- New sidebar checkbox univariate distribution
  - Selectbox with choose a column
  - Altair chart column
  - Wrap in `st.echo`
- New sidebar checkbox classification
  - Select target class
  - `train_rf(df, numeric_cols, target)` and st.balloons
  - `plot_confusion_matrix(clf, X_test, y_test, ax=ax)`
- Change dataset ? Go to gapminder to do something on this ?

## Resources

- https://pmbaumgartner.github.io/streamlitopedia/ https://github.com/pmbaumgartner/dank-data-explorer https://dank-data-explorer.herokuapp.com/
- https://datasetsformlapp.herokuapp.com/ https://www.youtube.com/watch?v=SIu2VL-RAXc
