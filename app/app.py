## Advanced Machine Learning Codealong: Introduction to Streamlitmodel
## Week 3: Lecture 1
## Objectives:  
## Create streamlit app to explore a dataset
## Include: Visualize dataframe, print descriptive statistics, and Generate EDA plots of columns

## Reference: https://docs.streamlit.io/library/api-reference

## Import necessary packages
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly as px
import custom_functions as fn
import os

## load_data with cahcing
# @st.cache_data
def load_data():
  df = joblib.load('Data-NLP/movie_reviews_preprocessed.pkl')
  return df

## Use function to load Data
df = load_data()
# df.head()

## Load training and test data


train_file_path = './app/Data-NLP/training_data.csv'
test_file_path = './app/Data-NLP/test_data.csv'

# @st.cache_data
def load_training_data():
  df = pd.read_csv(train_file_path, encoding='utf-8')
  return

# @st.cache_data
def load_testing_data():
  df = pd.read_csv(test_file_path, encoding='utf-8')
  return



# train = pd.DataFrame(joblib.load(train_file_path))
# test = pd.DataFrame(joblib.load(test_file_path))

# test = pd.DataFrame(joblib.load(test_file_path))



## Columns for EDA
columns = df.columns
target = 'Sentiment'
features = [col for col in columns if col != target]


## Image, title and Markdown subheader

## Display DataFrame
st.header('Movie Reviews')
st.dataframe(df)


## .info()
## Get info as text
buffer = StringIO()
df.info(buf=buffer)
info_text = buffer.getvalue()

#Display .info() with button trigger
st.sidebar.subheader('Show Dataframe Summary')
summary_text = st.sidebar.button('Summary Text')
if summary_text:
    st.text(info_text)

## Descriptive Statistics subheader
st.sidebar.subheader('Show Descriptive Statistics')

## Button for Statistics
show_stats = st.sidebar.button('Descriptive Statistics')
if show_stats:
    describe = df.describe()
    st.dataframe(describe)

## Eda Plots subheader
st.sidebar.subheader('Explore a Column')

## Columns for EDA
columns = df.columns
target = 'Sentiment'
features = [col for col in columns if col != target]

## selectbox for columns
eda_column = st.sidebar.selectbox('Column to Explore', columns, index=None)

## Conditional: if column was chosen
if eda_column:
    ## Check if column is object or numeric and use appropriate plot function
    if df[eda_column].dtype == 'object':
        fig = fn.explore_categorical(df, eda_column)
    else:
        fig = fn.explore_numeric(df, eda_column)

    ## Show plot
    st.subheader(f'Display Descriptive Plots for {eda_column}')
    st.pyplot(fig)

## Select box for features vs target
feature = st.sidebar.selectbox('Compare Feature to Target', features, index=None)

## Conditional: if feature was chosen
if feature:
    ## Check if feature is numeric or object
    if df[feature].dtype == 'object':
        comparison = df.groupby('Sentiment').count()
        title = f'Count of {feature} by {target}'
    else:
        comparison = df.groupby('Sentiment').mean()
        title = f'Mean {feature} by {target}'

    ## Display appropriate comparison
    pfig = px.bar(comparison, y=feature, title=title)
    st.plotly_chart(pfig)

# Create a streamlit app for getting predictions for a user-entered text from your loaded model
# Load the model
    
model = joblib.load('app/models/best_model.joblib')

label_encoder = joblib.load('app/models/label_encoder.joblib')

user_input = st.text_area('Enter your text here')
if st.button('Predict'):
  print(type(user_input))
  prediction = model.predict([user_input])
  sentiment = label_encoder.inverse_transform(prediction)
  st.write(sentiment)
  st.write('Sentiment:', sentiment[0])


train = load_training_data()
test = load_testing_data()

# Show training and test data
st.subheader('Training Data')
st.dataframe(train)

st.subheader('Testing Data')
st.dataframe(test)