import streamlit as st
import pandas as pd
import numpy as np

st.title('🤖 ML Project')

st.info('This is Machine Learning App')

with st.expander('Data'):
    df = pd.read_csv(r'https://raw.githubusercontent.com/Hackthecod/stock_prediction_LSTM/refs/heads/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X = df.drop('species', axis=1)
    X

    st.write('**X**')
    y = df['species']
    y

with st.expander('Data Visualization '):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data Preparation

with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
    gender = st.selectbox('Gender',('Female','Male'))
    bill_length_mm = st.slider('Bill Length(mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)

# Create a dataframe for the input features
