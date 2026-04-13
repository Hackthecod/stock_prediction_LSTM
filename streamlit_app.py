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

    st.scatter_chart(X)
