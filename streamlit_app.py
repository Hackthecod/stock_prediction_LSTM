import streamlit as st
import pandas as pd
import numpy as np

st.title('🤖 Stock Price Prediction')

st.info('This is Machine Learning App')

with st.expander('Data'):
  st.write("Raw Data")
  df = pd.read_csv(r'https://raw.githubusercontent.com/Hackthecod/stock_prediction_LSTM/refs/heads/master/penguins_cleaned.csv')
  df
