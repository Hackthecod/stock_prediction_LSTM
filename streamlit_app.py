import streamlit as st
import pandas as pd
import numpy as np

st.title('🤖 Stock Price Prediction')

st.info('This is Machine Learning App')

# df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df = pd.read_csv(r'penguins_cleaned.csv')
df
