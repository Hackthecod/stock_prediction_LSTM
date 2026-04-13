import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 ML Project')

st.info('This is Machine Learning App')

with st.expander('Data'):
    df = pd.read_csv(r'https://raw.githubusercontent.com/Hackthecod/stock_prediction_LSTM/refs/heads/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X_row = df.drop('species', axis=1)
    X_row

    st.write('**X**')
    y_row = df['species']
    y_row

with st.expander('Data Visualization '):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input Features
with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
    bill_length_mm = st.slider('Bill Length(mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender',('Female','Male'))


# Create a dataframe for the input features
    data = {'island':island,
           'bill_length_mm':bill_length_mm,
           'bill_depth_mm':bill_depth_mm,
           'flipper_length_mm':flipper_length_mm,
           'body_mass_g':body_mass_g,
           'sex':gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_row], axis=0)
    
# Data Preparation

# Encode X
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_row = df_penguins[:1]

with st.expander('Input Features'):
    st.write('**Input Penguins**')
    input_df
    st.write('**Combined penguins data**')
    input_penguins
# Encode Y
target_mapper = {'Adelie': 0,
                'Chinstrap': 1,
                'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]

y = y_row.apply(target_encode)

with st.expander('Data Preparation'):
    st.write('**Encoded X (input Penguin)**')
    input_row
    st.write('**Encoded y**')
    y

# Model Building
clf = RandomForestClassifier()
clf.fit(X_row, y)

# Apply the train Model
prediction = clf.predict(input_row)
prediction_prob = clf.prediction_prob(input_row)
prediction_prob
