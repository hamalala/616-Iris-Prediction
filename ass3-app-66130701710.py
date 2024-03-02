import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import Perceptron
model = pickle.load(open('per_model-66130701710.sav', 'rb'))

st.title('Iris Species Prediction using Perceptrin')

x1 = st.slider('Select input1', 0.0, 10.0, 3.0)
x2 = st.slider('Select input2', 0.0, 10.0, 5.0)
x3 = st.slider('Select input3', 0.0, 10.0, 4.0)
x4 = st.slider('Select input4', 0.0, 10.0, 7.0)

X_new = np.array([x1, x2, x3, x4]).reshape(1, -1)
model.predict(X_new)
