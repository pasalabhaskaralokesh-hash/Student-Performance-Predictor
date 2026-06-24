import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Student Performance Predictor")

data = {
    "Study_Hours": [2, 4, 6, 8, 10],
    "Marks": [50, 60, 70, 85, 95]
}

df = pd.DataFrame(data)

X = df[["Study_Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

hours = st.number_input("Enter Study Hours", min_value=0.0)

if st.button("Predict"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")
