import streamlit as st
st.write("THIS IS THE COURSE RECOMMENDATION APP")
st.title(" Course recommendation system Prediction")
import pickle
import pandas as pd

# Load model
model = pickle.load(open("recommendation_model.pkl", "rb"))
interest_encoder = pickle.load(open("interest_encoder.pkl", "rb"))
level_encoder = pickle.load(open("level_encoder.pkl", "rb"))
course_encoder = pickle.load(open("course_encoder.pkl", "rb"))

st.title("Course Recommendation System")

interest = st.selectbox(
    "Select Your Interest",
    interest_encoder.classes_
)

level = st.selectbox(
    "Skill Level",
    level_encoder.classes_
)

hours = st.number_input(
    "Study Hours Per Week",
    min_value=1,
    max_value=40,
    value=5
)

if st.button("Recommend Course"):

    interest_encoded = interest_encoder.transform([interest])[0]
    level_encoded = level_encoder.transform([level])[0]

    data = pd.DataFrame([[
        interest_encoded,
        level_encoded,
        hours
    ]], columns=[
        "Interest",
        "Skill_Level",
        "Study_Hours"
    ])

    prediction = model.predict(data)

    course = course_encoder.inverse_transform(prediction)

    st.success("Recommended Course")

    st.write(course[0])