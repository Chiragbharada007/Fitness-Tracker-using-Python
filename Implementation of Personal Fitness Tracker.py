import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

# Dynamic path detection
base_path = os.path.dirname(__file__)

# Construct relative paths
calories_path = os.path.join(base_path, "DataSet", "calories.csv")
exercise_path = os.path.join(base_path, "DataSet", "exercise.csv")

# Load datasets
try:
    calories_data = pd.read_csv(calories_path)
    exercise_data = pd.read_csv(exercise_path)
    
    # Ensure required columns exist
    required_columns = ["User_ID", "Gender", "Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp"]
    if not all(col in exercise_data.columns for col in required_columns) or "Calories" not in calories_data.columns:
        st.error("Error: Missing required columns in dataset.")
        st.stop()

except FileNotFoundError:
    st.error("Error: Data files not found. Ensure 'calories.csv' and 'exercise.csv' exist in './DataSet/' folder.")
    st.stop()

# Sidebar inputs
st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Display user parameters
st.write("---")
st.header("Your Parameters")
st.write(df)

# Merge datasets
exercise_df = exercise_data.merge(calories_data, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Create BMI column
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df = exercise_df.round({"BMI": 2})

# Prepare training and test sets
exercise_df = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_df = pd.get_dummies(exercise_df, drop_first=True)

train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]

X_test = test_data.drop("Calories", axis=1)
y_test = test_data["Calories"]

# Train the model
model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Align input data with training features
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = model.predict(df)

st.write("---")
st.header("Prediction")
st.write(f"You are estimated to burn **{round(prediction[0], 2)} kilocalories**.")

# Find similar results
st.write("---")
st.header("Similar Results")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]

if not similar_data.empty:
    st.write(similar_data.sample(min(5, len(similar_data))))
else:
    st.write("No similar records found.")

# General Information
st.write("---")
st.header("General Information")

st.write("You are older than", round((exercise_df["Age"] < df["Age"].values[0]).mean() * 100, 2), "% of other users.")
st.write("Your exercise duration is higher than", round((exercise_df["Duration"] < df["Duration"].values[0]).mean() * 100, 2), "% of other users.")
st.write("Your heart rate is higher than", round((exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).mean() * 100, 2), "% of other users.")
st.write("Your body temperature is higher than", round((exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).mean() * 100, 2), "% of other users.")
