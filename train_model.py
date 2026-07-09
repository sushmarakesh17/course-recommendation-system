import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("course_data.csv")

print(df.head())

# Label Encoding
interest_encoder = LabelEncoder()
level_encoder = LabelEncoder()
course_encoder = LabelEncoder()

df["Interest"] = interest_encoder.fit_transform(df["Interest"])
df["Skill_Level"] = level_encoder.fit_transform(df["Skill_Level"])
df["Recommended_Course"] = course_encoder.fit_transform(df["Recommended_Course"])

# Features and Target
X = df[["Interest", "Skill_Level", "Study_Hours"]]
y = df["Recommended_Course"]

# Train Model
model = RandomForestClassifier()

model.fit(X, y)

# Save Model
pickle.dump(model, open("recommendation_model.pkl", "wb"))
pickle.dump(interest_encoder, open("interest_encoder.pkl", "wb"))
pickle.dump(level_encoder, open("level_encoder.pkl", "wb"))
pickle.dump(course_encoder, open("course_encoder.pkl", "wb"))

print("Model Trained Successfully")