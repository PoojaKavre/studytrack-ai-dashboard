import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

st.set_page_config(page_title="StudyTrack AI", layout="wide")
st.title("📚 StudyTrack – AI Based Student Dashboard")

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("Students Performance Dataset.csv")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Remove unwanted first column
if df.columns[0].startswith("\\"):
    df = df.iloc[:, 1:]

st.success("Dataset Loaded Successfully ✅")

# ---------------- STUDENT SELECT ----------------
if "First_Name" in df.columns and "Last_Name" in df.columns:
    df["Full_Name"] = df["First_Name"] + " " + df["Last_Name"]
    student_name = st.sidebar.selectbox("Select Student", df["Full_Name"])
    selected_student = df[df["Full_Name"] == student_name]
else:
    st.error("First_Name / Last_Name column not found ❌")
    st.stop()

# ---------------- METRICS ----------------
st.subheader("📊 Student Overview")

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols[:4]:
    st.metric(col, f"{selected_student[col].values[0]}")

# ---------------- SCORE GRAPH ----------------
st.subheader("📈 Numeric Data Overview")

fig, ax = plt.subplots()
ax.bar(numeric_cols[:5], selected_student[numeric_cols[:5]].values[0])
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------- CLUSTERING ----------------
st.subheader("🧠 AI Clustering")

features = df[numeric_cols].fillna(0)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(features)

cluster_value = df[df["Full_Name"] == student_name]["Cluster"].values[0]

st.write(f"### Student Cluster Group: {cluster_value}")

# ---------------- PREDICTION ----------------
st.subheader("🎯 Score Prediction")

prediction = None
if len(numeric_cols) >= 2:
    X = df[numeric_cols[:-1]].fillna(0)
    y = df[numeric_cols[-1]]

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(selected_student[numeric_cols[:-1]])

    st.write(f"Predicted {numeric_cols[-1]}: {prediction[0]:.2f}")

# ---------------- RISK DETECTION ----------------
st.subheader("🚨 Risk Level")

if prediction is not None:
    if prediction[0] < 50:
        st.error("High Academic Risk ⚠️")
    elif prediction[0] < 70:
        st.warning("Moderate Risk ⚠️")
    else:
        st.success("Low Risk ✅")

# ---------------- DATA TABLE ----------------
st.subheader("📋 Full Dataset")
st.dataframe(df)