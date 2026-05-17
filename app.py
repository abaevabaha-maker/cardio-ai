import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL
# ======================

model = pickle.load(open("model.pkl", "rb"))

# ======================
# UI
# ======================

st.title("🏥 Cистема оценки риска сердечно-сосудистых заболеваний")

st.sidebar.header("Введите данные пациента")

age = st.sidebar.number_input("Возраст", 1, 120, 30)
height = st.sidebar.number_input("Рост", 100, 220, 170)
weight = st.sidebar.number_input("Вес", 30, 200, 70)

ap_hi = st.sidebar.number_input("Систолическое давление", 80, 250, 120)
ap_lo = st.sidebar.number_input("Диастолическое давление", 40, 150, 80)

chol = st.sidebar.number_input("Холестерин", 1, 50, 1)
gluc = st.sidebar.number_input("Глюкоза", 1, 50, 1)

# BMI
bmi = weight / ((height / 100) ** 2)

X = np.array([[age, height, weight, bmi, ap_hi, ap_lo, chol, gluc]])

# ======================
# PREDICT
# ======================

if st.sidebar.button("Анализ"):

    pred = model.predict(X)[0]
    risk = model.predict_proba(X)[0][1] * 100

    st.subheader("Результат")

    st.metric("Риск заболевания", f"{risk:.1f}%")
    st.metric("BMI", f"{bmi:.1f}")

    if risk < 30:
        st.success("Сбалансированное питание")
    elif risk < 60:
        st.warning("Низкожировая диета")
    else:
        st.error("Строгий контроль питания")

    # pie chart
    fig, ax = plt.subplots()
    ax.pie([risk, 100-risk], labels=["Risk","Safe"], autopct="%1.1f%%")
    st.pyplot(fig)

# ======================
# EDA
# ======================

st.markdown("---")
st.header("Анализ данных")

df = pd.read_csv("cardio_train.csv", sep=";")

if "id" in df.columns:
    df = df.drop(columns=["id"])

df["BMI"] = df["weight"] / ((df["height"]/100)**2)

st.bar_chart(df["cardio"].value_counts())

# Create a new figure and axes for the histogram
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df["BMI"], bins=30) # Plot the histogram on the new axes
st.pyplot(fig_hist)
