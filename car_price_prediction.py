
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction.joblib")   # сюда имя твоей модели

model = load_model()

st.title("🚗 Car Price Prediction")
st.markdown("Введите параметры автомобиля и получите предсказанную цену.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Основные данные")
    name = st.text_input("Название машины", value="Toyota Corolla")
    location = st.selectbox(
        "Город",
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
    )
    year = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2018, step=1)
    kilometers_driven = st.number_input("Пробег", min_value=0, value=45000, step=1000)
    fuel_type = st.selectbox(
        "Тип топлива",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )
    transmission = st.selectbox(
        "Коробка передач",
        ["Manual", "Automatic"]
    )

with col2:
    st.subheader("Технические данные")
    owner_type = st.selectbox(
        "Тип владельца",
        ["First", "Second", "Third", "Fourth & Above"]
    )
    mileage = st.number_input("Mileage", min_value=0.0, value=18.5, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=0.0, value=1498.0, step=100.0)
    power = st.number_input("Power (bhp)", min_value=0.0, value=103.0, step=1.0)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5, step=1)

st.divider()

if st.button("🔍 Predict Price", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Name": name,
        "Location": location,
        "Year": year,
        "Kilometers_Driven": kilometers_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats
    }])

    prediction = model.predict(input_data)[0]

    st.success("✅ Prediction completed")

    st.metric("Predicted Price", f"{prediction:.2f} Lakh")

    with st.expander("Показать введённые данные"):
        st.dataframe(input_data, use_container_width=True)

st.caption("ML project | Car Price Prediction")
