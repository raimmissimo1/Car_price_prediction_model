import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ===== Настройки страницы =====
st.set_page_config(
    page_title="Оценка цены автомобиля",
    page_icon="🚗",
    layout="wide"
)

# ===== Константы =====
INR_TO_KZT = 5.33   # можно менять вручную при необходимости
LUXURY_BRANDS = ["Mercedes-Benz", "BMW", "Audi", "Jaguar", "Lexus", "Porsche", "Volvo", "Land"]

LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]

# ===== Загрузка модели =====
@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction.joblib")

model = load_model()

# ===== Вспомогательные функции =====
def lakh_to_kzt(lakh: float) -> float:
    # 1 lakh = 100000 INR
    return lakh * 100000 * INR_TO_KZT

def format_kzt(value: float) -> str:
    return f"{value:,.0f} ₸".replace(",", " ")

def format_lakh(value: float) -> str:
    return f"{value:.2f} lakh"

def build_features(
    name: str,
    location: str,
    year: int,
    kilometers_driven: int,
    fuel_type: str,
    transmission: str,
    owner_type: str,
    mileage: float,
    engine_liters: float,
    power: float,
    seats: int
) -> pd.DataFrame:
    current_year = datetime.now().year

    brand = name.split()[0] if name.strip() else "Unknown"
    car_age = max(current_year - year, 1)
    km_per_year = kilometers_driven / car_age

    # модель у тебя, судя по данным, ждёт Engine_size и Engine в литрах/числе
    engine_size = engine_liters
    power_per_engine = power / engine_size if engine_size != 0 else 0
    is_luxury = 1 if brand in LUXURY_BRANDS else 0

    data = pd.DataFrame([{
        "Name": name,
        "Location": location,
        "Year": year,
        "Kilometers_Driven": kilometers_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Mileage": mileage,
        "Engine": engine_liters,
        "Power": power,
        "Seats": seats,
        "Brand": brand,
        "Car_Age": car_age,
        "Km_per_year": km_per_year,
        "Engine_size": engine_size,
        "Power_per_engine": power_per_engine,
        "Is_Luxury": is_luxury
    }])

    return data

# ===== Заголовок =====
st.title("🚗 Оценка цены автомобиля")
st.markdown(
    "Введите основные данные автомобиля и получите **примерную рыночную цену**."
)
st.divider()

# ===== Форма =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("Основная информация")
    name = st.text_input("Марка и модель", value="Toyota Corolla")
    location = st.selectbox("Город продажи", LOCATIONS, index=0)
    year = st.number_input("Год выпуска", min_value=1990, max_value=datetime.now().year, value=2018, step=1)
    kilometers_driven = st.number_input("Пробег (км)", min_value=0, value=45000, step=1000)
    owner_type = st.selectbox("Количество владельцев", OWNER_TYPES, index=0)

with col2:
    st.subheader("Технические характеристики")
    fuel_type = st.selectbox("Тип топлива", FUEL_TYPES, index=0)
    transmission = st.selectbox("Коробка передач", TRANSMISSIONS, index=0)
    mileage = st.number_input("Экономичность / mileage", min_value=0.0, value=18.5, step=0.1)
    engine_liters = st.number_input("Объём двигателя (л)", min_value=0.1, value=2.0, step=0.1)
    power = st.number_input("Мощность (л.с.)", min_value=1.0, value=103.0, step=1.0)
    seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5, step=1)

st.divider()

# ===== Кнопка =====
if st.button("🔎 Оценить стоимость", type="primary", use_container_width=True):
    try:
        input_data = build_features(
            name=name,
            location=location,
            year=year,
            kilometers_driven=kilometers_driven,
            fuel_type=fuel_type,
            transmission=transmission,
            owner_type=owner_type,
            mileage=mileage,
            engine_liters=engine_liters,
            power=power,
            seats=seats
        )

        prediction_lakh = float(model.predict(input_data)[0])
        prediction_kzt = lakh_to_kzt(prediction_lakh)

        # примерный диапазон ±10%
        low_kzt = prediction_kzt * 0.9
        high_kzt = prediction_kzt * 1.1

        st.success("✅ Оценка выполнена")

        m1, m2 = st.columns(2)
        m1.metric("Прогнозируемая цена", format_lakh(prediction_lakh))
        m2.metric("Примерно в тенге", format_kzt(prediction_kzt))

        st.info(
            f"Ориентировочный диапазон: **{format_kzt(low_kzt)} — {format_kzt(high_kzt)}**"
        )

        st.subheader("Краткая сводка")
        s1, s2, s3 = st.columns(3)
        s1.metric("Возраст авто", f"{int(input_data['Car_Age'].iloc[0])} лет")
        s2.metric("Пробег в год", f"{int(input_data['Km_per_year'].iloc[0]):,} км".replace(",", " "))
        s3.metric("Бренд", input_data["Brand"].iloc[0])

        with st.expander("Показать технические детали модели"):
            st.write("Эти признаки используются моделью для расчёта, но обычному пользователю они обычно не нужны.")
            tech_view = pd.DataFrame([{
                "Бренд": input_data["Brand"].iloc[0],
                "Возраст авто": int(input_data["Car_Age"].iloc[0]),
                "Пробег в год": round(float(input_data["Km_per_year"].iloc[0]), 2),
                "Объём двигателя": float(input_data["Engine_size"].iloc[0]),
                "Мощность / двигатель": round(float(input_data["Power_per_engine"].iloc[0]), 2),
                "Премиум-бренд": "Да" if int(input_data["Is_Luxury"].iloc[0]) == 1 else "Нет"
            }])
            st.dataframe(tech_view, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Ошибка при расчёте: {e}")

st.caption("Модель машинного обучения для оценки стоимости автомобиля")
