import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load pipeline
@st.cache_resource
def load_pipeline():
    pkl_path = BASE_DIR / "artifacts" / "pipeline.pkl"
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

# Page Config
st.set_page_config(page_title="Spaceship Titanic", page_icon="🚀", layout="centered")
st.title("🚀 Spaceship Titanic Prediction")
st.markdown("Isi data penumpang di bawah untuk memprediksi apakah mereka **Transported** ke dimensi lain.")
st.divider()

# Input form
st.subheader("🧑 Passenger Information")
col1, col2 = st.columns(2)

with col1:
    passenger_id = st.text_input("PassengerId", value="0001_01")
    home_planet  = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"], index=0)
    cryo_sleep   = st.selectbox("CryoSleep", [False, True], index=0)
    cabin        = st.text_input("Cabin (Deck/Num/Side)", value="F/123/S")
    destination  = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"], index=0)

with col2:
    age  = st.number_input("Age", min_value=0, max_value=100, value=28)
    vip  = st.selectbox("VIP", [False, True], index=0)
    name = st.text_input("Name", value="John Doe")

st.divider()
st.subheader("💰 Spending (in space credits)")
col3, col4, col5 = st.columns(3)

with col3:
    room_service = st.number_input("RoomService", min_value=0.0, value=0.0)
    food_court   = st.number_input("FoodCourt",   min_value=0.0, value=0.0)
with col4:
    shopping_mall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
    spa           = st.number_input("Spa",          min_value=0.0, value=0.0)
with col5:
    vr_deck = st.number_input("VRDeck", min_value=0.0, value=0.0)

st.divider()

# prredict
if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([{
        'PassengerId':  passenger_id,
        'HomePlanet':   home_planet,
        'CryoSleep':    cryo_sleep,
        'Cabin':        cabin,
        'Destination':  destination,
        'Age':          float(age),
        'VIP':          vip,
        'RoomService':  room_service,
        'FoodCourt':    food_court,
        'ShoppingMall': shopping_mall,
        'Spa':          spa,
        'VRDeck':       vr_deck,
        'Name':         name,
    }])

    prediction  = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success("🌀 **TRANSPORTED!** Penumpang ini dibawa ke dimensi lain.")
    else:
        st.error("🚢 **NOT TRANSPORTED.** Penumpang ini tetap di Spaceship Titanic.")

    st.metric("Transported Probability", f"{probability:.2%}")
    st.progress(float(probability))