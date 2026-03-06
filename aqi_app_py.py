import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import pickle
import time

# --- পেজ কনফিগ ---
st.set_page_config(
    page_title="Dhaka AQI Forecaster V3",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- কাস্টম CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 3.8rem; color: #00E5FF; text-align: center; margin: 1.5rem 0; font-weight: 900; text-shadow: 0 0 25px rgba(0, 229, 255, 0.7); }
    .sub-header { font-size: 1.6rem; color: #B0BEC5; text-align: center; margin-bottom: 3.5rem; }
    .aqi-card { padding: 3.5rem; border-radius: 25px; text-align: center; margin: 3rem auto; max-width: 800px; box-shadow: 0 20px 60px rgba(0,0,0,0.7); border: 3px solid; background: linear-gradient(135deg, rgba(15, 30, 55, 0.97), rgba(0, 15, 40, 0.97)); }
    .aqi-number { font-size: 7rem; font-weight: bold; margin: 0.6rem 0; }
    .aqi-level { font-size: 2.8rem; margin: 1rem 0; }
    .advice { font-size: 1.5rem; margin-top: 2.5rem; line-height: 1.8; color: #E0F7FA; }
    .action-list { background: rgba(255, 255, 255, 0.08); padding: 2rem; border-radius: 15px; margin-top: 2.5rem; border: 1px solid rgba(0, 229, 255, 0.25); }
    .footer { text-align: center; color: #78909C; margin-top: 6rem; padding: 2.5rem; border-top: 1px solid #444; font-size: 1.2rem; }
    .sidebar-title { font-size: 2rem; color: #00E5FF; margin-bottom: 1.5rem; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- Health advice function ---
def get_health_advice(aqi):
    if aqi <= 50:
        return "বাতাস ভালো — বাইরে যাওয়া সম্পূর্ণ নিরাপদ। 😊"
    elif aqi <= 100:
        return "মাঝারি দূষণ — সংবেদনশীল হলে সতর্ক থাকুন। 😐"
    elif aqi <= 150:
        return "অস্বাস্থ্যকর সংবেদনশীলদের জন্য — বাইরে কম যান। 😷"
    elif aqi <= 200:
        return "অস্বাস্থ্যকর — N95 মাস্ক পরুন, শারীরিক পরিশ্রম কমান। 🚨"
    elif aqi <= 300:
        return "খুব অস্বাস্থ্যকর — বাইরে না যাওয়াই ভালো। ☠️"
    else:
        return "বিপজ্জনক — জরুরি অবস্থা, ঘরে থাকুন, মাস্ক পরুন। ☢️"

# --- হেডার ---
st.markdown('<h1 class="main-header">Dhaka AQI Forecaster V3</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced LSTM Model • More Accurate Prediction • Real-time Health Insights</p>', unsafe_allow_html=True)

# --- সাইডবার ---
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">Dhaka AQI V3</h2>', unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=160)
    st.markdown("### Input Controls")
    st.info("Provide expected weather parameters for accurate AQI forecast")
    st.markdown("---")
    st.markdown("**All Variables Used**")
    st.markdown("- Average Temperature (°C)")
    st.markdown("- Minimum Temperature (°C)")
    st.markdown("- Maximum Temperature (°C)")
    st.markdown("- Rainfall (mm)")
    st.markdown("- Wind Speed (km/h)")
    st.markdown("- Atmospheric Pressure (hPa)")
    st.markdown("- Previous Day AQI")
    st.markdown("- Season (1=Winter, 4=Summer)")
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("Masud Hasan")
    st.markdown("[GitHub Repo](https://github.com/epicmasud/aqi-dhaka-forecast)")
    st.markdown("[Feedback / Suggestions](https://your-contact-link)")

# --- মেইন কনটেন্ট ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Weather Forecast Inputs")
    temp = st.slider("Average Temperature (°C)", 10.0, 40.0, 28.0, step=0.5)
    tmin = st.slider("Minimum Temperature (°C)", 5.0, 30.0, 20.0, step=0.5)
    tmax = st.slider("Maximum Temperature (°C)", 20.0, 45.0, 35.0, step=0.5)
    rain = st.slider("Expected Rainfall (mm)", 0.0, 100.0, 0.0, step=1.0)
    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, step=1.0)
    pressure = st.slider("Atmospheric Pressure (hPa)", 900.0, 1100.0, 1010.0, step=1.0)
    lag1 = st.number_input("Previous Day AQI", 0.0, 500.0, 100.0, step=1.0)
    season = st.slider("Season (1=Winter, 4=Summer)", 1, 4, 3)

    predict_button = st.button("Predict AQI", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Processing weather data & generating prediction..."):
        time.sleep(1.8)

        try:
            # মডেল ও স্কেলার লোড করো
            model = load_model('dhaka_aqi_lstm_v3.h5', custom_objects={'mse': MeanSquaredError()})
            with open('scaler_v3.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # ইনপুট ফিচার অর্ডার (ট্রেনিং-এ যে অর্ডারে ছিল সেটা মিলাতে হবে)
            # ধরে নেয়া হয়েছে features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'aqi_lag1', 'season', ...] (মোট 18)
            input_features = [temp, tmin, tmax, rain, wind, pressure, lag1, season, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 18 ফিচার
            input_array = np.array([input_features] * 7, dtype=np.float32)  # 7 timesteps
            input_scaled = scaler.transform(input_array.reshape(-1, len(input_features)))  # স্কেল করো
            input_scaled = input_scaled.reshape(1, 7, len(input_features))  # (1, timesteps, features)

            # প্রেডিক্ট করো
            pred_scaled = model.predict(input_scaled, verbose=0)[0][0]

            # সঠিকভাবে ইনভার্স ট্রান্সফর্ম করো (শুধু AQI কলামের জন্য)
            dummy = np.zeros((1, len(input_features) + 1))  # +1 for target
            dummy[0, -1] = pred_scaled
            pred_aqi = scaler.inverse_transform(dummy)[0, -1]
            pred_aqi = max(0, int(pred_aqi))  # নেগেটিভ এড়ানোর জন্য

            # AQI কার্ড
            if pred_aqi <= 50:
                color, level, icon = "#4CAF50", "ভালো", "😊"
            elif pred_aqi <= 100:
                color, level, icon = "#FFEB3B", "মাঝারি", "😐"
            elif pred_aqi <= 150:
                color, level, icon = "#FF9800", "অস্বাস্থ্যকর", "😷"
            elif pred_aqi <= 200:
                color, level, icon = "#F44336", "অস্বাস্থ্যকর", "🚨"
            elif pred_aqi <= 300:
                color, level, icon = "#9C27B0", "খুব অস্বাস্থ্যকর", "☠️"
            else:
                color, level, icon = "#B71C1C", "বিপজ্জনক", "☢️"

            st.markdown(
                f"""
                <div class="aqi-card" style="border-color: {color};">
                    <div class="aqi-number" style="color:{color};">{pred_aqi}</div>
                    <div class="aqi-level" style="color:{color};">{level} {icon}</div>
                    <div class="advice">
                        {get_health_advice(pred_aqi)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # যদি AQI 200+ হয় — জরুরি অ্যাকশন দেখাও
            if pred_aqi >= 200:
                st.markdown(
                    """
                    <div class="action-list">
                        <h3 style="color:#FF5252;">জরুরি অ্যাকশন নিন:</h3>
                        <ul style="font-size: 1.3rem; color:#FFCDD2;">
                            <li>বাইরে যাওয়া একদম এড়িয়ে চলুন</li>
                            <li>N95 বা KN95 মাস্ক ব্যবহার করুন (যদি জরুরি হয়)</li>
                            <li>ঘরের জানালা-দরজা বন্ধ রাখুন</li>
                            <li>এয়ার পিউরিফায়ার চালু রাখুন</li>
                            <li>শ্বাসকষ্ট বা বুকে ব্যথা হলে তাৎক্ষণিক হাসপাতালে যান</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"সমস্যা: {str(e)}\nমডেল বা স্কেলার ফাইল চেক করুন।")

# --- ফুটার ---
st.markdown(
    """
    <div class="footer">
        © 2026 Dhaka AQI Forecaster V3 | Built with ❤️ by Masud Hasan | 
        <a href="https://github.com/epicmasud/aqi-dhaka-forecast" style="color:#00E5FF; text-decoration:none;">Source Code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
