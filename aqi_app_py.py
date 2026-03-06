import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import time

# --- পেজ কনফিগ ---
st.set_page_config(
    page_title="Dhaka AQI Forecaster",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- কাস্টম CSS (প্রফেশনাল + কালারফুল) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3.8rem;
        color: #00E5FF;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 900;
        text-shadow: 0 0 25px rgba(0, 229, 255, 0.7);
    }
    .sub-header {
        font-size: 1.6rem;
        color: #B0BEC5;
        text-align: center;
        margin-bottom: 3.5rem;
    }
    .aqi-card {
        padding: 3.5rem;
        border-radius: 25px;
        text-align: center;
        margin: 3rem auto;
        max-width: 800px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.7);
        border: 3px solid;
        background: linear-gradient(135deg, rgba(15, 30, 55, 0.97), rgba(0, 15, 40, 0.97));
    }
    .aqi-number {
        font-size: 7rem;
        font-weight: bold;
        margin: 0.6rem 0;
    }
    .aqi-level {
        font-size: 2.8rem;
        margin: 1rem 0;
    }
    .advice {
        font-size: 1.5rem;
        margin-top: 2.5rem;
        line-height: 1.8;
        color: #E0F7FA;
    }
    .action-list {
        background: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2.5rem;
        border: 1px solid rgba(0, 229, 255, 0.25);
    }
    .footer {
        text-align: center;
        color: #78909C;
        margin-top: 6rem;
        padding: 2.5rem;
        border-top: 1px solid #444;
        font-size: 1.2rem;
    }
    .sidebar-title {
        font-size: 2rem;
        color: #00E5FF;
        margin-bottom: 1.5rem;
        text-align: center;
    }
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
st.markdown('<h1 class="main-header">Dhaka AQI Forecaster</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Air Quality Prediction for Dhaka using Advanced LSTM</p>', unsafe_allow_html=True)

# --- সাইডবার ---
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">Dhaka AQI</h2>', unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=160)
    st.markdown("### Input Controls")
    st.info("Enter expected weather conditions to predict AQI")
    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("- Advanced LSTM model")
    st.markdown("- Colorful AQI card & health advice")
    st.markdown("- Dark premium theme")
    st.markdown("- Responsive design")
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("Masud Hasan")
    st.markdown("[GitHub Repo](https://github.com/epicmasud/aqi-dhaka-forecast)")
    st.markdown("[Feedback / Suggestions](https://your-contact-link)")

# --- মেইন কনটেন্ট ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Weather Conditions")
    temp = st.slider("Expected Average Temperature (°C)", 10.0, 40.0, 28.0, step=0.5)
    rain = st.slider("Expected Rainfall (mm)", 0.0, 100.0, 0.0, step=1.0)

    predict_button = st.button("Predict AQI", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing weather data & predicting AQI..."):
        time.sleep(1.5)  # সিমুলেট লোডিং

        try:
            # মডেল লোড করো
            model = load_model('dhaka_aqi_lstm.h5', custom_objects={'mse': MeanSquaredError()})

            # তোমার মডেলের আসল input shape অনুযায়ী
            num_features = 19  # ← এখানে তোমার model.input_shape[2] দাও (যেমন 19)
            timesteps = 7

            dummy_features = np.zeros((1, timesteps, num_features), dtype=np.float32)
            dummy_features[0, 0, 0] = temp
            dummy_features[0, 0, 1] = rain
            # অন্য ফিচার যোগ করো যদি থাকে (যেমন lag1, season)

            pred_scaled = model.predict(dummy_features, verbose=0)[0][0]
            pred_aqi = int(pred_scaled * 350 + 30)  # আসল scaler দিয়ে বদলাও

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

        except Exception as e:
            st.error(f"সমস্যা: {str(e)}\nমডেল ফাইল 'dhaka_aqi_lstm.h5' আছে কি না চেক করুন।")

# --- ফুটার ---
st.markdown(
    """
    <div class="footer">
        © 2026 Dhaka AQI Forecaster | Built with ❤️ by Masud Hasan | 
        <a href="https://github.com/epicmasud/aqi-dhaka-forecast" style="color:#00E5FF; text-decoration:none;">Source Code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

# অ্যাডভাইস ফাংশন
def get_health_advice(aqi):
    if aqi <= 50:
        return "বাতাস ভালো — বাইরে যাওয়া সম্পূর্ণ নিরাপদ।"
    elif aqi <= 100:
        return "মাঝারি দূষণ — সংবেদনশীল হলে সতর্ক থাকুন।"
    elif aqi <= 150:
        return "অস্বাস্থ্যকর সংবেদনশীলদের জন্য — বাইরে কম যান।"
    elif aqi <= 200:
        return "অস্বাস্থ্যকর — N95 মাস্ক পরুন, শারীরিক পরিশ্রম কমান।"
    elif aqi <= 300:
        return "খুব অস্বাস্থ্যকর — বাইরে না যাওয়াই ভালো।"
    else:
        return "বিপজ্জনক — জরুরি অবস্থা, ঘরে থাকুন, মাস্ক পরুন।"
