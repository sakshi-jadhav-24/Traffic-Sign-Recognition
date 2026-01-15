import streamlit as st
from src import image_detect, realtime_detect

st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

# --- Header ---
st.markdown(
    "<h1 style='text-align: center; color: #007bff;'>🚦 Traffic Sign Recognition</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Deep Learning based real-time and image recognition system</p>",
    unsafe_allow_html=True
)

# --- Demo Image ---
st.image("demo.png", caption="Traffic Sign Detection Demo", use_container_width=True)

# --- Mode Selector ---
page = st.radio(
    "Choose Mode",
    ["🖼️ Image-Based Recognition", "🎥 Real-Time Recognition"],
    horizontal=True
)

if page == "🖼️ Image-Based Recognition":
    image_detect.run()
elif page == "🎥 Real-Time Recognition":
    realtime_detect.run()

# --- Footer ---
st.markdown("---")
st.markdown("<center>Made with ❤️ by Sakshi Jadhav and Rutuja Maske | © 2025</center>", unsafe_allow_html=True)
