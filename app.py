import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #fafafa; }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")
st.caption("Paste a news headline or article text below to check its credibility.")

text = st.text_area(
    "News text",
    height=180,
    placeholder="Paste a headline or article here...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 1])
with col1:
    check_clicked = st.button("🔍 Check Credibility", type="primary")
with col2:
    clear_clicked = st.button("🗑️ Clear")

if clear_clicked:
    st.rerun()

if check_clicked:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"text": text})
                response.raise_for_status()
                result = response.json()

                label = result["label"]
                confidence = result["confidence"]

                st.divider()

                if label == "Real":
                    st.success(f"✅ Prediction: **{label}**")
                else:
                    st.error(f"⚠️ Prediction: **{label}**")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Confidence Score", f"{confidence}%")
                with c2:
                    st.progress(confidence / 100)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Make sure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")