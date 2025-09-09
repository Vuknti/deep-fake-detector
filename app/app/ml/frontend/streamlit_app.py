import streamlit as st
import requests
import tempfile
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Deepfake Detector")

uploaded_file = st.file_uploader("Upload image or short video", type=["jpg","jpeg","png","mp4","avi","mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.write("Preview:")
    if uploaded_file.type.startswith("image"):
        st.image(tmp_path)
    else:
        st.video(tmp_path)

    if st.button("Analyze"):
        files = {"file": open(tmp_path,"rb")}
        try:
            res = requests.post(API_URL, files=files, timeout=60)
            if res.status_code == 200:
                st.json(res.json())
            else:
                st.error(f"API error: {res.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")
