# app.py
import streamlit as st
from yolo_utils import process_video
import tempfile

# Page setup
st.set_page_config(page_title="YOLO Video Detector 🎯", layout="centered")

# --- HEADER SECTION ---
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🎯 YOLOv8 Object Detection on Video</h1>
    <p style='text-align: center;'>Upload your video, run detection, count objects, and download results — all in one place 🚀</p>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# --- FILE UPLOAD ---
st.subheader("📁 Step 1: Upload Your Video File (.mp4)")
uploaded_file = st.file_uploader("", type=["mp4"])

if uploaded_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    st.subheader("⚡ Step 2: Run YOLOv8 Detection")

    if st.button("🚀 Run Detection Now"):
        with st.spinner("Running detection, please wait..."):
            output_path = process_video(tfile.name)
        st.success("✅ Detection Complete!")

        st.subheader("🎬 Step 3: View & Download Output")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Processed Video", f, file_name="yolo_result.mp4")

else:
    st.info("Please upload a video file to begin.")

# --- FOOTER ---
st.markdown("""
    <hr>
    <div style='text-align: center; color: grey; font-size: 14px;'>
        Built with ❤️ using YOLOv8 & Streamlit
    </div>
""", unsafe_allow_html=True)