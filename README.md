# 🧠 YOLOv8 Object Detection App  
> Real-time Object Detection on Images & Videos using YOLOv8 + Streamlit

![YOLOv8](https://img.shields.io/badge/Built%20With-YOLOv8-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square)

Detect and count objects in both images and videos using the powerful YOLOv8 model — all through a user-friendly Streamlit interface.

---

## 🚀 Features

- 📷 Upload **Images** or 🎞️ **Videos** for object detection  
- ✅ Detects 80+ object categories (trained on COCO dataset)  
- 📦 Bounding boxes + object labels displayed  
- 📽️ Downloadable processed videos  
- 🔢 Real-time object **counting** (for video)  
- 🧠 Powered by **YOLOv8 + OpenCV + Streamlit**  
- 🌐 Web-based — no installation needed on user end

---

## 📁 Folder Structure

yolov8_object_detection/
├── app.py                  # Streamlit UI
├── yolo_utils.py           # YOLOv8 detection logic
├── requirements.txt        # Python dependencies


---

## 🛠️ Installation & Setup

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/Vink-135/object_segmentation
cd yolov8-object-detection
###🧪 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
📦 3. Install Dependencies
pip install -r requirements.txt
⬇️ 4. Install YOLOv8
pip install ultralytics
▶️ Run the App
streamlit run yolo_streamlit.py
