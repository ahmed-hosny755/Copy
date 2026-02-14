import streamlit as st
import os
import tensorflow as tf
from tensorflow import keras

# مسار الموديل المحلي
MODEL_PATH = "plant_disease_model.keras"

# رابط مباشر من Google Drive
DOWNLOAD_URL = "https://drive.google.com/uc?id=1dBxiCkGL17RS1P5qsOhtRGezReiXMZyg"

def download_model():
    try:
        import gdown
    except ImportError:
        st.info("تثبيت gdown...")
        os.system("pip install gdown")
        import gdown
    st.info("تحميل الموديل من Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# التحقق من وجود الموديل
if not os.path.exists(MODEL_PATH):
    download_model()

# تحميل الموديل
try:
    model = keras.models.load_model(MODEL_PATH)
    st.success("تم تحميل الموديل بنجاح!")
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
    st.stop()  # إيقاف التطبيق لو الموديل مش جاهز

# --- واجهة Streamlit ---
st.title("Plant Disease Prediction 🌱")

uploaded_file = st.file_uploader("ارفع صورة النبات هنا", type=["jpg", "png"])
if uploaded_file:
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)
    
    # تحويل الصورة لمصفوفة للنموذج
    img_array = np.array(image.resize((224,224)))/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    st.write(f"توقع النموذج: **{predicted_class}**")
