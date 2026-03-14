import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

# --- वेबसाईटचे नाव आणि आयकॉन ---
st.set_page_config(page_title="Team Build Hub - AI Translator", page_icon="🤟", layout="centered")

# --- CSS Styling (डिझाईन सुधारण्यासाठी) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤟 AI Sign Language Translator")
st.subheader("Bridging the Gap with AI for Impact")
st.write("तुमचा हात कॅमेरासमोर दाखवा आणि 'Predict' बटण दाबा.")

# --- AI Model Load करणे ---
@st.cache_resource
def load_files():
    # तुमची 'model_unquant.tflite' फाईल लोड होत आहे
    interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    
    # 'labels.txt' मधून नाव वाचणे
    with open("labels.txt", "r") as f:
        labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return interpreter, labels

try:
    interpreter, labels = load_files()
except Exception as e:
    st.error(f"Error loading files: {e}. कृपया खात्री करा की 'model_unquant.tflite' आणि 'labels.txt' एकाच फोल्डरमध्ये आहेत.")

# --- कॅमेरा इंटरफेस ---
img_file = st.camera_input("कॅमेरा सुरू करा")

if img_file:
    # फोटो प्रोसेस करणे (224x224 size)
    image = Image.open(img_file).convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0  # Normalization

    # प्रेडिक्शन (Prediction) करणे
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)
    result = labels[index]
    confidence = output_data[0][index] * 100

    # निकाल दाखवणे
    st.markdown("---")
    st.markdown(f"""
    <div class="prediction-box">
        <h3>ओळखलेला शब्द:</h3>
        <h1 style='color: #ff4b4b;'>{result}</h1>
        <p>विश्वासार्हता: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

# --- फुटर ---
st.markdown("---")
st.caption("Powered by Team Build Hub | Created with Streamlit")