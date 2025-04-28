import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
from PIL import Image

# Setup page
st.set_page_config(page_title="PlantEase - Plant Doctor", page_icon="🌱", layout="centered")

# Groq API
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Load your model
model = tf.keras.models.load_model(r'C:\Users\Manthan Bhambhani\Desktop\AI Project\model\plant_disease_detection.h5')

# Class labels
class_names = ['Apple___Apple_scab',
               'Apple___Black_rot',
               'Apple___Cedar_apple_rust',
               'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Raspberry___healthy',
               'Soybean___healthy',
               'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch',
               'Strawberry___healthy',
               'Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


# Predict disease function
def predict_disease(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    input_arr = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    model_prediction = class_names[result_index]
    disease_name = model_prediction.split("__")
    plant = disease_name[0].replace("_","")
    condition = disease_name[1].replace("_"," ")
    disease = plant + " - " + condition
    return disease

# Groq chat function
def chat_with_groq(user_input):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert botanist and plant care advisor."},
            {"role": "user", "content": user_input}
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"⚠️ API Error: {response.status_code}"

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Display title
st.title("🌱 PlantEase")

# Display chat history
for role, content in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(content)
    elif role == "user_image":
        st.chat_message("user").image(content)
    else:
        st.chat_message("assistant").markdown(content)

# Chat input + Upload button together
with st.container():
    col1, col2, col3 = st.columns([7, 1, 1])
    
    with col1:
        st.session_state.user_input = st.text_input("Type here...", value=st.session_state.user_input, key="input", label_visibility="collapsed")
    
    with col2:
        upload_clicked = st.button("📷", use_container_width=True)
    
    with col3:
        send_clicked = st.button("➤", use_container_width=True)

# Handle Upload button
if upload_clicked:
    st.session_state.uploading = True  # Set a session flag

# If uploading is in progress
if st.session_state.get("uploading", False):
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="upload", label_visibility="hidden")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.chat_history.append(("user_image", image))  # User image
        disease_detected = predict_disease(image)
        st.session_state.disease_detected = disease_detected  # <-- Save disease into session state
        response = f"🩺 I analyzed the image and detected **{disease_detected}**.\n\n**What would you like to know more about it?**"
        st.session_state.chat_history.append(("bot", response))
        st.session_state.uploading = False  # Reset uploading
        st.rerun()

# Show Suggested questions if disease was detected
if "disease_detected" in st.session_state:
    disease_detected = st.session_state.disease_detected
    check = disease_detected.split(" ")
    if check[-1] != "healthy":
        suggestions = [
            f"What are the symptoms of {disease_detected}?",
            f"How can I treat {disease_detected} naturally?",
            f"How to prevent {disease_detected} from spreading?",
            f"Recommended fertilizers or fungicides for {disease_detected}?"
        ]
        for sugg in suggestions:
            if st.button(sugg):
                st.session_state.chat_history.append(("user", sugg))
                bot_reply = chat_with_groq(sugg)
                st.session_state.chat_history.append(("bot", bot_reply))
                st.session_state.user_input = ""
                del st.session_state.disease_detected  # After handling, remove it
                st.rerun()


# Handle Send button or ENTER key
if send_clicked or (st.session_state.user_input and st.session_state.user_input.endswith("\n")):
    user_query = st.session_state.user_input.strip()
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        bot_response = chat_with_groq(user_query)
        st.session_state.chat_history.append(("bot", bot_response))
        st.session_state.user_input = ""
        st.rerun()
