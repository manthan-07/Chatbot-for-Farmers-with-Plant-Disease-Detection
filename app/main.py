import tensorflow as tf
import cv2
import requests
import numpy as np

# Your Groq API Key (for text-based chatbot)
GROQ_API_KEY = "GROQ_API_KEY"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


model = tf.keras.models.load_model('..\model\plant_disease_detection.h5')

# Define the class names based on your model
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


def predict_disease(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    
    if img is None:
        print("Error: Image not found or cannot be read.")
        return
    
    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the required size (224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalize the image (rescale pixel values to [0, 1])
    img_normalized = img_resized / 255.0
    
    # Convert the image to a batch format
    input_arr = np.array([img_normalized])
    
    # Perform prediction
    prediction = model.predict(input_arr)
    
    # Get the index of the highest probability class
    result_index = np.argmax(prediction)
    
    # Retrieve the class name corresponding to the highest probability
    model_prediction = class_names[result_index]
    
    return model_prediction

# Function to interact with Groq API (Chatbot)
def chat_with_groq(user_input):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert botanist and plant care advisor. Answer questions only about plants, gardening, diseases, soil, fertilizers, sunlight, watering, and pest control."},
            {"role": "user", "content": user_input}
        ]
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"⚠️ API Error: {response.status_code}"


# Main chatbot flow
def main():
    print("\n🌱 Welcome to PlantEase 2.0 Chatbot! 🌱")
    print("Type 'text' for asking a question, 'image' for disease detection, or 'exit' to quit.\n")

    while True:
        mode = input("Choose mode (text/image/exit): ").strip().lower()
        
        if mode == "exit":
            print("👋 Goodbye! Stay green! 🌻")
            break

        elif mode == "text":
            user_input = input("👤 You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye! Stay green! 🌻")
                break

            # Call Groq API for plant-related queries
            response = chat_with_groq(user_input)
            print(f"🤖 PlantEase: {response}\n")

        elif mode == "image":
            img_path = input("🖼️ Enter the full path to the plant leaf image: ").strip()
            disease = predict_disease(img_path)
            print(f"🌿 Plant Disease Prediction: {disease}\n")

        else:
            print("⚠️ Invalid choice. Please type 'text', 'image', or 'exit'.\n")


# Run the chatbot
if __name__ == "__main__":
    main()
