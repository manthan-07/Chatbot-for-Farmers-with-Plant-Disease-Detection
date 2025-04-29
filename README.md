# 🌱 AgroHealth – Intelligent Plant Disease Detection with Chatbot Assistant

AgroHealth is an AI-powered web application designed to assist farmers and agriculturists in diagnosing plant diseases from leaf images using deep learning, and in receiving expert treatment suggestions via a chatbot. It combines a trained CNN model for image classification with Groq's LLaMA-3-based chatbot for personalized plant care guidance.

---

## 🔍 Features

- 🖼️ Upload plant leaf images to detect diseases instantly.
- 🧠 Deep learning CNN model classifies 38 plant diseases + healthy conditions.
- 💬 Interactive chatbot powered by Groq API provides expert guidance.
- ⚡ Built with Streamlit for an intuitive, fast, and responsive frontend.
- 📱 Mobile-compatible interface for ease of use by field users.

---

## 🛠️ Tech Stack

| Component        | Technology                    |
|------------------|-------------------------------|
| Frontend         | Streamlit (Python)            |
| Image Processing | OpenCV, Pillow, NumPy         |
| Deep Learning    | TensorFlow (.h5 model)        |
| Chatbot API      | Groq LLaMA-3 (OpenAI-compatible) |
| Deployment       | Local / Streamlit Cloud       |

---

## 🔄 How It Works

1. User uploads an image of a plant leaf.
2. The image is preprocessed and passed to a CNN model.
3. The model classifies it into one of 38 classes (disease or healthy).
4. Prediction is shown with suggested queries.
5. User can type a question or select a suggestion.
6. Chatbot responds with expert advice using the Groq API.

---

## 🤖 Model Details

- Pretrained on the **PlantVillage** dataset.
- Converted from [Kaggle PyTorch model](https://www.kaggle.com/code/imtkaggleteam/plant-diseases-detection-pytorch) to `.h5` format.
- Input size: 224x224, normalized
- Output: 38 softmax-activated class probabilities
