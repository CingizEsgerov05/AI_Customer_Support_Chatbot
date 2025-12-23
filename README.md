# 🤖 AI Customer Support Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/Hugging%20Face-BERT-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b)

A professional, hybrid **Customer Service Chatbot** built with **BERT (Bidirectional Encoder Representations from Transformers)** and **Streamlit**. 

This project simulates an intelligent assistant for E-commerce platforms, capable of handling inquiries about prices, shipping, returns, and products using a combination of Deep Learning and Rule-Based logic.

---

## 🚀 Key Features

* **🧠 Hybrid Intelligence:** Uses a **BERT-based classifier** for complex queries and **Keyword Matching (Fuzzy Logic)** for instant, accurate responses to common terms.
* **📊 Custom Data Augmentation:** Includes a pipeline to synthetically expand the training dataset using synonym replacement, improving model robustness with limited data.
* **💻 Interactive UI:** A clean, modern web interface built with **Streamlit**, featuring typing effects and session history.
* **🛑 Early Stopping:** The training loop includes an early stopping mechanism to prevent overfitting and save the best model state.
* **🌍 Multilingual Base:** Built on `bert-base-multilingual-cased`, allowing for easy adaptation to different languages (currently optimized for Azerbaijani).

---

## 📂 Project Structure

```bash
├── app.py                 # The Streamlit frontend interface
├── train.py               # Training pipeline (Data loading, Model training, Saving)
├── backend.py             # Core logic: Dataset, BERT Model architecture, & Inference
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
└── data/                  # (Generated files like .pth and .pkl will appear here)
```

## 🛠️ Installation & Setup

1. Clone the Repository

git clone https://github.com/CingizEsgerov05/AI_Customer_Support_Chatbot.git
cd AI_Customer_Support_Chatbot

2. Install Dependencies
Make sure you have Python installed. 

Then run:
pip install torch transformers streamlit scikit-learn numpy

3. Train the Model
Before running the app, you need to train the BERT model on the dataset.

python train.py
This process will generate best_chatbot_model.pth and chatbot_metadata.pkl.

4. Run the Chatbot
Launch the web interface:

streamlit run app.py

## ⚙️ How It Works
The system operates on a two-tier logic to ensure both speed and accuracy:

Tier 1: Keyword & Fuzzy Matching: The system first cleans the input and checks for strong keyword matches or fuzzy similarities (levenshtein distance). If a match is found with a high confidence score, it returns an instant response (Rule-based).

Tier 2: BERT Deep Learning: If no keywords are matched, the input is tokenized and passed through the BERT model. The model classifies the intent (e.g., shipping_query, return_policy) based on context. If the confidence score is above a threshold, the appropriate response is generated.

## Screenshots


## 🔮 Future Improvements
[ ] Translate internal dataset and logic to English (In Progress).

[ ] Add Voice-to-Text support (ASR).

[ ] Integrate with real E-commerce APIs for live stock checking.

[ ] Dockerize the application for easy deployment.

.

## 📝 Note
Currently, the dataset intents and responses inside backend.py are in Azerbaijani. The logic is language-agnostic and can be easily adapted to English or other languages by modifying the self.intents dictionary.
