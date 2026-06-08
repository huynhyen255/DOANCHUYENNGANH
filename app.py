import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors  # <-- Tích hợp mô hình Vector không gian
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- FIX LỖI NLTK TRÊN STREAMLIT CLOUD (CÁCH MẠNH TAY) ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.error(f"Lỗi tải NLTK: {e}")

download_nltk_resources()

# --- THIẾT KẾ GIAO DIỆN ---
st.set_page_config(page_title="Phân loại tin nhắn Spam", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 26px; font-weight: bold; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành - Hệ thống Phân loại tích hợp Naive Bayes & Vector Database")
st.write("---")

# --- HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN SONG SONG ---
@st.cache_resource
def load_trained_model():
    df = pd.read_csv("2cls_spam_text_cls.csv")
    ps = PorterStemmer()
    
    def preprocess(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]
        return [ps.stem(t) for t in tokens]

    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in dictionary: 
                # Ép kiểu int để phòng ngừa lỗi NumPy index
                features[int(dictionary.index(t))] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # 1. Khởi tạo & Huấn luyện Naive Bayes
    model = GaussianNB()
    model.fit(X, y)
