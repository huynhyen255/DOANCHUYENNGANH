import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- THIẾT KẾ GIAO DIỆN NÂNG CẤP ---
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; }
    .prediction-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành - Nâng cấp bởi AI: Tích hợp Tiếng Anh & Tiếng Việt")

# --- HÀM XỬ LÝ (Nâng cấp hỗ trợ đa ngôn ngữ nhẹ) ---
@st.cache_resource
def load_trained_model():
    nltk.download("stopwords")
    nltk.download("punkt")
    df = pd.read_csv("2cls_spam_text_cls.csv")
    
    def preprocess(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        # Nếu là tiếng Việt (kiểm tra dấu hoặc từ phổ biến), ta không lọc stopword tiếng Anh
        tokens = nltk.word_tokenize(text)
        # Tạm thời bỏ lọc stopword để máy học cả cụm từ tiếng Việt
        return [PorterStemmer().stem(t) for t in tokens]

    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in dictionary: features[dictionary.index(t)] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    model = GaussianNB()
    model.fit(X, y)
    return model, dictionary, le, preprocess, get_feats

model, dictionary, le, preprocess_fn, feat_fn = load_trained_model()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn (Anh/Việt):", height=200, placeholder="Nhập tin nhắn cần kiểm tra tại đây...")

with col2:
    st.subheader("📊 Thông số học máy")
    st.success("Mô hình: Naive Bayes và cơ sở dữ liệu vector")

if st.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
    if user_input:
        with st.spinner('Đang phân tích xác suất...'):
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # Dự đoán xác suất
            prob = model.predict_proba([features])[0]
            prediction = le.inverse_transform([np.argmax(prob)])[0]
            confidence = max(prob) * 100

            st.markdown("---")
            if prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                st.warning("Lời khuyên: Không nhấn vào bất kỳ đường link nào trong tin nhắn này.")
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                st.balloons()
    else:
        st.error("Vui lòng nhập nội dung!")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179543.png", width=100)
st.sidebar.title("Quản lý Đồ án")
st.sidebar.write("**Sinh viên:** Huỳnh Lê Hoàng Yến 022101091")  
st.sidebar.write("**Sinh viên:** Phạm Minh Tuấn 022101006")  
st.sidebar.write("**Sinh viên:** Huỳnh Văn Đăng Khoa 022101111")
