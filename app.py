import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- FIX LỖI NLTK TRÊN STREAMLIT CLOUD (CÁCH MẠNH TAY) ---
# Tạo thư mục nltk_data ngay tại project để máy chủ không bị lạc đường dẫn
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Ép tải dữ liệu vào đúng thư mục vừa tạo
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        nltk.download('punkt_tab', download_dir=nltk_data_path) # Tải thêm bản tab để phòng hờ
    except Exception as e:
        st.error(f"Lỗi tải NLTK: {e}")

download_nltk_resources()

# --- THIẾT KẾ GIAO DIỆN ---
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 26px; font-weight: bold; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành - Công nghệ Machine Learning")

# --- HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_resource
def load_trained_model():
    df = pd.read_csv("2cls_spam_text_cls.csv")
    ps = PorterStemmer()
    
    def preprocess(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        # Sử dụng hàm tokenize với đường dẫn đã config
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
                features[dictionary.index(t)] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    model = GaussianNB()
    model.fit(X, y)
    
    return model, dictionary, le, preprocess, get_feats

# Khởi tạo model
try:
    model, dictionary, le, preprocess_fn, feat_fn = load_trained_model()
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi huấn luyện mô hình: {e}")
    st.stop()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra:", height=180, placeholder="Dán nội dung tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH")

with col2:
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Mô hình: Gaussian Naive Bayes")
    st.info(f"📁 Từ điển: {len(dictionary)} từ vựng")

# --- XỬ LÝ KẾT QUẢ ---
if btn_click:
    if user_input:
        with st.spinner('Hệ thống đang phân tích...'):
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            prob = model.predict_proba([features])[0]
            raw_prediction = le.inverse_transform([np.argmax(prob)])[0]
            confidence = max(prob) * 100
            
            # Tầng lọc xác suất
            if confidence < 95 or len(user_input.strip()) < 15:
                final_prediction = 'ham'
            else:
                final_prediction = raw_prediction

            st.markdown("---")
            if final_prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
    else:
        st.error("Vui lòng không để trống nội dung!")

# --- THANH BÊN (Sidebar) ---
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown(f"""
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
""")
