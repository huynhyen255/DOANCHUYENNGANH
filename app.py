import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
import csv  # <-- SỬA LỖI: Bổ sung thư viện csv để xử lý quotechar cho Pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- FIX LỖI NLTK TRÊN STREAMLIT CLOUD ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

if nltk_data_path not in nltk.data.path:
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
st.write("Đồ án Chuyên Ngành")

# --- HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_resource
def load_trained_model():
    # SỬA LỖI: Đọc file an toàn với đầy đủ tham số cấu hình tránh tràn cột
    df = pd.read_csv(
        "2cls_spam_text_cls.csv",
        quotechar='"',                 
        quoting=csv.QUOTE_MINIMAL,     
        on_bad_lines='skip',           
        encoding='utf-8'
    )
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    def preprocess(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        return [ps.stem(t) for t in tokens]

    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))
    
    # TỐI ƯU: Chuyển sang dạng dict tìm kiếm O(1) tăng tốc độ map đặc trưng gấp 10 lần
    word_to_idx = {word: i for i, word in enumerate(dictionary)}

    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in word_to_idx: 
                features[word_to_idx[t]] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    model = GaussianNB()
    model.fit(X, y)
    
    return model, dictionary, le, preprocess, get_feats

# Khởi tạo model an toàn
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
    st.success("🤖 Mô hình: Naive Bayes (Bag-of-Words)")
    st.info(f"📁 Từ điển: {len(dictionary)} từ vựng")

# --- XỬ LÝ KẾT QUẢ ---
if btn_click:
    if user_input.strip():
        with st.spinner('Hệ thống đang phân tích...'):
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            prob = model.predict_proba([features])[0]
            
            # Lấy vị trí lớp có xác suất cao nhất
            pred_idx = np.argmax(prob)
            raw_prediction = le.inverse_transform([pred_idx])[0]
            confidence = prob[pred_idx] * 100
            
            # SỬA LOGIC LỌC NGƯỠNG: Giữ nguyên kết quả mô hình, tránh ép uổng tin nhắn ngắn thành hợp lệ
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
st.sidebar.markdown("""
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
""")
