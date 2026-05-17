import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
from sklearn.naive_bayes import MultinomialNB
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
        st.error(f"Lỗi tải tài nguyên NLTK: {e}")

download_nltk_resources()

# --- THIẾT KẾ GIAO DIỆN MÀU SẮC CHUYÊN NGHIỆP ---
st.set_page_config(page_title="Phân loại tin nhắn Spam", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 26px; font-weight: bold; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành - Công nghệ Học máy kết hợp Tiền xử lý NLP")
st.write("---")

# --- HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH TẠI CHỖ ---
@st.cache_resource
def load_trained_model():
    file_path = "2cls_spam_text_cls.csv"
    if not os.path.exists(file_path):
        return None, None, None, None, None, f"Không tìm thấy file dữ liệu '{file_path}' trong thư mục!"

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        ps = PorterStemmer()
        
        # Hàm tiền xử lý nội bộ văn bản
        def preprocess(text):
            text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
            tokens = nltk.word_tokenize(text)
            stop_words = set(stopwords.words("english"))
            tokens = [t for t in tokens if t not in stop_words]
            return [ps.stem(t) for t in tokens]

        # Xử lý toàn bộ tập dữ liệu nền
        processed_msgs = [preprocess(msg) for msg in df["Message"]]
        # Xây dựng không gian từ điển độc nhất
        dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

        # Hàm trích xuất đặc trưng Bag of Words (Tần suất xuất hiện từ)
        def get_feats(tokens):
            features = np.zeros(len(dictionary), dtype=np.float64)
            for t in tokens:
                if t in dictionary:
                    # Ép kiểu int thuần Python cho vị trí index để phòng ngừa lỗi NumPy Indexing
                    idx = int(dictionary.index(t))
                    features[idx] += 1
            return features

        # Chuyển đổi dữ liệu sang mảng ma trận đầu vào học máy
        X = np.array([get_feats(t) for t in processed_msgs], dtype=np.float64)
        
        le = LabelEncoder()
        y = le.fit_transform(df["Category"])
        
        # Sử dụng MultinomialNB tối ưu hơn rất nhiều cho dữ liệu đếm tần suất (Bag of Words) so với GaussianNB
        model = MultinomialNB()
        model.fit(X, y)
        
        return model, dictionary, le, preprocess, get_feats, "Thành công"
    except Exception as e:
        return None, None, None, None, None, str(e)

# Khởi tạo nạp pipeline lõi hệ thống
model, dictionary, le, preprocess_fn, feat_fn, status_msg = load_trained_model()

if status_msg != "Thành công":
    st.error(f"Đã xảy ra lỗi khi cấu hình hoặc huấn luyện mô hình: {status_msg}")
    st.stop()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra (Hỗ trợ phân tích ngữ nghĩa):", height=180, placeholder="Dán nội dung tin nhắn cần quét tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH")

with col2:
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Mô hình: Naive Bayes nâng cao")
    st.info(f"📁 Kích thước từ điển nền: {len(dictionary)} từ vựng")

# --- XỬ LÝ KẾT QUẢ PHÂN LOẠI PIPELINE ---
if btn_click:
    if user_input.strip():
        with st.spinner('Hệ thống đang trích xuất đặc trưng toán học...'):
            # 1. Pipeline NLP xử lý văn bản đầu vào
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # 2. Tính toán phân phối xác suất lớp tin nhắn
            prob = model.predict_proba([features])[0]
            
            # Ép chỉ số mảng max về kiểu int thuần Python giải quyết triệt để lỗi Scalar Index
            max_idx = int(np.argmax(prob))
            raw_prediction = str(le.inverse_transform([max_idx])[0])
            confidence = float(prob[max_idx]) * 100
            
            # 3. Tầng lọc ngưỡng an toàn phòng chống Overfitting do tin ngắn
            if confidence < 95.0 or len(user_input.strip()) < 12:
                final_prediction = 'ham'
            else:
                final_prediction = raw_prediction

            st.markdown("---")
            # Hiển thị giao diện cảnh báo trực quan
            if final_prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
    else:
        st.error("Vui lòng nhập hoặc dán nội dung tin nhắn trước khi bấm nút kiểm tra!")

# --- THANH BÊN (Sidebar - Thông tin bản quyền Nhóm) ---
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown("""
* **Huỳnh Lê Hoàng Yến** - `022101091`
* **Phạm Minh Tuấn** - `022101006`
* **Huỳnh Văn Đăng Khoa** - `022101111`
---
*Đồ án chuyên ngành ĐH CNTT*
""")
