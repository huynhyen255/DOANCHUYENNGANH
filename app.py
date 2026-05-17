import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors  # <-- Cơ sở dữ liệu Vector (KNN Cosine)
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- FIX LỖI NLTK TRÊN STREAMLIT CLOUD (CÁCH MẠNH TAY) ---
# Tạo thư mục nltk_data ngay tại project để máy chủ không bị lạc đường dẫn
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Ép tải dữ liệu vào đúng thư mục vừa tạo
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.error(f"Lỗi tải NLTK: {e}")
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        nltk.download('punkt_tab', download_dir=nltk_data_path) # Tải thêm bản tab để phòng hờ
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
                # Ép kiểu int thuần Python để tránh lỗi NumPy Indexing
                features[int(dictionary.index(t))] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # 1. Huấn luyện mô hình Naive Bayes gốc
    model = GaussianNB()
    model.fit(X, y)
    
    # 2. Thiết lập cơ sở dữ liệu Vector (KNN dùng Metric Cosine)
    knn_vector_db = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_vector_db.fit(X)
    
    # Lưu lại tập dữ liệu gốc để phục vụ truy vấn hiển thị không gian vector lân cận
    all_texts = df["Message"].astype(str).values
    all_labels = df["Category"].astype(str).values
    
    return model, knn_vector_db, dictionary, le, preprocess, get_feats, all_texts, all_labels
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
    model, knn_vector_db, dictionary, le, preprocess_fn, feat_fn, all_texts, all_labels = load_trained_model()
    model, dictionary, le, preprocess_fn, feat_fn = load_trained_model()
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi huấn luyện mô hình: {e}")
    st.stop()
    st.error(f"Đã xảy ra lỗi khi huấn luyện mô hình: {e}")
    st.stop()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra:", height=180, placeholder="Dán nội dung tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH")
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra:", height=180, placeholder="Dán nội dung tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH")

with col2:
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Mô hình: Naive Bayes và cơ sở dữ liệu vector")
    st.info(f"📁 Từ điển: {len(dictionary)} từ vựng")
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Mô hình: Naive Bayes và cơ sở dữ liệu vector")
    st.info(f"📁 Từ điển: {len(dictionary)} từ vựng")

# --- XỬ LÝ KẾT QUẢ ---
if btn_click:
    if user_input:
        with st.spinner('Hệ thống đang phân tích...'):
            # Trích xuất đặc trưng câu đầu vào
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # --- LUỒNG 1: DỰ ĐOÁN BẰNG NAIVE BAYES ---
            prob = model.predict_proba([features])[0]
            nb_pred_idx = int(np.argmax(prob)) # Tránh lỗi scalar index bằng ép kiểu int()
            raw_prediction = le.inverse_transform([nb_pred_idx])[0]
            confidence = float(prob[nb_pred_idx]) * 100
            
            # --- LUỒNG 2: TRUY VẤN KHÔNG GIAN VECTOR (VECTOR DB) ---
            distances, indices = knn_vector_db.kneighbors([features], n_neighbors=3)
            
            knn_labels = []
            similar_samples = []
            for i in range(3):
                idx = int(indices[0][i])
                similarity_score = 1 - float(distances[0][i]) # Chuyển từ khoảng cách sang độ tương đồng
                knn_labels.append(all_labels[idx].lower())
                similar_samples.append({
                    "text": all_texts[idx],
                    "label": all_labels[idx],
                    "score": similarity_score
                })
            
            # Lấy kết quả biểu quyết số đông từ Vector DB
            knn_prediction = max(set(knn_labels), key=knn_labels.count)
            
            # --- TẦNG QUYẾT ĐỊNH LAI (HYBRID DECISION) ---
            if confidence < 95 or len(user_input.strip()) < 15:
                final_prediction = 'ham'
            else:
                # Nếu một trong hai bộ lọc (Naive Bayes HOẶC không gian hình học Vector) báo là Spam
                if raw_prediction.lower() == 'spam' or knn_prediction == 'spam':
                    final_prediction = 'spam'
                else:
                    final_prediction = 'ham'

            # --- HIỂN THỊ KẾT QUẢ ---
            st.markdown("---")
            if final_prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
                
            # Log thông tin chi tiết phục vụ báo cáo đồ án
            st.write("")
            with st.expander("🔍 Xem chi tiết log truy vấn Cơ sở dữ liệu Vector (Vector DB)"):
                st.markdown(f"**Kết quả phân tích thuật toán:**")
                st.write(f"- Nhãn dự đoán của Naive Bayes: `{raw_prediction.upper()}` (Độ tự tin: {confidence:.2f}%)")
                st.write(f"- Nhãn biểu quyết của không gian hình học Vector: `{knn_prediction.upper()}`")
                st.write(f"**Top 3 tin nhắn tương đồng gần nhất tìm thấy trong cơ sở dữ liệu nền:**")
                for rank, sample in enumerate(similar_samples, 1):
                    st.caption(f"*{rank}. Nhãn gốc:* `{sample['label'].upper()}` | *Độ khớp:* `{sample['score']:.4f}` -> *Nội dung:* \"{sample['text']}\"")
    else:
        st.error("Vui lòng không để trống nội dung!")

# --- THANH BÊN (Sidebar) ---
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


ode phân loại spam tiếng việt đi# --- THANH BÊN (Sidebar) ---
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown(f"""
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
""")
