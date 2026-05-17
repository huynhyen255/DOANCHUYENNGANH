import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import re
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
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
st.write("Đồ án Chuyên Ngành - Hệ thống Phân loại Đa ngôn ngữ (Anh - Việt) tích hợp Naive Bayes & Vector DB")
st.write("---")

# --- HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN SONG SONG ---
@st.cache_resource
def load_trained_model():
    df = pd.read_csv("2cls_spam_text_cls.csv")
    ps = PorterStemmer()
    
    def preprocess(text):
        # Chuyển chữ thường và giữ lại các ký tự tiếng Việt có dấu + chữ tiếng Anh + số
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text)
        tokens = nltk.word_tokenize(text)
        
        # Kết hợp Stopwords Anh - Việt rác hay gặp
        stop_words = set(stopwords.words("english"))
        vi_stopwords = {'và', 'với', 'là', 'thì', 'mà', 'bị', 'được', 'cho', 'của', 'các', 'này', 'trong', 'để'}
        stop_words = stop_words.union(vi_stopwords)
        
        tokens = [t for t in tokens if t not in stop_words]
        return [ps.stem(t) for t in tokens]

    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

    # Bổ sung các từ khóa kích hoạt nhận diện Spam tiếng Việt phổ biến vào từ điển nền
    vi_spam_keywords = ['vay', 'von', 'nhan', 'thuong', 'ca', 'cuoc', 'uu', 'dai', 'tai', 'khoan', 'rut', 'tien', 'lua', 'dao', 'trieu', 'qua', 'tang']
    for kw in vi_spam_keywords:
        if kw not in dictionary:
            dictionary.append(kw)

    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in dictionary: 
                features[int(dictionary.index(t))] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # 1. Khởi tạo & Huấn luyện Naive Bayes
    model = GaussianNB()
    model.fit(X, y)
    
    # 2. Khởi tạo & Thiết lập index cho Cơ sở dữ liệu Vector (KNN Cosine)
    knn_vector_db = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_vector_db.fit(X)
    
    all_texts = df["Message"].astype(str).values
    all_labels = df["Category"].astype(str).values
    
    return model, knn_vector_db, dictionary, le, preprocess, get_feats, all_texts, all_labels

# Khởi tạo toàn bộ pipeline hệ thống
try:
    model, knn_vector_db, dictionary, le, preprocess_fn, feat_fn, all_texts, all_labels = load_trained_model()
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi huấn luyện mô hình: {e}")
    st.stop()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra (Hỗ trợ cả Tiếng Anh và Tiếng Việt):", height=180, placeholder="Dán nội dung tin nhắn cần quét tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH")

with col2:
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Mô hình: Song hành Naive Bayes & Vector DB")
    st.info(f"📁 Tổng từ vựng từ điển: {len(dictionary)} từ")

# --- XỬ LÝ KẾT QUẢ TÍCH HỢP ĐA NGÔN NGỮ ---
if btn_click:
    if user_input.strip():
        with st.spinner('Hệ thống đang thực thi phân tích chuỗi văn bản...'):
            # 1. Chạy tầng kiểm tra heuristic bằng biểu thức chính quy (Regex) dành riêng cho Spam tiếng Việt
            # Giúp hệ thống bắt cứng ngay lập tức các tin nhắn lừa đảo tài chính, cờ bạc bằng tiếng Việt
            vietnamese_spam_pattern = re.compile(
                r'(vay\s+vốn|nhận\s+thưởng|quà\s+tặng|trúng\s+thưởng|lì\s+xì|cá\s+cược|tài\s+khoản\s+bị|khóa\s+thẻ|kiếm\s+tiền\s+online|nhấp\s+vào\s+liên\s+kết|nhan\s+thuong|trung\s+thuong|vay\s+von|ca\s+cuoc)', 
                re.IGNORECASE
            )
            
            is_vi_spam_rule = bool(vietnamese_spam_pattern.search(user_input))

            # 2. Trích xuất đặc trưng toán học của câu đầu vào thông qua NLP nâng cấp
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # --- PHẦN A: TÍNH TOÁN XÁC SUẤT BẰNG NAIVE BAYES ---
            prob = model.predict_proba([features])[0]
            nb_pred_idx = int(np.argmax(prob)) 
            raw_prediction = le.inverse_transform([nb_pred_idx])[0]
            confidence = float(prob[nb_pred_idx]) * 100
            
            # --- PHẦN B: TRUY VẤN KHÔNG GIAN VECTOR (VECTOR DB) ---
            distances, indices = knn_vector_db.kneighbors([features], n_neighbors=3)
            
            knn_labels = []
            similar_samples = []
            
            for i in range(3):
                idx = int(indices[0][i]) 
                similarity_score = 1 - float(distances[0][i]) 
                knn_labels.append(all_labels[idx].lower())
                similar_samples.append({
                    "text": all_texts[idx],
                    "label": all_labels[idx],
                    "score": similarity_score
                })
            
            knn_prediction = max(set(knn_labels), key=knn_labels.count)
            
            # --- PHẦN C: ĐA TẦNG QUYẾT ĐỊNH LAI (HYBRID DECISION) ---
            # Nếu dính luật từ khóa tiếng Việt HOẶC Naive Bayes báo spam HOẶC Vector DB báo spam -> Kết luận SPAM
            if is_vi_spam_rule:
                final_prediction = 'spam'
                confidence = 100.0  # Định danh tuyệt đối bằng bộ lọc Heuristic tiếng Việt
            elif confidence < 90 or len(user_input.strip()) < 8:
                final_prediction = 'ham'
            else:
                if raw_prediction.lower() == 'spam' or knn_prediction == 'spam':
                    final_prediction = 'spam'
                else:
                    final_prediction = 'ham'

            # --- HIỂN THỊ KẾT QUẢ GIAO DIỆN ---
            st.markdown("---")
            if final_prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN SPAM / LỪA ĐẢO ({confidence:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN HỢP LỆ ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
            
            # Khối hiển thị log chi tiết
            st.write("")
            with st.expander("🔍 Chi tiết phân tích kỹ thuật của mô hình lai (Hybrid Pipeline)"):
                st.markdown(f"**Kết quả kiểm tra từng tầng toán học:**")
                st.write(f"- Khớp quy tắc Heuristic Tiếng Việt: `{is_vi_spam_rule}`")
                st.write(f"- Nhãn dự đoán của Naive Bayes: `{raw_prediction.upper()}`")
                st.write(f"- Nhãn biểu quyết của Cơ sở dữ liệu Vector: `{knn_prediction.upper()}`")
                st.write(f"**Top 3 tin nhắn tương đồng trong Database nền tiếng Anh:**")
                for rank, sample in enumerate(similar_samples, 1):
                    st.caption(f"*{rank}. Nhãn gốc:* `{sample['label'].upper()}` | *Độ khớp hình học:* `{sample['score']:.4f}` -> *Nội dung:* \"{sample['text']}\"")
    else:
        st.error("Vui lòng nhập nội dung tin nhắn trước!")

# --- THANH BÊN (Sidebar) ---
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown(f"""
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
---
*Đồ án Chuyên ngành ĐH CNTT*
""")
