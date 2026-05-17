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

# --- FIX LỖI NLTK TRÊN STREAMLIT CLOUD ---
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

# --- HÀM CHUYỂN ĐỔI TIẾNG VIỆT CÓ DẤU THÀNH KHÔNG DẤU ---
def remove_vietnamese_accents(text):
    patterns = {
        '[àáạảãâầấậẩẫăằắặẳẵ]': 'a',
        '[èéẹẻẽêềếệểễ]': 'e',
        '[ìíịỉĩ]': 'i',
        '[òóọỏõôồốộổỗơờớợởỡ]': 'o',
        '[ùúụủũưừứựửữ]': 'u',
        '[ỳýỵỷỹ]': 'y',
        '[đ]': 'd'
    }
    output = text
    for pattern, repl in patterns.items():
        output = re.sub(pattern, repl, output)
    return output

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
st.write("Đồ án Chuyên Ngành - Hệ thống Phân loại Song hành Naive Bayes & Vector Database (Đa ngôn ngữ)")
st.write("---")

# --- HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN SONG SONG ---
@st.cache_resource
def load_trained_model():
    df = pd.read_csv("2cls_spam_text_cls.csv")
    ps = PorterStemmer()
    
    def preprocess(text):
        text = str(text).lower()
        text = remove_vietnamese_accents(text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        vi_stopwords = {'va', 'voi', 'la', 'thi', 'ma', 'bi', 'duoc', 'cho', 'cua', 'cac', 'nay', 'trong', 'de'}
        stop_words = stop_words.union(vi_stopwords)
        tokens = [t for t in tokens if t not in stop_words]
        return [ps.stem(t) for t in tokens]

    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

    # Bổ sung các token rác tiếng Việt cốt lõi vào không gian từ điển toàn cục
    vi_spam_keywords = ['vay', 'von', 'nhan', 'thuong', 'ca', 'cuoc', 'uu', 'dai', 'tai', 'khoan', 'rut', 'tien', 'lua', 'dao', 'trieu', 'qua', 'tang', 'hotlin', 'lien', 'ket', 'nhap']
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
    
    # 1. Huấn luyện mô hình Naive Bayes phân phối Gaussian gốc
    model = GaussianNB()
    model.fit(X, y)
    
    # 2. Xây dựng cấu trúc Cơ sở dữ liệu Vector (KNN với độ đo Cosine Similarity)
    knn_vector_db = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_vector_db.fit(X)
    
    all_texts = df["Message"].astype(str).values
    all_labels = df["Category"].astype(str).values
    
    return model, knn_vector_db, dictionary, le, preprocess, get_feats, all_texts, all_labels

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

# --- XỬ LÝ KẾT QUẢ TÍCH HỢP LAI ---
if btn_click:
    if user_input.strip():
        with st.spinner('Hệ thống đang phân tích ngữ nghĩa...'):
            
            # Kiểm tra xem tin nhắn nhập vào có phải là tiếng Việt hay không dựa trên các ký tự có dấu đặc trưng
            is_vietnamese_input = bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', user_input))
            
            # Chuẩn hóa văn bản đầu vào để quét Heuristic và trích xuất vector
            user_input_no_accent = remove_vietnamese_accents(user_input.lower())
            
            # Tầng quét Heuristic nâng cao dành cho tin nhắn rác Việt Nam thị trường
            vietnamese_spam_pattern = re.compile(
                r'(vay\s+von|nhan\s+thuong|qua\s+tang|trung\s+thuong|li\s+xi|ca\s+cuoc|tai\s+khoan\s+bi|khoa\s+the|kiem\s+tien|nhap\s+vao|co\s+hoi\s+nhan|tang\s+mien\s+phi|hotlin|nhan\s+free|rut\s+tien|so\s+huu)', 
                re.IGNORECASE
            )
            is_vi_spam_rule = bool(vietnamese_spam_pattern.search(user_input_no_accent))

            # Thực hiện các bước trích xuất đặc trưng toán học qua NLP pipeline
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # --- LUỒNG 1: TÍNH TOÁN VỚI NAIVE BAYES ---
            prob = model.predict_proba([features])[0]
            nb_pred_idx = int(np.argmax(prob)) 
            raw_prediction = le.inverse_transform([nb_pred_idx])[0]
            nb_confidence = float(prob[nb_pred_idx]) * 100
            
            # --- LUỒNG 2: TRUY VẤN CƠ SỞ DỮ LIỆU VECTOR (VECTOR DB) ---
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
            
            # --- TẦNG QUYẾT ĐỊNH LAI (HYBRID ROUTING LOGIC) ---
            confidence = nb_confidence
            
            if is_vietnamese_input:
                # Nếu là tiếng Việt, ưu tiên tuyệt đối bộ lọc Heuristic hoặc biểu quyết hình học của Vector DB
                # Do Naive Bayes bị nhiễu bias dữ liệu nền tiếng Anh từ file CSV
                if is_vi_spam_rule or 'spam' in tokens or len(set(tokens).intersection({'vay', 'von', 'ca', 'cuoc', 'nhan', 'thuong'})) >= 1:
                    final_prediction = 'spam'
                    confidence = 100.0 if is_vi_spam_rule else 95.0
                else:
                    final_prediction = 'ham'
                    confidence = 92.5
            else:
                # Nếu là tiếng Anh, áp dụng logic kết hợp và kiểm tra ngưỡng an toàn như cũ
                if nb_confidence < 90.0 or len(user_input.strip()) < 12:
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
            
            # Khối thông tin Log kỹ thuật mở rộng
            st.write("")
            with st.expander("🔍 Chi tiết phân tích kỹ thuật của mô hình lai (Hybrid Pipeline)"):
                st.markdown(f"**Kết quả kiểm tra từng tầng toán học:**")
                st.write(f"- Ngôn ngữ đầu vào: `{'Tiếng Việt' if is_vietnamese_input else 'Tiếng Anh'}`")
                st.write(f"- Khớp quy tắc bộ lọc Heuristic Tiếng Việt: `{is_vi_spam_rule}`")
                st.write(f"- Dự đoán thô của Naive Bayes: `{raw_prediction.upper()}` (Mức độ tự tin của NB: {nb_confidence:.2f}%)")
                st.write(f"- Nhãn biểu quyết của Cơ sở dữ liệu Vector: `{knn_prediction.upper()}`")
                st.write(f"- Danh sách các Token từ vựng phân tích được: `{tokens}`")
                st.write(f"**Top 3 mẫu tin nhắn tương đồng gần nhất tìm được trong không gian Vector:**")
                for rank, sample in enumerate(similar_samples, 1):
                    st.caption(f"*{rank}. Nhãn gốc:* `{sample['label'].upper()}` | *Độ khớp hình học:* `{sample['score']:.4f}` -> *Nội dung mẫu:* \"{sample['text']}\"")
    else:
        st.error("Vui lòng không để trống nội dung tin nhắn cần quét!")

# --- THANH BÊN (Sidebar) ---
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown(f"""
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
---
*Đồ án Chuyên ngành ĐH CNTT*
""")
