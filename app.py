import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# --- FIX LỖI NLTK TRÊN SERVER / STREAMLIT CLOUD ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

def download_nltk_resources():
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.error(f"Lỗi tải tài nguyên NLTK: {e}")

download_nltk_resources()

# ---------------------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Phân loại tin nhắn Spam", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Đồ án Chuyên ngành: Phân loại tin nhắn Spam")
st.write("Sử dụng giải thuật Naive Bayes kết hợp Cơ sở dữ liệu Vector (Hỗ trợ Anh - Việt)")
st.write("---")

# ---------------------------------------------------------------------------
# HÀM TIỀN XỬ LÝ VĂN BẢN (Chuẩn NLP hỗ trợ Tiếng Việt)
# ---------------------------------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    # Giữ lại chữ cái Tiếng Việt có dấu, chữ tiếng Anh và số
    text = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text)
    tokens = word_tokenize(text)
    
    # Kết hợp từ dừng Anh - Việt
    stop_words = set(stopwords.words('english'))
    vietnamese_stopwords = {'và', 'với', 'là', 'thì', 'mà', 'bị', 'được', 'cho', 'của', 'các', 'này', 'trong', 'để'}
    stop_words = stop_words.union(vietnamese_stopwords)
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    stemmer = PorterStemmer()
    # Stemming cho từ tiếng Anh, từ tiếng Việt giữ nguyên
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    return " ".join(stemmed_tokens)

# ---------------------------------------------------------------------------
# LÕI HỆ THỐNG: HUẤN LUYỆN MODEL & THIẾT LẬP VECTOR DATABASE (ĐÃ BỎ CACHE)
# ---------------------------------------------------------------------------
def load_model_pipeline_fresh():  
    file_path = '2cls_spam_text_cls.csv'
    if not os.path.exists(file_path):
        return None, f"Không tìm thấy file '{file_path}' trong thư mục dự án!"
    
    try:
        df = pd.read_csv(file_path, quotechar='"', quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip', encoding='utf-8')
        
        # Nhận diện và đồng bộ hóa tên cột động linh hoạt
        if 'Category' in df.columns and 'Message' in df.columns:
            df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)
        elif 'label' not in df.columns or 'text' not in df.columns:
            df.columns = ['label', 'text'] + list(df.columns[2:])
            
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Trích xuất mảng numpy độc lập tách rời khỏi DataFrame để chống lỗi chỉ mục
        all_labels = df['label'].astype(str).values
        all_texts = df['text'].astype(str).values
        
        # === PHẦN 1: MÔ HÌNH NAIVE BAYES ===
        cv = CountVectorizer()
        X_nb = cv.fit_transform(df['processed_text']).toarray()
        y_nb = all_labels
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)
        
        nb_model = MultinomialNB()
        nb_model.fit(X_train_nb, y_train_nb)
        nb_acc = accuracy_score(y_test_nb, nb_model.predict(X_test_nb))
        
        # === PHẦN 2: CƠ SỞ DỮ LIỆU VECTOR (TF-IDF Embedder & KNN) ===
        tfidf = TfidfVectorizer()
        X_vector_db = tfidf.fit_transform(df['processed_text']).toarray()
        
        knn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_index.fit(X_vector_db)
        
        return {
            "cv": cv, "nb_model": nb_model, "nb_acc": nb_acc,
            "tfidf": tfidf, "knn_index": knn_index, 
            "all_labels": all_labels, "all_texts": all_texts, "total_records": len(df)
        }, "Hệ thống Naive Bayes & Vector Database khởi tạo thành công!"
    except Exception as e:
        return None, f"Lỗi nạp hệ thống: {str(e)}"

# Kích hoạt lõi công nghệ mới (Chạy tươi mới hoàn toàn)
system_core, status_msg = load_model_pipeline_fresh()

# ---------------------------------------------------------------------------
# BỐ CỤC GIAO DIỆN
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra (Hỗ trợ cả Tiếng Anh và Tiếng Việt):", 
                              height=180, placeholder="Dán nội dung tin nhắn hoặc đoạn hội thoại tại đây...")
    k_neighbors = st.slider("Số lượng láng giềng k truy vấn không gian Vector DB:", min_value=1, max_value=5, value=3)
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH PIPELINE")

with col2:
    st.subheader("📊 Thông số hệ thống")
    if system_core is None:
        st.error(status_msg)
    else:
        st.success("🤖 Trạng thái: Hệ thống đã sẵn sàng")
        st.metric(label="Độ chính xác Naive Bayes (Test)", value=f"{system_core['nb_acc']*100:.2f}%")
        st.info(f"📁 Bản ghi nền: {system_core['total_records']} tin nhắn dữ liệu")

# ---------------------------------------------------------------------------
# XỬ LÝ KẾT QUẢ KHI BẤM NÚT
# ---------------------------------------------------------------------------
if btn_click:
    if not user_input.strip():
        st.error("Vui lòng không để trống nội dung văn bản cần kiểm tra!")
    elif system_core is None:
        st.error("Hệ thống chưa thiết lập thành công dữ liệu nền!")
    else:
        with st.spinner('Đang chạy thực thi quy trình phân loại kết hợp...'):
            cv = system_core["cv"]
            nb_model = system_core["nb_model"]
            tfidf = system_core["tfidf"]
            knn_index = system_core["knn_index"]
            all_labels = system_core["all_labels"]
            all_texts = system_core["all_texts"]
            
            # 1. Tiền xử lý NLP đầu vào
            processed_input = preprocess_text(user_input)
            
            # KHỞI TẠO TẬP TRUNG TẤT CẢ BIẾN ĐỂ TRÁNH LỖI PHÂN NHÁNH TRÊN GIAO DIỆN
            final_decision = "ham"
            nb_prediction_str = "ham"
            knn_prediction_str = "ham"
            nb_confidence = 100.0
            neighbors_results = []
            
            # 2. Xử lý bộ lọc tin nhắn quá ngắn
            if len(user_input.strip()) < 8:
                st.warning("⚠️ Nhận diện: Tin nhắn quá ngắn. Hệ thống đưa vào vùng an toàn.")
            else:
                # === TIẾN TRÌNH 1: DỰ ĐOÁN XÁC SUẤT NAIVE BAYES ===
                vectorized_nb = cv.transform([processed_input]).toarray()
                nb_pred = nb_model.predict(vectorized_nb)[0]
                nb_prediction_str = str(nb_pred).strip().lower()
                
                # Trích xuất chỉ số an toàn từ mảng thuần Python
                classes_list = [str(c).strip().lower() for c in nb_model.classes_]
                nb_proba = nb_model.predict_proba(vectorized_nb)[0]
                
                if nb_prediction_str in classes_list:
                    pred_index = classes_list.index(nb_prediction_str)
                    nb_confidence = nb_proba[pred_index] * 100
                else:
                    nb_confidence = 50.0
                
                # === TIẾN TRÌNH 2: TRUY VẤN CƠ SỞ DỮ LIỆU VECTOR ===
                vectorized_vector_db = tfidf.transform([processed_input]).toarray()
                distances, indices = knn_index.kneighbors(vectorized_vector_db, n_neighbors=k_neighbors)
                
                for i in range(k_neighbors):
                    idx = indices[0][i]
                    score = 1 - distances[0][i] # Cosine Similarity
                    neighbors_results.append({
                        "label": str(all_labels[idx]).strip(),
                        "text": str(all_texts[idx]).strip(),
                        "score": score
                    })
                
                # Biểu quyết số đông từ không gian Vector
                neighbor_labels = [n["label"].lower() for n in neighbors_results]
                if neighbor_labels:
                    knn_prediction_str = max(set(neighbor_labels), key=neighbor_labels.count)
                
                # MA TRẬN BIỂU QUYẾT TỔNG HỢP PIPELINE
                if nb_prediction_str == "spam" or knn_prediction_str == "spam":
                    final_decision = "spam"
                else:
                    final_decision = "ham"
            
            # --- HIỂN THỊ HỘP KẾT QUẢ MÀU SẮC ĐẸP ---
            st.markdown("---")
            if final_decision == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">🚨 CẢNH BÁO: TIN NHẮN SPAM / RÁC</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN HỢP LỆ (HAM)</div>', unsafe_allow_html=True)
                st.balloons()
            
            # --- HIỂN THỊ CHI TIẾT TABS ---
            tab1, tab2 = st.tabs(["📐 Chi tiết Thống kê toán học", "📝 Nhật ký chuỗi sau NLP"])
            with tab1:
                st.write(f"- Dự đoán Naive Bayes: **`{nb_prediction_str.upper()}`** (Độ tự tin: {nb_confidence:.2f}%)")
                st.write(f"- Biểu quyết Vector DB: **`{knn_prediction_str.upper()}`**")
                if len(neighbors_results) > 0:
                    st.write(f"**Top mẫu tương đồng nhất trong Cơ sở dữ liệu Vector:**")
                    for r, n in enumerate(neighbors_results, 1):
                        st.caption(f"{r}. Nhãn: `{n['label'].upper()}` | Độ khớp hình học: {n['score']:.4f} -> Txt: *\"{n['text']}\"*")
            with tab2:
                st.write("Mã Token sau khi làm sạch chữ thường, xóa kí tự đặc biệt và lọc Stopwords:")
                st.code(processed_input if processed_input.strip() else "[Chuỗi trống]", language="text")

# ---------------------------------------------------------------------------
# THANH BÊN (Sidebar)
# ---------------------------------------------------------------------------
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown("""
* **Giảng viên hướng dẫn:** ThS. Phạm Ngọc Giàu
* **Thành viên thực hiện:**
  - Huỳnh Lê Hoàng Yến - `022101091`
  - Phạm Minh Tuấn - `022101006`
  - Huỳnh Văn Đăng Khoa - `022101111`
* **Lớp:** ĐH CNTT22B
""")
