import streamlit as st
import pandas as pd
import numpy as np
import re
import os

# --- THƯ VIỆN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP) ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- THƯ VIỆN HỌC MÁY & VECTOR DATABASE (CỦA CÔ) ---
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors # Để xây dựng Cơ sở dữ liệu Vector và KNN

# Tải tài nguyên NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN STREAMLIT CHUẨN BÁO CÁO
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Message Classification using Naive Bayes and Vector Database",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Đồ án: Phân loại tin nhắn Spam sử dụng Naive Bayes và Cơ sở dữ liệu Vector")
st.caption("Giảng viên hướng dẫn: ThS. Phạm Ngọc Giàu | Sinh viên thực hiện: Huỳnh Lê Hoàng Yến - Huỳnh Văn Đăng Khoa - Phạm Minh Tuấn")
st.write("---")

# ---------------------------------------------------------------------------
# PHẦN II.4: TIỀN XỬ LÝ DỮ LIỆU 
# ---------------------------------------------------------------------------
def preprocess_text(text):
    # 1. Lowercase (Chuyển về chữ thường)
    text = str(text).lower()
    # 2. Xóa ký tự đặc biệt, giữ lại ký tự chữ, số và khoảng trắng (hỗ trợ Tiếng Việt và Tiếng Anh)
    text = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text)
    # 3. Tokenization (Tách từ)
    tokens = word_tokenize(text)
    # 4. Stopwords (Loại bỏ từ dừng)
    stop_words = set(stopwords.words('english'))
    vietnamese_stopwords = {'và', 'với', 'là', 'thì', 'mà', 'bị', 'được', 'cho', 'của', 'các', 'này'}
    stop_words = stop_words.union(vietnamese_stopwords)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    # 5. Stemming (Trích xuất gốc từ) bằng PorterStemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    return " ".join(stemmed_tokens)

# ---------------------------------------------------------------------------
# HÀM HUẤN LUYỆN VÀ TẠO PIPELINE TỰ ĐỘNG (Streamlit Cache để tối ưu hệ thống)
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_system():
    file_path = '2cls_spam_text_cls.csv'
    if not os.path.exists(file_path):
        return None, "Thiếu file '2cls_spam_text_cls.csv' trong thư mục gốc!"
    
    try:
        # Đọc dữ liệu thô
        df = pd.read_csv(file_path)
        # Thực hiện tiền xử lý đồng bộ văn bản
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # === PHẦN II: XÂY DỰNG MÔ HÌNH NAIVE BAYES ===
        cv = CountVectorizer()
        X_nb = cv.fit_transform(df['processed_text']).toarray()
        y_nb = df['label'].values
        
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)
        nb_model = MultinomialNB()
        nb_model.fit(X_train_nb, y_train_nb)
        nb_acc = accuracy_score(y_test_nb, nb_model.predict(X_test_nb))
        
        # === PHẦN III: XÂY DỰNG CƠ SỞ DỮ LIỆU VECTOR & KNN ===
        # Sử dụng TfidfVectorizer làm Mô hình Embedding toán học như cô định hướng
        tfidf = TfidfVectorizer()
        X_vector_db = tfidf.fit_transform(df['processed_text']).toarray()
        
        # Khởi tạo mô hình định tuyến không gian láng giềng NearestNeighbors (Vector Database Index)
        knn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_index.fit(X_vector_db)
        
        return {
            "cv": cv, "nb_model": nb_model, "nb_acc": nb_acc,
            "tfidf": tfidf, "knn_index": knn_index, "df": df
        }, "Hệ thống kết hợp Naive Bayes & Vector Database khởi tạo thành công!"
    except Exception as e:
        return None, f"Lỗi xử lý hệ thống: {str(e)}"

# Chạy kích hoạt lõi hệ thống
system_core, status_msg = initialize_system()

# Tách giao diện thành 2 phần tách biệt
layout_left, layout_right = st.columns([1, 2])

with layout_left:
    st.header("📊 Trạng thái mô hình học máy")
    if system_core is None:
        st.error(status_msg)
    else:
        st.success(status_msg)
        st.metric(label="Độ chính xác mô hình Naive Bayes", value=f"{system_core['nb_acc']*100:.2f}%")
        
        st.info(
            "**Cấu trúc kỹ thuật hệ thống:**\n"
            "- **Mô hình 1:** Multinomial Naive Bayes (Phần II)\n"
            "- **Mô hình 2:** Vector Database Index + KNN Cosine Distance (Phần III)\n"
            "- **Quy trình NLP:** Tokenize -> Stopwords -> PorterStemmer"
        )

with layout_right:
    st.header("🔍 Hệ thống kiểm thử & Phân loại trực quan")
    user_input = st.text_area("Nhập nội dung tin nhắn cần phân loại (Hỗ trợ cả Tiếng Anh và Tiếng Việt):", height=120, 
                              placeholder="Ví dụ: FREE!! Click here to win $1000 NOW! Limited time offer!...")
    
    # Cho phép người dùng tùy chọn k (Số lượng láng giềng gần nhất trong cơ sở dữ liệu Vector - chuẩn III.8)
    k_neighbors = st.slider("Chọn số lượng láng giềng k tra cứu trong Vector DB (Phần III):", min_value=1, max_value=5, value=3)
    
    if st.button("🚀 Thực thi Pipeline phân loại kết hợp", type="primary"):
        if not user_input.strip():
            st.warning("Vui lòng nhập nội dung văn bản cần kiểm tra!")
        elif system_core is None:
            st.error("Hệ thống chưa sẵn sàng!")
        else:
            # Lấy các thành phần từ lõi hệ thống ra để xử lý
            cv = system_core["cv"]
            nb_model = system_core["nb_model"]
            tfidf = system_core["tfidf"]
            knn_index = system_core["knn_index"]
            df = system_core["df"]
            
            # 1. TIỀN XỬ LÝ VĂN BẢN ĐẦU VÀO
            processed_input = preprocess_text(user_input)
            
            st.write("---")
            st.subheader("📊 Kết quả phân tích chi tiết từ hệ thống pipeline:")
            
            # =======================================================================
            # KIỂM TRA MÔ HÌNH 1: NAIVE BAYES (PHẦN II.8)
            # =======================================================================
            vectorized_nb = cv.transform([processed_input]).toarray()
            nb_prediction = nb_model.predict(vectorized_nb)[0]
            nb_proba = nb_model.predict_proba(vectorized_nb)[0]
            labels_mapping = nb_model.classes_
            pred_index = np.where(labels_mapping == nb_prediction)[0][0]
            nb_confidence = nb_proba[pred_index] * 100
            
            # =======================================================================
            # KIỂM TRA MÔ HÌNH 2: VECTOR DATABASE & KNN (PHẦN III.8 & III.9 Y chang cô)
            # =======================================================================
            # Thực hiện Embedding văn bản đầu vào thành Vector đặc trưng hình học
            vectorized_vector_db = tfidf.transform([processed_input]).toarray()
            
            # Thực hiện truy vấn khoảng cách Cosine Distance tìm k láng giềng gần nhất trong DB
            distances, indices = knn_index.kneighbors(vectorized_vector_db, n_neighbors=k_neighbors)
            
            # Trích xuất nhãn và nội dung của các láng giềng tương đồng nhất
            neighbors_results = []
            for i in range(k_neighbors):
                idx = indices[0][i]
                score = 1 - distances[0][i] # Độ tương đồng Cosine Similarity
                neighbors_results.append({
                    "label": df.iloc[idx]['label'],
                    "text": df.iloc[idx]['text'],
                    "score": score
                })
            
            # Biểu quyết số đông nhãn từ danh sách láng giềng (KNN Voting Logic)
            neighbor_labels = [n["label"] for n in neighbors_results]
            knn_prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            
            # =======================================================================
            # ĐÁNH GIÁ CHUNG VÀ HIỂN THỊ GIAO DIỆN
            # =======================================================================
            tab1, tab2, tab3 = st.tabs(["🛡️ Kết quả tổng hợp", "📐 Chi tiết Vector Database (KNN)", "📝 Nhật ký chuỗi NLP thô"])
            
            with tab1:
                st.write("#### 🔹 Dự đoán từ Mô hình Học máy Naive Bayes:")
                if nb_prediction == "Spam":
                    st.error(f"🚨 **Dán nhãn: SPAM** (Độ tự tin thuật toán: {nb_confidence:.2f}%)")
                else:
                    st.success(f"✅ **Dán nhãn: HAM (An toàn)** (Độ tự tin thuật toán: {nb_confidence:.2f}%)")
                
                st.write("#### 🔹 Dự đoán từ Không gian Cơ sở dữ liệu Vector:")
                if knn_prediction == "Spam":
                    st.error(f"🚨 **Dán nhãn: SPAM** (Dựa theo biểu quyết {k_neighbors} mẫu tương đồng nhất)")
                else:
                    st.success(f"✅ **Dán nhãn: HAM (An toàn)** (Dựa theo biểu quyết {k_neighbors} mẫu tương đồng nhất)")
            
            with tab2:
                st.write(f"#### 🔍 Top {k_neighbors} dữ liệu Vector tương đồng nhất trong Database (Top Neighbors):")
                for rank, neighbor in enumerate(neighbors_results, 1):
                    with st.container():
                        st.write(f"**Top {rank}. Nhãn gốc trong DB: `{neighbor['label']}` | Độ tương đồng hình học (Score): `{neighbor['score']:.4f}`**")
                        st.caption(f"Nội dung mẫu văn bản gốc: *\"{neighbor['text']}\"*")
                        st.write("---")
            
            with tab3:
                st.write("#### 👁️ Chuỗi văn bản sau khi hoàn thành tiền xử lý NLP Pipeline thô:")
                st.code(processed_input, language="text")
