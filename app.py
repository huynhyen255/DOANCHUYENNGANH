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
from sklearn.neighbors import NearestNeighbors

# Tải tài nguyên NLTK ẩn
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

st.title("🛡️ Đồ án Chuyên ngành: Phân loại tin nhắn Spam sử dụng Naive Bayes và Cơ sở dữ liệu Vector")
st.write("---")

# ---------------------------------------------------------------------------
# PHẦN II.4: TIỀN XỬ LÝ DỮ LIỆU (Y chang hàm xử lý chuỗi của cô)
# ---------------------------------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    vietnamese_stopwords = {'và', 'với', 'là', 'thì', 'mà', 'bị', 'được', 'cho', 'của', 'các', 'này'}
    stop_words = stop_words.union(vietnamese_stopwords)
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    return " ".join(stemmed_tokens)

# ---------------------------------------------------------------------------
# HÀM HUẤN LUYỆN VÀ TẠO PIPELINE TỰ ĐỘNG
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_system():
    file_path = '2cls_spam_text_cls.csv'
    if not os.path.exists(file_path):
        return None, "Thiếu file '2cls_spam_text_cls.csv' trong thư mục dự án!"
    
    try:
        df = pd.read_csv(file_path)
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # === PHẦN II: NAIVE BAYES ===
        cv = CountVectorizer()
        X_nb = cv.fit_transform(df['processed_text']).toarray()
        y_nb = df['label'].values
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)
        nb_model = MultinomialNB()
        nb_model.fit(X_train_nb, y_train_nb)
        nb_acc = accuracy_score(y_test_nb, nb_model.predict(X_test_nb))
        
        # === PHẦN III: VECTOR DATABASE & KNN ===
        tfidf = TfidfVectorizer()
        X_vector_db = tfidf.fit_transform(df['processed_text']).toarray()
        knn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_index.fit(X_vector_db)
        
        return {
            "cv": cv, "nb_model": nb_model, "nb_acc": nb_acc,
            "tfidf": tfidf, "knn_index": knn_index, "df": df
        }, "Hệ thống kết hợp Naive Bayes & Vector Database khởi tạo thành công!"
    except Exception as e:
        return None, f"Lỗi xử lý hệ thống: {str(e)}"

system_core, status_msg = initialize_system()

# Tách giao diện thành 2 cột: Cột trái (Thông tin nhóm), Cột phải (Phần chạy thử nghiệm)
layout_left, layout_right = st.columns([1, 2])

with layout_left:
    st.header("📊 Thông tin đồ án")
    if system_core is None:
        st.error(status_msg)
    else:
        st.success(status_msg)
        st.metric(label="Độ chính xác mô hình Naive Bayes", value=f"{system_core['nb_acc']*100:.2f}%")
        
        # --- PHẦN THÔNG TIN ĐÃ TÁCH DÒNG VÀ GHI ĐẦY ĐỦ THEO YÊU CẦU ---
        st.markdown("### 👨‍🏫 Giảng viên hướng dẫn:")
        st.write("- **ThS. Phạm Ngọc Giàu**")
        
        st.markdown("### 👥 Sinh viên thực hiện:")
        st.write("- 🧑‍💻 **Huỳnh Lê Hoàng Yến** — MSSV: `022101091` (Lớp: ĐH CNTT22B)")
        st.write("- 🧑‍💻 **Huỳnh Văn Đăng Khoa** — MSSV: `022101111` (Lớp: ĐH CNTT22B)")
        st.write("- 🧑‍💻 **Phạm Minh Tuấn** — MSSV: `022101006` (Lớp: ĐH CNTT22B)")
        
        st.write("---")
        st.info(
            "**Cấu trúc kỹ thuật lõi:**\n"
            "- Mô hình 1: Multinomial Naive Bayes\n"
            "- Mô hình 2: Vector Database + KNN Index\n"
            "- Độ tương đồng hình học: Cosine Distance"
        )

with layout_right:
    st.header("🔍 Hệ thống kiểm thử Pipeline")
    user_input = st.text_area("Nhập nội dung tin nhắn cần phân loại (Hỗ trợ cả Tiếng Anh và Tiếng Việt):", height=120, 
                              placeholder="Ví dụ: FREE!! Click here to win $1000 NOW! Limited time offer!...")
    
    k_neighbors = st.slider("Chọn số lượng láng giềng k tra cứu trong Vector DB (Phần III):", min_value=1, max_value=5, value=3)
    
    if st.button("🚀 Thực thi Pipeline phân loại kết hợp", type="primary"):
        if not user_input.strip():
            st.warning("Vui lòng nhập nội dung văn bản trước khi thực hiện!")
        elif system_core is None:
            st.error("Hệ thống chưa sẵn sàng!")
        else:
            cv = system_core["cv"]
            nb_model = system_core["nb_model"]
            tfidf = system_core["tfidf"]
            knn_index = system_core["knn_index"]
            df = system_core["df"]
            
            # 1. Tiến hành tiền xử lý văn bản đầu vào
            processed_input = preprocess_text(user_input)
            
            st.write("---")
            st.subheader("📊 Kết quả kiểm thử thành phần:")
            
            # =======================================================================
            # MÔ HÌNH 1: NAIVE BAYES (PHẦN II)
            # =======================================================================
            vectorized_nb = cv.transform([processed_input]).toarray()
            nb_prediction = nb_model.predict(vectorized_nb)[0]
            nb_proba = nb_model.predict_proba(vectorized_nb)[0]
            labels_mapping = nb_model.classes_
            pred_index = np.where(labels_mapping == nb_prediction)[0][0]
            nb_confidence = nb_proba[pred_index] * 100
            
            # =======================================================================
            # MÔ HÌNH 2: VECTOR DATABASE & KNN (PHẦN III)
            # =======================================================================
            vectorized_vector_db = tfidf.transform([processed_input]).toarray()
            distances, indices = knn_index.kneighbors(vectorized_vector_db, n_neighbors=k_neighbors)
            
            neighbors_results = []
            for i in range(k_neighbors):
                idx = indices[0][i]
                score = 1 - distances[0][i]
                neighbors_results.append({
                    "label": df.iloc[idx]['label'],
                    "text": df.iloc[idx]['text'],
                    "score": score
                })
            
            neighbor_labels = [n["label"] for n in neighbors_results]
            knn_prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            
            # =======================================================================
            # HIỂN THỊ KẾT QUẢ GIAO DIỆN TABS
            # =======================================================================
            tab1, tab2, tab3 = st.tabs(["🛡️ Dự đoán tổng hợp", "📐 Tra cứu Vector DB", "📝 Chuỗi dữ liệu sau NLP"])
            
            with tab1:
                st.write("#### 🔹 Kết quả từ bộ phân loại Naive Bayes:")
                if nb_prediction == "Spam":
                    st.error(f"🚨 **Dán nhãn: SPAM** (Độ tin cậy: {nb_confidence:.2f}%)")
                else:
                    st.success(f"✅ **Dán nhãn: HAM (An toàn)** (Độ tin cậy: {nb_confidence:.2f}%)")
                
                st.write("#### 🔹 Kết quả biểu quyết từ không gian Vector Database:")
                if knn_prediction == "Spam":
                    st.error(f"🚨 **Dán nhãn: SPAM** (Biểu quyết từ {k_neighbors} mẫu láng giềng gần nhất)")
                else:
                    st.success(f"✅ **Dán nhãn: HAM (An toàn)** (Biểu quyết từ {k_neighbors} mẫu láng giềng gần nhất)")
            
            with tab2:
                st.write(f"#### 🔍 Chi tiết {k_neighbors} mẫu dữ liệu có khoảng cách Vector gần nhất (Top Neighbors):")
                for rank, neighbor in enumerate(neighbors_results, 1):
                    with st.container():
                        st.write(f"**Top {rank}. Nhãn gốc: `{neighbor['label']}` | Độ tương đồng Cosine: `{neighbor['score']:.4f}`**")
                        st.caption(f"Văn bản mẫu: *\"{neighbor['text']}\"*")
                        st.write("---")
            
            with tab3:
                st.write("#### 👁️ Chuỗi ký tự sau khi đi qua Pipeline tiền xử lý văn bản thô (Token sau Stemming):")
                st.code(processed_input, language="text")
