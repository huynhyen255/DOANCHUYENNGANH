import streamlit as st
import pandas as pd
import numpy as np  # Khai báo numpy chuẩn để sửa lỗi NameError
import re
import os

# --- THƯ VIỆN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP) ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- THƯ VIỆN HỌC MÁY & VECTOR DATABASE ---
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Tải tài nguyên NLTK ẩn bảo đảm không gây chậm hệ thống
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN STREAMLIT CHUẨN BÁO CÁO ĐỒ ÁN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Message Classification using Naive Bayes and Vector Database",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Đồ án Chuyên ngành: Phân loại tin nhắn Spam sử dụng Naive Bayes và Cơ sở dữ liệu Vector")
st.write("---")

# ---------------------------------------------------------------------------
# PHẦN II.4: TIỀN XỬ LÝ DỮ LIỆU
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
# HÀM HUẤN LUYỆN VÀ TẠO CHỈ MỤC VECTOR DATABASE INTERACTIVE
# ---------------------------------------------------------------------------
@st.cache_resource
def initialize_system():
    file_path = '2cls_spam_text_cls.csv'
    if not os.path.exists(file_path):
        return None, "Không tìm thấy file dữ liệu mẫu '2cls_spam_text_cls.csv'. Vui lòng kiểm tra lại thư mục gốc!"
    
    try:
        df = pd.read_csv(file_path)
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # === PHẦN II: PHÂN LOẠI MÔ HÌNH HỌC MÁY NAIVE BAYES ===
        cv = CountVectorizer()
        X_nb = cv.fit_transform(df['processed_text']).toarray()
        y_nb = df['label'].values
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)
        
        nb_model = MultinomialNB()
        nb_model.fit(X_train_nb, y_train_nb)
        nb_acc = accuracy_score(y_test_nb, nb_model.predict(X_test_nb))
        
        # === PHẦN III: THIẾT LẬP CƠ SỞ DỮ LIỆU VECTOR & CHỈ MỤC KNN ===
        tfidf = TfidfVectorizer()
        X_vector_db = tfidf.fit_transform(df['processed_text']).toarray()
        
        knn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_index.fit(X_vector_db)
        
        return {
            "cv": cv, "nb_model": nb_model, "nb_acc": nb_acc,
            "tfidf": tfidf, "knn_index": knn_index, "df": df
        }, "Hệ thống tích hợp Naive Bayes & Cơ sở dữ liệu Vector khởi tạo thành công!"
    except Exception as e:
        return None, f"Lỗi nghiêm trọng trong quá trình nạp hệ thống: {str(e)}"

system_core, status_msg = initialize_system()

# Bố cục giao diện
layout_left, layout_right = st.columns([1, 2])

with layout_left:
    st.header("📊 Thông tin đồ án")
    if system_core is None:
        st.error(status_msg)
    else:
        st.success(status_msg)
        st.metric(label="Độ chính xác kiểm thử Naive Bayes", value=f"{system_core['nb_acc']*100:.2f}%")
        
        st.markdown("### 👨‍🏫 Giảng viên hướng dẫn:")
        st.write("- **ThS. Phạm Ngọc Giàu**")
        
        st.markdown("### 👥 Sinh viên thực hiện:")
        st.write("- 🧑‍💻 **Huỳnh Lê Hoàng Yến** — MSSV: `022101091` (Lớp: ĐH CNTT22B)")
        st.write("- 🧑‍💻 **Huỳnh Văn Đăng Khoa** — MSSV: `022101111` (Lớp: ĐH CNTT22B)")
        st.write("- 🧑‍💻 **Phạm Minh Tuấn** — MSSV: `022101006` (Lớp: ĐH CNTT22B)")
        
        st.write("---")
        st.info(
            "**Thông số kỹ thuật lõi:**\n"
            "- Pipeline 1: Multinomial Naive Bayes\n"
            "- Pipeline 2: Vector DB Index\n"
            "- Hàm khoảng cách: Cosine Distance"
        )

with layout_right:
    st.header("🔍 Hệ thống kiểm thử Pipeline")
    user_input = st.text_area("Nhập nội dung văn bản / tin nhắn cần phân loại kiểm thử:", height=120, 
                              placeholder="Nhập hoặc dán nội dung đoạn tin nhắn tại đây...")
    
    k_neighbors = st.slider("Cấu hình số lượng láng giềng k tra cứu trong Vector DB (Phần III):", min_value=1, max_value=5, value=3)
    
    if st.button("🚀 Thực thi Pipeline phân loại kết hợp", type="primary"):
        if not user_input.strip():
            st.warning("Hệ thống yêu cầu nhập nội dung văn bản trước khi bấm thực thi!")
        elif system_core is None:
            st.error("Lỗi lõi hệ thống chưa được giải phóng!")
        else:
            cv = system_core["cv"]
            nb_model = system_core["nb_model"]
            tfidf = system_core["tfidf"]
            knn_index = system_core["knn_index"]
            df = system_core["df"]
            
            processed_input = preprocess_text(user_input)
            
            st.write("---")
            st.subheader("📊 Kết quả thực nghiệm phân tích:")
            
            # Khống chế tin nhắn quá ngắn
            if len(user_input.strip()) < 10:
                final_decision = "Ham"
                nb_prediction = "Ham"
                knn_prediction = "Ham"
                nb_confidence = 100.0
                neighbors_results = []
                st.warning("⚠️ Nhận diện: Tin nhắn có độ dài quá ngắn. Hệ thống tự động đưa vào danh sách an toàn.")
            else:
                # =======================================================================
                # MÔ HÌNH 1: NAIVE BAYES (Sửa lỗi triệt để bằng việc gọi thư viện đồng bộ)
                # =======================================================================
                vectorized_nb = cv.transform([processed_input]).toarray()
                nb_prediction = nb_model.predict(vectorized_nb)[0]
                nb_proba = nb_model.predict_proba(vectorized_nb)[0]
                
                # Trích xuất vị trí nhãn thủ công an toàn, tránh xung đột mảng numpy
                labels_list = list(nb_model.classes_)
                pred_index = labels_list.index(nb_prediction)
                nb_confidence = nb_proba[pred_index] * 100
                
                # =======================================================================
                # MÔ HÌNH 2: VECTOR DATABASE & KNN
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
                
                # BIỂU QUYẾT TỔNG HỢP
                if nb_prediction == "Spam" or knn_prediction == "Spam":
                    final_decision = "Spam"
                else:
                    final_decision = "Ham"

            # =======================================================================
            # TABS HIỂN THỊ
            # =======================================================================
            tab1, tab2, tab3 = st.tabs(["🛡️ Kết luận Pipeline chung", "📐 Nhật ký Vector Database (KNN)", "📝 Chuỗi NLP Token thô"])
            
            with tab1:
                st.write("### 🎯 Kết luận nhãn cuối cùng:")
                if final_decision == "Spam":
                    st.error(f"🚨 **HỆ THỐNG DÁN NHÃN: SPAM (TIN NHẮN RÁC)**")
                else:
                    st.success(f"✅ **HỆ THỐNG DÁN NHÃN: HAM (AN TOÀN)**")
                
                st.write("---")
                st.write("#### 🔹 Kết quả chi tiết cấu phần:")
                st.write(f"- Phân loại Naive Bayes: **`{nb_prediction.upper()}`** (Độ tự tin: {nb_confidence:.2f}%)")
                st.write(f"- Không gian Vector DB: **`{knn_prediction.upper()}`**")
            
            with tab2:
                if len(neighbors_results) == 0:
                    st.write("*Không có láng giềng do tin nhắn quá ngắn.*")
                else:
                    st.write(f"#### 🔍 Top {k_neighbors} tọa độ Vector láng giềng có độ tương đồng cao nhất:")
                    for rank, neighbor in enumerate(neighbors_results, 1):
                        with st.container():
                            st.write(f"**Top {rank}. Nhãn gốc: `{neighbor['label']}` | Độ khớp Cosine: `{neighbor['score']:.4f}`**")
                            st.caption(f"Nội dung mẫu: *\"{neighbor['text']}\"*")
                            st.write("---")
            
            with tab3:
                st.write("#### 👁️ Dữ liệu sau bộ lọc NLP Pipeline:")
                st.code(processed_input if processed_input.strip() else "[Chuỗi trống]", language="text")
