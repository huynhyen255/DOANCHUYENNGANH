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
# PHẦN II.4: TIỀN XỬ LÝ DỮ LIỆU (Đồng bộ ngôn ngữ theo tài liệu hướng dẫn)
# ---------------------------------------------------------------------------
def preprocess_text(text):
    # 1. Chuyển về chữ thường
    text = str(text).lower()
    # 2. Xóa ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng (hỗ trợ Tiếng Việt và Tiếng Anh)
    text = re.sub(r'[^a-zA-Z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text)
    # 3. Tách từ (Tokenization)
    tokens = word_tokenize(text)
    
    # 4. Loại bỏ từ dừng (Stopwords) - Kết hợp đa ngôn ngữ
    stop_words = set(stopwords.words('english'))
    vietnamese_stopwords = {'và', 'với', 'là', 'thì', 'mà', 'bị', 'được', 'cho', 'của', 'các', 'này'}
    stop_words = stop_words.union(vietnamese_stopwords)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Rút gọn gốc từ (Stemming) bằng thuật toán PorterStemmer
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
        # Đọc kho ngữ liệu nền
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
        
        # Tạo cấu trúc chỉ mục tìm kiếm không gian láng giềng dựa trên độ tương đồng Cosine
        knn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_index.fit(X_vector_db)
        
        return {
            "cv": cv, "nb_model": nb_model, "nb_acc": nb_acc,
            "tfidf": tfidf, "knn_index": knn_index, "df": df
        }, "Hệ thống tích hợp Naive Bayes & Cơ sở dữ liệu Vector khởi tạo thành công!"
    except Exception as e:
        return None, f"Lỗi nghiêm trọng trong quá trình nạp hệ thống: {str(e)}"

# Kích hoạt lõi công nghệ của ứng dụng
system_core, status_msg = initialize_system()

# Phân chia bố cục giao diện (Layout Responsive)
layout_left, layout_right = st.columns([1, 2])

with layout_left:
    st.header("📊 Thông tin đồ án")
    if system_core is None:
        st.error(status_msg)
    else:
        st.success(status_msg)
        st.metric(label="Độ chính xác kiểm thử Naive Bayes", value=f"{system_core['nb_acc']*100:.2f}%")
        
        # --- THÔNG TIN THÀNH VIÊN NHÓM THỰC HIỆN ĐÃ ĐỒNG BỘ ---
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
            "- Pipeline 2: Vector DB Index (TF-IDF Embedder)\n"
            "- Hàm khoảng cách: Cosine Distance Metric\n"
            "- Ngưỡng lọc an toàn: 95%"
        )

with layout_right:
    st.header("🔍 Hệ thống kiểm thử Pipeline")
    user_input = st.text_area("Nhập nội dung văn bản / tin nhắn cần phân loại kiểm thử:", height=120, 
                              placeholder="Nhập hoặc dán nội dung đoạn tin nhắn tại đây...")
    
    # Cho phép điều chỉnh tham số K láng giềng trực quan
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
            
            # Thực hiện tiền xử lý văn bản đầu vào thông qua NLP Pipeline
            processed_input = preprocess_text(user_input)
            
            st.write("---")
            st.subheader("📊 Kết quả thực nghiệm phân tích:")
            
            # --- KIỂM SOÁT LOGIC CHUỖI KÝ TỰ QUÁ NGẮN (THRESHOLD CONTROL) ---
            if len(user_input.strip()) < 10:
                final_decision = "Ham"
                nb_prediction = "Ham"
                knn_prediction = "Ham"
                nb_confidence = 100.0
                neighbors_results = []
                st.warning("⚠️ Nhận diện: Tin nhắn có độ dài quá ngắn. Hệ thống tự động đưa vào danh sách trắng an toàn.")
            else:
                # =======================================================================
                # MÔ HÌNH 1: TRÍCH XUẤT XÁC SUẤT NAIVE BAYES (PHẦN II)
                # =======================================================================
                vectorized_nb = cv.transform([processed_input]).toarray()
                nb_prediction = nb_model.predict(vectorized_nb)[0]
                nb_proba = nb_model.predict_proba(vectorized_nb)[0]
                labels_mapping = nb_model.classes_
                pred_index = np.where(labels_mapping == nb_prediction)[0][0]
                nb_confidence = nb_proba[pred_index] * 100
                
                # =======================================================================
                # MÔ HÌNH 2: TRUY VẤN KHÔNG GIAN VECTOR DATABASE (PHẦN III)
                # =======================================================================
                vectorized_vector_db = tfidf.transform([processed_input]).toarray()
                distances, indices = knn_index.kneighbors(vectorized_vector_db, n_neighbors=k_neighbors)
                
                neighbors_results = []
                for i in range(k_neighbors):
                    idx = indices[0][i]
                    score = 1 - distances[0][i]  # Chuyển đổi từ Distance sang Cosine Similarity Score
                    neighbors_results.append({
                        "label": df.iloc[idx]['label'],
                        "text": df.iloc[idx]['text'],
                        "score": score
                    })
                
                # Trích xuất biểu quyết số đông láng giềng (Voting Method)
                neighbor_labels = [n["label"] for n in neighbors_results]
                knn_prediction = max(set(neighbor_labels), key=neighbor_labels.count)
                
                # =======================================================================
                # BỘ QUYẾT ĐỊNH PIPELINE TỔNG HỢP (HYBRID DECISION MATRIX)
                # =======================================================================
                # Nếu hai mô hình đồng thuận hoặc Naive Bayes có độ tự tin tối cao > 95%
                if nb_prediction == "Spam" and nb_confidence >= 95.0:
                    final_decision = "Spam"
                elif nb_prediction == "Spam" and knn_prediction == "Spam":
                    final_decision = "Spam"
                else:
                    final_decision = "Ham"

            # =======================================================================
            # KẾT CẤU TABS HIỂN THỊ TRỰC QUAN CHUẨN ĐÁNH GIÁ CỦA GIÁO VIÊN
            # =======================================================================
            tab1, tab2, tab3 = st.tabs(["🛡️ Kết luận Pipeline chung", "📐 Nhật ký Vector Database (KNN)", "📝 Chuỗi NLP Token thô"])
            
            with tab1:
                st.write("### 🎯 Kết luận nhãn cuối cùng của hệ thống:")
                if final_decision == "Spam":
                    st.error(f"🚨 **HỆ THỐNG DÁN NHÃN: SPAM (TIN NHẮN RÁC / LỪA ĐẢO)**")
                else:
                    st.success(f"✅ **HỆ THỐNG DÁN NHÃN: HAM (TIN NHẮN AN TOÀN TRẮNG)**")
                
                st.write("---")
                st.write("#### 🔹 Kết quả chi tiết cấu phần học máy:")
                st.write(f"- Phân loại Naive Bayes dự đoán nhãn: **`{nb_prediction.upper()}`** (Độ tự tin toán học: {nb_confidence:.2f}%)")
                st.write(f"- Biểu quyết không gian Vector DB dự đoán nhãn: **`{knn_prediction.upper()}`**")
            
            with tab2:
                if len(neighbors_results) == 0:
                    st.write("*Không có láng giềng nào được truy vấn do tin nhắn thuộc diện danh sách trắng.*")
                else:
                    st.write(f"#### 🔍 Top {k_neighbors} tọa độ Vector láng giềng có độ tương đồng hình học cao nhất:")
                    for rank, neighbor in enumerate(neighbors_results, 1):
                        with st.container():
                            st.write(f"**Top {rank}. Nhãn lưu trữ gốc: `{neighbor['label']}` | Độ khớp Cosine Similarity: `{neighbor['score']:.4f}`**")
                            st.caption(f"Nội dung trích mẫu trong DB: *\"{neighbor['text']}\"*")
                            st.write("---")
            
            with tab3:
                st.write("#### 👁️ Dữ liệu sau khi đi qua bộ lọc làm sạch văn bản (Tokens sau xử lý PorterStemmer):")
                st.code(processed_input if processed_input.strip() else "[Chuỗi trống hoặc chứa toàn bộ từ dừng bị loại bỏ]", language="text")
