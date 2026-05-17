import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# ==============================================================================
# 1. FIX LỖI NLTK TRÊN STREAMLIT CLOUD (ĐƯỜNG DẪN CỤC BỘ)
# ==============================================================================
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        nltk.download('punkt_tab', download_dir=nltk_data_path)
    except Exception as e:
        st.error(f"Lỗi tải tài nguyên NLTK: {e}")

download_nltk_resources()

# ==============================================================================
# 2. CẤU HÌNH GIAO DIỆN CHÍNH (UI/UX)
# ==============================================================================
st.set_page_config(page_title="Phân loại tin nhắn Spam", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; 
        border-radius: 20px; 
        height: 3.5em; 
        background-color: #ff4b4b; 
        color: white; 
        font-weight: bold; 
    }
    .prediction-box { 
        padding: 25px; 
        border-radius: 15px; 
        text-align: center; 
        font-size: 26px; 
        font-weight: bold; 
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1); 
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành")

# ==============================================================================
# 3. HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH LAI (HYBRID)
# ==============================================================================
@st.cache_resource
def load_trained_model():
    # Đọc tệp dữ liệu huấn luyện nền
    df = pd.read_csv("2cls_spam_text_cls.csv")
    ps = PorterStemmer()
    
    # Hàm tiền xử lý ngôn ngữ tự nhiên độc lập
    def preprocess(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]
        return [ps.stem(t) for t in tokens]

    # Xây dựng không gian từ điển toàn cục (Vocabulary)
    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

    # Hàm ánh xạ văn bản thành Vector đặc trưng toán học (Bag-of-Words)
    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in dictionary: 
                # Ép kiểu int tường minh để tránh lỗi chỉ mục mảng vô hướng
                features[int(dictionary.index(t))] += 1
        return features

    # Chuyển đổi toàn bộ tập dữ liệu thành ma trận không gian vector X
    X = np.array([get_feats(t) for t in processed_msgs])
    
    # Số hóa nhãn mục tiêu (Category: ham, spam -> 0, 1)
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # 3.1. Khởi tạo và huấn luyện khối Xác suất Gaussian Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X, y)
    
    # 3.2. Khởi tạo và cấu hình khối Cơ sở dữ liệu Vector (KNN Cosine)
    # Tìm kiếm K=3 điểm lân cận gần nhất dựa trên góc lệch hướng Cosine
    vector_db = NearestNeighbors(n_neighbors=3, metric='cosine')
    vector_db.fit(X)
    
    return nb_model, vector_db, dictionary, le, preprocess, get_feats, df

# Tiến hành kích hoạt nạp mô hình hệ thống lai
try:
    nb_model, vector_db, dictionary, preprocess_fn, feat_fn, le, raw_df = load_trained_model()
except Exception as e:
    st.error(f"Đã xảy ra lỗi hệ thống khi thiết lập mô hình lai: {e}")
    st.stop()

# ==============================================================================
# 4. TRIỂN KHAI BỐ CỤC GIAO DIỆN (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra:", height=180, placeholder="Dán nội dung tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH LAI SONG HÀNH")

with col2:
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Hệ thống lai: Naive Bayes & Cơ sở dữ liệu Vector")
    st.info(f"📁 Quy mô từ điển: {len(dictionary)} từ vựng")

# ==============================================================================
# 5. XỬ LÝ KẾT QUẢ THEO KIẾN TRÚC ĐA TẦNG QUYẾT ĐỊNH (HYBRID INFEPENCE PIPELINE)
# ==============================================================================
if btn_click:
    if user_input.strip():
        with st.spinner('Hệ thống đang tiến hành phân tích đa tầng...'):
            # Bước 1: Trích xuất đặc trưng NLP của tin nhắn đầu vào
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # Khối A: Dự đoán bằng xác suất Toán học (Naive Bayes)
            prob = nb_model.predict_proba([features])[0]
            nb_raw_pred = le.inverse_transform([np.argmax(prob)])[0]
            confidence = max(prob) * 100
            
            # Khối B: Truy vấn không gian Cơ sở dữ liệu Vector (Vector DB)
            # Tìm khoảng cách và chỉ số index của 3 vector tương đồng nhất
            distances, indices = vector_db.kneighbors([features])
            neighbor_labels = [raw_df["Category"].iloc[int(idx)] for idx in indices[0]]
            
            # Thực hiện biểu quyết số đông (Majority Voting) giữa 3 lân cận gần nhất
            vdb_pred = max(set(neighbor_labels), key=neighbor_labels.count)
            
            # Bước 2: Sàng lọc qua Tầng kiểm soát ngưỡng an toàn (Decision Gate)
            if confidence < 95 or len(user_input.strip()) < 15:
                final_prediction = 'ham'
            else:
                # Nếu tin nhắn dài và rõ ràng, chỉ cần 1 trong 2 khối (NB hoặc Vector DB) 
                # nhận diện là 'spam' thì hệ thống sẽ lập tức phát cảnh báo nguy hiểm.
                if nb_raw_pred == 'spam' or vdb_pred == 'spam':
                    final_prediction = 'spam'
                else:
                    final_prediction = 'ham'

            # Bước 3: Kết xuất kết quả trực quan ra màn hình UI
            st.markdown("---")
            if final_prediction == 'spam':
                st.markdown(
                    f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">'
                    f'⚠️ CẢNH BÁO: TIN NHẮN RÁC (Độ tin cậy toán học: {confidence:.1f}%)'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">'
                    f'✅ AN TOÀN: TIN NHẮN THƯỜNG (Độ tin cậy toán học: {confidence:.1f}%)'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                st.balloons()
                
            # Khối bổ sung: Log phân tích kỹ thuật phục vụ phản biện đồ án
            with st.expander("🔍 Xem nhật ký phân tích kỹ thuật nâng cao"):
                st.write("**1. Kết quả khối toán học Naive Bayes:**")
                st.write(f"- Nhãn thô dự đoán: `{nb_raw_pred.upper()}` | Xác suất phân phối chính xác: `{confidence:.2f}%`")
                
                st.write("**2. Kết quả khối không gian Vector DB (Top 3 lân cận gần nhất):**")
                st.write(f"- Nhãn biểu quyết số đông: `{vdb_pred.upper()}`")
                for i in range(3):
                    idx_val = int(indices[0][i])
                    dist_val = float(distances[0][i])
                    st.text(f"  + Lân cận {i+1}: Nhãn [{raw_df['Category'].iloc[idx_val].upper()}] | Khoảng cách Cosine: {dist_val:.4f}")
                    st.caption(f"    Nội dung gốc: \"{raw_df['Message'].iloc[idx_val]}\"")

    else:
        st.error("Vui lòng nhập hoặc dán nội dung văn bản cần kiểm tra!")

# ==============================================================================
# 6. THANH BÊN THÔNG TIN THÀNH VIÊN (SIDEBAR)
# ==============================================================================
st.sidebar.title("Thông tin Nhóm")
st.sidebar.markdown("""
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
""")
