import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.naive_bayes import MultinomialNB  # Đổi sang MultinomialNB chuyên cho text
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

# --- TẢI TÀI NGUYÊN NLTK NGOÀI HÀM CACHE ---
@st.cache_resource
def download_nltk_deps():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt", quiet=True)

download_nltk_deps()

# --- THIẾT KẾ GIAO DIỆN NÂNG CẤP ---
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.caption("Đồ án Chuyên Ngành - Thành viên: Yến, Tuấn, Khoa")

# --- TIỀN XỬ LÝ TEXT (Hỗ trợ Song ngữ Anh - Việt) ---
def preprocess(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    # Không dùng PorterStemmer cho tiếng Việt để tránh lỗi mất dấu/biến dạng từ
    return tokens

# --- HÀM TẢI VÀ HUẤN LUYỆN MÔ HÌNH (Đã tối ưu hóa tốc độ) ---
@st.cache_resource
def load_trained_model():
    # Đọc dữ liệu
    df = pd.read_csv("2cls_spam_text_cls.csv")
    
    # Tiền xử lý tập dữ liệu
    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    
    # Tạo từ điển (Vocabulary)
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))
    word_to_idx = {word: i for i, word in enumerate(dictionary)} # Dùng dict để tăng tốc độ lookup từ O(N) lên O(1)

    # Hàm biến đổi vector đặc trưng nhanh
    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in word_to_idx: 
                features[word_to_idx[t]] += 1
        return features

    # Tạo ma trận đặc trưng X
    X = np.array([get_feats(t) for t in processed_msgs])
    
    # Mã hóa nhãn y
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # Huấn luyện mô hình MultinomialNB (Nhanh hơn GaussianNB gấp 10-100 lần trên ma trận thưa)
    model = MultinomialNB()
    model.fit(X, y)
    
    return model, dictionary, le, get_feats

# Khởi tạo mô hình
with st.spinner('🔄 Đang khởi tạo hệ thống AI (Vui lòng đợi trong giây lát)...'):
    model, dictionary, le, feat_fn = load_trained_model()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn (Anh/Việt):", height=180, placeholder="Nhập hoặc dán tin nhắn cần kiểm tra tại đây...")

with col2:
    st.subheader("📊 Thông số học máy")
    st.info(f"📚 Kích thước từ điển: **{len(dictionary)}** từ")
    st.success("🤖 Thuật toán: **Multinomial Naive Bayes**")

# Nút kích hoạt phân tích nằm ngoài cột để căn đều giao diện
if st.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
    if user_input.strip():
        with st.spinner('Đang phân tích xác suất...'):
            tokens = preprocess(user_input)
            features = feat_fn(tokens)
            
            # Dự đoán xác suất
            prob = model.predict_proba([features])[0]
            pred_class_idx = np.argmax(prob)
            prediction = le.inverse_transform([pred_class_idx])[0]
            confidence = prob[pred_class_idx] * 100

            st.markdown("---")
            # Hiển thị kết quả trực quan
            if prediction.lower() == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                st.warning("👉 **Lời khuyên:** Tin nhắn này có dấu hiệu lừa đảo hoặc quảng cáo phiền toái. Không nhấn vào bất kỳ đường link nào!")
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                st.balloons()
    else:
        st.error("⚠️ Vui lòng nhập nội dung tin nhắn trước khi phân tích!")

# --- THANH SIDEBAR QUẢN LÝ ĐỒ ÁN ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179543.png", width=100)
st.sidebar.title("Quản lý Đồ án")
st.sidebar.markdown("---")
st.sidebar.write("**Thành viên thực hiện:**")
st.sidebar.write("1. 👩‍🎓 Huỳnh Lê Hoàng Yến - *022101091*")  
st.sidebar.write("2. 👨‍🎓 Phạm Minh Tuấn - *022101006*")  
st.sidebar.write("3. 👨‍🎓 Huỳnh Văn Đăng Khoa - *022101111*")
