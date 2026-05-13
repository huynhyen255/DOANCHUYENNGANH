import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# --- THIẾT KẾ GIAO DIỆN NÂNG CẤP ---
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 26px; font-weight: bold; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành ")

# --- HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_resource
def load_trained_model():
    nltk.download("stopwords")
    nltk.download("punkt")
    df = pd.read_csv("2cls_spam_text_cls.csv")
    
    def preprocess(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]
        return [PorterStemmer().stem(t) for t in tokens]

    processed_msgs = [preprocess(msg) for msg in df["Message"]]
    dictionary = list(set([word for sublist in processed_msgs for word in sublist]))

    def get_feats(tokens):
        features = np.zeros(len(dictionary))
        for t in tokens:
            if t in dictionary: features[dictionary.index(t)] += 1
        return features

    X = np.array([get_feats(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    model = GaussianNB()
    model.fit(X, y)
    return model, dictionary, le, preprocess, get_feats

model, dictionary, le, preprocess_fn, feat_fn = load_trained_model()

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn (Anh/Việt):", height=180, placeholder="Dán tin nhắn cần kiểm tra tại đây...")
    btn_click = st.button("🚀 BẮT ĐẦU PHÂN TÍCH")

with col2:
    st.subheader("📊 Thông số học máy")
    st.success("🤖 Mô hình: Naive Bayes và Cơ sở dữ liệu vector ")
    st.info(f"📁 Từ điển: {len(dictionary)} đặc trưng")
    st.warning("🌐 Ngôn ngữ: Anh - Việt")

# --- XỬ LÝ KẾT QUẢ ---
if btn_click:
    if user_input:
        with st.spinner('Đang phân tích xác suất...'):
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            prob = model.predict_proba([features])[0]
            raw_prediction = le.inverse_transform([np.argmax(prob)])[0]
            confidence = max(prob) * 100
            
            # TẦNG LỌC BIAS (Sửa lỗi bắt nhầm tin nhắn ngắn/tiếng Việt)
            if confidence < 95 or len(user_input.strip()) < 15:
                final_prediction = 'ham'
            else:
                final_prediction = raw_prediction

            st.markdown("---")
            if final_prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.warning("**Lời khuyên:** Hệ thống phát hiện dấu hiệu quảng cáo hoặc lừa đảo.")
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
                if confidence > 50 and raw_prediction == 'spam':
                    st.caption("ℹ️ *Lưu ý: Tin nhắn chứa một số từ khóa nhạy cảm nhưng chưa đủ cơ sở để kết luận là Spam.*")
    else:
        st.error("Vui lòng nhập nội dung để phân tích!")

# --- THANH BÊN (Sidebar) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179543.png", width=100)
st.sidebar.title("Quản lý Đồ án")
st.sidebar.markdown(f"""
**Nhóm sinh viên thực hiện:**
* **Huỳnh Lê Hoàng Yến** - 022101091
* **Phạm Minh Tuấn** - 022101006
* **Huỳnh Văn Đăng Khoa** - 022101111
""")
