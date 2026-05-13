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
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; }
    .prediction-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành - Nâng cấp bởi AI: Tích hợp Tiếng Anh & Tiếng Việt")

# --- HIỂN THỊ KẾT QUẢ (Bản sửa lỗi Bias) ---
if btn_click:
    if user_input:
        with st.spinner('Đang phân tích xác suất...'):
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            prob = model.predict_proba([features])[0]
            raw_prediction = le.inverse_transform([np.argmax(prob)])[0]
            confidence = max(prob) * 100
            
            # --- ĐIỀU KIỆN LỌC BỔ SUNG ---
            # 1. Nếu độ tin cậy thấp hơn 95%
            # 2. Hoặc tin nhắn quá ngắn (dưới 15 ký tự)
            if confidence < 95 or len(user_input.strip()) < 15:
                final_prediction = 'ham'
            else:
                final_prediction = raw_prediction

            st.markdown("---")
            if final_prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.write("**Ghi chú:** Hệ thống phát hiện các dấu hiệu quảng cáo hoặc lừa đảo rõ rệt.")
            else:
                # Nếu máy đoán là Spam nhưng bị điều kiện phụ chặn lại, ta báo là An toàn
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
                if confidence > 50 and raw_prediction == 'spam':
                    st.caption("ℹ️ *Lưu ý: Tin nhắn có một vài từ khóa nhạy cảm nhưng chưa đủ cơ sở để kết luận là Spam.*")
    else:
        st.error("Vui lòng nhập nội dung để phân tích!")

# --- GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn (Anh/Việt):", height=200, placeholder="Nhập tin nhắn cần kiểm tra tại đây...")

with col2:
    st.subheader("📊 Thông số học máy")
    st.success("Mô hình: Naive Bayes và cơ sở dữ liệu vector")

if st.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
    if user_input:
        with st.spinner('Đang phân tích xác suất...'):
            tokens = preprocess_fn(user_input)
            features = feat_fn(tokens)
            
            # Dự đoán xác suất
            prob = model.predict_proba([features])[0]
            prediction = le.inverse_transform([np.argmax(prob)])[0]
            confidence = max(prob) * 100

            st.markdown("---")
            if prediction == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                st.warning("Lời khuyên: Không nhấn vào bất kỳ đường link nào trong tin nhắn này.")
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                st.balloons()
    else:
        st.error("Vui lòng nhập nội dung!")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179543.png", width=100)
st.sidebar.title("Quản lý Đồ án")
st.sidebar.write("**Sinh viên:** Huỳnh Lê Hoàng Yến 022101091")  
st.sidebar.write("**Sinh viên:** Phạm Minh Tuấn 022101006")  
st.sidebar.write("**Sinh viên:** Huỳnh Văn Đăng Khoa 022101111")
