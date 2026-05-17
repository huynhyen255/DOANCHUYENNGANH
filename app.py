import streamlit as st
import pandas as pd
import numpy as np
import string
import os
import nltk
import csv
import torch
import torch.nn.functional as F
import faiss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel

# --- THIẾT LẬP THƯ MỤC NLTK AN TOÀN TRÊN CLOUD ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.error(f"Lỗi tải tài nguyên NLTK: {e}")

download_nltk_resources()

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; font-size: 16px; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 26px; font-weight: bold; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .neighbor-card { background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.write("Đồ án Chuyên Ngành: Kết hợp Naive Bayes & Cơ sở dữ liệu Vector")

# --- HÀM BỔ TRỢ VECTOR POOLING (Theo tài liệu của Cô) ---
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# --- PHẦN 1: TIỀN XỬ LÝ VÀ HUẤN LUYỆN MÔ HÌNH (LOAD HỆ THỐNG NGẦM) ---
@st.cache_resource
def initialize_all_models():
    # 1. Đọc dữ liệu an toàn, chống lỗi tràn cột ParserError
    DATASET_PATH = "2cls_spam_text_cls.csv"
    df = pd.read_csv(
        DATASET_PATH,
        quotechar='"',                 
        quoting=csv.QUOTE_MINIMAL,     
        on_bad_lines='skip',           
        encoding='utf-8'
    )
    
    messages = df["Message"].values.tolist()
    labels = df["Category"].values.tolist()
    
    # ---- LOGIC PHƯƠNG PHÁP 1: NAIVE BAYES ----
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    def preprocess_text(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        return [ps.stem(token) for token in tokens]

    processed_messages = [preprocess_text(msg) for msg in messages]
    dictionary = list(set([word for sublist in processed_messages for word in sublist]))
    word_to_idx = {word: i for i, word in enumerate(dictionary)}

    def create_features(tokens):
        features = np.zeros(len(dictionary))
        for token in tokens:
            if token in word_to_idx: 
                features[word_to_idx[token]] += 1
        return features

    X_nb = np.array([create_features(tokens) for tokens in processed_messages])
    le_nb = LabelEncoder()
    y_nb = le_nb.fit_transform(labels)
    
    # Huấn luyện mô hình Naive Bayes gốc
    nb_model = GaussianNB()
    nb_model.fit(X_nb, y_nb)
    
    # ---- LOGIC PHƯƠNG PHÁP 2: VECTOR DATABASE (FAISS) ----
    MODEL_NAME = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    v_model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_model.to(device)
    v_model.eval()
    
    # Tạo Embeddings theo từng cụm (Batch) nhỏ để tránh treo RAM
    embeddings = []
    batch_size = 16
    for i in range(0, len(messages), batch_size):
        batch_texts = messages[i:i+batch_size]
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        batch_dict = tokenizer(batch_texts_with_prefix, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = v_model(**batch_dict)
            batch_emb = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
            embeddings.append(batch_emb.cpu().numpy())
            
    X_embeddings = np.vstack(embeddings)
    
    le_v = LabelEncoder()
    y_v = le_v.fit_transform(labels)
    metadata = [
        {"index": i, "message": msg, "label": lbl, "label_encoded": y_v[i]} 
        for i, (msg, lbl) in enumerate(zip(messages, labels))
    ]
    
    train_indices, _ = train_test_split(range(len(messages)), test_size=0.1, stratify=y_v, random_state=42)
    X_train_emb = X_embeddings[train_indices]
    train_metadata = [metadata[idx] for idx in train_indices]
    
    # Nạp dữ liệu vào FAISS Vector DB
    embedding_dim = X_train_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(X_train_emb.astype("float32"))
    
    return nb_model, dictionary, le_nb, preprocess_text, create_features, v_model, tokenizer, device, faiss_index, train_metadata

# Khởi động hệ thống
with st.spinner('🔄 Hệ thống đang nạp song song hai mô hình AI và Vector Database...'):
    nb_model, dictionary, le_nb, preprocess_text_fn, create_features_fn, v_model, tokenizer, device, faiss_index, train_metadata = initialize_all_models()

# --- GIAO DIỆN ĐIỀU KHIỂN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Phân tích nội dung")
    user_input = st.text_area("Nhập tin nhắn cần kiểm tra (Hỗ trợ Anh/Việt):", height=150, placeholder="Dán nội dung tin nhắn vào đây...")
    
    # Cho phép giảng viên lựa chọn thuật toán đối chiếu ngay trên giao diện Web
    method = st.radio("Lựa chọn phương pháp phân loại tin nhắn:", 
                      ("Phương pháp 1: Học máy Naive Bayes (Bag-of-Words)", 
                       "Phương pháp 2: Cơ sở dữ liệu Vector (FAISS + KNN)"))

with col2:
    st.subheader("📊 Thông số hệ thống")
    st.success("🤖 Mô hình Naive Bayes: Đã sẵn sàng")
    st.success("🗄️ Vector Database (FAISS): Đã kết nối")
    st.info(f"📁 Số lượng từ vựng trong từ điển: {len(dictionary)} từ")
    
    # Cấu hình tham số K nếu chọn phương pháp 2
    k_value = st.slider("Số láng giềng đối chiếu (Tham số K cho KNN):", min_value=1, max_value=7, value=3, disabled=(method == "Phương pháp 1: Học máy Naive Bayes (Bag-of-Words)"))

# --- XỬ LÝ LOGIC PHÂN LOẠI THEO TỪNG PHƯƠNG PHÁP ---
if st.button("🚀 BẮT ĐẦU KIỂM TRA TIN NHẮN"):
    if user_input.strip():
        with st.spinner('Hệ thống đang xử lý toán học...'):
            
            # --- TRƯỜNG HỢP 1: CHẠY NAIVE BAYES ---
            if "Naive Bayes" in method:
                tokens = preprocess_text_fn(user_input)
                features = create_features_fn(tokens)
                prob = nb_model.predict_proba([features])[0]
                
                # Sửa lỗi Ép kiểu scalar index để không bị sập app
                pred_idx = int(np.argmax(prob))
                final_prediction = le_nb.inverse_transform([pred_idx])[0]
                confidence = prob[pred_idx] * 100
                
                st.markdown("---")
                st.subheader("📊 Kết quả phân tích từ Naive Bayes")
                if final_prediction == 'spam':
                    st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN RÁC ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                    st.balloons()
            
            # --- TRƯỜNG HỢP 2: CHẠY VECTOR DATABASE + KNN ---
            else:
                query_with_prefix = f"query: {user_input}"
                batch_dict = tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
                
                with torch.no_grad():
                    outputs = v_model(**batch_dict)
                    query_embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                    query_embedding = F.normalize(query_embedding, p=2, dim=1)
                
                query_embedding_np = query_embedding.cpu().numpy().astype("float32")
                scores, indices = faiss_index.search(query_embedding_np, k_value)
                
                predictions = []
                neighbor_info = []
                for i in range(k_value):
                    neighbor_idx = indices[0][i]
                    neighbor_score = scores[0][i]
                    n_label = train_metadata[neighbor_idx]['label']
                    n_msg = train_metadata[neighbor_idx]['message']
                    
                    predictions.append(n_label)
                    neighbor_info.append({'score': float(neighbor_score), 'label': n_label, 'message': n_msg})
                
                unique_labels, counts = np.unique(predictions, return_counts=True)
                final_prediction = unique_labels[np.argmax(counts)]
                confidence_rate = (max(counts) / k_value) * 100
                
                st.markdown("---")
                st.subheader("📊 Kết quả phân tích từ Vector Database")
                if final_prediction.lower() == 'spam':
                    st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN SPAM ĐỘC HẠI ({confidence_rate:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN HỢP LỆ ({confidence_rate:.1f}%)</div>', unsafe_allow_html=True)
                    st.balloons()
                
                # Khối hiển thị Explainable AI giải thích thuật toán KNN cho giảng viên xem
                with st.expander("🔍 Chi tiết các láng giềng gần nhất trong cơ sở dữ liệu vector"):
                    for idx, neighbor in enumerate(neighbor_info, 1):
                        badge_color = "#f8d7da" if neighbor['label'].lower() == 'spam' else "#d4edda"
                        text_color = "#721c24" if neighbor['label'].lower() == 'spam' else "#155724"
                        st.markdown(f"""
                        <div class="neighbor-card">
                            <strong>Top {idx} - Độ tương đồng (Cosine Score):</strong> <code style="color:blue;">{neighbor['score']:.4f}</code> | 
                            Nhãn gốc: <span style="background-color:{badge_color}; color:{text_color}; padding:2px 6px; border-radius:5px; font-weight:bold;">{neighbor['label'].upper()}</span>
                            <br><p style="margin-top:8px; color:#555; font-style:italic;">"Nội dung: {neighbor['message']}"</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Vui lòng nhập nội dung tin nhắn trước khi nhấn phân tích!")

# --- THANH SIDEBAR QUẢN LÝ ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179543.png", width=90)
st.sidebar.title("Quản lý Đồ án")
st.sidebar.markdown("---")
st.sidebar.markdown("### **Thành viên thực hiện:**")
st.sidebar.write("1. 👩‍🎓 Huỳnh Lê Hoàng Yến - *022101091*")  
st.sidebar.write("2. 👨‍🎓 Phạm Minh Tuấn - *022101006*")  
st.sidebar.write("3. 👨‍🎓 Huỳnh Văn Đăng Khoa - *022101111*")
