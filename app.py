import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import string
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- THIẾT KẾ GIAO DIỆN WEB ---
st.set_page_config(page_title="AI Spam Shield Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3.2em; background-color: #ff4b4b; color: white; font-weight: bold; font-size: 16px; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 26px; font-weight: bold; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .neighbor-card { background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam Đa Ngôn Ngữ")
st.write("Đồ án Chuyên Ngành | Công nghệ: **Sentence Embeddings & Vector Database (FAISS)**")

# --- HÀM HỖ TRỢ XỬ LÝ VECTOR (Theo tài liệu đồ án) ---
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# --- TẢI MÔ HÌNH EMBEDDING & KHỞI TẠO VECTOR DB (Cấu hình tối ưu Cache) ---
@st.cache_resource
def initialize_system():
    # 1. Đọc và chuẩn bị dữ liệu đầu vào
    DATASET_PATH = "2cls_spam_text_cls.csv"
    try:
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', encoding='utf-8')
    except Exception:
        # Tạo dữ liệu giả lập chuẩn nếu không tìm thấy file để tránh crash giao diện
        df = pd.DataFrame({
            "Message": ["Trúng thưởng thẻ cào 500k bấm vào đây", "Bạn có hẹn lúc 2h chiều nay nhé", "FREE!! Click here to win money"],
            "Category": ["spam", "ham", "spam"]
        })

    messages = df["Message"].values.tolist()
    labels = df["Category"].values.tolist()
    
    # 2. Khởi tạo Mô hình Embedding đa ngôn ngữ (Hỗ trợ tiếng Việt rất mạnh)
    MODEL_NAME = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 3. Tiến hành tạo Vector Embedding cho tập dữ liệu mẫu (Batch size = 16 để tránh tràn RAM Cloud)
    embeddings = []
    batch_size = 16
    for i in range(0, len(messages), batch_size):
        batch_texts = messages[i : i + batch_size]
        # Thêm tiền tố "passage: " chuẩn quy định của mô hình E5
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        batch_dict = tokenizer(batch_texts_with_prefix, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_emb = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
            embeddings.append(batch_emb.cpu().numpy())
            
    X_embeddings = np.vstack(embeddings)
    
    # 4. Lưu nhãn và Metadata để đối chiếu kết quả KNN
    le = LabelEncoder()
    y = le.fit_transform(labels)
    metadata = [
        {"index": i, "message": msg, "label": lbl, "label_encoded": y[i]} 
        for i, (msg, lbl) in enumerate(zip(messages, labels))
    ]
    
    # Tách dữ liệu Train/Test tỉ lệ 90/10 theo tài liệu
    train_indices, _ = train_test_split(range(len(messages)), test_size=0.1, stratify=y, random_state=42)
    X_train_emb = X_embeddings[train_indices]
    train_metadata = [metadata[i] for i in train_indices]
    
    # 5. Xây dựng Cơ sở dữ liệu Vector với FAISS IndexFlatIP (Inner Product)
    embedding_dim = X_train_emb.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(X_train_emb.astype("float32"))
    
    return model, tokenizer, device, index, train_metadata

# Thực hiện tải và kích hoạt hệ thống ngầm
with st.spinner('🔄 Hệ thống đang nạp Mô hình Ngôn ngữ và Khởi tạo Cơ sở dữ liệu Vector...'):
    model, tokenizer, device, index, train_metadata = initialize_system()

# --- BỐ CỤC GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([5, 3])

with col1:
    st.subheader("📝 Phân tích Nội dung Tin nhắn")
    user_input = st.text_area(
        "Nhập tin nhắn cần kiểm tra (Hỗ trợ tiếng Việt, tiếng Anh và Telex):", 
        height=180, 
        placeholder="Ví dụ: Chuc mung ban da trung thuong 1 chiec xe may! Vui long truy cap link..."
    )
    
    # Thanh chọn tham số K láng giềng cho cơ sở dữ liệu vector
    k_value = st.slider("Cấu hình số lượng láng giềng đối chiếu (Tham số K-KNN):", min_value=1, max_value=7, value=3)

with col2:
    st.subheader("⚙️ Trạng thái Hệ thống AI")
    st.info(f"🧬 Mã nguồn Embedding: `intfloat/multilingual-e5-base` [cite: 144]")
    st.success(f"🗄️ Vector Database: **FAISS (Facebook AI Similarity Search)** [cite: 19, 124, 184]")
    st.metric(label="Tổng số tin nhắn trong cơ sở dữ liệu mẫu", value=len(train_metadata))

st.markdown("---")

# --- XỬ LÝ LOGIC PHÂN LOẠI KHI NGƯỜI DÙNG BẤM NÚT ---
if st.button("🚀 KÍCH HOẠT SHIELD KIỂM TRA TIN NHẮN"):
    if user_input.strip():
        with st.spinner('🔍 Đang trích xuất vector đặc trưng và truy vấn láng giềng gần nhất...'):
            # 1. Định dạng text truy vấn với tiền tố "query: " cho mô hình E5
            query_with_prefix = f"query: {user_input}"
            
            # 2. Vector hóa nội dung tin nhắn người dùng nhập vào
            batch_dict = tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = model(**batch_dict)
                query_embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            query_embedding_np = query_embedding.cpu().numpy().astype("float32")
            
            # 3. Truy vấn tìm kiếm độ tương đồng trên Vector Database FAISS
            scores, indices = index.search(query_embedding_np, k_value)
            
            # 4. Trích xuất thông tin láng giềng phục vụ bầu chọn Majority Vote
            predictions = []
            neighbor_info = []
            for i in range(k_value):
                neighbor_idx = indices[0][i]
                neighbor_score = scores[0][i]
                n_label = train_metadata[neighbor_idx]['label']
                n_msg = train_metadata[neighbor_idx]['message']
                
                predictions.append(n_label)
                neighbor_info.append({
                    'score': float(neighbor_score),
                    'label': n_label,
                    'message': n_msg
                })
            
            # 5. Thực hiện bầu chọn số đông (Majority Vote) để ra nhãn cuối cùng
            unique_labels, counts = np.unique(predictions, return_counts=True)
            final_prediction = unique_labels[np.argmax(counts)]
            
            # Tính toán phần trăm tin cậy dựa trên tỷ lệ phiếu bầu của láng giềng
            confidence_rate = (max(counts) / k_value) * 100

            # --- HIỂN THỊ KẾT QUẢ RA WEB ---
            st.subheader("📊 Kết quả Phân tích Ngữ nghĩa")
            
            if final_prediction.lower() == 'spam':
                st.markdown(f'<div class="prediction-box" style="background-color: #ffebee; color: #c62828;">⚠️ CẢNH BÁO: TIN NHẮN SPAM / LỪA ĐẢO ĐỘ ĐỘC HẠI CAO ({confidence_rate:.1f}%)</div>', unsafe_allow_html=True)
                st.warning("🚨 **Khuyến nghị an toàn:** Nội dung này có cấu trúc tương đồng dữ liệu rác phổ biến. Tuyệt đối không click vào đường link đi kèm hoặc cung cấp OTP.")
            else:
                st.markdown(f'<div class="prediction-box" style="background-color: #e8f5e9; color: #2e7d32;">✅ AN TOÀN: TIN NHẮN THƯỜNG / HỢP LỆ ({confidence_rate:.1f}%)</div>', unsafe_allow_html=True)
                st.balloons()
            
            # Hiển thị giải thích thuật toán: Các văn bản tương đồng nhất trong DB
            with st.expander("🔍 Xem chi tiết các tin nhắn tương đồng nhất trong Vector Database (Giải thích thuật toán KNN)"):
                st.write("Dưới đây là các tin nhắn có khoảng cách ngữ nghĩa gần nhất với tin nhắn của bạn được tìm thấy:")
                for idx, neighbor in enumerate(neighbor_info, 1):
                    badge_color = "#f8d7da" if neighbor['label'].lower() == 'spam' else "#d4edda"
                    text_color = "#721c24" if neighbor['label'].lower() == 'spam' else "#155724"
                    
                    st.markdown(f"""
                    <div class="neighbor-card">
                        <strong>Top {idx} - Độ tương đồng hình học (Cosine Score):</strong> <code style="color:blue;">{neighbor['score']:.4f}</code> | 
                        Nhãn gốc: <span style="background-color:{badge_color}; color:{text_color}; padding:2px 6px; border-radius:5px; font-weight:bold;">{neighbor['label'].upper()}</span>
                        <br><p style="margin-top:8px; color:#555; font-style:italic;">"Nội dung đối chiếu: {neighbor['message']}"</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Vui lòng điền nội dung tin nhắn vào ô văn bản trước khi phân tích!")

# --- THANH SIDEBAR THÀNH VIÊN ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179543.png", width=90)
st.sidebar.title("Quản lý Đồ án")
st.sidebar.markdown("---")
st.sidebar.markdown("### **Thành viên thực hiện:**")
st.sidebar.write("1. 👩‍🎓 Huỳnh Lê Hoàng Yến - *022101091*")  
st.sidebar.write("2. 👨‍🎓 Phạm Minh Tuấn - *022101006*")  
st.sidebar.write("3. 👨‍🎓 Huỳnh Văn Đăng Khoa - *022101111*")
