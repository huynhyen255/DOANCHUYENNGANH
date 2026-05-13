import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Cấu hình giao diện Web
st.set_page_config(page_title="Spam Detector AI", page_icon="🛡️")
st.title("🛡️ Ứng dụng Phân loại Tin nhắn Spam")
st.write("Đồ án chuyên ngành: Sử dụng thuật toán Naive Bayes")

# Logic AI (Y chang tài liệu của cô)
@st.cache_resource
def load_and_train_model():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    df = pd.read_csv("2cls_spam_text_cls.csv")
    messages = df["Message"].values.tolist()
    labels = df["Category"].values.tolist()

    def preprocess_text(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    processed_msgs = [preprocess_text(msg) for msg in messages]
    dictionary = []
    for tokens in processed_msgs:
        for token in tokens:
            if token not in dictionary: dictionary.append(token)

    def get_features(tokens):
        features = np.zeros(len(dictionary))
        for token in tokens:
            if token in dictionary: features[dictionary.index(token)] += 1
        return features

    X = np.array([get_features(t) for t in processed_msgs])
    le = LabelEncoder()
    y = le.fit_transform(labels)
    model = GaussianNB()
    model.fit(X, y)
    return model, dictionary, le, preprocess_text, get_features

model, dictionary, le, preprocess_fn, feature_fn = load_and_train_model()

# Giao diện nhập liệu
user_input = st.text_area("Nhập tin nhắn cần kiểm tra:", placeholder="Ví dụ: Free entry to win FA Cup...")

if st.button("Phân loại ngay"):
    if user_input:
        tokens = preprocess_fn(user_input)
        features = feature_fn(tokens)
        prediction = model.predict([features])
        result = le.inverse_transform(prediction)[0]
        if result == 'spam':
            st.error(f"CẢNH BÁO: Đây là tin nhắn SPAM!")
        else:
            st.success(f"THÔNG BÁO: Đây là tin nhắn An toàn (Ham).")