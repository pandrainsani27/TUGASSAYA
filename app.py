import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Mushroom AI Detector",
    page_icon="🍄",
    layout="wide"
)

# ==========================================
# 2. FUNGSI LOAD MODEL PINTAR (AUTO-DETECT)
# ==========================================
@st.cache_resource
def load_model_smart():
    # Cari lokasi file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "model.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"❌ File tidak ditemukan di: {model_path}")
        return None, None

    try:
        loaded_object = joblib.load(model_path)
        
        # SKENARIO A: File berisi Dictionary {"model": X, "encoder": Y}
        if isinstance(loaded_object, dict):
            return loaded_object.get("model"), loaded_object.get("encoder")
        
        # SKENARIO B: File langsung berupa Model/Pipeline (Penyebab error kamu sebelumnya)
        else:
            return loaded_object, None

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# Load Data
model, encoder = load_model_smart()

if model is None:
    st.stop()

# ==========================================
# 3. TAMPILAN UTAMA
# ==========================================
st.title("🍄 Mushroom Classification AI")
st.write("Prediksi apakah jamur aman atau beracun.")

# ==========================================
# 4. FORM INPUT FLEKSIBEL
# ==========================================
with st.sidebar.form("input_form"):
    st.header("Masukkan Ciri-Ciri")
    inputs = {}
    
    # Cek fitur dari Model atau Encoder
    # Kita prioritaskan feature_names_in_ dari model
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
    elif encoder and hasattr(encoder, 'feature_names_in_'):
        features = encoder.feature_names_in_
    else:
        # Fallback darurat jika tidak ada nama fitur
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"] # Contoh dummy
        st.warning("⚠️ Nama fitur tidak terdeteksi otomatis. Menggunakan default.")

    # Loop membuat input
    for col in features:
        # LOGIKA DROPDOWN PINTAR
        # Kita coba cari kategori. Jika encoder ada, pakai encoder.
        # Jika tidak, kita cek apakah model itu Pipeline yang punya langkah preprocessor
        options = None
        
        if encoder and hasattr(encoder, 'categories_'):
             idx = list(features).index(col)
             if idx < len(encoder.categories_):
                 options = encoder.categories_[idx]
        
        # Tampilkan Input
        if options is not None:
            inputs[col] = st.selectbox(col, options)
        else:
            # Jika tidak ada opsi kategori, jadikan text input (aman dari crash)
            inputs[col] = st.text_input(col)

    submitted = st.form_submit_button("🔍 Prediksi")

# ==========================================
# 5. PREDIKSI
# ==========================================
if submitted:
    try:
        df = pd.DataFrame([inputs])
        
        # Jika encoder terpisah, transform dulu
        if encoder:
            X_final = encoder.transform(df)
        else:
            # Jika encoder menyatu di model (Pipeline), langsung masukkan df
            X_final = df
            
        pred = model.predict(X_final)[0]
        
        # Tampilkan Hasil
        if pred == 1:
            st.error("☠️ POISONOUS (Beracun)")
        else:
            st.success("🥗 EDIBLE (Aman Dimakan)")
            
    except Exception as e:
        st.error("Terjadi kesalahan saat prediksi.")
        st.info("Kemungkinan format input tidak cocok dengan model.")
        st.code(e)
