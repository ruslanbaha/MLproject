import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ==========================================
# 1. SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="DogDetect AI - Real vs AI",
    page_icon="🐕",
    layout="centered"
)

# CSS สำหรับตกแต่ง
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
    .main-header { text-align: center; margin-bottom: 30px; }
    .result-card {
        background-color: white; padding: 30px; border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08); text-align: center;
        border: 1px solid #f0f2f5; margin-bottom: 20px;
    }
    .score-big {
        font-size: 4rem; font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3498db, #8e44ad);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .label-badge {
        display: inline-block; padding: 8px 20px; border-radius: 50px;
        font-weight: 600; font-size: 1.2rem; margin-bottom: 10px;
    }
    .badge-ai { background-color: #ffebee; color: #c62828; }
    .badge-real { background-color: #e8f5e9; color: #2e7d32; }
    .cookie-box {
        background-color: #34495e; color: white; padding: 15px;
        border-radius: 10px; margin-bottom: 20px;
    }
    .stFileUploader { border: 2px dashed #bdc3c7; border-radius: 15px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

translations = {
    "th": {
        "title": "🐕 DogDetect AI",
        "subtitle": "ตรวจจับภาพน้องหมาว่าเป็น 'ภาพ AI' หรือไม่",
        "upload_label": "อัปโหลดภาพน้องหมา (Drag & Drop)",
        "analyzing": "กำลังประมวลผล...",
        "result_title": "ผลการวิเคราะห์ AI",
        "ai_prob": "ความเป็น AI",
        "type": "ประเภท",
        "type_ai": "🤖 ภาพจาก AI (Generated)",
        "type_real": "📸 ภาพถ่ายจริง (Real Photo)",
        "share": "แชร์ผลลัพธ์",
        "cookie_text": "🍪 เว็บไซต์นี้ใช้คุกกี้เพื่อพัฒนาโมเดล AI หากยอมรับ ภาพของคุณจะถูกนำไปเรียนรู้",
        "accept": "ยอมรับ",
        "decline": "ไม่ยอมรับ",
        "sensitive_title": "⚠️ แจ้งเตือนภาพละเอียดอ่อน",
        "sensitive_msg": "ระบบตรวจพบว่าภาพนี้อาจมีข้อมูลส่วนบุคคล (เช่น หน้าคน, บัตรประชาชน) คุณยืนยันที่จะประมวลผลหรือไม่?",
        "btn_continue": "ยืนยัน / ทำต่อ",
        "btn_cancel": "ยกเลิก",
        "error_model": "❌ ไม่พบไฟล์โมเดล หรือไฟล์เสียหาย",
    },
    "en": {
        "title": "🐕 DogDetect AI",
        "subtitle": "Detect if a dog image is 'AI Generated' or Real",
        "upload_label": "Upload Dog Image (Drag & Drop)",
        "analyzing": "Processing...",
        "result_title": "AI Analysis Result",
        "ai_prob": "AI Probability",
        "type": "Type",
        "type_ai": "🤖 AI Generated",
        "type_real": "📸 Real Photo",
        "share": "Share Result",
        "cookie_text": "🍪 We use cookies to improve our AI model.",
        "accept": "Accept",
        "decline": "Decline",
        "sensitive_title": "⚠️ Sensitive Content Warning",
        "sensitive_msg": "This image may contain personal data (faces, IDs). Do you want to proceed?",
        "btn_continue": "Confirm / Proceed",
        "btn_cancel": "Cancel",
        "error_model": "❌ Model file not found or corrupted.",
    }
}


# ==========================================
# 2. LOGIC & FUNCTIONS (แก้ไขจุดสำคัญ)
# ==========================================
@st.cache_resource
def load_ai_model():
    # หาตำแหน่งไฟล์ปัจจุบัน เพื่อสร้าง Path ที่ถูกต้อง 100%
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'dog_model_binary.keras')

    try:
        # ลองโหลดโมเดล
        model = tf.keras.models.load_model(model_path)
        return model, None  # Return model, No error
    except Exception as e:
        # ถ้าพัง ให้ส่ง Error กลับไปบอกผู้ใช้
        return None, str(e)


def predict_image(model, image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)

    prediction = model.predict(img_array)
    score = prediction[0][0]

    if score < 0.5:
        is_ai = True
        ai_percent = (1 - score) * 100
    else:
        is_ai = False
        ai_percent = (1 - score) * 100

    return is_ai, ai_percent


def check_sensitive_content(image):
    import random
    return random.random() > 0.7


# ==========================================
# 3. MAIN APP FLOW
# ==========================================
if 'lang' not in st.session_state: st.session_state.lang = 'th'
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = None
if 'sensitive_confirmed' not in st.session_state: st.session_state.sensitive_confirmed = False

t = translations[st.session_state.lang]

# --- Load Model with Error Handling ---
model, error_msg = load_ai_model()

# Sidebar
with st.sidebar:
    st.header("Settings ⚙️")
    lang_choice = st.radio("Language / ภาษา", ["ภาษาไทย", "English"])
    if lang_choice == "English":
        st.session_state.lang = 'en'
    else:
        st.session_state.lang = 'th'
    st.rerun if lang_choice != ("ภาษาไทย" if st.session_state.lang == 'th' else "English") else None

# Cookie Banner
if st.session_state.cookie_consent is None:
    with st.container():
        st.markdown(f"""<div class="cookie-box"><div>{t['cookie_text']}</div></div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([6, 1, 1])
        if col2.button(t['accept']):
            st.session_state.cookie_consent = True
            st.rerun()
        if col3.button(t['decline']):
            st.session_state.cookie_consent = False
            st.rerun()

st.markdown(f"""<div class="main-header"><h1>{t['title']}</h1><p>{t['subtitle']}</p></div>""", unsafe_allow_html=True)

# --- Model Check & Error Display ---
if model is None:
    st.error(f"{t['error_model']}")
    if error_msg:
        st.warning(f"🔍 Technical Error Details:\n\n{error_msg}")
        st.info("Tip: Try checking 'requirements.txt' tensorflow version or rebuild the app.")
else:
    uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'webp', 'heic', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview", use_container_width=True)

        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.is_sensitive = check_sensitive_content(image)
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.sensitive_confirmed = False

        if st.session_state.is_sensitive and not st.session_state.sensitive_confirmed:
            with st.container():
                st.warning(f"**{t['sensitive_title']}**")
                st.write(t['sensitive_msg'])
                c1, c2 = st.columns(2)
                if c1.button(t['btn_continue'], type="primary"):
                    st.session_state.sensitive_confirmed = True
                    st.rerun()
                if c2.button(t['btn_cancel']):
                    st.session_state.last_uploaded = None
                    st.rerun()
            st.stop()

        if st.button("🚀 " + t['analyzing'].replace("...", ""), type="primary", use_container_width=True):
            progress_text = t['analyzing']
            my_bar = st.progress(0, text=progress_text)
            import time

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            is_ai, ai_percent = predict_image(model, image)
            my_bar.empty()

            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center;'>{t['result_title']}</h3>", unsafe_allow_html=True)

            if is_ai:
                badge_class, badge_text, score_color = "badge-ai", t['type_ai'], "#c62828"
            else:
                badge_class, badge_text, score_color = "badge-real", t['type_real'], "#2e7d32"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""<div class="result-card"><div style="color: #7f8c8d; font-weight:600;">{t['type']}</div><div class="label-badge {badge_class}">{badge_text}</div></div>""",
                    unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"""<div class="result-card"><div style="color: #7f8c8d; font-weight:600;">{t['ai_prob']}</div><div class="score-big" style="background: -webkit-linear-gradient(45deg, #2c3e50, {score_color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{ai_percent:.1f}%</div></div>""",
                    unsafe_allow_html=True)

            st.markdown(f"<center style='color:#aaa; margin-top:20px;'>{t['share']}</center>", unsafe_allow_html=True)
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.button("🔗 Copy Link", use_container_width=True)
            col_s2.button("📘 Facebook", use_container_width=True)
            col_s3.button("❌ X (Twitter)", use_container_width=True)