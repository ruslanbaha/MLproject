import streamlit as st
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os

# ==========================================
# 1. SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="DogDetect AI - Real vs AI",
    page_icon="🐕",
    layout="centered"
)

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

    /* Custom Button Styles for Finished State */
    .success-box {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- แก้ไขข้อความ Subtitle ตรงนี้ครับ ---
translations = {
    "th": {
        "title": "🐕 DogDetect AI",
        "subtitle": "ตรวจจับภาพน้องหมาว่าเป็น 'ภาพ AI'",  # ลบข้อความส่วนเกินออกแล้ว
        "upload_label": "อัปโหลดภาพน้องหมา (Drag & Drop)",
        "btn_start": "ประมวลผล",
        "processing": "กำลังประมวลผล...",
        "btn_done": "✅ ประมวลผลเสร็จสิ้น",
        "result_title": "ผลการวิเคราะห์ AI",
        "ai_prob": "ความเป็น AI",
        "type": "ประเภท",
        "type_ai": "🤖 ภาพจาก AI (Generated)",
        "type_real": "📸 ภาพถ่ายจริง (Real Photo)",
        "cookie_text": "🍪 เว็บไซต์นี้ใช้คุกกี้เพื่อพัฒนาโมเดล AI",
        "accept": "ยอมรับ",
        "decline": "ไม่ยอมรับ",
        "error_model": "❌ ไม่พบไฟล์โมเดล (.pth) หรือไฟล์เสียหาย"
    },
    "en": {
        "title": "🐕 DogDetect AI",
        "subtitle": "Detect if a dog image is 'AI Generated'",  # ปรับให้สั้นกระชับเช่นกัน
        "upload_label": "Upload Dog Image",
        "btn_start": "Analyze",
        "processing": "Processing...",
        "btn_done": "✅ Analysis Complete",
        "result_title": "AI Analysis Result",
        "ai_prob": "AI Probability",
        "type": "Type",
        "type_ai": "🤖 AI Generated",
        "type_real": "📸 Real Photo",
        "cookie_text": "🍪 Cookies used.",
        "accept": "Accept",
        "decline": "Decline",
        "error_model": "❌ Model file (.pth) not found."
    }
}


# ==========================================
# 2. LOGIC & FUNCTIONS (PYTORCH)
# ==========================================
@st.cache_resource
def load_pytorch_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'dog_model_pytorch.pth')

    if not os.path.exists(model_path):
        return None, f"File not found at: {model_path}"

    try:
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


def predict_image(model, image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    if prob < 0.5:
        is_ai = True
        ai_percent = (1 - prob) * 100
    else:
        is_ai = False
        ai_percent = (1 - prob) * 100

    return is_ai, ai_percent


# ==========================================
# 3. MAIN APP FLOW
# ==========================================
# 1. Init State
if 'lang' not in st.session_state: st.session_state.lang = 'th'
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'result_is_ai' not in st.session_state: st.session_state.result_is_ai = None
if 'result_percent' not in st.session_state: st.session_state.result_percent = None
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None

# 2. Sidebar & Language Logic (ย้ายมาไว้ตรงนี้เพื่อให้เปลี่ยนภาษาทันที)
with st.sidebar:
    st.header("Settings ⚙️")
    # หา Index ปัจจุบันเพื่อให้ Radio Button แสดงถูกต้อง
    current_index = 0 if st.session_state.lang == 'th' else 1
    lang_choice = st.radio("Language / ภาษา", ["ภาษาไทย", "English"], index=current_index)

    # แปลงค่าที่เลือกกลับเป็นรหัสภาษา
    selected_lang_code = 'th' if lang_choice == "ภาษาไทย" else 'en'

    # ถ้าค่าเปลี่ยน ให้ update state และ rerun ทันที
    if selected_lang_code != st.session_state.lang:
        st.session_state.lang = selected_lang_code
        st.rerun()

# 3. Load Text Dictionary (ดึงหลังจากจัดการภาษาเสร็จแล้ว)
t = translations[st.session_state.lang]

# Load Model
model, error = load_pytorch_model()

# Cookie Banner
if st.session_state.cookie_consent is None:
    with st.container():
        st.markdown(f"""<div class="cookie-box"><div>{t['cookie_text']}</div></div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([6, 1, 1])
        if c2.button(t['accept']): st.session_state.cookie_consent = True; st.rerun()
        if c3.button(t['decline']): st.session_state.cookie_consent = False; st.rerun()

# Main Header
st.markdown(f"""<div class="main-header"><h1>{t['title']}</h1><p>{t['subtitle']}</p></div>""", unsafe_allow_html=True)

if model is None:
    st.error(t['error_model'])
    if error: st.warning(f"Error Detail: {error}")
else:
    uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        # เช็คว่าไฟล์เปลี่ยนหรือไม่ หากเปลี่ยนให้รีเซ็ตผลลัพธ์และปุ่ม
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.analysis_done = False
            st.session_state.result_is_ai = None
            st.session_state.result_percent = None

        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Preview", use_container_width=True)

        # Logic การแสดงปุ่ม
        if not st.session_state.analysis_done:
            # ปุ่มประมวลผล (Start)
            if st.button(f"⚡ {t['btn_start']}", type="primary", use_container_width=True):
                # 1. แสดง Progress bar
                progress_text = t['processing']
                my_bar = st.progress(0, text=progress_text)

                for i in range(100):
                    time.sleep(0.01)
                    my_bar.progress(i + 1)

                # 2. ประมวลผล
                is_ai, ai_percent = predict_image(model, image)

                # 3. เก็บค่าลง Session State
                st.session_state.result_is_ai = is_ai
                st.session_state.result_percent = ai_percent
                st.session_state.analysis_done = True

                my_bar.empty()
                st.rerun()  # รีเฟรชหน้าเพื่อเปลี่ยนปุ่มเป็นสีเขียว
        else:
            # ปุ่มสีเขียว (Finished)
            st.markdown(f"""<div class="success-box">{t['btn_done']}</div>""", unsafe_allow_html=True)

            # แสดงผลลัพธ์
            is_ai = st.session_state.result_is_ai
            ai_percent = st.session_state.result_percent

            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center;'>{t['result_title']}</h3>", unsafe_allow_html=True)

            if is_ai:
                badge_class, badge_text, score_color = "badge-ai", t['type_ai'], "#c62828"
            else:
                badge_class, badge_text, score_color = "badge-real", t['type_real'], "#2e7d32"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""<div class="result-card"><div style="color:#7f8c8d;">{t['type']}</div><div class="label-badge {badge_class}">{badge_text}</div></div>""",
                    unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"""<div class="result-card"><div style="color:#7f8c8d;">{t['ai_prob']}</div><div class="score-big" style="background:-webkit-linear-gradient(45deg,#2c3e50,{score_color});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{ai_percent:.1f}%</div></div>""",
                    unsafe_allow_html=True)