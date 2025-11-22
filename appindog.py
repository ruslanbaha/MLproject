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

# ... (CSS เดิมใช้ได้เลยครับ ไม่ต้องแก้) ...
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
        "subtitle": "ตรวจจับภาพน้องหมาว่าเป็น 'ภาพ AI' หรือไม่ (PyTorch Version)",
        "upload_label": "อัปโหลดภาพน้องหมา (Drag & Drop)",
        "analyzing": "กำลังประมวลผล...",
        "result_title": "ผลการวิเคราะห์ AI",
        "ai_prob": "ความเป็น AI",
        "type": "ประเภท",
        "type_ai": "🤖 ภาพจาก AI (Generated)",
        "type_real": "📸 ภาพถ่ายจริง (Real Photo)",
        "share": "แชร์ผลลัพธ์",
        "cookie_text": "🍪 เว็บไซต์นี้ใช้คุกกี้เพื่อพัฒนาโมเดล AI",
        "accept": "ยอมรับ",
        "decline": "ไม่ยอมรับ",
        "sensitive_title": "⚠️ แจ้งเตือนภาพละเอียดอ่อน",
        "sensitive_msg": "ตรวจพบเนื้อหาละเอียดอ่อน ยืนยันที่จะทำต่อ?",
        "btn_continue": "ยืนยัน / ทำต่อ",
        "btn_cancel": "ยกเลิก",
        "error_model": "❌ ไม่พบไฟล์โมเดล (.pth) หรือไฟล์เสียหาย"
    },
    "en": {
        "title": "🐕 DogDetect AI",
        "subtitle": "Detect if a dog image is 'AI Generated' or Real (PyTorch)",
        "upload_label": "Upload Dog Image",
        "analyzing": "Processing...",
        "result_title": "AI Analysis Result",
        "ai_prob": "AI Probability",
        "type": "Type",
        "type_ai": "🤖 AI Generated",
        "type_real": "📸 Real Photo",
        "share": "Share Result",
        "cookie_text": "🍪 Cookies used.",
        "accept": "Accept",
        "decline": "Decline",
        "sensitive_title": "⚠️ Sensitive Warning",
        "sensitive_msg": "Proceed with sensitive content?",
        "btn_continue": "Confirm",
        "btn_cancel": "Cancel",
        "error_model": "❌ Model file (.pth) not found."
    }
}


# ==========================================
# 2. LOGIC & FUNCTIONS (PYTORCH)
# ==========================================
@st.cache_resource
def load_pytorch_model():
    # 1. ประกาศตัวแปร model_path ให้ถูกต้องภายในฟังก์ชัน
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'dog_model_pytorch.pth')

    # 2. เช็คว่าไฟล์มีไหม
    if not os.path.exists(model_path):
        return None, f"File not found at: {model_path}"

    try:
        # 3. โหลดโมเดล (ใส่ weights_only=False ตรงนี้)
        # map_location='cpu' เพื่อให้รันได้แม้ไม่มี GPU
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()  # ปรับเป็น mode ประมวลผล
        return model, None
    except Exception as e:
        return None, str(e)


def predict_image(model, image):
    # Preprocess ให้เหมือนตอนเทรนเป๊ะๆ
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # แปลงภาพ
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # เพิ่ม Batch dimension (1, 3, 224, 224)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()  # แปลง Logits เป็น Probability (0-1)

    # Logic:
    # ตอนเทรน Class 0 = ai, Class 1 = real (เพราะมันเรียงตามตัวอักษร)
    # ถ้า prob < 0.5 คือค่อนไปทาง AI
    # ถ้า prob > 0.5 คือค่อนไปทาง Real

    if prob < 0.5:
        is_ai = True
        ai_percent = (1 - prob) * 100
    else:
        is_ai = False
        ai_percent = (1 - prob) * 100  # ถ้าเป็น Real ก็โชว์เปอร์เซ็นต์ AI น้อยๆ (หรือจะโชว์ Real % ก็ได้แล้วแต่ดีไซน์)

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

# Load Model
model, error = load_pytorch_model()

# Sidebar
with st.sidebar:
    st.header("Settings ⚙️")
    lang_choice = st.radio("Language / ภาษา", ["ภาษาไทย", "English"])
    st.session_state.lang = 'en' if lang_choice == "English" else 'th'
    if lang_choice != ("ภาษาไทย" if st.session_state.lang == 'th' else "English"): st.rerun()

# Cookie
if st.session_state.cookie_consent is None:
    with st.container():
        st.markdown(f"""<div class="cookie-box"><div>{t['cookie_text']}</div></div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([6, 1, 1])
        if c2.button(t['accept']): st.session_state.cookie_consent = True; st.rerun()
        if c3.button(t['decline']): st.session_state.cookie_consent = False; st.rerun()

st.markdown(f"""<div class="main-header"><h1>{t['title']}</h1><p>{t['subtitle']}</p></div>""", unsafe_allow_html=True)

if model is None:
    st.error(t['error_model'])
    if error: st.warning(f"Error Detail: {error}")
else:
    uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')  # บังคับแปลงเป็น RGB กัน Error
        st.image(image, caption="Preview", use_container_width=True)

        # Sensitive check logic (เหมือนเดิม)
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.is_sensitive = check_sensitive_content(image)
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.sensitive_confirmed = False

        if st.session_state.is_sensitive and not st.session_state.sensitive_confirmed:
            st.warning(f"**{t['sensitive_title']}**")
            st.write(t['sensitive_msg'])
            c1, c2 = st.columns(2)
            if c1.button(t['btn_continue'], type="primary"): st.session_state.sensitive_confirmed = True; st.rerun()
            if c2.button(t['btn_cancel']): st.session_state.last_uploaded = None; st.rerun()
            st.stop()

        if st.button("🚀 " + t['analyzing'].replace("...", ""), type="primary", use_container_width=True):
            my_bar = st.progress(0, text=t['analyzing'])
            for i in range(100):
                time.sleep(0.01)
                my_bar.progress(i + 1)

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
                    f"""<div class="result-card"><div style="color:#7f8c8d;">{t['type']}</div><div class="label-badge {badge_class}">{badge_text}</div></div>""",
                    unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"""<div class="result-card"><div style="color:#7f8c8d;">{t['ai_prob']}</div><div class="score-big" style="background:-webkit-linear-gradient(45deg,#2c3e50,{score_color});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{ai_percent:.1f}%</div></div>""",
                    unsafe_allow_html=True)

            # Share buttons... (เหมือนเดิม)