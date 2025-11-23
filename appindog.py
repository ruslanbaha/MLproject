import streamlit as st
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torchvision import transforms
import os
import io
import base64

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
    .share-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

translations = {
    "th": {
        "title": "🐕 DogDetect AI",
        "subtitle": "ตรวจจับภาพน้องหมาว่าเป็น 'ภาพ AI'",
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
        "error_model": "❌ ไม่พบไฟล์โมเดล (.pth) หรือไฟล์เสียหาย",
        "share_title": "📤 แชร์ผลลัพธ์",
        "download_btn": "ดาวน์โหลดรูปผลลัพธ์ (Image)",
        "copy_link": "คัดลอกลิงก์ผลลัพธ์",
        "link_copied": "คัดลอกลิงก์แล้ว!",
        "shared_view": "🔗 นี่คือผลลัพธ์ที่ถูกแชร์มา",
        "shared_note": "(รูปภาพต้นฉบับไม่สามารถแสดงได้ในโหมดลิงก์แชร์ เนื่องจากไม่ได้ถูกบันทึกบนเซิร์ฟเวอร์)"
    },
    "en": {
        "title": "🐕 DogDetect AI",
        "subtitle": "Detect if a dog image is 'AI Generated'",
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
        "error_model": "❌ Model file (.pth) not found.",
        "share_title": "📤 Share Result",
        "download_btn": "Download Result Image",
        "copy_link": "Copy Share Link",
        "link_copied": "Link Copied!",
        "shared_view": "🔗 Shared Result View",
        "shared_note": "(Original image cannot be displayed in shared link mode as it's not hosted.)"
    }
}


# ==========================================
# 2. LOGIC & FUNCTIONS
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
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    if prob < 0.5:
        return True, (1 - prob) * 100
    else:
        return False, (1 - prob) * 100


def create_result_card(original_image, is_ai, percent):
    # สร้าง Canvas พื้นหลังสีขาว
    width, height = 600, 800
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)

    # 1. Header สีพื้นหลัง
    header_color = "#ffebee" if is_ai else "#e8f5e9"
    draw.rectangle([0, 0, width, 150], fill=header_color)

    # 2. ใส่รูป User (Resize & Center)
    # Resize ให้กว้าง 500px
    base_width = 500
    w_percent = (base_width / float(original_image.size[0]))
    h_size = int((float(original_image.size[1]) * float(w_percent)))
    img_resized = original_image.resize((base_width, h_size), Image.Resampling.LANCZOS)

    # ถ้าสูงเกินไป ให้ Crop
    if h_size > 400:
        img_resized = ImageOps.fit(original_image, (500, 400), Image.Resampling.LANCZOS)
        h_size = 400

    y_pos = 180
    x_pos = (width - img_resized.width) // 2
    card.paste(img_resized, (x_pos, y_pos))

    # 3. วาดขอบรูป
    draw.rectangle([x_pos - 5, y_pos - 5, x_pos + img_resized.width + 5, y_pos + img_resized.height + 5],
                   outline="#bdc3c7", width=5)

    # 4. Text (ใช้ default font แต่ปรับขนาดไม่ได้ใน standard PIL ถ้าไม่มีไฟล์ .ttf)
    # เพื่อความง่ายและรันได้ทุกเครื่อง เราจะใช้ LoadDefault แต่เทคนิคคือวาดใหญ่ๆ
    # หากต้องการสวยงามต้องมีไฟล์ font.ttf ในโฟลเดอร์ แต่ขอใช้แบบ default เพื่อความชัวร์ว่ารันผ่าน

    # Text Config
    text_result = "AI GENERATED" if is_ai else "REAL PHOTO"
    text_color = "#c62828" if is_ai else "#2e7d32"

    # เนื่องจากข้อจำกัด Default Font เล็กมาก เราจะใช้ Logic ง่ายๆ ในการแปะ Text
    # หรือถ้า Server มี Font เราจะใช้ (ในที่นี้ขอสมมติว่าไม่มีเพื่อความ Safe)
    # *แต่วิธีที่ดีที่สุดสำหรับ Card คือใช้ HTML Render หรือหา Font มาใส่*
    # *ในโค้ดนี้ผมจะพยายามโหลด Font มาตรฐานของระบบ*

    try:
        # พยายามโหลด Font มาตรฐาน (รองรับ Eng)
        font_large = ImageFont.truetype("arial.ttf", 60)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except:
        try:
            # Linux path
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        except:
            # Fallback สุดท้าย
            font_large = None
            font_small = None

    # Helper function วาด text กึ่งกลาง
    def draw_centered_text(text, y, font, fill):
        if font:
            # ใช้ textbbox เพื่อหาขนาด (PIL เวอร์ชั่นใหม่)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(((width - text_w) / 2, y), text, font=font, fill=fill)
        else:
            # Fallback แบบไม่มี Font (ตัวเล็กหน่อย)
            draw.text((width / 2 - 50, y), text, fill=fill)

    # วาดข้อความ
    draw_centered_text("DogDetect AI", 50, font_large, "black")

    result_y = y_pos + h_size + 40
    draw_centered_text(text_result, result_y, font_large, text_color)

    score_text = f"Confidence: {percent:.1f}%"
    draw_centered_text(score_text, result_y + 80, font_small, "#7f8c8d")

    draw_centered_text("Scan at: dogdetect-ai.streamlit.app", height - 60, font_small, "#bdc3c7")

    return card


# ==========================================
# 3. MAIN APP FLOW
# ==========================================

# 1. Check Query Params (โหมดลิงก์แชร์)
query_params = st.query_params
is_shared_mode = "shared" in query_params and query_params["shared"] == "true"

# 2. Init State
if 'lang' not in st.session_state: st.session_state.lang = 'th'
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'result_is_ai' not in st.session_state: st.session_state.result_is_ai = None
if 'result_percent' not in st.session_state: st.session_state.result_percent = None
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None

# Sidebar Language
with st.sidebar:
    st.header("Settings ⚙️")
    current_index = 0 if st.session_state.lang == 'th' else 1
    lang_choice = st.radio("Language / ภาษา", ["ภาษาไทย", "English"], index=current_index)
    selected_lang_code = 'th' if lang_choice == "ภาษาไทย" else 'en'
    if selected_lang_code != st.session_state.lang:
        st.session_state.lang = selected_lang_code
        st.rerun()

t = translations[st.session_state.lang]

# Cookie
if st.session_state.cookie_consent is None:
    with st.container():
        st.markdown(f"""<div class="cookie-box"><div>{t['cookie_text']}</div></div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([6, 1, 1])
        if c2.button(t['accept']): st.session_state.cookie_consent = True; st.rerun()
        if c3.button(t['decline']): st.session_state.cookie_consent = False; st.rerun()

# Header
st.markdown(f"""<div class="main-header"><h1>{t['title']}</h1><p>{t['subtitle']}</p></div>""", unsafe_allow_html=True)

# -------------------------------------------
# CASE A: SHARED LINK MODE (เปิดจากลิงก์)
# -------------------------------------------
if is_shared_mode:
    st.info(f"**{t['shared_view']}**")

    # ดึงค่าจาก URL
    try:
        shared_ai = query_params.get("ai") == "true"
        shared_score = float(query_params.get("score"))
    except:
        shared_ai = False
        shared_score = 0.0

    # แสดงผลลัพธ์ (Copy Logic จากด้านล่างมา)
    if shared_ai:
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
            f"""<div class="result-card"><div style="color:#7f8c8d;">{t['ai_prob']}</div><div class="score-big" style="background:-webkit-linear-gradient(45deg,#2c3e50,{score_color});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{shared_score:.1f}%</div></div>""",
            unsafe_allow_html=True)

    st.caption(t['shared_note'])

    if st.button("🏠 Home / Start New"):
        st.query_params.clear()
        st.rerun()

# -------------------------------------------
# CASE B: NORMAL MODE (อัปโหลดเอง)
# -------------------------------------------
else:
    # Load Model
    model, error = load_pytorch_model()

    if model is None:
        st.error(t['error_model'])
        if error: st.warning(f"Error Detail: {error}")
    else:
        uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.analysis_done = False
                st.session_state.result_is_ai = None
                st.session_state.result_percent = None

            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Preview", use_container_width=True)

            if not st.session_state.analysis_done:
                if st.button(f"⚡ {t['btn_start']}", type="primary", use_container_width=True):
                    progress_text = t['processing']
                    my_bar = st.progress(0, text=progress_text)
                    for i in range(100):
                        time.sleep(0.01)
                        my_bar.progress(i + 1)

                    is_ai, ai_percent = predict_image(model, image)

                    st.session_state.result_is_ai = is_ai
                    st.session_state.result_percent = ai_percent
                    st.session_state.analysis_done = True
                    my_bar.empty()
                    st.rerun()
            else:
                st.markdown(f"""<div class="success-box">{t['btn_done']}</div>""", unsafe_allow_html=True)

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

                # ==========================================
                # 4. SHARE & DOWNLOAD SECTION
                # ==========================================
                st.markdown(f"<h4 style='text-align:center; margin-top:20px;'>{t['share_title']}</h4>",
                            unsafe_allow_html=True)

                # A. สร้างรูป Result Card สำหรับดาวน์โหลด
                card_img = create_result_card(image, is_ai, ai_percent)

                # แปลงรูปเป็น Bytes เพื่อใส่ในปุ่ม Download
                buf = io.BytesIO()
                card_img.save(buf, format="PNG")
                byte_im = buf.getvalue()

                col_share1, col_share2 = st.columns(2)

                with col_share1:
                    st.download_button(
                        label=f"⬇️ {t['download_btn']}",
                        data=byte_im,
                        file_name="dogdetect_result.png",
                        mime="image/png",
                        use_container_width=True,
                        type="secondary"
                    )

                # B. สร้างลิงก์แชร์ (Query Params)
                # หมายเหตุ: ลิงก์นี้จะใช้ได้จริงเมื่อ Deploy ขึ้น Server แล้ว (localhost ส่งให้เพื่อนไม่ได้)
                # เราจะใช้ base_url ของ Streamlit (ซึ่งถ้า run local มันคือ localhost)

                # สร้าง URL string
                base_url = "http://localhost:8501"  # หรือ URL จริงของคุณถ้า Deploy แล้ว
                # ตรวจสอบว่ารันบน Cloud หรือไม่ (Optional logic)

                is_ai_str = "true" if is_ai else "false"
                query_str = f"?shared=true&ai={is_ai_str}&score={ai_percent:.2f}"

                # เนื่องจาก st.write(link) มันไม่สวย เราใช้ Code block ให้ user copy
                with col_share2:
                    # สร้าง URL ปัจจุบันแบบ Dynamic (พยายาม)
                    st.write("🔗 **Link:**")
                    st.code(f"{base_url}/{query_str}", language="text")
                    st.caption("*(Copy URL นี้ส่งให้เพื่อนดูคะแนนได้เลย)*")