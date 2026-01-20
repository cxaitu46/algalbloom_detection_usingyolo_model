import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Try importing YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Tech Powered Aquaculture",
    page_icon="🐟",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.info-box {background:#e7f3ff;padding:15px;border-left:5px solid #0066cc;border-radius:8px;}
.success-box {background:#d4edda;padding:15px;border-left:5px solid #28a745;border-radius:8px;}
.warning-box {background:#fff3cd;padding:15px;border-left:5px solid #ffc107;border-radius:8px;}
.danger-box {background:#f8d7da;padding:15px;border-left:5px solid #dc3545;border-radius:8px;}
h1,h2,h3 {color:#0066cc;}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🐟 Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["PPE in Aquaculture", "Water Quality Checks", "Algal Bloom Detection"]
)

st.sidebar.info(
    "Offline AI system for safe & green aquaculture.\n\n"
  
    "✔ YOLOv8 Local Model"
)

# =========================
# TITLE
# =========================
st.title("🌊 Tech Powered Approaches to Safe & Green Aquaculture")

# ======================================================
# PAGE 1: PPE
# ======================================================
if page == "PPE in Aquaculture":
    st.header("👷 Personal Protective Equipment (PPE)")

    st.markdown("""
    <div class="info-box">
    <h3>Why PPE is Important</h3>
    <ul>
        <li>Protection from biological hazards</li>
        <li>Prevents chemical exposure</li>
        <li>Reduces physical injuries</li>
        <li>Ensures worker safety in wet conditions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <h3>Recommended PPE</h3>
    <ul>
        <li>🧤 Waterproof Gloves</li>
        <li>👢 Rubber Boots</li>
        <li>🥽 Safety Goggles</li>
        <li>😷 Face Masks</li>
        <li>🦺 Protective Aprons</li>
        <li>🧥 Waterproof Jackets</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE 2: WATER QUALITY
# ======================================================
elif page == "Water Quality Checks":
    st.header("💧 Water Quality Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.slider("Water pH", 0.0, 14.0, 7.0, 0.1)

    with col2:
        color = st.selectbox(
            "Water Color",
            ["Clear", "Light Green", "Dark Green", "Brown", "Murky"]
        )

    with col3:
        smell = st.selectbox(
            "Water Smell",
            ["Fresh", "Earthy", "Fishy", "Rotten"]
        )

    if st.button("Diagnose Water Quality"):
        if 6.5 <= ph <= 9:
            status = "GOOD"
            box = "success-box"
        elif 5.5 <= ph < 6.5 or 9 < ph <= 10:
            status = "MODERATE"
            box = "warning-box"
        else:
            status = "CRITICAL"
            box = "danger-box"

        st.markdown(f"""
        <div class="{box}">
        <h3>Water Quality Status: {status}</h3>
        <p><strong>pH:</strong> {ph}</p>
        <p><strong>Color:</strong> {color}</p>
        <p><strong>Smell:</strong> {smell}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h4>Recommendations</h4>
        <ul>
            <li>Maintain proper aeration</li>
            <li>Monitor ammonia & nitrite</li>
            <li>Control feeding</li>
            <li>Perform partial water exchange</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# PAGE 3: ALGAL BLOOM DETECTION
# ======================================================
elif page == "Algal Bloom Detection":
    st.header("🔬 Algal Bloom Detection (YOLOv8)")

    if not YOLO_AVAILABLE:
        st.error("Ultralytics YOLO not installed. Run: pip install ultralytics")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Pond Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width="stretch", caption="Uploaded Image")

        if st.button("Analyze Image"):
            try:
                model = YOLO("algaldetection.pt")

                img_np = np.array(image)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                results = model(img_bgr)
                detections = results[0].boxes

                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, width="stretch", caption="Detection Result")

                if len(detections) > 0:
                    count = len(detections)
                    severity = (
                        "SEVERE" if count >= 5 else
                        "MODERATE" if count >= 2 else
                        "MILD"
                    )

                    st.markdown(f"""
                    <div class="danger-box">
                    <h3>🚨 Algal Bloom Detected</h3>
                    <p>Detected Regions: {count}</p>
                    <p>Severity: {severity}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="warning-box">
                    <h4>Immediate Actions</h4>
                    <ul>
                        <li>Increase aeration immediately</li>
                        <li>Reduce feeding</li>
                        <li>Partial water exchange</li>
                        <li>Monitor dissolved oxygen</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="success-box">
                    <h3>✅ No Algal Bloom Detected</h3>
                    <p>Water condition appears normal.</p>
                    </div>
                    """, unsafe_allow_html=True)

            except FileNotFoundError:
                st.error("algaldetection.pt not found in project folder.")
            except Exception as e:
                st.error(f"Error during detection: {e}")

    else:
        st.info("Upload a pond image to begin detection.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;">
<b>Tech Powered Aquaculture</b><br>
Offline YOLOv8 Based Detection System<br>
Educational & Research Use
</div>
""", unsafe_allow_html=True)
