from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps

import config
import inference
import report

st.set_page_config(page_title="Damage Inspector", page_icon="🚗", layout="wide")
st.markdown(f"<style>{(config.BASE / 'assets' / 'style.css').read_text()}</style>",
            unsafe_allow_html=True)
IMAGE_TYPES = ["jpg", "jpeg", "jfif", "png", "webp", "bmp", "tif", "tiff", "gif", "heic", "heif"]

@st.cache_resource
def get_models():
    return inference.load_models()


yolo, resnet = get_models()

# Sidebar
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection sensitivity", 0.10, 0.90, config.YOLO_CONF, 0.05,
                     help="Lower = finds more (and possibly false) damage. Higher = only confident detections.")
    st.caption("Tip: use a clear, well-lit photo that shows the whole damaged area.")

# Hero
st.markdown("""
<div class="hero">
  <div class="tag">AI-POWERED INSPECTION</div>
  <h1>Vehicle <span>Damage</span> Inspector</h1>
  <p>Upload a photo of a vehicle. We locate each damaged area, identify the type of damage,
     rate the overall severity, and give you a downloadable assessment report.</p>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Upload a vehicle photo", type=IMAGE_TYPES,
                        label_visibility="collapsed")

if not file:
    st.markdown("""
    <div class="card empty">
      <div style="font-size:2.2rem">📷</div>
      <div style="margin-top:.4rem">Drop a photo above to start the inspection</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

try:
    image = inference.load_image(file)
except Exception:
    st.error("Sorry, this file couldn't be read as an image. Please try another photo.")
    st.stop()
with st.spinner("Inspecting vehicle..."):
    result = inference.analyze(image, yolo, resnet, conf=conf)

dets = report.enrich_detections(result["detections"], image.size)
sev = result["severity"]
color = report.SEVERITY_INFO[sev]["color"]
sev_conf = result["severity_probs"][sev] * 100

# Top stats
c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="card"><div class="label">Overall severity</div>
  <div class="value" style="color:{color}">{sev.title()}</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="card"><div class="label">Damage areas found</div>
  <div class="value">{len(dets)}</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="card"><div class="label">Severity confidence</div>
  <div class="value">{sev_conf:.0f}%</div></div>""", unsafe_allow_html=True)

# Images + details
left, right = st.columns([3, 2], gap="large")

with left:
    tab1, tab2 = st.tabs(["Detected damage", "Original"])
    tab1.image(result["annotated_image"], use_container_width=True)
    tab2.image(image, use_container_width=True)

with right:
    bars = ""
    for name, p in result["severity_probs"].items():
        c = report.SEVERITY_INFO[name]["color"]
        bars += f"""<div class="bar-row"><div class="bar-top"><span>{name.title()}</span>
        <span>{p * 100:.0f}%</span></div>
        <div class="bar"><span style="width:{p * 100:.0f}%;background:{c}"></span></div></div>"""
    st.markdown(f"""<div class="card"><div class="label">Severity breakdown</div>{bars}</div>""",
                unsafe_allow_html=True)

    if dets:
        rows = ""
        for d in sorted(dets, key=lambda x: -x["confidence"]):
            rows += f"""<div class="det"><div><div class="name">{d['type']}</div>
            <div class="meta">{d['location'].title()} · {d['coverage']:.1f}% of image</div></div>
            <span class="chip">{d['confidence'] * 100:.0f}%</span></div>"""
        st.markdown(f"""<div class="card"><div class="label">Detected damage</div>{rows}</div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div class="card"><div class="label">Detected damage</div>
        <p style="color:#8E9BB3;margin-top:.5rem">No damage areas found at this sensitivity.
        Try lowering it in the sidebar.</p></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="card"><div class="label">Recommended action</div>
      <p style="margin-top:.5rem;color:#C9D2E3">{report.SEVERITY_INFO[sev]['advice']}</p></div>""",
                unsafe_allow_html=True)

# PDF
pdf_bytes = report.build_pdf(image, result["annotated_image"], result, dets)
st.download_button("⬇  Download PDF assessment report", data=pdf_bytes,
                   file_name=f"damage_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                   mime="application/pdf", type="primary")

st.caption("AI-assisted estimate. Not a substitute for a professional inspection.")